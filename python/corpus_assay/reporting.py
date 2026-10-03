import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corpus_assay.attribution import AttributionIndex
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.fingerprint import library_version
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.schema import ScanSummary
from corpus_assay.schemas import RunStopGramsInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReportOutputLayout:
    out_dir: Path

    @property
    def report_json(self) -> Path:
        return self.out_dir / "report.json"

    @property
    def report_md(self) -> Path:
        return self.out_dir / "report.md"

    @property
    def leak_by_benchmark_json(self) -> Path:
        return self.out_dir / "leak_by_benchmark.json"

    @property
    def leak_by_benchmark_csv(self) -> Path:
        return self.out_dir / "leak_by_benchmark.csv"

    @property
    def leak_by_shard_json(self) -> Path:
        return self.out_dir / "leak_by_shard.json"

    @property
    def leak_by_shard_csv(self) -> Path:
        return self.out_dir / "leak_by_shard.csv"

    @property
    def leak_by_item_json(self) -> Path:
        return self.out_dir / "leak_by_item.json"

    @property
    def leak_by_item_csv(self) -> Path:
        return self.out_dir / "leak_by_item.csv"

    def ensure_dir(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    config: ScanConfig
    library_version: str | None = None
    stopgrams: RunStopGramsInfo | None = None


class FinalSummary(BaseModel):
    total_files_scanned: int = 0
    total_docs_scanned: int = 0
    total_contaminated: int = 0
    contamination_rate: float = 0.0
    total_empty_after_filter: int = 0
    failed_files: int = 0
    total_files: int = 0


class ReportScanSummary(BaseModel):
    total_files: int
    ok_files: int
    failed_files: int
    total_docs_scanned: int
    total_contaminated: int
    contamination_rate: float
    total_empty_after_filter: int


class LeakByBenchmarkRow(BaseModel):
    source_id: int
    benchmark: str
    total_hashes_in_index: int
    leaked_hashes: int
    leak_fraction: float


class LeakByItemRow(BaseModel):
    """Exact per-item leak: how many of a protected item's index hashes appeared
    anywhere in the scanned corpus, split by the hashes' item fan-out. Exhaustive
    (computed from the full hits.bin leaked-hash set joined to the static `.items`
    sidecar), not a sample. Only items with at least one leaked hash are emitted."""

    item_idx: int
    item_id: str
    benchmark: str
    source_id: int
    n_protected_hashes: int
    leaked_hashes: int
    leak_fraction: float
    leaked_unique: int
    leaked_shared: int
    leaked_boilerplate: int


class ShardLeakRow(BaseModel):
    shard: str
    hits_file: str
    summary_file: str | None
    sample_docs_file: str | None
    docs_scanned: int
    docs_contaminated: int
    contamination_rate: float
    unique_leaked_hashes_total: int
    unique_leaked_hashes_per_1k_docs: float
    unique_leaked_hashes_by_benchmark: dict[str, int]


class BadShards(BaseModel):
    top_overall: list[ShardLeakRow]
    top_by_benchmark: dict[str, list[ShardLeakRow]]


class ExamplesSection(BaseModel):
    num_sample_files_found: int
    example_files: list[str]


class IndexBenchmarkInfo(BaseModel):
    source_id: int
    benchmark: str
    total_hashes_in_index: int


class IndexInfo(BaseModel):
    index_path: str
    ngram: int | None
    benchmarks: list[IndexBenchmarkInfo]


class RunInfo(BaseModel):
    config: ScanConfig


class ReportPayload(BaseModel):
    report_version: int
    created_at: str
    library_version: str
    scan_dir: str
    run: RunInfo
    index: IndexInfo
    scan_summary: ReportScanSummary
    leak_by_benchmark: list[LeakByBenchmarkRow]
    # Exact per-item leak, populated when the index has an `.items` sidecar; empty
    # otherwise. Ordered by leak fraction descending.
    leak_by_item: list[LeakByItemRow] = Field(default_factory=list)
    bad_shards: BadShards
    examples: ExamplesSection
    warnings: list[str]
    stopgrams: dict[str, Any] | None = None


def _popcount(buf: bytearray) -> int:
    """Exact number of set bits in a bitmap (one bit per index position)."""
    return int.from_bytes(buf, "little").bit_count()


def _relative_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _index_sidecar(index_path: Path, suffix: str) -> Path:
    return index_path.with_name(f"{index_path.name}{suffix}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_index_path(scan_dir: Path, raw_index_path: str) -> Path:
    index_path = Path(raw_index_path).expanduser()
    if index_path.is_absolute():
        if index_path.exists():
            return index_path.resolve()
    else:
        if index_path.exists():
            return index_path.resolve()
        scan_relative = scan_dir / index_path
        if scan_relative.exists():
            return scan_relative.resolve()

    raise FileNotFoundError(
        "Index path not found. Tried:\n"
        f"  - {index_path.resolve()}\n"
        f"  - {(scan_dir / index_path).resolve()}"
    )


def _load_run_manifest(layout: OutputLayout) -> RunManifest:
    return RunManifest.model_validate_json(
        layout.run_manifest_json.read_text(encoding="utf-8")
    )


def _load_scan_summary(
    layout: OutputLayout,
    summary_paths: list[Path],
) -> ReportScanSummary:
    if layout.final_summary_json.exists():
        final_summary = FinalSummary.model_validate_json(
            layout.final_summary_json.read_text(encoding="utf-8")
        )
        total_files = (
            final_summary.total_files
            if final_summary.total_files
            else final_summary.total_files_scanned + final_summary.failed_files
        )
        return ReportScanSummary(
            total_files=total_files,
            ok_files=final_summary.total_files_scanned,
            failed_files=final_summary.failed_files,
            total_docs_scanned=final_summary.total_docs_scanned,
            total_contaminated=final_summary.total_contaminated,
            contamination_rate=final_summary.contamination_rate,
            total_empty_after_filter=final_summary.total_empty_after_filter,
        )

    total_files = len(summary_paths)
    ok_files = 0
    failed_files = 0
    total_docs_scanned = 0
    total_contaminated = 0
    total_empty_after_filter = 0

    for path in summary_paths:
        summary = ScanSummary.model_validate_json(path.read_text(encoding="utf-8"))
        if summary.error:
            failed_files += 1
            continue
        ok_files += 1
        total_docs_scanned += int(summary.scanned)
        total_contaminated += int(summary.contaminated)
        total_empty_after_filter += int(summary.empty_after_filter)

    contamination_rate = (
        (total_contaminated / total_docs_scanned) if total_docs_scanned else 0.0
    )
    return ReportScanSummary(
        total_files=total_files,
        ok_files=ok_files,
        failed_files=failed_files,
        total_docs_scanned=total_docs_scanned,
        total_contaminated=total_contaminated,
        contamination_rate=contamination_rate,
        total_empty_after_filter=total_empty_after_filter,
    )


def _load_index_metadata(
    index_path: Path,
    *,
    ngram_fallback: int | None,
) -> tuple[list[str], dict[int, int], int | None, dict[int, int]]:
    attr_path = _index_sidecar(index_path, ".attr")
    sources_path = _index_sidecar(index_path, ".sources.json")
    meta_path = _index_sidecar(index_path, ".meta.json")

    for path in (attr_path, sources_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing index sidecar: {path}")

    legend = AttributionIndex.load_legend(str(sources_path))
    id_to_name = legend.id_to_name or []

    # Read the attribution sidecar once. It stores every unique index hash with its
    # source mask. That gives us both the exact per-source totals (denominators) and a
    # stable position for each index hash (hash_to_pos) -- the position space for the
    # exact leak-fraction bitmap in _collect_hits. Positions come from enumeration order,
    # so correctness does not depend on the sidecar being sorted; any injective map works.
    hash_to_pos: dict[int, int] = {}
    totals: dict[int, int] = defaultdict(int)
    for _hash, mask in AttributionIndex.read_hash_mask_pairs(
        str(attr_path), expected_file_type=AttributionIndex.ATTR_FILE_TYPE
    ):
        hash_to_pos[_hash] = len(hash_to_pos)
        for sid in AttributionIndex.iter_mask_source_ids(mask):
            totals[sid] += 1
    total_per_src = dict(totals)

    meta = _load_json(meta_path)
    ngram = meta.get("ngram", ngram_fallback)
    return id_to_name, total_per_src, ngram, hash_to_pos


def _collect_hits(
    layout: OutputLayout,
    *,
    scan_dir: Path,
    hash_to_pos: dict[int, int],
) -> tuple[list[ShardLeakRow], dict[int, bytearray], set[int]]:
    # Exact leak fraction via a bitmap over index positions. Each index hash has a stable
    # position (hash_to_pos, an O(1) lookup); one bit per position per source, distinct
    # matches = popcount. Setting bits into the shared per-source bytearray across every
    # shard is the cross-shard OR-merge, so the count is exact regardless of sharding.
    n_bytes = (len(hash_to_pos) + 7) // 8
    bitmap_by_source: dict[int, bytearray] = defaultdict(lambda: bytearray(n_bytes))

    shard_records: list[ShardLeakRow] = []
    # Exhaustive set of distinct index hashes that leaked anywhere in the corpus,
    # de-duplicated across shards. Joined to the static `.items` sidecar downstream to
    # produce the exact per-item leak rollup.
    leaked_hashes: set[int] = set()

    hits_files = sorted(layout.hits_dir.glob("*.hits.bin"))
    if not hits_files:
        raise FileNotFoundError(f"No *.hits.bin files found in {layout.hits_dir}")

    for hits_file in hits_files:
        shard_name = hits_file.name.replace(".hits.bin", "")
        unique_total = 0
        by_benchmark: dict[int, int] = defaultdict(int)

        for h, mask in AttributionIndex.read_hash_mask_pairs(
            str(hits_file), expected_file_type=AttributionIndex.HITS_FILE_TYPE
        ):
            unique_total += 1
            leaked_hashes.add(h)
            pos = hash_to_pos.get(h)
            for sid in AttributionIndex.iter_mask_source_ids(mask):
                by_benchmark[sid] += 1
                if pos is not None:
                    bitmap_by_source[sid][pos >> 3] |= 1 << (pos & 7)

        summary_path = layout.summary_path(shard_name)
        if summary_path.exists():
            summary = ScanSummary.model_validate_json(
                summary_path.read_text(encoding="utf-8")
            )
            docs_scanned = int(summary.scanned)
            docs_contaminated = int(summary.contaminated)
            contamination_rate = summary.contam_rate
        else:
            docs_scanned = 0
            docs_contaminated = 0
            contamination_rate = 0.0
        unique_per_1k = (unique_total / docs_scanned * 1000) if docs_scanned else 0.0
        sample_path = layout.contaminated_docs_path(shard_name)
        sample_file = (
            _relative_path(sample_path, scan_dir) if sample_path.exists() else None
        )

        shard_records.append(
            ShardLeakRow(
                shard=shard_name,
                hits_file=_relative_path(hits_file, scan_dir),
                summary_file=_relative_path(summary_path, scan_dir)
                if summary_path.exists()
                else None,
                sample_docs_file=sample_file,
                docs_scanned=docs_scanned,
                docs_contaminated=docs_contaminated,
                contamination_rate=contamination_rate,
                unique_leaked_hashes_total=unique_total,
                unique_leaked_hashes_per_1k_docs=unique_per_1k,
                unique_leaked_hashes_by_benchmark={
                    str(sid): count for sid, count in sorted(by_benchmark.items())
                },
            )
        )

    return shard_records, bitmap_by_source, leaked_hashes


def _build_leak_by_benchmark(
    *,
    bitmap_by_source: dict[int, bytearray],
    total_per_src: dict[int, int],
    id_to_name: list[str],
) -> list[LeakByBenchmarkRow]:
    leaked_per_src = {sid: _popcount(bm) for sid, bm in bitmap_by_source.items()}

    rows: list[LeakByBenchmarkRow] = []
    all_sids = set(total_per_src.keys()) | set(leaked_per_src.keys())
    for sid in sorted(all_sids):
        name = id_to_name[sid] if sid < len(id_to_name) else f"sid_{sid}"
        denom = int(total_per_src.get(sid, 0))
        leaked = int(leaked_per_src.get(sid, 0))
        leak_fraction = (leaked / denom) if denom else 0.0
        rows.append(
            LeakByBenchmarkRow(
                source_id=sid,
                benchmark=name,
                total_hashes_in_index=denom,
                leaked_hashes=leaked,
                leak_fraction=leak_fraction,
            )
        )

    rows.sort(key=lambda row: (-row.leak_fraction, row.benchmark))
    return rows


def _load_item_metadata(index_path: Path) -> list[dict[str, Any]] | None:
    """Per-item metadata from the `.items.meta.json` sidecar, or None if the index
    has no item postings. Each entry is `{benchmark, item_id, n_protected_hashes}`,
    positionally indexed by item_idx."""
    items_meta_path = _index_sidecar(index_path, ".items.meta.json")
    items_path = _index_sidecar(index_path, ".items")
    # Both sidecars are required for the rollup: the meta gives denominators, the
    # `.items` binary gives per-hash item postings. Bail if either is missing.
    if not items_meta_path.exists() or not items_path.exists():
        return None
    return _load_json(items_meta_path).get("items", [])


def _build_leak_by_item(
    *,
    index_path: Path,
    leaked_hashes: set[int],
    items_meta: list[dict[str, Any]],
    id_to_name: list[str],
    boilerplate_threshold: int,
) -> list[LeakByItemRow]:
    # Exact, exhaustive per-item leak. We stream the static `.items` sidecar and keep
    # only hashes that actually leaked (the exhaustive hits.bin set from _collect_hits),
    # so per-item counts are exact -- independent of the contaminated-docs sample cap.
    # Each leaked hash's class is fixed by its global item fan-out: fan-out 1 -> unique,
    # >= boilerplate_threshold -> boilerplate, else shared (mirrors the runtime split in
    # the Rust scanner).
    items_path = _index_sidecar(index_path, ".items")
    name_to_id = {name: sid for sid, name in enumerate(id_to_name)}

    leaked: dict[int, int] = defaultdict(int)
    leaked_unique: dict[int, int] = defaultdict(int)
    leaked_shared: dict[int, int] = defaultdict(int)
    leaked_boiler: dict[int, int] = defaultdict(int)

    for h, item_idxs in AttributionIndex.read_items_sidecar(str(items_path)):
        if h not in leaked_hashes:
            continue
        fan_out = len(item_idxs)
        if fan_out <= 1:
            bucket = leaked_unique
        elif fan_out >= boilerplate_threshold:
            bucket = leaked_boiler
        else:
            bucket = leaked_shared
        for it in item_idxs:
            leaked[it] += 1
            bucket[it] += 1

    rows: list[LeakByItemRow] = []
    for it, n_leaked in leaked.items():
        meta = items_meta[it] if 0 <= it < len(items_meta) else {}
        benchmark = str(meta.get("benchmark", ""))
        n_protected = int(meta.get("n_protected_hashes", 0))
        rows.append(
            LeakByItemRow(
                item_idx=it,
                item_id=str(meta.get("item_id", it)),
                benchmark=benchmark,
                source_id=name_to_id.get(benchmark, -1),
                n_protected_hashes=n_protected,
                leaked_hashes=n_leaked,
                leak_fraction=(n_leaked / n_protected) if n_protected else 0.0,
                leaked_unique=leaked_unique.get(it, 0),
                leaked_shared=leaked_shared.get(it, 0),
                leaked_boilerplate=leaked_boiler.get(it, 0),
            )
        )

    rows.sort(
        key=lambda r: (-r.leak_fraction, -r.leaked_hashes, r.benchmark, r.item_id)
    )
    return rows


def _build_bad_shards(
    *,
    shard_rows: list[ShardLeakRow],
    id_to_name: list[str],
    topk_overall: int,
    topk_per_benchmark: int,
) -> BadShards:
    top_overall = sorted(
        shard_rows,
        key=lambda row: (
            -row.unique_leaked_hashes_total,
            -row.unique_leaked_hashes_per_1k_docs,
            row.shard,
        ),
    )[:topk_overall]

    top_by_benchmark: dict[str, list[ShardLeakRow]] = {}
    all_sids = {
        int(sid)
        for row in shard_rows
        for sid in row.unique_leaked_hashes_by_benchmark.keys()
    }

    for sid in sorted(all_sids):
        name = id_to_name[sid] if sid < len(id_to_name) else f"sid_{sid}"
        ranked = sorted(
            shard_rows,
            key=lambda row: (
                -row.unique_leaked_hashes_by_benchmark.get(str(sid), 0),
                row.shard,
            ),
        )
        top_entries = [
            row
            for row in ranked
            if row.unique_leaked_hashes_by_benchmark.get(str(sid), 0)
        ][:topk_per_benchmark]
        top_by_benchmark[name] = top_entries

    return BadShards(top_overall=top_overall, top_by_benchmark=top_by_benchmark)


def _build_examples(layout: OutputLayout, *, scan_dir: Path) -> ExamplesSection:
    sample_files = sorted(layout.contaminated_dir.glob("*.decontam.sample.ndjson"))
    example_files = [_relative_path(path, scan_dir) for path in sample_files[:10]]
    return ExamplesSection(
        num_sample_files_found=len(sample_files),
        example_files=example_files,
    )


def _render_markdown_report(
    report: ReportPayload,
    *,
    top_benchmark_rows: list[LeakByBenchmarkRow],
    top_shards: list[ShardLeakRow],
    top_by_benchmark: dict[str, list[ShardLeakRow]],
    example_files: list[str],
    benchmark_filter: set[str],
) -> str:
    run_cfg = report.run.config
    scan_summary = report.scan_summary
    index_info = report.index
    benchmark_to_id = {
        entry.benchmark: entry.source_id for entry in report.index.benchmarks
    }

    if report.stopgrams:
        stopgrams_line = "Stop-grams: enabled"
        details = []
        tau = report.stopgrams.get("tau")
        if tau is not None:
            details.append(f"tau={tau}")
        df_threshold = report.stopgrams.get("df_threshold")
        if df_threshold is not None:
            details.append(f"df_threshold={df_threshold}")
        stopgram_count = report.stopgrams.get("stopgram_count")
        if stopgram_count is not None:
            details.append(f"count={stopgram_count}")
        path = report.stopgrams.get("path")
        if path:
            details.append(f"path={path}")
        if details:
            stopgrams_line = f"{stopgrams_line}, " + ", ".join(details)
    else:
        stopgrams_line = "Stop-grams: disabled"

    lines = [
        "# Decontamination Report",
        "",
        "## Run overview",
        f"- Scan dir: `{report.scan_dir}`",
        f"- Index path: `{index_info.index_path}`",
        f"- N-gram: `{run_cfg.ngram}`",
        f"- Min hits: `{run_cfg.min_hits}`",
        f"- Min coverage: `{run_cfg.min_coverage}`",
        f"- {stopgrams_line}",
        "",
        "## Scan summary",
        f"- Total files: `{scan_summary.total_files}`",
        f"- OK files: `{scan_summary.ok_files}`",
        f"- Failed files: `{scan_summary.failed_files}`",
        f"- Docs scanned: `{scan_summary.total_docs_scanned}`",
        f"- Docs contaminated: `{scan_summary.total_contaminated}`",
        f"- Contamination rate: `{scan_summary.contamination_rate:.4%}`",
        f"- Docs empty after stop-grams: `{scan_summary.total_empty_after_filter}`",
        "",
        "## Leak by benchmark (top 10)",
        "",
        "| Benchmark | Leaked hashes | Total hashes in index | Leak fraction |",
        "| --- | --- | --- | --- |",
    ]

    for row in top_benchmark_rows:
        lines.append(
            "| {benchmark} | {leaked} | {total} | {fraction:.4%} |".format(
                benchmark=row.benchmark,
                leaked=row.leaked_hashes,
                total=row.total_hashes_in_index,
                fraction=row.leak_fraction,
            )
        )

    lines.extend(
        [
            "",
            "See `leak_by_benchmark.csv` for the full table.",
            "",
        ]
    )

    if report.leak_by_item:
        lines.extend(
            [
                "## Leak by item (top 10)",
                "",
                "Exact per-item leak (item hashes seen in the corpus / item hashes in "
                "the index), split by hash fan-out. Exhaustive, not sampled.",
                "",
                "| Item | Benchmark | Leaked | Protected | Leak fraction | "
                "Unique/Shared/Boiler |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in report.leak_by_item[:10]:
            lines.append(
                "| {item} | {benchmark} | {leaked} | {protected} | {fraction:.4%} | "
                "{u}/{s}/{b} |".format(
                    item=row.item_id.replace("|", "\\|"),
                    benchmark=row.benchmark.replace("|", "\\|"),
                    leaked=row.leaked_hashes,
                    protected=row.n_protected_hashes,
                    fraction=row.leak_fraction,
                    u=row.leaked_unique,
                    s=row.leaked_shared,
                    b=row.leaked_boilerplate,
                )
            )
        lines.extend(
            [
                "",
                "See `leak_by_item.csv` for the full table.",
                "",
            ]
        )

    lines.extend(
        [
            "## Worst leaking shards (overall, top 10)",
            "",
            "| Shard | Unique leaked hashes | Per 1k docs | Docs scanned | Contamination rate |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for row in top_shards:
        lines.append(
            "| {shard} | {unique} | {per_1k:.2f} | {docs} | {rate:.4%} |".format(
                shard=row.shard,
                unique=row.unique_leaked_hashes_total,
                per_1k=row.unique_leaked_hashes_per_1k_docs,
                docs=row.docs_scanned,
                rate=row.contamination_rate,
            )
        )

    lines.extend(
        [
            "",
            "See `leak_by_shard.csv` for the full table.",
            "",
            "## Worst leaking shards by benchmark (top 5)",
            "",
        ]
    )

    for benchmark, rows in top_by_benchmark.items():
        if not rows:
            continue
        if benchmark not in benchmark_filter:
            continue
        lines.extend(
            [
                f"### {benchmark}",
                "",
                "| Shard | Leaked hashes | Docs scanned | Contamination rate |",
                "| --- | --- | --- | --- |",
            ]
        )
        sid = benchmark_to_id.get(benchmark, 0)
        for row in rows[:5]:
            leaked = row.unique_leaked_hashes_by_benchmark.get(str(sid), 0)
            lines.append(
                "| {shard} | {leaked} | {docs} | {rate:.4%} |".format(
                    shard=row.shard,
                    leaked=leaked,
                    docs=row.docs_scanned,
                    rate=row.contamination_rate,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Examples",
            "Sample contaminated documents live under `contaminated_docs/`.",
            "",
        ]
    )
    if example_files:
        lines.append("Example files:")
        lines.extend([f"- `{path}`" for path in example_files])
    else:
        lines.append("No sample files found.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    if report.warnings:
        lines.extend([f"- {warning}" for warning in report.warnings])
    else:
        lines.append("No warnings.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def _write_report_artifacts(
    *,
    layout: ReportOutputLayout,
    report: ReportPayload,
    leak_by_benchmark: list[LeakByBenchmarkRow],
    leak_by_item: list[LeakByItemRow],
    leak_by_shard: list[ShardLeakRow],
    id_to_name: list[str],
) -> None:
    _write_json(layout.report_json, report.model_dump(mode="json"))
    _write_json(
        layout.leak_by_benchmark_json,
        [row.model_dump(mode="json") for row in leak_by_benchmark],
    )
    _write_json(
        layout.leak_by_shard_json,
        [row.model_dump(mode="json") for row in leak_by_shard],
    )
    _write_json(
        layout.leak_by_item_json,
        [row.model_dump(mode="json") for row in leak_by_item],
    )

    with layout.leak_by_benchmark_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_id",
            "benchmark",
            "total_hashes_in_index",
            "leaked_hashes",
            "leak_fraction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row.model_dump(mode="json") for row in leak_by_benchmark])

    with layout.leak_by_item_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "item_idx",
            "item_id",
            "benchmark",
            "source_id",
            "n_protected_hashes",
            "leaked_hashes",
            "leak_fraction",
            "leaked_unique",
            "leaked_shared",
            "leaked_boilerplate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row.model_dump(mode="json") for row in leak_by_item])

    with layout.leak_by_shard_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "shard",
            "docs_scanned",
            "docs_contaminated",
            "contamination_rate",
            "unique_leaked_hashes_total",
            "unique_leaked_hashes_per_1k_docs",
            "top_benchmarks",
            "hits_file",
            "summary_file",
            "sample_docs_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in leak_by_shard:
            pairs = []
            for sid, count in sorted(
                row.unique_leaked_hashes_by_benchmark.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                sid_int = int(sid)
                name = (
                    id_to_name[sid_int] if sid_int < len(id_to_name) else f"sid_{sid}"
                )
                pairs.append(f"{name}:{count}")
            writer.writerow(
                {
                    "shard": row.shard,
                    "docs_scanned": row.docs_scanned,
                    "docs_contaminated": row.docs_contaminated,
                    "contamination_rate": row.contamination_rate,
                    "unique_leaked_hashes_total": row.unique_leaked_hashes_total,
                    "unique_leaked_hashes_per_1k_docs": row.unique_leaked_hashes_per_1k_docs,
                    "top_benchmarks": ";".join(pairs),
                    "hits_file": row.hits_file,
                    "summary_file": row.summary_file,
                    "sample_docs_file": row.sample_docs_file,
                }
            )

    top_benchmark_names = {row.benchmark for row in leak_by_benchmark[:10]}
    layout.report_md.write_text(
        _render_markdown_report(
            report,
            top_benchmark_rows=leak_by_benchmark[:10],
            top_shards=report.bad_shards.top_overall[:10],
            top_by_benchmark=report.bad_shards.top_by_benchmark,
            example_files=report.examples.example_files,
            benchmark_filter=top_benchmark_names,
        ),
        encoding="utf-8",
    )


def generate_report_from_scan_dir(
    scan_dir: str,
    *,
    out_dir: str | None = None,
    topk_overall: int = 50,
    topk_per_benchmark: int = 20,
) -> ReportPayload:
    scan_dir_path = Path(scan_dir).expanduser().resolve()
    layout = OutputLayout(scan_dir_path)
    layout.validate_layout()

    manifest = _load_run_manifest(layout)
    index_path = _resolve_index_path(scan_dir_path, manifest.config.index_path)

    id_to_name, total_per_src, ngram, hash_to_pos = _load_index_metadata(
        index_path,
        ngram_fallback=manifest.config.ngram,
    )

    summary_paths = sorted(layout.summaries_dir.glob("*.summary.json"))
    scan_summary = _load_scan_summary(layout, summary_paths)

    shard_rows, bitmap_by_source, leaked_hashes = _collect_hits(
        layout,
        scan_dir=scan_dir_path,
        hash_to_pos=hash_to_pos,
    )

    leak_by_benchmark = _build_leak_by_benchmark(
        bitmap_by_source=bitmap_by_source,
        total_per_src=total_per_src,
        id_to_name=id_to_name,
    )

    # Exact per-item leak, only when the index carries item postings.
    items_meta = _load_item_metadata(index_path)
    leak_by_item: list[LeakByItemRow] = []
    if items_meta is not None:
        leak_by_item = _build_leak_by_item(
            index_path=index_path,
            leaked_hashes=leaked_hashes,
            items_meta=items_meta,
            id_to_name=id_to_name,
            boilerplate_threshold=manifest.config.boilerplate_threshold,
        )

    leak_by_shard = sorted(
        shard_rows,
        key=lambda row: (-row.unique_leaked_hashes_total, row.shard),
    )

    bad_shards = _build_bad_shards(
        shard_rows=shard_rows,
        id_to_name=id_to_name,
        topk_overall=topk_overall,
        topk_per_benchmark=topk_per_benchmark,
    )

    examples = _build_examples(layout, scan_dir=scan_dir_path)

    warnings: list[str] = []

    report = ReportPayload(
        # v2: leak fields are exact (leaked_hashes / leak_fraction), replacing the v1
        # HLL++ estimates (leaked_hashes_estimate / leak_fraction_estimate) and dropping
        # the index.hll block.
        report_version=2,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        library_version=library_version(),
        scan_dir=str(scan_dir_path),
        run=RunInfo(config=manifest.config),
        index=IndexInfo(
            index_path=str(index_path),
            ngram=ngram,
            benchmarks=[
                IndexBenchmarkInfo(
                    source_id=row.source_id,
                    benchmark=row.benchmark,
                    total_hashes_in_index=row.total_hashes_in_index,
                )
                for row in leak_by_benchmark
            ],
        ),
        scan_summary=scan_summary,
        leak_by_benchmark=leak_by_benchmark,
        leak_by_item=leak_by_item,
        bad_shards=bad_shards,
        examples=examples,
        warnings=warnings,
        stopgrams=manifest.stopgrams.model_dump(mode="json")
        if manifest.stopgrams
        else None,
    )

    report_layout = ReportOutputLayout(
        Path(out_dir).expanduser().resolve() if out_dir else scan_dir_path / "report"
    )
    report_layout.ensure_dir()
    _write_report_artifacts(
        layout=report_layout,
        report=report,
        leak_by_benchmark=leak_by_benchmark,
        leak_by_item=leak_by_item,
        leak_by_shard=leak_by_shard,
        id_to_name=id_to_name,
    )

    logger.info("Wrote report to %s", report_layout.out_dir)
    return report


__all__ = ["generate_report_from_scan_dir", "ReportOutputLayout", "ReportPayload"]
