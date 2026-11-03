"""Write decontaminated copies of scanned shards (the ``filter`` command).

The scan runs first, uncapped, into ``<out_dir>/_scan/`` (kept as evidence). Every
flagged sub-document record carries the row it came from (``row_index``) and
where in the row it is (``field``, ``turn``, ``part``), so the shards are
filtered without parsing doc ids:

- ``record`` granularity drops every row with a flagged sub-document;
- ``subdoc`` granularity removes just the flagged sub-documents: flagged packed
  parts are dropped and the rest rejoined with the separator, a text field left
  with nothing is set to ``""``, and a chat turn left with nothing is removed
  from the messages list. A row left with no text at all is dropped.

Cleaned shards keep the input format at the input's path relative to the
common directory of all inputs. Parquet keeps the schema (and its metadata)
and every column; JSONL keeps every line it does not remove byte for byte
(blank lines included), and its compression.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from corpus_assay.attribution import AttributionIndex
from corpus_assay.config import ScanConfig
from corpus_assay.constants import ITEMS_TEXT_SUFFIX
from corpus_assay.scanner.fingerprint import (
    _file_sha256,
    library_version,
    rust_extension_version,
)
from corpus_assay.scanner.io import (
    build_index_info,
    build_scan_config_info,
    parse_contamination_record,
    run_fingerprint_of,
)
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.runner import collect_files
from corpus_assay.scanner.schema import ContaminationRecord, RunSummary
from corpus_assay.schemas import (
    FILTER_MANIFEST_SCHEMA_VERSION,
    FilterCounts,
    FilterManifest,
    FilterRemovals,
    FilterShard,
)
from corpus_assay.schemas.filter_manifest import FilterGranularity

FILTER_MANIFEST_NAME = "_FILTER_MANIFEST.json"
REMOVED_IDS_NAME = "removed_ids.ndjson"
DATASET_CARD_SNIPPET_NAME = "DATASET_CARD_SNIPPET.md"
SCAN_DIR_NAME = "_scan"

# Whitespace pyarrow's JSON reader skips as a blank line (the JSON whitespace).
_JSON_WS = b" \t\r\n"


class FilterError(ValueError):
    """The filter cannot run as asked (bad output directory, failed scan...)."""


# -----------------------------
# Output directory and shard paths
# -----------------------------


@dataclass(frozen=True)
class FilterPlan:
    """Input shards and where their cleaned copies go."""

    out_dir: Path
    files: list[Path]
    # Input shard -> its path relative to the output directory.
    outputs: dict[Path, Path]

    @property
    def scan_dir(self) -> Path:
        return self.out_dir / SCAN_DIR_NAME


def _shard_format(path: Path) -> str:
    for fmt in ("parquet", "jsonl.gz", "jsonl.zst", "jsonl"):
        if path.name.endswith(f".{fmt}"):
            return fmt
    raise FilterError(f"Unsupported shard format: {path}")


def plan_filter(inputs: list[str], out_dir: Path, *, overwrite: bool) -> FilterPlan:
    """Resolve the input shards and check the output directory.

    Refuses a non-empty ``out_dir`` unless ``overwrite`` (then the outputs of a
    previous filter run there are removed first), an ``out_dir`` inside an
    input directory or holding an input shard, and shards whose names would
    share scan outputs (the scan keys them by file stem).
    """
    files = collect_files(inputs)
    out_dir = out_dir.resolve()

    stems: dict[str, Path] = {}
    for path in files:
        stem = OutputLayout.stem_from_path(path)
        if stem in stems:
            raise FilterError(
                f"Shards {stems[stem]} and {path} share the stem '{stem}', so their "
                "scan outputs would collide; filter them in separate runs."
            )
        stems[stem] = path

    for raw in inputs:
        given = Path(raw)
        if given.is_dir() and out_dir.is_relative_to(given.resolve()):
            raise FilterError(
                f"--out-dir {out_dir} is inside the input directory {given}; "
                "a later scan of that directory would pick up the cleaned shards."
            )
    inside = [path for path in files if path.is_relative_to(out_dir)]
    if inside:
        raise FilterError(
            f"--out-dir {out_dir} holds input shards (e.g. {inside[0]}); "
            "choose another output directory."
        )

    base = Path(os.path.commonpath([path.parent for path in files]))
    outputs = {path: path.relative_to(base) for path in files}
    reserved = {
        SCAN_DIR_NAME,
        FILTER_MANIFEST_NAME,
        REMOVED_IDS_NAME,
        DATASET_CARD_SNIPPET_NAME,
    }
    clash = [rel for rel in outputs.values() if rel.parts[0] in reserved]
    if clash:
        raise FilterError(
            f"The cleaned shard {clash[0]} would land on a name the filter writes "
            f"itself ({', '.join(sorted(reserved))}); rename or move it."
        )

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FilterError(
                f"--out-dir {out_dir} is not empty; pass --overwrite to replace a "
                "previous filter run there."
            )
        _remove_previous_outputs(out_dir, outputs.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    return FilterPlan(out_dir=out_dir, files=files, outputs=outputs)


def _remove_previous_outputs(out_dir: Path, outputs: Iterable[Path]) -> None:
    """Remove what a filter run writes: its scan directory, manifest, removed
    ids, card snippet and cleaned shards (those the old manifest lists and
    those this run is about to write). Other files are left alone."""
    stale: set[Path] = {out_dir / rel for rel in outputs}
    manifest_path = out_dir / FILTER_MANIFEST_NAME
    if manifest_path.exists():
        try:
            old = FilterManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            stale.update(out_dir / shard.output_path for shard in old.shards)
        except (OSError, ValidationError):
            pass
    for name in (FILTER_MANIFEST_NAME, REMOVED_IDS_NAME, DATASET_CARD_SNIPPET_NAME):
        stale.add(out_dir / name)
    for path in stale:
        if path.resolve().is_relative_to(out_dir):
            path.unlink(missing_ok=True)
    shutil.rmtree(out_dir / SCAN_DIR_NAME, ignore_errors=True)


# -----------------------------
# Flags and their benchmarks
# -----------------------------


class BenchmarkNames:
    """Benchmark names for a record's evidence, from the index sidecars."""

    def __init__(self, index_path: Path) -> None:
        self.sources: list[str] = []
        legend = index_path.with_name(f"{index_path.name}.sources.json")
        if legend.exists():
            self.sources = list(AttributionIndex.load_legend(str(legend)).id_to_name)
        self.items: list[str | None] = _item_benchmarks(index_path)

    def _item(self, item: int) -> str | None:
        return self.items[item] if 0 <= item < len(self.items) else None

    def of(self, rec: ContaminationRecord) -> list[str]:
        names: set[str] = set()
        for hit in rec.src_hits:
            if hit.hits > 0:
                names.add(
                    self.sources[hit.source_id]
                    if hit.source_id < len(self.sources)
                    else f"source#{hit.source_id}"
                )
        items = [hit.item_id for hit in rec.item_hits]
        items += [hit.item_id for hit in rec.exact_hits]
        if rec.spaced_item is not None:
            items.append(rec.spaced_item)
        for item in items:
            name = self._item(item)
            if name:
                names.add(name)
        return sorted(names)


def _item_benchmarks(index_path: Path) -> list[str | None]:
    """Benchmark of each item, by item index (empty when the index has no item
    sidecar)."""
    meta = index_path.with_name(f"{index_path.name}.items.meta.json")
    if meta.exists():
        try:
            items = json.loads(meta.read_text(encoding="utf-8")).get("items", [])
            return [entry.get("benchmark") for entry in items]
        except (OSError, json.JSONDecodeError):
            pass
    text = index_path.with_name(f"{index_path.name}{ITEMS_TEXT_SUFFIX}")
    out: list[str | None] = []
    if text.exists():
        with text.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                idx = int(entry["item_idx"])
                out.extend([None] * (idx + 1 - len(out)))
                out[idx] = entry.get("benchmark")
    return out


def _read_flags(path: Path) -> list[ContaminationRecord]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [parse_contamination_record(line) for line in handle if line.strip()]


# -----------------------------
# Sub-document removal
# -----------------------------


@dataclass
class _RowFlags:
    """The flagged sub-documents of one row: per (field, turn), the flagged
    packed parts (None: the whole field or turn)."""

    records: list[ContaminationRecord] = field(default_factory=list)
    segments: dict[tuple[str | None, int | None], set[int] | None] = field(
        default_factory=dict
    )

    def add(self, rec: ContaminationRecord) -> None:
        self.records.append(rec)
        key = (rec.field, rec.turn)
        if rec.part is None:
            self.segments[key] = None
        elif key not in self.segments:
            self.segments[key] = {rec.part}
        else:
            parts = self.segments[key]
            if parts is not None:
                parts.add(rec.part)


def _drop_parts(text: Any, parts: set[int] | None, cfg: ScanConfig) -> str:
    """``text`` without its flagged packed parts (``""`` when none is left).

    Splits like the scanner: the typo separator is first replaced by the
    separator, so the rejoined text uses the separator throughout.
    """
    sep = cfg.packed_doc_sep
    if parts is None or sep is None or not isinstance(text, str):
        return ""
    typo = cfg.packed_doc_sep_typo
    if typo and typo in text:
        text = text.replace(typo, sep)
    kept = [part for k, part in enumerate(text.split(sep)) if k not in parts]
    if not any(part.strip() for part in kept):
        return ""
    return sep.join(kept)


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def clean_row(row: dict[str, Any], flags: _RowFlags, cfg: ScanConfig) -> bool:
    """Remove the flagged sub-documents from ``row`` (its scanned columns) in place.

    Returns False when the row has no text left in any scanned column and
    should be dropped.
    """
    first_key = cfg.text_keys[0] if cfg.text_keys else None
    drop_turns: set[int] = set()
    for (column, turn), parts in flags.segments.items():
        column = column or first_key
        if column is None or column not in row:
            continue
        if turn is None:
            row[column] = _drop_parts(row[column], parts, cfg)
            continue
        turns = row[column]
        if not isinstance(turns, list) or not 0 <= turn < len(turns):
            continue
        message = turns[turn]
        content = message.get("content") if isinstance(message, dict) else None
        cleaned = _drop_parts(content, parts, cfg)
        if cleaned:
            message["content"] = cleaned
        else:
            drop_turns.add(turn)
    if drop_turns and cfg.messages_key in row:
        row[cfg.messages_key] = [
            turn for k, turn in enumerate(row[cfg.messages_key]) if k not in drop_turns
        ]

    if any(_has_text(row.get(key)) for key in cfg.text_keys):
        return True
    turns = row.get(cfg.messages_key) if cfg.messages_key else None
    if isinstance(turns, list):
        return any(
            isinstance(turn, dict) and _has_text(turn.get("content")) for turn in turns
        )
    return False


# -----------------------------
# Shard writers
# -----------------------------


@dataclass
class _ShardResult:
    rows_before: int
    rows_after: int
    # Rows dropped from the output, and rows kept with sub-documents removed.
    removed_rows: set[int]
    modified_rows: set[int]


def _parquet_compression(path: Path) -> str:
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).metadata
    if meta.num_row_groups == 0 or meta.num_columns == 0:
        return "snappy"
    codec = str(meta.row_group(0).column(0).compression).lower()
    return "none" if codec == "uncompressed" else codec


def _rebuild_column(values: list[Any], dtype: Any) -> Any:
    import pyarrow as pa

    try:
        return pa.array(values, type=dtype)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        return pa.array(values).cast(dtype)


def filter_parquet(
    src: Path,
    dst: Path,
    flags: dict[int, _RowFlags],
    cfg: ScanConfig,
    granularity: FilterGranularity,
) -> _ShardResult:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(src)
    schema = source.schema_arrow
    scanned = [key for key in [*cfg.text_keys, cfg.messages_key] if key]
    removed: set[int] = set()
    modified: set[int] = set()
    rows_before = 0
    rows_after = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(dst, schema, compression=_parquet_compression(src)) as out:
        for batch in source.iter_batches(batch_size=65536):
            start = rows_before
            rows_before += batch.num_rows
            hit_rows = [r for r in range(start, rows_before) if r in flags]
            if not hit_rows:
                out.write_batch(batch)
                rows_after += batch.num_rows
                continue
            keep = [True] * batch.num_rows
            if granularity == "record":
                for row in hit_rows:
                    keep[row - start] = False
                    removed.add(row)
            else:
                columns = {
                    name: batch.column(name).to_pylist()
                    for name in scanned
                    if name in schema.names
                }
                for row in hit_rows:
                    values = {name: col[row - start] for name, col in columns.items()}
                    if clean_row(values, flags[row], cfg):
                        modified.add(row)
                        for name, value in values.items():
                            columns[name][row - start] = value
                    else:
                        keep[row - start] = False
                        removed.add(row)
                arrays = [
                    _rebuild_column(columns[name], schema.field(name).type)
                    if name in columns
                    else batch.column(name)
                    for name in schema.names
                ]
                batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            kept = batch.filter(pa.array(keep))
            out.write_batch(kept)
            rows_after += kept.num_rows
    return _ShardResult(rows_before, rows_after, removed, modified)


def _line_ending(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return b"\r\n"
    return b"\n" if line.endswith(b"\n") else b""


def filter_jsonl(
    src: Path,
    dst: Path,
    flags: dict[int, _RowFlags],
    cfg: ScanConfig,
    granularity: FilterGranularity,
) -> _ShardResult:
    """Copy a JSONL shard line by line, dropping (or cleaning) flagged rows.

    Row ``k`` is the ``k``-th non-blank line, as pyarrow (and so the scan)
    numbers them. Kept lines, blank ones included, are written unchanged;
    a cleaned line is re-serialized. Compression follows the file extension.
    """
    import pyarrow as pa

    removed: set[int] = set()
    modified: set[int] = set()
    row = -1
    rows_after = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with (
        pa.input_stream(str(src)) as raw_in,
        pa.output_stream(str(dst)) as out,
    ):
        for line in io.BufferedReader(raw_in):  # type: ignore[arg-type]
            if not line.strip(_JSON_WS):
                out.write(line)
                continue
            row += 1
            row_flags = flags.get(row)
            if row_flags is None:
                out.write(line)
                rows_after += 1
                continue
            if granularity == "record":
                removed.add(row)
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or not clean_row(obj, row_flags, cfg):
                removed.add(row)
                continue
            modified.add(row)
            rows_after += 1
            out.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
            out.write(_line_ending(line))
    return _ShardResult(row + 1, rows_after, removed, modified)


# -----------------------------
# The filter run
# -----------------------------


def _counts(counts: dict[str, FilterCounts], name: str) -> FilterCounts:
    if name not in counts:
        counts[name] = FilterCounts()
    return counts[name]


def write_filtered(
    plan: FilterPlan,
    cfg: ScanConfig,
    run: RunSummary,
    *,
    granularity: FilterGranularity,
    config_path: str | None = None,
) -> FilterManifest:
    """Write the cleaned shards, ``removed_ids.ndjson``, the dataset card
    snippet and ``_FILTER_MANIFEST.json`` from a finished uncapped scan."""
    if cfg.max_return_records is not None:
        raise FilterError("filter needs an uncapped scan (max_return_records=None)")
    failed = [s for s in run.per_file if s.error is not None]
    if failed:
        details = "\n".join(f"  {s.file}: {s.error}" for s in failed)
        raise FilterError(
            "Some shards failed to scan, so no cleaned data was written:\n" + details
        )
    summaries = {s.file: s for s in run.per_file}
    layout = OutputLayout(Path(cfg.out_dir))
    index_path = Path(cfg.index_path)
    names = BenchmarkNames(index_path)

    shards: list[FilterShard] = []
    totals = FilterRemovals()
    by_benchmark: dict[str, FilterCounts] = {}
    by_channel: dict[str, int] = {}
    removed_ids_path = plan.out_dir / REMOVED_IDS_NAME
    with removed_ids_path.open("w", encoding="utf-8") as removed_ids:
        for src in plan.files:
            rel = plan.outputs[src]
            stem = OutputLayout.stem_from_path(src)
            records = _read_flags(layout.contaminated_docs_path(stem))
            summary = summaries.get(src.name)
            expected = summary.contaminated if summary is not None else len(records)
            if len(records) != expected:
                raise FilterError(
                    f"{src}: the scan flagged {expected} sub-documents but wrote "
                    f"{len(records)} records; rerun the filter."
                )
            flags: dict[int, _RowFlags] = {}
            for rec in records:
                if rec.row_index is None:
                    raise FilterError(
                        f"{src}: a flagged record has no row_index (the scan "
                        f"outputs in {layout.out_dir} predate it); rerun the filter."
                    )
                flags.setdefault(rec.row_index, _RowFlags()).add(rec)

            fmt = _shard_format(src)
            dst = plan.out_dir / rel
            writer = filter_parquet if fmt == "parquet" else filter_jsonl
            result = writer(src, dst, flags, cfg, granularity)
            bad_rows = [r for r in flags if r >= result.rows_before]
            if bad_rows:
                raise FilterError(
                    f"{src}: flagged row {bad_rows[0]} is past the shard's "
                    f"{result.rows_before} rows; the shard changed during the run?"
                )

            shard_rel = rel.as_posix()
            for row in sorted(flags):
                row_flags = flags[row]
                row_removed = row in result.removed_rows
                row_benchmarks: set[str] = set()
                for rec in row_flags.records:
                    benchmarks = names.of(rec)
                    row_benchmarks.update(benchmarks)
                    for channel in rec.channels:
                        by_channel[channel] = by_channel.get(channel, 0) + 1
                    for name in benchmarks:
                        _counts(by_benchmark, name).flagged_subdocs += 1
                    entry: dict[str, Any] = {
                        "shard": shard_rel,
                        "row_index": row,
                        "doc_id": rec.doc_id,
                    }
                    for key in ("field", "role", "turn", "part"):
                        value = getattr(rec, key)
                        if value is not None:
                            entry[key] = value
                    entry["benchmarks"] = benchmarks
                    entry["channels"] = list(rec.channels)
                    entry["row_removed"] = row_removed
                    removed_ids.write(json.dumps(entry, ensure_ascii=False) + "\n")
                for name in row_benchmarks:
                    counts = _counts(by_benchmark, name)
                    if row_removed:
                        counts.rows_removed += 1
                    else:
                        counts.rows_modified += 1

            shards.append(
                FilterShard(
                    input_path=str(src),
                    input_sha256=_file_sha256(src),
                    output_path=shard_rel,
                    output_sha256=_file_sha256(dst),
                    format=fmt,
                    rows_before=result.rows_before,
                    rows_after=result.rows_after,
                    rows_removed=len(result.removed_rows),
                    rows_modified=len(result.modified_rows),
                    flagged_subdocs=len(records),
                )
            )
            totals.rows_before += result.rows_before
            totals.rows_after += result.rows_after
            totals.rows_removed += len(result.removed_rows)
            totals.rows_modified += len(result.modified_rows)
            totals.flagged_subdocs += len(records)

    totals.by_benchmark = dict(sorted(by_benchmark.items()))
    totals.by_channel = dict(sorted(by_channel.items()))
    manifest = FilterManifest(
        schema_version=FILTER_MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        library_version=library_version(),
        rust_extension_version=rust_extension_version(),
        granularity=granularity,
        config_path=config_path,
        index=build_index_info(index_path, run_fingerprint_of(run, cfg)),
        scan_config=build_scan_config_info(cfg),
        input_paths=list(cfg.inputs),
        shards=shards,
        removals=totals,
        scan_dir=SCAN_DIR_NAME,
        removed_ids=REMOVED_IDS_NAME,
        dataset_card_snippet=DATASET_CARD_SNIPPET_NAME,
        config=cfg.model_dump(),
    )
    (plan.out_dir / DATASET_CARD_SNIPPET_NAME).write_text(
        dataset_card_snippet(manifest), encoding="utf-8"
    )
    (plan.out_dir / FILTER_MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


# -----------------------------
# Dataset card snippet
# -----------------------------


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("|", "\\|")


def _reproduce_command(manifest: FilterManifest) -> str:
    sc = manifest.scan_config
    args = ["corpus-assay filter"]
    if manifest.config_path:
        args.append(f"--config {manifest.config_path}")
    args.append(f"--index {Path(manifest.index.path).name}")
    args.extend(f"--input {path}" for path in manifest.input_paths)
    args.append("--out-dir <out-dir>")
    if manifest.granularity != "record":
        args.append(f"--granularity {manifest.granularity}")
    keys = sc.text_keys or ([sc.text_key] if sc.text_key else [])
    if keys != ["text"] and keys:
        args.extend(f"--text-key {key}" for key in keys)
    if sc.messages_key:
        args.append(f"--messages-key {sc.messages_key}")
    if sc.id_key:
        args.append(f"--id-key {sc.id_key}")
    if sc.packed_doc_sep is None:
        args.append("--no-split")
    if sc.gate_mode and sc.gate_mode not in ("union", "auto"):
        args.append(f"--gate-mode {sc.gate_mode}")
    if sc.min_longest_run:
        args.append(f"--min-longest-run {sc.min_longest_run}")
    if sc.min_item_coverage:
        args.append(f"--min-item-coverage {sc.min_item_coverage}")
    if sc.exact_channel != "off":
        args.append(f"--exact-channel {sc.exact_channel}")
        args.append(f"--exact-min-chars {sc.exact_min_chars}")
    if sc.stopgrams_path:
        args.append(f"--stopgrams {Path(sc.stopgrams_path).name}")
    return " \\\n  ".join(args)


def dataset_card_snippet(manifest: FilterManifest) -> str:
    """A markdown section for a dataset card, generated from the manifest."""
    idx = manifest.index
    sc = manifest.scan_config
    rem = manifest.removals
    stats = {s.name: s for s in idx.benchmark_stats}
    unit = "rows" if manifest.granularity == "record" else "rows or sub-documents"
    lines = [
        "## Benchmark decontamination",
        "",
        f"This data was decontaminated with "
        f"[corpus-assay](https://github.com/mrcabbage972/corpus-assay) "
        f"{manifest.library_version} (Rust extension "
        f"{manifest.rust_extension_version}) on {manifest.created_at}, by removing "
        f"{unit} that overlap the benchmarks below ({manifest.granularity} "
        "granularity).",
        "",
        f"- Rows: {rem.rows_before:,} before, {rem.rows_after:,} after "
        f"({rem.rows_removed:,} removed"
        + (
            f", {rem.rows_modified:,} with sub-documents removed"
            if manifest.granularity == "subdoc"
            else ""
        )
        + f"); {rem.flagged_subdocs:,} flagged sub-documents.",
        f"- Index: `{Path(idx.path).name}`, SHA-256 `{_cell(idx.native_sha256)}` "
        f"(metadata SHA-256 `{_cell(idx.meta_sha256)}`).",
        f"- Normalization config SHA-256 `{_cell(idx.normalization_config_sha)}`, "
        f"n-gram config hash `{_cell(idx.ngram_config_hash)}`.",
        "",
        "| Benchmark | Dataset | Config | Split | Revision | Resolved revision "
        "| Items | N-grams | Rows removed |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    benchmarks = idx.benchmarks or []
    listed: set[str] = set()
    for b in benchmarks:
        name = b.name or "-"
        # A benchmark with several splits gets one row per split; its counts go
        # on the first.
        first = name not in listed
        listed.add(name)
        stat = stats.get(name) if first else None
        removed: int | None = None
        if first:
            counts = rem.by_benchmark.get(name)
            removed = counts.rows_removed if counts else 0
        lines.append(
            f"| {_cell(name)} | {_cell(b.hf_repo)} | {_cell(b.config_name)} "
            f"| {_cell(b.split)} | {_cell(b.revision)} | {_cell(b.resolved_revision)} "
            f"| {_cell(stat.items if stat else None)} "
            f"| {_cell(stat.ngrams if stat else None)} | {_cell(removed)} |"
        )
    for name, counts in rem.by_benchmark.items():
        if name not in listed:
            lines.append(
                f"| {_cell(name)} | - | - | - | - | - | - | - | {counts.rows_removed} |"
            )
    gate = sc.gate_mode or "union"
    settings = [
        f"n={sc.n}",
        f"min_hits={sc.min_hits}",
        f"min_coverage={sc.min_coverage}",
        f"gate={gate}",
        f"min_item_coverage={sc.min_item_coverage}",
    ]
    if sc.min_longest_run:
        settings.append(f"min_longest_run={sc.min_longest_run}")
    settings.append(f"exact_channel={sc.exact_channel}")
    if sc.exact_channel != "off":
        settings.append(f"exact_min_chars={sc.exact_min_chars}")
    settings.append(
        "packed splitting off"
        if sc.packed_doc_sep is None
        else f"records split on `{sc.packed_doc_sep}`"
    )
    if sc.stopgrams_path:
        settings.append(f"stop-grams {Path(sc.stopgrams_path).name}")
    lines += [
        "",
        "A row counts for every benchmark its flagged text matched, so the "
        "per-benchmark counts can add up to more than the total.",
        "",
        "Matching settings: " + ", ".join(settings) + ".",
        "",
        "To reproduce (the index must have the SHA-256 above):",
        "",
        "```bash",
        _reproduce_command(manifest),
        "```",
        "",
        f"The full record is in `{FILTER_MANIFEST_NAME}`, the removed ids in "
        f"`{REMOVED_IDS_NAME}`, and the scan evidence in `{SCAN_DIR_NAME}/`. "
        "Please cite corpus-assay (DOI "
        "[10.5281/zenodo.23125328](https://doi.org/10.5281/zenodo.23125328)) "
        "with the index SHA-256 when reporting results on this data.",
        "",
    ]
    return "\n".join(lines)
