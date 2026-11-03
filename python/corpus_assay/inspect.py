from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from corpus_assay.attribution import AttributionIndex
from corpus_assay.formats import (
    HEADER_SIZE,
    BinaryHeader,
    FileType,
    read_header_optional,
)
from corpus_assay.scanner.io import parse_contamination_record
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.schemas import (
    DocResult,
    FinalSummaryOutput,
    IndexMeta,
    RunManifest,
    ScanSummaryOutput,
)


def _validate_binary_file(
    path: Path, *, expected_file_type: FileType, record_size: int
) -> tuple[bool, int, BinaryHeader | None]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = read_header_optional(handle, expected_file_type=expected_file_type)

    if header:
        data_size = size - HEADER_SIZE
        if data_size < 0:
            raise ValueError(f"{path} is smaller than header size.")
        if data_size % record_size != 0:
            raise ValueError(
                f"{path} has invalid data size {data_size}, not a multiple of {record_size}."
            )
        if data_size != header.record_count * record_size:
            raise ValueError(
                f"{path} record count mismatch: header says {header.record_count}, "
                f"but size implies {data_size // record_size}."
            )
        return True, data_size // record_size, header

    if size % record_size != 0:
        raise ValueError(
            f"{path} has invalid size {size}, not a multiple of {record_size}."
        )
    return False, size // record_size, None


def _sidecar(index_path: Path, suffix: str) -> Path:
    return index_path.with_name(f"{index_path.name}{suffix}")


def index_benchmark_stats(
    index_path: Path, meta: IndexMeta, *, min_hits: int | None = None
) -> tuple[list[dict[str, object]], int]:
    """Per-benchmark counts for an index, and the min_hits they were counted at.

    Prefers the per-item table in ``.items.meta.json`` (so any ``min_hits`` can be
    applied), then the counts recorded in the meta (schema v2), then the per-source
    totals in ``.sources.json`` (old indexes). Counts that cannot be recovered are
    None.
    """
    recorded = {b.name: b for b in meta.benchmark_stats}
    effective_min_hits = min_hits
    if effective_min_hits is None:
        effective_min_hits = next(
            (b.min_hits for b in meta.benchmark_stats if b.min_hits is not None), 3
        )

    names: list[str] = [b.name for b in meta.benchmark_stats]
    for spec in meta.benchmark_specs:
        name = spec.get("name")
        if isinstance(name, str) and name not in names:
            names.append(name)

    ngrams_by_name: dict[str, int] = {}
    sources_path = _sidecar(index_path, ".sources.json")
    if sources_path.exists():
        legend = AttributionIndex.load_legend(str(sources_path))
        for sid, name in enumerate(legend.id_to_name):
            ngrams_by_name[name] = (legend.per_source_total or {}).get(sid, 0)
            if name not in names:
                names.append(name)

    items_by_name: dict[str, list[int]] | None = None
    items_meta_path = _sidecar(index_path, ".items.meta.json")
    if items_meta_path.exists():
        payload = json.loads(items_meta_path.read_text(encoding="utf-8"))
        items_by_name = {}
        for item in payload.get("items", []):
            name = str(item.get("benchmark"))
            items_by_name.setdefault(name, []).append(
                int(item.get("n_protected_hashes", 0))
            )
            if name not in names:
                names.append(name)

    rows: list[dict[str, object]] = []
    for name in names:
        rec = recorded.get(name)
        row: dict[str, object] = {
            "name": name,
            "items": rec.items if rec else None,
            "ngrams": (
                rec.ngrams
                if rec
                else ngrams_by_name.get(name, 0)
                if sources_path.exists()
                else None
            ),
            "items_zero_ngrams": rec.items_zero_ngrams if rec else None,
            "items_below_min_hits": (
                rec.items_below_min_hits
                if rec and rec.min_hits == effective_min_hits
                else None
            ),
        }
        if items_by_name is not None:
            # The sidecar lists every registered item, including those with no
            # n-grams, so it is authoritative when present.
            counts = items_by_name.get(name, [])
            row["items"] = len(counts)
            row["items_zero_ngrams"] = sum(1 for c in counts if c == 0)
            row["items_below_min_hits"] = sum(
                1 for c in counts if c < effective_min_hits
            )
        rows.append(row)
    return rows, effective_min_hits


def inspect_index(
    index_path: Path, *, min_hits: int | None = None
) -> dict[str, object]:
    meta_path = index_path.with_name(f"{index_path.name}.meta.json")
    attr_path = index_path.with_name(f"{index_path.name}.attr")

    if not index_path.exists():
        raise FileNotFoundError(f"Missing native index: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing index metadata: {meta_path}")
    if not attr_path.exists():
        raise FileNotFoundError(f"Missing attribution sidecar: {attr_path}")

    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = IndexMeta.model_validate(meta_payload)

    native_has_header, native_count, native_header = _validate_binary_file(
        index_path, expected_file_type=FileType.NATIVE_HASHES, record_size=8
    )
    attr_has_header, attr_count, attr_header = _validate_binary_file(
        attr_path, expected_file_type=FileType.ATTR_MASKS, record_size=16
    )
    if native_header and native_header.ngram_n and native_header.ngram_n != meta.ngram:
        raise ValueError(
            f"{index_path} header ngram {native_header.ngram_n} != meta ngram {meta.ngram}."
        )
    if attr_header and attr_header.ngram_n and attr_header.ngram_n != meta.ngram:
        raise ValueError(
            f"{attr_path} header ngram {attr_header.ngram_n} != meta ngram {meta.ngram}."
        )

    benchmark_stats, stats_min_hits = index_benchmark_stats(
        index_path, meta, min_hits=min_hits
    )
    items_path = _sidecar(index_path, ".items")
    items_format = (
        AttributionIndex.items_sidecar_version(str(items_path))
        if items_path.exists()
        else None
    )
    return {
        "ngram": meta.ngram,
        "schema_version": meta.schema_version,
        "index_format_version": meta.index_format_version,
        "native_records": native_count,
        "native_header": native_has_header,
        "attr_records": attr_count,
        "attr_header": attr_has_header,
        "benchmark_stats": benchmark_stats,
        "min_hits": stats_min_hits,
        "items_format": items_format,
        "item_stopgrams": (
            meta.item_stopgrams.model_dump() if meta.item_stopgrams else None
        ),
    }


def format_benchmark_table(
    rows: list[tuple[str, str, str, str, str]],
    *,
    min_hits: int | None,
    indent: str = "",
) -> str:
    """Render per-benchmark (name, items, ngrams, 0 n-grams, < min_hits) rows."""
    below = f"< {min_hits} n-grams" if min_hits is not None else "< min_hits"
    header = ("benchmark", "items", "n-grams", "0 n-grams", below)
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def _fmt(row: tuple[str, ...]) -> str:
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(widths[i]) for i, cell in enumerate(row[1:], start=1)]
        return indent + "  ".join(cells).rstrip()

    return "\n".join([_fmt(header), *(_fmt(r) for r in rows)])


def _iter_ndjson_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".ndjson":
            yield path


def inspect_run(
    scan_dir: Path, *, validate_ndjson: bool = True, max_ndjson_lines: int | None = None
) -> dict[str, object]:
    layout = OutputLayout(scan_dir)
    layout.validate_layout()

    manifest_payload = json.loads(layout.run_manifest_json.read_text(encoding="utf-8"))
    manifest = RunManifest.model_validate(manifest_payload)

    final_payload = json.loads(layout.final_summary_json.read_text(encoding="utf-8"))
    FinalSummaryOutput.model_validate(final_payload)

    summary_files = sorted(layout.summaries_dir.glob("*.summary.json"))
    for summary_path in summary_files:
        ScanSummaryOutput.model_validate_json(summary_path.read_text(encoding="utf-8"))

    hits_files = sorted(layout.hits_dir.glob("*.hits.bin"))
    for hits_path in hits_files:
        _, _, hits_header = _validate_binary_file(
            hits_path, expected_file_type=FileType.HITS_PAIRS, record_size=16
        )
        run_ngram = manifest.scan_config.n
        if hits_header and hits_header.ngram_n and run_ngram is not None:
            if hits_header.ngram_n != run_ngram:
                raise ValueError(
                    f"{hits_path} header ngram {hits_header.ngram_n} != run ngram {run_ngram}."
                )

    if validate_ndjson:
        sample_files = _iter_ndjson_files(layout.contaminated_dir.glob("*.ndjson"))
        validated_lines = 0
        validated_files = 0
        for sample_path in sample_files:
            validated_files += 1
            with sample_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if (
                        max_ndjson_lines is not None
                        and validated_lines >= max_ndjson_lines
                    ):
                        break
                    parsed = json.loads(line)
                    DocResult.model_validate(parsed)
                    parse_contamination_record(line)
                    validated_lines += 1
            if max_ndjson_lines is not None and validated_lines >= max_ndjson_lines:
                break
    else:
        validated_files = 0
        validated_lines = 0

    return {
        "summaries": len(summary_files),
        "hits_files": len(hits_files),
        "validated_ndjson": bool(validate_ndjson),
        "validated_ndjson_files": validated_files if validate_ndjson else 0,
        "validated_ndjson_lines": validated_lines if validate_ndjson else 0,
    }
