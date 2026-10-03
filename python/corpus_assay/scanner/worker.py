import io
from pathlib import Path
from typing import IO, Any

from corpus_assay._native import scan_stream_rust
from corpus_assay.attribution import AttributionIndex
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.io import (
    maybe_write_contaminated_docs,
    try_read_existing_summary,
    write_summary,
)
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.schema import (
    PerSourceAggregate,
    PerSourceStats,
    ScanSummary,
)


def aggregate_per_source_hits(hits_path: Path) -> PerSourceAggregate:
    if not hits_path.exists():
        return PerSourceAggregate()

    per_src_hits: dict[int, int] = {}
    for _hash, mask in AttributionIndex.read_hash_mask_pairs(
        str(hits_path), expected_file_type=AttributionIndex.HITS_FILE_TYPE
    ):
        for sid in AttributionIndex.iter_mask_source_ids(mask):
            per_src_hits[sid] = per_src_hits.get(sid, 0) + 1

    all_sids = set(per_src_hits)
    return PerSourceAggregate(
        by_source={
            str(sid): PerSourceStats(
                unique_hashes=int(per_src_hits.get(sid, 0)),
            )
            for sid in all_sids
        }
    )


def scan_file(
    file_path: Path,
    cfg: ScanConfig,
    layout: OutputLayout,
) -> tuple[int, int, int, list[str]]:
    stem = layout.stem_from_path(file_path)
    hits_path = layout.per_hits_path(stem)
    index_backend = _resolve_index_backend(cfg)

    with _open_scan_input(file_path, cfg) as f:
        scanned, contaminated, json_results, empty_after_filter = scan_stream_rust(
            input=f,
            text_key=cfg.text_key,
            id_key=cfg.id_key,
            index_path=str(cfg.index_path),
            index_backend=index_backend,
            n=cfg.ngram,
            min_hits=cfg.min_hits,
            min_coverage=cfg.min_coverage,
            packed_doc_sep=cfg.packed_doc_sep,
            packed_doc_sep_typo=cfg.packed_doc_sep_typo,
            out_hits_path=str(hits_path),
            max_return_records=cfg.max_return_records,
            stopgrams_path=cfg.stopgrams_path,
            gate_mode=cfg.gate_mode,
            min_longest_run=cfg.min_longest_run,
            boilerplate_threshold=cfg.boilerplate_threshold,
            spaced_path=cfg.spaced_path,
            spaced_min_loci=cfg.spaced_min_loci,
            spaced_ver_min_span=cfg.spaced_ver_min_span,
            spaced_ver_identity=cfg.spaced_ver_identity,
        )

    return int(scanned), int(contaminated), int(empty_after_filter), list(json_results)


def _open_scan_input(file_path: Path, cfg: ScanConfig) -> IO[bytes]:
    """Open a shard as the Parquet byte stream the Rust scanner reads.

    Parquet passes through untouched. JSONL (optionally ``.gz`` / ``.zst``) is
    loaded into memory and re-encoded as Parquet, keeping only the text and id
    columns as UTF-8 strings.
    """
    if file_path.name.endswith(".parquet"):
        return file_path.open("rb")
    return _jsonl_as_parquet(file_path, cfg)


def _jsonl_as_parquet(file_path: Path, cfg: ScanConfig) -> io.BytesIO:
    import pyarrow as pa
    import pyarrow.json as pa_json
    import pyarrow.parquet as pq

    # pyarrow decompresses by file extension (.gz, .zst, ...).
    table = pa_json.read_json(str(file_path))
    if cfg.text_key not in table.column_names:
        raise ValueError(f"Text key '{cfg.text_key}' not found in {file_path.name}")
    columns = {cfg.text_key: table[cfg.text_key].cast(pa.string())}
    if cfg.id_key and cfg.id_key in table.column_names:
        columns[cfg.id_key] = table[cfg.id_key].cast(pa.string())

    buffer = io.BytesIO()
    pq.write_table(pa.table(columns), buffer)
    buffer.seek(0)
    return buffer


def _resolve_index_backend(cfg: ScanConfig) -> str:
    if cfg.index_backend == "auto":
        return "mmap" if cfg.workers > 1 else "memory"
    return cfg.index_backend


def build_scan_summary(
    filename: str,
    scanned: int,
    contaminated: int,
    empty_after_filter: int,
    hits_path: Path,
    run_fingerprint: dict[str, Any],
) -> ScanSummary:
    per_source = aggregate_per_source_hits(hits_path)
    return ScanSummary(
        file=filename,
        scanned=scanned,
        contaminated=contaminated,
        contam_rate=(contaminated / scanned) if scanned else 0.0,
        empty_after_filter=empty_after_filter,
        per_source=per_source.to_legacy_dict(),
        run_fingerprint=run_fingerprint,
    )


def scan_one_file_worker(
    file_path: Path, cfg: ScanConfig, run_fingerprint: dict[str, Any]
) -> ScanSummary:
    layout = OutputLayout(Path(cfg.out_dir))
    stem = layout.stem_from_path(file_path)
    hits_path = layout.per_hits_path(stem)
    summary_path = layout.summary_path(stem)

    existing = try_read_existing_summary(summary_path, run_fingerprint)
    if existing is not None:
        return existing

    layout.ensure_dirs()

    try:
        scanned, contaminated, empty_after_filter, json_results = scan_file(
            file_path, cfg, layout
        )
    except Exception as e:
        return ScanSummary(file=file_path.name, error=str(e))

    maybe_write_contaminated_docs(
        layout.contaminated_docs_path(stem), contaminated, json_results
    )

    summary = build_scan_summary(
        file_path.name,
        scanned,
        contaminated,
        empty_after_filter,
        hits_path,
        run_fingerprint,
    )
    write_summary(summary_path, summary)
    return summary
