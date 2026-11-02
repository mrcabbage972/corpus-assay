import io
import logging
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)


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
    channel_counts: dict[str, int] | None = None,
) -> tuple[int, int, int, list[str]]:
    """Scan one shard: ``(scanned, contaminated, empty_after_filter, records)``.

    ``channel_counts``, when given, receives the flagged sub-documents per
    enabled channel (``ngram``, ``spaced``, ``exact``).
    """
    stem = layout.stem_from_path(file_path)
    hits_path = layout.per_hits_path(stem)
    item_best_path = layout.per_item_best_path(stem)
    # Written only when the index has item postings; drop one from an earlier run.
    item_best_path.unlink(missing_ok=True)
    index_backend = _resolve_index_backend(cfg)
    counts: dict[str, int] = {}

    with _open_scan_input(file_path, cfg) as f:
        scanned, contaminated, json_results, empty_after_filter = scan_stream_rust(
            input=f,
            text_key=cfg.text_keys,
            messages_key=cfg.messages_key,
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
            min_item_coverage=cfg.min_item_coverage,
            out_item_best_path=str(item_best_path),
            spaced_path=cfg.spaced_path,
            spaced_min_loci=cfg.spaced_min_loci,
            spaced_ver_min_span=cfg.spaced_ver_min_span,
            spaced_ver_identity=cfg.spaced_ver_identity,
            exact_channel=cfg.exact_channel,
            exact_min_chars=cfg.exact_min_chars,
            confirm=cfg.confirm,
            channel_counts=counts,
        )

    if channel_counts is not None:
        channel_counts.update({str(k): int(v) for k, v in counts.items()})
    return int(scanned), int(contaminated), int(empty_after_filter), list(json_results)


def _open_scan_input(file_path: Path, cfg: ScanConfig) -> IO[bytes]:
    """Open a shard as the Parquet byte stream the Rust scanner reads.

    Parquet passes through untouched. JSONL (optionally ``.gz`` / ``.zst``) is
    loaded into memory and re-encoded as Parquet, keeping only the text,
    messages and id columns. Either way, string columns the scan does not read
    are reported once per file (see ``warn_unscanned_columns``).
    """
    if file_path.name.endswith(".parquet"):
        import pyarrow.parquet as pq

        warn_unscanned_columns(pq.read_schema(file_path), cfg, file_path.name)
        return file_path.open("rb")
    return _jsonl_as_parquet(file_path, cfg)


def _jsonl_as_parquet(file_path: Path, cfg: ScanConfig) -> io.BytesIO:
    import pyarrow as pa
    import pyarrow.json as pa_json
    import pyarrow.parquet as pq

    # pyarrow decompresses by file extension (.gz, .zst, ...).
    table = pa_json.read_json(str(file_path))
    warn_unscanned_columns(table.schema, cfg, file_path.name)
    columns: dict[str, pa.ChunkedArray] = {}
    for key in cfg.text_keys:
        if key not in table.column_names:
            raise ValueError(f"Text key '{key}' not found in {file_path.name}")
        columns[key] = table[key].cast(pa.string())
    if cfg.messages_key is not None:
        if cfg.messages_key not in table.column_names:
            raise ValueError(
                f"Messages key '{cfg.messages_key}' not found in {file_path.name}"
            )
        # Kept as read (list<struct<role, content, ...>>); Rust validates it.
        columns[cfg.messages_key] = table[cfg.messages_key]
    if cfg.id_key and cfg.id_key in table.column_names:
        columns[cfg.id_key] = table[cfg.id_key].cast(pa.string())

    buffer = io.BytesIO()
    pq.write_table(pa.table(columns), buffer)
    buffer.seek(0)
    return buffer


def _is_text_like(dtype: "pa.DataType") -> bool:
    import pyarrow.types as pat

    if pat.is_dictionary(dtype):
        dtype = dtype.value_type
    return bool(
        pat.is_string(dtype) or pat.is_large_string(dtype) or pat.is_string_view(dtype)
    )


def _is_chat_like(dtype: "pa.DataType") -> bool:
    """A list of structs with a string ``content`` field (a chat column)."""
    import pyarrow.types as pat

    if not (pat.is_list(dtype) or pat.is_large_list(dtype)):
        return False
    item = dtype.value_type
    if not pat.is_struct(item):
        return False
    index = item.get_field_index("content")
    return index >= 0 and _is_text_like(item.field(index).type)


def warn_unscanned_columns(
    schema: "pa.Schema", cfg: ScanConfig, file_name: str
) -> None:
    """Warn about text columns of a shard that this scan does not read.

    A record may carry several texts (a prompt and its answers, chat turns);
    scanning only one leaves the others unchecked. Also warns when the
    ``--id-key`` column is missing, since doc ids then fall back to text hashes.
    """
    used = {*cfg.text_keys, cfg.messages_key, cfg.id_key}
    text_cols = [
        field.name
        for field in schema
        if field.name not in used and _is_text_like(field.type)
    ]
    chat_cols = [
        field.name
        for field in schema
        if field.name not in used and _is_chat_like(field.type)
    ]
    if text_cols or chat_cols:
        parts = []
        if text_cols:
            parts.append(
                f"string column(s) {', '.join(text_cols)} (add with --text-key)"
            )
        if chat_cols:
            parts.append(
                f"chat column(s) {', '.join(chat_cols)} (scan with --messages-key)"
            )
        logger.warning("%s: not scanning %s", file_name, "; ".join(parts))
    if cfg.id_key and cfg.id_key not in schema.names:
        logger.warning(
            "%s: id column '%s' not found; doc ids fall back to text hashes",
            file_name,
            cfg.id_key,
        )


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
    channel_counts: dict[str, int] | None = None,
) -> ScanSummary:
    per_source = aggregate_per_source_hits(hits_path)
    return ScanSummary(
        file=filename,
        scanned=scanned,
        contaminated=contaminated,
        contam_rate=(contaminated / scanned) if scanned else 0.0,
        empty_after_filter=empty_after_filter,
        per_source=per_source.to_legacy_dict(),
        contaminated_by_channel=dict(channel_counts or {}),
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
        channels: dict[str, int] = {}
        scanned, contaminated, empty_after_filter, json_results = scan_file(
            file_path, cfg, layout, channel_counts=channels
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
        channel_counts=channels,
    )
    write_summary(summary_path, summary)
    return summary
