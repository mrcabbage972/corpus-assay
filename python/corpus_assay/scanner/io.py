import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from pydantic import ValidationError

from corpus_assay.config import ScanConfig
from corpus_assay.constants import INDEX_FORMAT_VERSION
from corpus_assay.scanner.fingerprint import (
    _file_sha256,
    library_version,
    rust_extension_version,
)
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.schema import (
    ContaminationRecord,
    RunSummary,
    ScanSummary,
)
from corpus_assay.schemas import (
    FINAL_SUMMARY_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    SCAN_SUMMARY_SCHEMA_VERSION,
    FinalSummaryOutput,
    RunHFDatasetInput,
    RunIndexInfo,
    RunInputs,
    RunManifest,
    RunOutputs,
    RunParquetShard,
    RunScanConfig,
    RunStopGramsInfo,
    ScanSummaryOutput,
)

logger = logging.getLogger(__name__)


def parse_contamination_record(line: str) -> ContaminationRecord:
    raw = json.loads(line)
    pairs = raw.get("src_hits", []) or []
    if pairs and isinstance(pairs[0], dict):
        raw["src_hits"] = [
            {"source_id": int(rec["source_id"]), "hits": int(rec["hits"])}
            for rec in pairs
        ]
    else:
        raw["src_hits"] = [
            {"source_id": int(sid), "hits": int(cnt)} for sid, cnt in pairs
        ]
    # Legacy pre-v2 scan outputs serialized item_hits as tuples
    # `[item_id, distinct_hits, longest_run_tokens]`; normalize them to the v2
    # object form (unique/shared/boilerplate unknown -> 0) so old NDJSON still parses.
    items = raw.get("item_hits", []) or []
    if items and not isinstance(items[0], dict):
        raw["item_hits"] = [
            {
                "item_id": int(t[0]),
                "hits": int(t[1]),
                "longest_run_tokens": int(t[2]) if len(t) > 2 else 0,
            }
            for t in items
        ]
    return ContaminationRecord.model_validate(raw)


def maybe_write_contaminated_docs(
    path: Path, contaminated: int, json_results: Iterable[str]
) -> None:
    """
    Write a sampled NDJSON file of contaminated docs (json_results is capped upstream).
    """
    if contaminated <= 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in json_results:
            f.write(line)
            f.write("\n")


def write_run_outputs(layout: OutputLayout, run: RunSummary, cfg: ScanConfig) -> None:
    layout.out_dir.mkdir(parents=True, exist_ok=True)

    index_path = Path(cfg.index_path)
    meta_path = index_path.with_name(f"{index_path.name}.meta.json")

    parquet_shards: list[RunParquetShard] = []
    for path_str in cfg.inputs:
        path = Path(path_str)
        if path.suffix != ".parquet" or not path.exists():
            continue
        sha = _file_sha256(path)
        bytes_size = path.stat().st_size if path.exists() else None
        try:
            rel_path = path.relative_to(layout.out_dir).as_posix()
        except ValueError:
            rel_path = str(path)
        parquet_shards.append(
            RunParquetShard(path=rel_path, sha256=sha, bytes=bytes_size)
        )

    hf_datasets: list[RunHFDatasetInput] = []
    if cfg.target_hf_dataset:
        hf_datasets.append(
            RunHFDatasetInput(
                hf_repo=cfg.target_hf_dataset,
                split=cfg.target_hf_split,
                revision=cfg.target_hf_revision,
                dataset_fingerprint=cfg.target_hf_fingerprint,
            )
        )

    layout.run_manifest_json.write_text(
        RunManifest(
            schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            library_version=library_version(),
            rust_extension_version=rust_extension_version(),
            run_id=uuid4(),
            started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            index=RunIndexInfo(
                path=str(index_path.resolve()),
                meta_path=str(meta_path.resolve()),
                index_format_version=INDEX_FORMAT_VERSION,
                meta_sha256=_file_sha256(meta_path),
            ),
            scan_config=RunScanConfig(
                text_key=cfg.text_key,
                id_key=cfg.id_key,
                n=cfg.ngram,
                min_hits=cfg.min_hits,
                min_coverage=cfg.min_coverage,
                packed_doc_sep=cfg.packed_doc_sep,
                packed_doc_sep_typo=cfg.packed_doc_sep_typo,
                stopgrams_path=cfg.stopgrams_path,
            ),
            inputs=RunInputs(
                input_paths=list(cfg.inputs),
                hf_datasets=hf_datasets,
                parquet_shards=parquet_shards,
            ),
            outputs=RunOutputs(
                results_ndjson=str(layout.contaminated_dir.relative_to(layout.out_dir)),
                hits_bin=str(layout.hits_dir.relative_to(layout.out_dir)),
                final_summary_json=layout.final_summary_json.name,
            ),
            config=cfg.model_dump(),
            stopgrams=_build_stopgrams_manifest(cfg),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    # JSON
    layout.final_summary_json.write_text(
        FinalSummaryOutput(
            schema_version=FINAL_SUMMARY_SCHEMA_VERSION,
            total_files_scanned=run.ok_files,
            total_docs_scanned=run.total_docs_scanned,
            total_contaminated=run.total_contaminated,
            contamination_rate=run.contamination_rate,
            total_empty_after_filter=run.total_empty_after_filter,
            failed_files=run.failed_files,
            total_files=run.total_files,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    # CSV (optional)
    try:
        rows = [summary.model_dump() for summary in run.per_file]
        fieldnames = sorted(ScanSummary.model_fields.keys())
        with layout.final_summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        logger.warning("Failed to write CSV summary: %s", e)


def _build_stopgrams_manifest(cfg: ScanConfig) -> RunStopGramsInfo | None:
    if not cfg.stopgrams_path:
        return None
    from corpus_assay.stopgrams import stopgrams_meta_path

    stopgrams_path = Path(cfg.stopgrams_path)
    native_sha = _file_sha256(stopgrams_path) if stopgrams_path.exists() else None
    meta_path = stopgrams_meta_path(stopgrams_path)
    meta_sha = _file_sha256(meta_path) if meta_path.exists() else None
    payload = None
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    return RunStopGramsInfo(
        path=str(stopgrams_path.resolve()),
        meta_path=str(meta_path.resolve()) if meta_path.exists() else None,
        format_version=payload.get("format_version") if payload else None,
        meta_sha256=meta_sha,
        native_sha256=native_sha,
        tau=payload.get("tau") if payload else None,
        df_threshold=payload.get("df_threshold") if payload else None,
        stopgram_count=payload.get("stopgram_count") if payload else None,
    )


def try_read_existing_summary(
    path: Path, run_fingerprint: dict[str, Any]
) -> ScanSummary | None:
    if not path.exists():
        return None
    try:
        summary = ScanSummary.model_validate_json(path.read_text(encoding="utf-8"))
        if summary.run_fingerprint != run_fingerprint:
            return None
        return summary
    except (OSError, ValidationError):
        return None
    return None


def write_summary(path: Path, summary: ScanSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ScanSummaryOutput(
        schema_version=SCAN_SUMMARY_SCHEMA_VERSION,
        **summary.model_dump(),
    ).model_dump()
    path.write_text(json.dumps(payload), encoding="utf-8")
