"""Record schema v3 (field / role / turn) and the run-manifest scan config.

v2 records and manifests written before ``text_keys`` existed must still load.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from corpus_assay import reporting
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.io import parse_contamination_record, write_run_outputs
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.schemas import DOC_RESULT_SCHEMA_VERSION, DocResult, RunManifest

V2_RECORD = {
    "schema_version": 2,
    "doc_id": "doc500-part-0",
    "match_count": 85,
    "src_hits": [[0, 85]],
    "item_hits": [
        {
            "item_id": 7,
            "hits": 85,
            "longest_run_tokens": 75,
            "unique_hits": 85,
            "shared_hits": 0,
            "boilerplate_hits": 0,
        }
    ],
}

V3_RECORD = {
    "schema_version": 3,
    "doc_id": "c1/messages/1-part-0",
    "match_count": 4,
    "src_hits": [[0, 4]],
    "field": "messages",
    "role": "user",
    "turn": 1,
}


def test_record_schema_version_is_3() -> None:
    assert DOC_RESULT_SCHEMA_VERSION == 3


def test_v2_records_still_parse() -> None:
    line = json.dumps(V2_RECORD)
    doc = DocResult.model_validate(V2_RECORD)
    rec = parse_contamination_record(line)

    assert (doc.field, doc.role, doc.turn) == (None, None, None)
    assert (rec.field, rec.role, rec.turn) == (None, None, None)
    assert rec.item_hits[0].unique_hits == 85


def test_v3_records_parse_with_location() -> None:
    doc = DocResult.model_validate(V3_RECORD)
    rec = parse_contamination_record(json.dumps(V3_RECORD))

    assert (doc.field, doc.role, doc.turn) == ("messages", "user", 1)
    assert (rec.field, rec.role, rec.turn) == ("messages", "user", 1)


def test_scan_config_accepts_legacy_text_key() -> None:
    base = dict(inputs=["x"], id_key=None, index_path="i", out_dir="o")

    assert ScanConfig(**base, text_key="body").text_keys == ["body"]  # type: ignore[arg-type]
    assert ScanConfig(**base, text_key=["a", "b"]).text_keys == ["a", "b"]  # type: ignore[arg-type]
    assert ScanConfig(**base).text_keys == ["text"]  # type: ignore[arg-type]
    # A chat-only scan has no default text column.
    chat_only = ScanConfig(**base, messages_key="messages")  # type: ignore[arg-type]
    assert chat_only.text_keys == [] and chat_only.text_key is None
    # A config dumped before text_keys existed (e.g. a manifest's `config`).
    old_dump = {**base, "text_key": "text", "packed_doc_sep": "<|endoftext|>"}
    assert ScanConfig.model_validate(old_dump).text_keys == ["text"]


def test_manifest_records_text_keys_and_messages_key(tmp_path: Path) -> None:
    index = tmp_path / "idx.native"
    index.write_bytes(b"")
    (tmp_path / "idx.native.meta.json").write_text("{}", encoding="utf-8")
    cfg = ScanConfig(
        inputs=[],
        text_keys=["prompt", "chosen"],
        messages_key="messages",
        id_key="id",
        index_path=str(index),
        out_dir=str(tmp_path / "out"),
        packed_doc_sep=None,
    )
    run = SimpleNamespace(
        ok_files=0,
        total_docs_scanned=0,
        total_contaminated=0,
        contamination_rate=0.0,
        total_empty_after_filter=0,
        total_contaminated_by_channel={},
        failed_files=0,
        total_files=0,
        per_file=[],
    )
    layout = OutputLayout(Path(cfg.out_dir))

    write_run_outputs(layout, run, cfg)  # type: ignore[arg-type]

    manifest = RunManifest.model_validate_json(layout.run_manifest_json.read_text())
    assert manifest.scan_config.text_keys == ["prompt", "chosen"]
    assert manifest.scan_config.text_key == "prompt"
    assert manifest.scan_config.messages_key == "messages"
    assert manifest.scan_config.packed_doc_sep is None
    assert manifest.config is not None
    assert ScanConfig.model_validate(manifest.config) == cfg


def test_old_manifest_scan_config_still_loads() -> None:
    old_scan_config = {
        "text_key": "text",
        "id_key": None,
        "n": 13,
        "min_hits": 3,
        "min_coverage": 0.001,
        "packed_doc_sep": "<|endoftext|>",
        "packed_doc_sep_typo": "<|endoftext}>",
        "stopgrams_path": None,
    }
    old_config = {
        "inputs": ["data/"],
        "text_key": "text",
        "id_key": None,
        "index_path": "idx.native",
        "out_dir": "out",
    }
    manifest = RunManifest.model_validate(
        {
            "schema_version": 1,
            "run_id": "00000000-0000-0000-0000-000000000000",
            "started_at": "2026-10-03T00:00:00Z",
            "library_version": "0.1.0",
            "rust_extension_version": "0.1.0",
            "index": {"path": "idx.native", "index_format_version": 1},
            "scan_config": old_scan_config,
            "inputs": {},
            "outputs": {
                "results_ndjson": "contaminated_docs",
                "hits_bin": "per_source_hits",
                "final_summary_json": "_FINAL_SUMMARY.json",
            },
            "config": old_config,
        }
    )
    assert manifest.scan_config.text_keys == []
    assert manifest.scan_config.text_key == "text"
    # `report` reads the full config through its own manifest model.
    report_manifest = reporting.RunManifest.model_validate({"config": old_config})
    assert report_manifest.config.text_keys == ["text"]
