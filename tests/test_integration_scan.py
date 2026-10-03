from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from _index_helpers import write_native_index
from corpus_assay._native import (
    hash_ngram,
    ngram_allowed,
    normalize_text,
)
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.fingerprint import build_run_fingerprint
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.worker import scan_one_file_worker
from corpus_assay.text_utils import iter_ngrams


def _build_index(tmp_path: Path) -> dict[str, Any]:
    index_path = tmp_path / "index.native"

    text = "alpha beta gamma delta"
    ngram = 2
    tokens = normalize_text(text)
    ngrams = [ng for ng in iter_ngrams(tokens, ngram) if ngram_allowed(ng)]
    target_hash = hash_ngram(ngrams[0])

    write_native_index(
        index_path,
        ngram=ngram,
        hashes={target_hash},
        sources={target_hash: "source_a"},
    )

    return {
        "index_path": str(index_path),
        "ngram": ngram,
        "text": text,
    }


def _write_parquet(path: Path, records: list[dict[str, str]]) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def test_scan_one_file_worker_integration(tmp_path: Path) -> None:
    index = _build_index(tmp_path)

    docs = [
        {"doc_id": "doc-1", "text": index["text"]},
        {"doc_id": "doc-2", "text": "this sentence is unrelated"},
        {"doc_id": "doc-3", "text": "nothing to see here"},
    ]
    parquet_path = tmp_path / "docs.parquet"
    _write_parquet(parquet_path, docs)

    cfg = ScanConfig(
        inputs=[str(parquet_path)],
        text_key="text",
        id_key="doc_id",
        index_path=index["index_path"],
        out_dir=str(tmp_path / "out"),
        min_hits=1,
        ngram=index["ngram"],
        min_coverage=0.0,
        workers=1,
        max_return_records=10,
    )

    run_fingerprint = build_run_fingerprint(cfg)
    summary = scan_one_file_worker(parquet_path, cfg, run_fingerprint)

    assert summary.error is None
    assert summary.file == parquet_path.name
    assert summary.scanned == len(docs)
    assert summary.contaminated == 1

    layout = OutputLayout(Path(cfg.out_dir))
    stem = layout.stem_from_path(parquet_path)
    summary_path = layout.summary_path(stem)
    contaminated_path = layout.contaminated_docs_path(stem)

    assert summary_path.exists()
    assert contaminated_path.exists()

    contaminated_lines = contaminated_path.read_text(encoding="utf-8").splitlines()
    assert len(contaminated_lines) == 1

    record = json.loads(contaminated_lines[0])
    assert record["doc_id"].startswith("doc-1")
