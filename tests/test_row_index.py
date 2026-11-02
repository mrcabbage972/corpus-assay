"""Records locate their sub-document: ``row_index`` (the row in the shard) and
``part`` (the packed part within the field or turn)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _index_helpers import write_native_index
from corpus_assay._native import hash_ngram, ngram_allowed, normalize_text
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.io import parse_contamination_record
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.worker import scan_file
from corpus_assay.schemas import DocResult
from corpus_assay.text_utils import iter_ngrams

NGRAM = 3
LEAK = "photosynthesis converts light energy into chemical energy stored in glucose"
CLEAN = "completely unrelated sentence about sailing boats and harbour weather"
SEP = "<|endoftext|>"


@pytest.fixture
def index_path(tmp_path: Path) -> str:
    grams = [g for g in iter_ngrams(normalize_text(LEAK), NGRAM) if ngram_allowed(g)]
    hashes = {hash_ngram(g) for g in grams}
    return write_native_index(
        tmp_path / "idx.native",
        ngram=NGRAM,
        hashes=hashes,
        sources={h: "bench" for h in hashes},
    )


def _scan(tmp_path: Path, index: str, shard: Path, **kwargs) -> list:
    cfg = ScanConfig(
        inputs=[str(shard)],
        id_key=None,
        index_path=index,
        out_dir=str(tmp_path / "out"),
        ngram=NGRAM,
        min_hits=2,
        min_coverage=0.0,
        workers=1,
        max_return_records=None,
        **kwargs,
    )
    layout = OutputLayout(Path(cfg.out_dir))
    layout.ensure_dirs()
    _scanned, _contaminated, _empty, records = scan_file(shard, cfg, layout)
    for line in records:
        DocResult.model_validate_json(line)
    return [parse_contamination_record(r) for r in records]


def test_jsonl_row_index_counts_non_blank_lines(
    tmp_path: Path, index_path: str
) -> None:
    shard = tmp_path / "s.jsonl"
    shard.write_text(
        "\n".join(
            [
                json.dumps({"text": CLEAN}),
                "",
                "  \t",
                json.dumps({"text": LEAK}),
                "",
                json.dumps({"text": CLEAN}),
                json.dumps({"text": f"{CLEAN}{SEP}{LEAK}"}),
            ]
        )
        + "\n\n",
        encoding="utf-8",
    )
    records = _scan(tmp_path, index_path, shard)
    assert [(r.row_index, r.part) for r in records] == [(1, 0), (3, 1)]


def test_parquet_row_index_spans_batches(tmp_path: Path, index_path: str) -> None:
    # The scanner reads 1,024 rows per batch; row indexes continue across them.
    texts = [CLEAN] * 2500
    texts[3] = LEAK
    texts[2049] = LEAK
    shard = tmp_path / "s.parquet"
    pq.write_table(pa.table({"text": texts}), shard, row_group_size=1000)
    records = _scan(tmp_path, index_path, shard)
    assert [r.row_index for r in records] == [3, 2049]


def test_no_split_records_have_no_part(tmp_path: Path, index_path: str) -> None:
    shard = tmp_path / "s.parquet"
    pq.write_table(pa.table({"text": [CLEAN, f"{CLEAN}{SEP}{LEAK}"]}), shard)
    records = _scan(tmp_path, index_path, shard, packed_doc_sep=None)
    assert [(r.row_index, r.part) for r in records] == [(1, None)]
