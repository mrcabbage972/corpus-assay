"""JSONL shards go through the real Rust scanner (no mocks).

The scanner reads Parquet only; the worker converts JSONL (plain, ``.gz`` or
``.zst``) to an in-memory Parquet buffer first.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _index_helpers import write_native_index
from corpus_assay._native import hash_ngram, ngram_allowed, normalize_text
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.worker import scan_file
from corpus_assay.text_utils import iter_ngrams

NGRAM = 3
LEAK = "photosynthesis converts light energy into chemical energy stored in glucose"
CLEAN = "completely unrelated sentence about sailing boats and harbour weather"


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


def _records() -> list[dict[str, object]]:
    # Integer ids exercise the cast to string.
    return [{"id": 1, "text": LEAK}, {"id": 2, "text": CLEAN}]


def _write(path: Path) -> None:
    payload = "".join(json.dumps(r) + "\n" for r in _records()).encode("utf-8")
    if path.name.endswith(".gz"):
        payload = gzip.compress(payload)
    elif path.name.endswith(".zst"):
        with pa.output_stream(str(path), compression="zstd") as out:
            out.write(payload)
        return
    path.write_bytes(payload)


def _cfg(tmp_path: Path, index: str) -> ScanConfig:
    return ScanConfig(
        inputs=[str(tmp_path)],
        text_key="text",
        id_key="id",
        index_path=index,
        out_dir=str(tmp_path / "out"),
        ngram=NGRAM,
        min_hits=2,
        min_coverage=0.0,
        workers=1,
    )


@pytest.mark.parametrize("name", ["a.jsonl", "a.jsonl.gz", "a.jsonl.zst"])
def test_jsonl_shard_is_scanned(tmp_path: Path, index_path: str, name: str) -> None:
    shard = tmp_path / name
    _write(shard)
    cfg = _cfg(tmp_path, index_path)
    layout = OutputLayout(Path(cfg.out_dir))
    layout.ensure_dirs()

    scanned, contaminated, _empty, records = scan_file(shard, cfg, layout)

    assert (scanned, contaminated) == (2, 1)
    # Records are per packed sub-document: "<id>-part-<k>".
    assert [json.loads(r)["doc_id"] for r in records] == ["1-part-0"]


def test_jsonl_missing_text_key_errors(tmp_path: Path, index_path: str) -> None:
    shard = tmp_path / "a.jsonl"
    shard.write_text(json.dumps({"body": LEAK}) + "\n", encoding="utf-8")
    cfg = _cfg(tmp_path, index_path)
    layout = OutputLayout(Path(cfg.out_dir))
    layout.ensure_dirs()

    with pytest.raises(ValueError, match="Text key 'text' not found"):
        scan_file(shard, cfg, layout)


def test_parquet_large_string_text_column_is_rejected(
    tmp_path: Path, index_path: str
) -> None:
    """Documented limitation: the Rust scanner needs a plain UTF-8 text column."""
    shard = tmp_path / "a.parquet"
    table = pa.table({"text": pa.array([LEAK], type=pa.large_string())})
    pq.write_table(table, shard)
    cfg = _cfg(tmp_path, index_path)
    layout = OutputLayout(Path(cfg.out_dir))
    layout.ensure_dirs()

    with pytest.raises(ValueError, match="not a string type"):
        scan_file(shard, cfg, layout)
