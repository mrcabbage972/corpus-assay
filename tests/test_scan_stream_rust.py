import io
import json
import struct
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _index_helpers import write_native_index
from corpus_assay._native import (
    hash_ngram,
    ngram_allowed,
    normalize_text,
    scan_stream_rust,
)
from corpus_assay.text_utils import iter_ngrams


@pytest.fixture(scope="session")
def scan_index(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base_dir = tmp_path_factory.mktemp("scan_index")
    index_path = base_dir / "index.native"

    text = "alpha beta gamma delta"
    ngram = 2
    tokens = normalize_text(text)
    ngrams = [ng for ng in iter_ngrams(tokens, ngram) if ngram_allowed(ng)]
    target_ngram = ngrams[0]
    target_hash = hash_ngram(target_ngram)

    write_native_index(
        index_path,
        ngram=ngram,
        hashes={target_hash},
        sources={target_hash: "source_a"},
    )

    return {
        "index_path": str(index_path),
        "ngram": ngram,
        "hash": target_hash,
        "text": text,
    }


def _parquet_bytes(texts: list[str], ids: list[str]) -> bytes:
    table = pa.Table.from_pydict({"text": texts, "doc_id": ids})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_scan_stream_rust_matches(tmp_path: Path, scan_index: dict[str, Any]) -> None:
    parquet_data = _parquet_bytes(
        [scan_index["text"], "epsilon zeta"],  # type: ignore
        ["doc-1", "doc-2"],
    )
    out_hits_path = tmp_path / "hits.bin"

    scanned, contaminated, results, _ = scan_stream_rust(
        input=io.BytesIO(parquet_data),
        text_key="text",
        id_key="doc_id",
        index_path=scan_index["index_path"],
        n=scan_index["ngram"],
        min_hits=1,
        min_coverage=0.0,
        out_hits_path=str(out_hits_path),
    )

    assert scanned == 2
    assert contaminated == 1
    assert len(results) == 1

    record = json.loads(results[0])
    assert record["doc_id"] == "doc-1-part-0"
    assert record["match_count"] == 1
    assert dict(record["src_hits"]) == {0: 1}

    with open(out_hits_path, "rb") as f:
        content = f.read()

    assert len(content) == 52
    hashed, mask = struct.unpack("<QQ", content[36:])
    assert hashed == scan_index["hash"]
    assert mask == 1


def test_scan_stream_rust_caps_returned_records(
    tmp_path: Path, scan_index: dict[str, Any]
) -> None:
    parquet_data = _parquet_bytes(
        [scan_index["text"], scan_index["text"]],  # type: ignore
        ["doc-1", "doc-2"],
    )
    out_hits_path = tmp_path / "hits.bin"

    scanned, contaminated, results, _ = scan_stream_rust(
        input=io.BytesIO(parquet_data),
        text_key="text",
        id_key="doc_id",
        index_path=scan_index["index_path"],
        n=scan_index["ngram"],
        min_hits=1,
        min_coverage=0.0,
        out_hits_path=str(out_hits_path),
        max_return_records=1,
    )

    assert scanned == 2
    assert contaminated == 2
    assert len(results) == 1


@pytest.mark.parametrize("min_coverage", [-0.1, 1.1])
def test_scan_stream_rust_rejects_invalid_coverage(
    scan_index: dict[str, Any],
    min_coverage: float,
) -> None:
    parquet_data = _parquet_bytes([scan_index["text"]], ["doc-1"])  # type: ignore

    with pytest.raises(ValueError, match=r"min_coverage must be in \[0,1\]"):
        scan_stream_rust(
            input=io.BytesIO(parquet_data),
            text_key="text",
            id_key="doc_id",
            index_path=scan_index["index_path"],
            n=scan_index["ngram"],
            min_hits=1,
            min_coverage=min_coverage,
        )


def test_scan_stream_rust_rejects_zero_min_hits(scan_index: dict[str, Any]) -> None:
    parquet_data = _parquet_bytes([scan_index["text"]], ["doc-1"])  # type: ignore

    with pytest.raises(ValueError, match=r"min_hits must be >= 1"):
        scan_stream_rust(
            input=io.BytesIO(parquet_data),
            text_key="text",
            id_key="doc_id",
            index_path=scan_index["index_path"],
            n=scan_index["ngram"],
            min_hits=0,
            min_coverage=0.0,
        )
