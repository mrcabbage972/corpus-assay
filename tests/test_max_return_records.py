"""``--max-return-records 0`` / ``--all-records`` write every flagged record (#10)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _index_helpers import write_native_index
from corpus_assay._native import hash_ngram, ngram_allowed, normalize_text
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.io import maybe_write_contaminated_docs
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.worker import scan_one_file_worker
from corpus_assay.text_utils import iter_ngrams

NGRAM = 3
LEAK = "photosynthesis converts light energy into chemical energy stored in glucose"


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


def _cfg(tmp_path: Path, index: str, **kwargs: object) -> ScanConfig:
    return ScanConfig(
        inputs=[str(tmp_path / "in")],
        text_key="text",
        id_key="id",
        index_path=index,
        out_dir=str(tmp_path / "out"),
        ngram=NGRAM,
        min_hits=2,
        min_coverage=0.0,
        workers=1,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("value", [0, None])
def test_scan_config_zero_means_no_cap(
    tmp_path: Path, index_path: str, value: int | None
) -> None:
    assert (
        _cfg(tmp_path, index_path, max_return_records=value).max_return_records is None
    )


def test_scan_config_rejects_negative_cap(tmp_path: Path, index_path: str) -> None:
    with pytest.raises(ValueError, match="max_return_records"):
        _cfg(tmp_path, index_path, max_return_records=-1)


@pytest.mark.parametrize(("cap", "expected"), [(2, 2), (0, 5), (None, 5)])
def test_worker_writes_every_flagged_record_when_uncapped(
    tmp_path: Path, index_path: str, cap: int | None, expected: int
) -> None:
    shard = tmp_path / "in" / "shard.parquet"
    shard.parent.mkdir()
    pq.write_table(
        pa.table({"id": [str(i) for i in range(5)], "text": [LEAK] * 5}), shard
    )
    cfg = _cfg(tmp_path, index_path, max_return_records=cap)

    summary = scan_one_file_worker(shard, cfg, run_fingerprint={})

    assert summary.error is None
    assert summary.contaminated == 5
    layout = OutputLayout(Path(cfg.out_dir))
    lines = layout.contaminated_docs_path("shard").read_text().splitlines()
    assert len(lines) == expected
    assert all(json.loads(line)["doc_id"].endswith("-part-0") for line in lines)


def test_no_records_means_no_sample_file(tmp_path: Path) -> None:
    path = tmp_path / "contaminated_docs" / "shard.decontam.sample.ndjson"
    maybe_write_contaminated_docs(path, contaminated=0, json_results=[])
    assert not path.exists()

    # A stale sample from an earlier run of the same shard is removed.
    path.parent.mkdir(parents=True)
    path.write_text('{"doc_id": "old"}\n', encoding="utf-8")
    maybe_write_contaminated_docs(path, contaminated=3, json_results=[])
    assert not path.exists()
