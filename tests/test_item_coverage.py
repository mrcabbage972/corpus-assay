"""Per-item coverage: the share of one item's n-grams a single document holds.

``item_coverage = hits / item_ngrams`` is reported for each item in a record,
``min_item_coverage`` adds an OR path to the union gate (and an extra per-item
test to the item gate), and each shard writes the best single-document coverage
of every item it hit to ``per_item_best/``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from corpus_assay._native import hash_ngram, normalize_text, scan_stream_rust
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import _field_texts
from corpus_assay.indexing import build_index, write_native_index_artifacts
from corpus_assay.scanner.schema import ItemBestFile
from corpus_assay.text_utils import iter_allowed_ngrams

NGRAM = 3
ITEM = (
    "a cyclist accelerates uniformly from rest reaching eighteen metres per second "
    "after twelve seconds compute the displacement covered"
)
OTHER = "photosynthesis converts sunlight carbon dioxide water into glucose oxygen"
# Long unrelated text: every 3-gram distinct, so document coverage stays tiny.
LONG_FILLER = " ".join(f"filler{i}" for i in range(2000))


def _ngrams(text: str) -> set[int]:
    return {hash_ngram(ng) for ng in iter_allowed_ngrams(normalize_text(text), NGRAM)}


N_ITEM = len(_ngrams(ITEM))


class _Loader:
    dataset_fingerprints: list = []

    def __init__(self, spec: BenchmarkSpec, rows: list[dict]) -> None:
        self.spec, self.rows = spec, rows

    def load_positive_fields(self):
        for row in self.rows:
            yield _field_texts(row, self.spec.fields)

    def load_negative(self):
        return iter(())


@pytest.fixture
def index(tmp_path: Path) -> str:
    spec = BenchmarkSpec(name="Q", dataset="x", fields=["question"])
    rows = [{"question": ITEM}, {"question": OTHER}]
    result = build_index(
        benchmarks=[spec],
        ngram=NGRAM,
        loader_factory=lambda s: _Loader(s, rows),  # type: ignore[arg-type,return-value]
    )
    path = tmp_path / "q.native"
    write_native_index_artifacts(result, str(path))
    return str(path)


def _scan(index: str, texts: list[str], **kwargs) -> tuple[int, list[dict]]:
    buffer = io.BytesIO()
    pq.write_table(
        pa.table({"text": texts, "id": [f"d{i}" for i in range(len(texts))]}),
        buffer,
    )
    buffer.seek(0)
    args = {"min_hits": 3, "min_coverage": 0.01, **kwargs}
    _, contaminated, records, _ = scan_stream_rust(
        input=buffer, text_key="text", id_key="id", index_path=index, n=NGRAM, **args
    )
    return contaminated, sorted(
        (json.loads(r) for r in records), key=lambda r: r["doc_id"]
    )


def _half_item() -> str:
    words = ITEM.split()
    return " ".join(words[: len(words) // 2 + 1])


def test_long_document_holding_a_whole_item_needs_min_item_coverage(
    index: str,
) -> None:
    long_doc = f"{LONG_FILLER} {ITEM} {LONG_FILLER.replace('filler', 'more')}"
    # Document coverage is far below min_coverage=0.01.
    assert N_ITEM / len(_ngrams(long_doc)) < 0.01

    contaminated, _ = _scan(index, [long_doc], gate_mode="union")
    assert contaminated == 0

    contaminated, records = _scan(
        index, [long_doc], gate_mode="union", min_item_coverage=0.5
    )
    assert contaminated == 1
    (hit,) = records[0]["item_hits"]
    assert hit["item_id"] == 0
    assert hit["hits"] == hit["item_ngrams"] == N_ITEM
    assert hit["item_coverage"] == 1.0
    assert records[0]["match_count"] == N_ITEM

    # The coverage path needs min_hits too.
    contaminated, _ = _scan(
        index,
        [long_doc],
        gate_mode="union",
        min_item_coverage=0.5,
        min_hits=N_ITEM + 1,
    )
    assert contaminated == 0


def test_item_mode_takes_min_item_coverage_as_an_extra_requirement(
    index: str,
) -> None:
    half = _half_item()
    hits = len(_ngrams(half))
    coverage = hits / N_ITEM
    assert 0.3 < coverage < 0.7

    for threshold, flagged in [(0.0, 1), (coverage, 1), (coverage + 0.01, 0)]:
        contaminated, records = _scan(
            index,
            [half],
            gate_mode="item",
            min_coverage=0.0,
            min_item_coverage=threshold,
        )
        assert contaminated == flagged, threshold
        if flagged:
            (hit,) = records[0]["item_hits"]
            assert hit["hits"] == hits and hit["item_ngrams"] == N_ITEM
            assert hit["item_coverage"] == pytest.approx(coverage)

    # In item mode the document-coverage test still applies.
    long_doc = f"{LONG_FILLER} {ITEM}"
    contaminated, _ = _scan(index, [long_doc], gate_mode="item", min_item_coverage=0.5)
    assert contaminated == 0


def test_union_evidence_lists_items_passing_either_path(index: str) -> None:
    # Flagged by the union rule (short doc, high doc coverage): the item holding
    # half of its n-grams is listed, with its coverage.
    half = _half_item()
    contaminated, records = _scan(index, [half], gate_mode="union")
    assert contaminated == 1
    (hit,) = records[0]["item_hits"]
    assert hit["item_coverage"] == pytest.approx(len(_ngrams(half)) / N_ITEM)


def test_min_item_coverage_requires_item_postings(tmp_path: Path, index: str) -> None:
    Path(index + ".items").unlink()
    with pytest.raises(ValueError, match="min_item_coverage requires an item sidecar"):
        _scan(index, [ITEM], min_item_coverage=0.5)
    _scan(index, [ITEM])  # still fine without the threshold
    with pytest.raises(ValueError, match=r"min_item_coverage must be in \[0,1\]"):
        _scan(index, [ITEM], min_item_coverage=1.5)


def test_per_item_best_keeps_the_best_document_of_each_item(
    tmp_path: Path, index: str
) -> None:
    out = tmp_path / "best.json"
    half = _half_item()
    two_hits = " ".join(ITEM.split()[:4])  # 2 n-grams: below min_hits, not flagged
    docs = [two_hits, half, f"{LONG_FILLER} {ITEM}", ITEM, "nothing to see here at all"]
    contaminated, _ = _scan(index, docs, gate_mode="union", out_item_best_path=str(out))
    assert contaminated == 2  # half and the bare item; the long doc is too diluted

    best = ItemBestFile.model_validate_json(out.read_text("utf-8"))
    (entry,) = best.items  # only items with a hit
    assert entry.item_id == 0
    # The long (unflagged) doc holds the whole item first; ties keep it.
    assert (entry.hits, entry.item_ngrams, entry.coverage) == (N_ITEM, N_ITEM, 1.0)
    assert entry.doc_id == "d2-part-0"

    _scan(index, docs[:2], out_item_best_path=str(out))
    (entry,) = ItemBestFile.model_validate_json(out.read_text("utf-8")).items
    assert entry.doc_id == "d1-part-0"
    assert entry.coverage == pytest.approx(len(_ngrams(half)) / N_ITEM)


def test_no_per_item_best_without_item_postings(tmp_path: Path, index: str) -> None:
    Path(index + ".items").unlink()
    out = tmp_path / "best.json"
    _scan(index, [ITEM], out_item_best_path=str(out))
    assert not out.exists()
