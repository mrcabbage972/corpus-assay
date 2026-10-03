"""Item-aware scan gating: a doc must reproduce a *single* protected item to be
flagged, not accumulate hits scattered across unrelated items.

Builds a tiny real index (native hashes + attr + item-postings sidecar) via the
production writer, then scans crafted parquet docs through the Rust worker in both
gate modes.
"""

import io
import json
from pathlib import Path

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
from corpus_assay.attribution import AttributionIndex
from corpus_assay.indexing import (
    BuildIndexResult,
    BuildIndexStats,
    write_native_index_artifacts,
)
from corpus_assay.text_utils import iter_ngrams

NGRAM = 2
# Two protected evaluation items in one benchmark. Each contributes two bigrams.
ITEMS = [
    ("benchA", "alpha beta gamma"),  # bigrams: "alpha beta", "beta gamma"
    ("benchA", "delta epsilon zeta"),  # bigrams: "delta epsilon", "epsilon zeta"
]


@pytest.fixture(scope="module")
def item_index(tmp_path_factory: pytest.TempPathFactory) -> str:
    base = tmp_path_factory.mktemp("item_index")
    index_path = base / "index.native"

    attribution = AttributionIndex()
    for benchmark, text in ITEMS:
        item_idx = attribution.register_item(benchmark)
        toks = normalize_text(text)
        for ng in iter_ngrams(toks, NGRAM):
            if not ngram_allowed(ng):
                continue
            h = hash_ngram(ng)
            attribution.add_hash_for_source(h, benchmark)
            attribution.add_hash_for_item(h, item_idx)

    result = BuildIndexResult(
        ngram=NGRAM,
        hashes=set(attribution.hash2mask),
        attribution=attribution,
        stats=BuildIndexStats(),
        benchmark_specs=[],
        dataset_fingerprints=[],
    )
    write_native_index_artifacts(result, str(index_path))
    assert (Path(str(index_path) + ".items")).exists()
    return str(index_path)


def _parquet_bytes(texts: list[str], ids: list[str]) -> bytes:
    table = pa.Table.from_pydict({"text": texts, "doc_id": ids})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


# A doc reproducing one item (2 bigrams from item 0); a doc with one bigram from
# each of two items (union=2, but no single item reaches 2); and a clean doc.
SINGLE_ITEM_DOC = "alpha beta gamma"
CROSS_ITEM_DOC = "alpha beta frog delta epsilon"
CLEAN_DOC = "lorem ipsum dolor"


def _scan(index_path: str, gate_mode: str, min_longest_run: int = 0):
    data = _parquet_bytes(
        [SINGLE_ITEM_DOC, CROSS_ITEM_DOC, CLEAN_DOC],
        ["single", "cross", "clean"],
    )
    scanned, contaminated, results, _ = scan_stream_rust(
        input=io.BytesIO(data),
        text_key="text",
        id_key="doc_id",
        index_path=index_path,
        n=NGRAM,
        min_hits=2,
        min_coverage=0.0,
        gate_mode=gate_mode,
        min_longest_run=min_longest_run,
    )
    return scanned, contaminated, [json.loads(r) for r in results]


def test_union_mode_flags_cross_item_doc(item_index: str) -> None:
    """Default behavior: union gate flags both the single- and cross-item docs,
    and attaches per-item attribution where a single item is responsible."""
    scanned, contaminated, records = _scan(item_index, "union")
    assert scanned == 3
    assert contaminated == 2
    recs = {r["doc_id"]: r for r in records}
    assert set(recs) == {"single-part-0", "cross-part-0"}
    # The single-item doc gets per-item attribution (item 0, 2 hits, run 3 tokens).
    # Both hashes are unique to item 0, so the split is 2 unique / 0 shared / 0 boiler.
    assert recs["single-part-0"].get("item_hits") == [
        {
            "item_id": 0,
            "hits": 2,
            "longest_run_tokens": 3,
            "unique_hits": 2,
            "shared_hits": 0,
            "boilerplate_hits": 0,
        }
    ]
    # The cross-item doc is flagged by aggregation; no single item reaches min_hits,
    # so it carries no single-item attribution (the scatter is itself informative).
    assert "item_hits" not in recs["cross-part-0"]


def test_auto_aliases_union(item_index: str) -> None:
    """'auto' resolves to the union gate (max recall), not item gating."""
    _, contaminated, _ = _scan(item_index, "auto")
    assert contaminated == 2


def test_item_mode_rejects_cross_item_doc(item_index: str) -> None:
    """Item gate: only the single-item doc is flagged; cross-item is not."""
    scanned, contaminated, records = _scan(item_index, "item")
    assert scanned == 3
    assert contaminated == 1
    assert len(records) == 1
    rec = records[0]
    assert rec["doc_id"] == "single-part-0"
    # per-item evidence: item 0 with 2 hits and a contiguous run of 3 tokens
    assert rec["item_hits"] == [
        {
            "item_id": 0,
            "hits": 2,
            "longest_run_tokens": 3,
            "unique_hits": 2,
            "shared_hits": 0,
            "boilerplate_hits": 0,
        }
    ]
    assert rec["match_count"] == 2


def test_longest_run_clause(item_index: str) -> None:
    """The optional locality clause can veto a sub-run match."""
    # The single-item doc has a contiguous run of 3 tokens; requiring 4 vetoes it.
    _, contaminated_strict, _ = _scan(item_index, "item", min_longest_run=4)
    assert contaminated_strict == 0
    _, contaminated_ok, _ = _scan(item_index, "item", min_longest_run=3)
    assert contaminated_ok == 1


def _build_item_index(base: Path, items: list[tuple[str, str]]) -> str:
    """Build a native+attr+items index directly from (benchmark, text) items,
    registering each item so per-hash item fan-out is exactly as constructed."""
    index_path = base / "split_index.native"
    attribution = AttributionIndex()
    for benchmark, text in items:
        item_idx = attribution.register_item(benchmark)
        toks = normalize_text(text)
        for ng in iter_ngrams(toks, NGRAM):
            if not ngram_allowed(ng):
                continue
            h = hash_ngram(ng)
            attribution.add_hash_for_source(h, benchmark)
            attribution.add_hash_for_item(h, item_idx)
    result = BuildIndexResult(
        ngram=NGRAM,
        hashes=set(attribution.hash2mask),
        attribution=attribution,
        stats=BuildIndexStats(),
        benchmark_specs=[],
        dataset_fingerprints=[],
    )
    write_native_index_artifacts(result, str(index_path))
    return str(index_path)


def test_item_hits_unique_shared_boilerplate_split(tmp_path: Path) -> None:
    """Each per-item hit is bucketed by the matched n-gram's global item fan-out:
    fan-out 1 -> unique, >= boilerplate_threshold -> boilerplate, else shared."""
    # Overlap structure (bigrams): table-chair & chair-couch are unique to item 0;
    # couch-shelf & shelf-bench are shared by items 0-1 (fan-out 2); bench-stool is
    # in items 0-3 (fan-out 4).
    items = [
        ("benchA", "table chair couch shelf bench stool"),
        ("benchA", "couch shelf bench stool"),
        ("benchA", "plant bench stool"),
        ("benchA", "bench stool"),
    ]
    index_path = _build_item_index(tmp_path, items)

    doc = "table chair couch shelf bench stool"
    data = _parquet_bytes([doc], ["d0"])
    _, contaminated, results, _ = scan_stream_rust(
        input=io.BytesIO(data),
        text_key="text",
        id_key="doc_id",
        index_path=index_path,
        n=NGRAM,
        min_hits=1,
        min_coverage=0.0,
        gate_mode="item",
        boilerplate_threshold=3,
    )
    assert contaminated == 1
    rec = json.loads(results[0])
    # Strongest evidence first: item 0 reproduces all 5 indexed bigrams.
    item0 = rec["item_hits"][0]
    assert item0["item_id"] == 0
    assert item0["hits"] == 5
    assert item0["unique_hits"] == 2  # table-chair, chair-couch
    assert item0["shared_hits"] == 2  # couch-shelf, shelf-bench
    assert item0["boilerplate_hits"] == 1  # bench-stool (fan-out 4 >= 3)
    # Split partitions the distinct hits.
    assert (
        item0["unique_hits"] + item0["shared_hits"] + item0["boilerplate_hits"]
        == item0["hits"]
    )


def _build_index(tmp: Path, name: str, items: list[tuple[str, str]]) -> str:
    """Build a real native+attr+items index from (benchmark, text) items."""
    from corpus_assay.benchmark_spec import BenchmarkSpec
    from corpus_assay.indexing import build_index, write_native_index_artifacts

    mapping: dict[str, list[str]] = {}
    for benchmark, text in items:
        mapping.setdefault(benchmark, []).append(text)

    def factory(spec: BenchmarkSpec):
        class _L:
            dataset_fingerprints: list = []

            def load_positive(self):
                yield from mapping[spec.name]

            def load_negative(self):
                return iter(())

        return _L()

    specs = [BenchmarkSpec(name=b, dataset="x", fields=["text"]) for b in mapping]
    res = build_index(benchmarks=specs, ngram=NGRAM, loader_factory=factory)  # type: ignore[arg-type]
    idx = tmp / f"{name}.native"
    write_native_index_artifacts(res, str(idx))
    return str(idx)


def test_item_mode_ignores_stale_sidecar(tmp_path: Path) -> None:
    """A `.items` sidecar whose hashes are absent from the active index must not
    decide a scan (guards against a stale/mismatched sidecar)."""
    idx_a = _build_index(tmp_path, "a", [("A", "alpha beta gamma")])
    idx_b = _build_index(tmp_path, "b", [("B", "kappa lambda mu")])
    # Plant index A's item sidecar next to index B (its hashes aren't in B).
    (Path(idx_b + ".items")).write_bytes(Path(idx_a + ".items").read_bytes())

    data = _parquet_bytes(["alpha beta gamma"], ["a-doc"])
    _, contaminated, _, _ = scan_stream_rust(
        input=io.BytesIO(data),
        text_key="text",
        id_key="doc_id",
        index_path=idx_b,
        n=NGRAM,
        min_hits=2,
        min_coverage=0.0,
        gate_mode="item",
    )
    # The doc's hashes are in the stale sidecar but not in index B -> no flag.
    assert contaminated == 0


def test_item_mode_requires_sidecar(tmp_path: Path) -> None:
    """Explicit item mode against an index without postings errors clearly."""
    # Build a native+attr index WITHOUT item postings (no .items sidecar).
    text = "alpha beta gamma delta"
    toks = normalize_text(text)
    ngrams = [ng for ng in iter_ngrams(toks, NGRAM) if ngram_allowed(ng)]
    h = hash_ngram(ngrams[0])
    native = tmp_path / "noitems.native"
    write_native_index(native, ngram=NGRAM, hashes={h}, sources={h: "src"})
    assert not Path(str(native) + ".items").exists()

    data = _parquet_bytes([text], ["d1"])
    with pytest.raises(ValueError, match=r"requires an item sidecar"):
        scan_stream_rust(
            input=io.BytesIO(data),
            text_key="text",
            id_key="doc_id",
            index_path=str(native),
            n=NGRAM,
            min_hits=1,
            min_coverage=0.0,
            gate_mode="item",
        )
