from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TypeAlias

import pytest

from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.indexing import _benchmark_spec_meta, build_index

# -------------------------
# Types
# -------------------------

LoaderFactory: TypeAlias = Callable[[BenchmarkSpec], "FakeLoader"]
LoaderMapping: TypeAlias = Mapping[str, tuple[list[str], list[str]]]


# -------------------------
# Test fakes / helpers
# -------------------------


class FakeLoader:
    def __init__(self, spec: BenchmarkSpec, positive: list[str], negative: list[str]):
        self.spec = spec
        self._positive = positive
        self._negative = negative

    def load_positive(self) -> Iterator[str]:
        yield from self._positive

    def load_negative(self) -> Iterator[str]:
        yield from self._negative


def make_loader_factory(mapping: LoaderMapping) -> LoaderFactory:
    """
    mapping:
      benchmark_name -> (positive_texts, negative_texts)

    Note: negative_texts is always a list (possibly empty) to keep typing simple.
    """

    def factory(spec: BenchmarkSpec) -> FakeLoader:
        pos, neg = mapping[spec.name]
        return FakeLoader(spec, pos, neg)

    return factory


# -------------------------
# Global monkeypatches
# -------------------------


@pytest.fixture(autouse=True)
def no_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    import corpus_assay.indexing as mod

    monkeypatch.setattr(mod, "tqdm", lambda it, **_kw: it)


@pytest.fixture(autouse=True)
def tiny_deterministic_ngram_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import corpus_assay.indexing as mod

    # Normalize: simple whitespace split (deterministic)
    monkeypatch.setattr(mod, "normalize_text", lambda s: s.split())

    # N-gram generation + filtering: indexing now uses iter_ngrams_with_filter(tokens, n)
    # which yields (ngram_str, allowed_bool, reject_idx). For n==2, yield bigrams as strings.
    def fake_iter_ngrams_with_filter(toks: list[str], n: int):
        if n != 2:
            return iter(())
        pairs = (
            (" ".join([toks[i], toks[i + 1]]), True, None) for i in range(len(toks) - 1)
        )
        return iter(pairs)

    monkeypatch.setattr(mod, "iter_ngrams_with_filter", fake_iter_ngrams_with_filter)

    # Hashing: indexing calls hash_ngram(ngram_str)
    monkeypatch.setattr(mod, "hash_ngram", lambda ng: hash(ng))


# -------------------------
# Tests
# -------------------------


def test_single_benchmark_no_subtraction() -> None:
    spec = BenchmarkSpec(
        name="A",
        dataset="dummy",
        splits=["test"],
        fields=["text"],
        subtraction_splits=[],
    )

    factory = make_loader_factory({"A": (["a b c"], [])})

    res = build_index(benchmarks=[spec], ngram=2, loader_factory=factory)  # type: ignore

    expected = {hash("a b"), hash("b c")}
    assert res.hashes == expected
    assert set(res.attribution.hash2mask.keys()) == expected


def test_subtraction_removes_overlapping_ngrams() -> None:
    spec = BenchmarkSpec(
        name="A",
        dataset="dummy",
        splits=["test"],
        subtraction_splits=["train"],
        fields=["text"],
    )

    factory = make_loader_factory({"A": (["a b c"], ["b c d"])})

    res = build_index(benchmarks=[spec], ngram=2, loader_factory=factory)  # type: ignore

    assert res.hashes == {hash("a b")}


def test_full_subtraction_results_in_empty_index() -> None:
    spec = BenchmarkSpec(
        name="A",
        dataset="dummy",
        splits=["test"],
        subtraction_splits=["train"],
        fields=["text"],
    )

    factory = make_loader_factory({"A": (["a b c"], ["a b c"])})

    res = build_index(benchmarks=[spec], ngram=2, loader_factory=factory)  # type: ignore

    assert res.hashes == set()
    assert res.attribution.hash2mask == {}


def test_multiple_benchmarks_union_of_hashes() -> None:
    specs = [
        BenchmarkSpec(
            name="A",
            dataset="x",
            splits=["test"],
            fields=["text"],
            subtraction_splits=[],
        ),
        BenchmarkSpec(
            name="B",
            dataset="y",
            splits=["test"],
            fields=["text"],
            subtraction_splits=[],
        ),
    ]

    factory = make_loader_factory(
        {
            "A": (["a b"], []),
            "B": (["b c"], []),
        }
    )

    res = build_index(benchmarks=specs, ngram=2, loader_factory=factory)  # type: ignore

    assert res.hashes == {hash("a b"), hash("b c")}


def test_shared_ngram_attributed_to_multiple_sources() -> None:
    specs = [
        BenchmarkSpec(
            name="A",
            dataset="x",
            splits=["test"],
            fields=["text"],
            subtraction_splits=[],
        ),
        BenchmarkSpec(
            name="B",
            dataset="y",
            splits=["test"],
            fields=["text"],
            subtraction_splits=[],
        ),
    ]

    factory = make_loader_factory({"A": (["a b"], []), "B": (["a b"], [])})

    res = build_index(benchmarks=specs, ngram=2, loader_factory=factory)  # type: ignore

    h = hash("a b")
    assert res.hashes == {h}
    assert h in res.attribution.hash2mask


@pytest.mark.parametrize(
    ("pos", "neg", "expected"),
    [
        (["a b c"], [], {"a b", "b c"}),
        (["a b c"], ["a b"], {"b c"}),
        (["a b"], ["a b"], set()),
    ],
)
def test_parametrized_subtraction_cases(
    pos: list[str], neg: list[str], expected: set[str]
) -> None:
    spec = BenchmarkSpec(
        name="A",
        dataset="dummy",
        splits=["test"],
        subtraction_splits=["train"] if neg else [],
        fields=["text"],
    )

    factory = make_loader_factory({"A": (pos, neg)})

    res = build_index(benchmarks=[spec], ngram=2, loader_factory=factory)  # type: ignore

    assert res.hashes == {hash(x) for x in expected}


def test_benchmark_spec_meta_includes_revision() -> None:
    spec = BenchmarkSpec(
        name="Meta",
        dataset="demo/meta",
        revision="abc123",
        splits=["test"],
        fields=["text"],
        configs=["cfg"],
        drop_configs=["drop"],
        trust_remote_code=True,
    )

    meta = _benchmark_spec_meta(spec)

    assert meta["revision"] == "abc123"
    assert meta["configs"] == ["cfg"]
    assert meta["drop_configs"] == ["drop"]
    assert meta["trust_remote_code"] is True


def _spec(name: str, dataset: str = "dummy") -> BenchmarkSpec:
    return BenchmarkSpec(name=name, dataset=dataset, splits=["test"], fields=["t"])


def test_build_index_counts_items_and_ngrams_per_benchmark() -> None:
    factory = make_loader_factory(
        {
            # 3 items: 3 bigrams, 1 bigram, 0 bigrams (a single token).
            "A": (["a b c d", "e f", "g"], []),
            "B": (["b c x y z"], []),
        }
    )
    res = build_index(
        benchmarks=[_spec("A"), _spec("B")],
        ngram=2,
        min_hits=2,
        loader_factory=factory,  # type: ignore[arg-type]
    )
    a, b = res.benchmark_stats
    assert (a.name, a.items, a.ngrams) == ("A", 3, 4)
    assert (a.items_zero_ngrams, a.items_below_min_hits, a.min_hits) == (1, 2, 2)
    # "b c" is shared with A and counts for both benchmarks.
    assert (b.name, b.items, b.ngrams) == ("B", 1, 4)
    assert (b.items_zero_ngrams, b.items_below_min_hits) == (0, 0)


def test_empty_benchmark_problems_name_benchmark_and_reason() -> None:
    from corpus_assay.indexing import BenchmarkBuildStats, empty_benchmark_problems

    class StatusLoader(FakeLoader):
        def __init__(self, spec: BenchmarkSpec, positive: list[str]) -> None:
            super().__init__(spec, positive, [])
            from corpus_assay.benchmarks import BenchmarkLoadStatus

            self.load_status = BenchmarkLoadStatus(
                failures=[] if positive else ["config=None split='test': boom"]
            )

    texts = {"Full": ["a b c"], "Missing": [], "Short": ["a"]}
    res = build_index(
        benchmarks=[_spec("Full"), _spec("Missing"), _spec("Short")],
        ngram=2,
        loader_factory=lambda spec: StatusLoader(spec, texts[spec.name]),  # type: ignore[arg-type,return-value]
    )
    missing, short = empty_benchmark_problems(res)
    assert missing.startswith("Missing (dummy): loaded no rows.")
    assert "config=None split='test': boom" in missing
    assert "gated" not in missing
    assert short.startswith("Short (dummy): loaded 1 items but indexed no n-grams.")
    assert "ngram=2" in short

    # A gated registry dataset (or a Hub error saying so) gets the login hint.
    gated = _result_with_stats(
        BenchmarkBuildStats(name="GPQA-Diamond", dataset="Idavidrein/gpqa"),
        BenchmarkBuildStats(
            name="Other",
            dataset="someone/private",
            load_failures=["DatasetNotFoundError: is a gated dataset on the Hub"],
        ),
    )
    for message in empty_benchmark_problems(gated):
        assert "hf auth login" in message
        assert "accept the dataset terms" in message


def _result_with_stats(*stats):
    from corpus_assay.attribution import AttributionIndex
    from corpus_assay.indexing import BuildIndexResult, BuildIndexStats

    return BuildIndexResult(
        ngram=13,
        hashes=set(),
        attribution=AttributionIndex(),
        stats=BuildIndexStats(),
        benchmark_specs=[],
        dataset_fingerprints=[],
        benchmark_stats=list(stats),
    )


def test_check_benchmarks_not_empty_raises_unless_allowed() -> None:
    from corpus_assay.indexing import (
        BenchmarkBuildStats,
        EmptyBenchmarkError,
        check_benchmarks_not_empty,
    )

    res = _result_with_stats(
        BenchmarkBuildStats(name="Ok", dataset="d", items=2, ngrams=5),
        BenchmarkBuildStats(name="Empty", dataset="d"),
    )
    with pytest.raises(EmptyBenchmarkError) as excinfo:
        check_benchmarks_not_empty(res)
    message = str(excinfo.value)
    assert "Empty (d): loaded no rows" in message
    assert "Ok" not in message
    assert "--allow-empty-benchmarks" in message
    assert "No index was written" in message

    problems = check_benchmarks_not_empty(res, allow_empty_benchmarks=True)
    assert len(problems) == 1


def test_build_stats_print_benchmark_table(capsys) -> None:
    from corpus_assay.indexing import BenchmarkBuildStats, _report_build_stats

    res = _result_with_stats(
        BenchmarkBuildStats(
            name="Long",
            dataset="d",
            items=10,
            ngrams=1234,
            items_zero_ngrams=1,
            items_below_min_hits=2,
            warnings=["Long: every 'label' value in split 'test' is missing"],
        ),
        BenchmarkBuildStats(name="Short", dataset="d", items=3, ngrams=0),
    )
    _report_build_stats(res, problems=["Short (d): loaded 3 items but ..."])
    out = capsys.readouterr().out
    lines = out.splitlines()
    header = next(line for line in lines if "benchmark" in line and "items" in line)
    assert "< 3 n-grams" in header
    assert any(line.split()[:3] == ["Long", "10", "1,234"] for line in lines)
    assert "2 item(s) have fewer than min_hits=3 n-grams" in out
    assert "WARNING (--allow-empty-benchmarks): Short (d)" in out
    assert "WARNING: Long: every 'label' value" in out
