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


def test_build_stats_warn_about_benchmarks_with_no_ngrams(capsys) -> None:
    from corpus_assay.attribution import AttributionIndex
    from corpus_assay.indexing import (
        BuildIndexResult,
        BuildIndexStats,
        _report_build_stats,
    )

    attribution = AttributionIndex()
    attribution.add_hash_for_source(123, "Long")
    result = BuildIndexResult(
        ngram=13,
        hashes={123},
        attribution=attribution,
        stats=BuildIndexStats(),
        benchmark_specs=[{"name": "Long"}, {"name": "Short"}],
        dataset_fingerprints=[],
    )
    _report_build_stats(result)
    out = capsys.readouterr().out
    assert "no n-grams indexed for: Short." in out
    assert "Long" not in out.split("no n-grams indexed for:")[1].split(".")[0]
