from __future__ import annotations

import pytest

from corpus_assay.benchmark_registry import BenchmarkRegistry
from corpus_assay.benchmark_spec import BenchmarkSpec


def test_registry_lists_keys_sorted() -> None:
    registry = BenchmarkRegistry(
        {
            "B": BenchmarkSpec(
                name="B",
                dataset="ds/b",
                splits=["test"],
                fields=["q"],
            ),
            "a": BenchmarkSpec(
                name="A",
                dataset="ds/a",
                splits=["test"],
                fields=["q"],
            ),
        }
    )

    assert registry.list_registry() == ["a", "b"]


def test_registry_resolve_returns_copy() -> None:
    spec = BenchmarkSpec(
        name="Demo",
        dataset="demo/ds",
        splits=["test"],
        fields=["question"],
    )
    registry = BenchmarkRegistry({"demo": spec})

    resolved = registry.resolve_ref("demo")
    resolved.fields.append("answer")

    assert spec.fields == ["question"]


def test_registry_unknown_ref_raises() -> None:
    registry = BenchmarkRegistry(
        {
            "demo": BenchmarkSpec(
                name="Demo",
                dataset="demo/ds",
                splits=["test"],
                fields=["question"],
            )
        }
    )

    with pytest.raises(KeyError, match="Unknown benchmark ref"):
        registry.resolve_ref("missing")
