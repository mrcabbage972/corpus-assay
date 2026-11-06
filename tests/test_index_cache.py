"""The run/run-hf index cache key must change whenever the resolved specs do."""

from __future__ import annotations

from pathlib import Path

import pytest

import corpus_assay.config as config_mod
import corpus_assay.services.index_cache as index_cache
from corpus_assay.benchmark_registry import BenchmarkRegistry
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.formats import FileType, write_header
from corpus_assay.services.index_cache import (
    benchmark_specs_hash,
    ensure_native_index,
    index_cache_path,
    native_index_ready,
    native_sidecar,
)


def _registry(fields: list[str]) -> BenchmarkRegistry:
    return BenchmarkRegistry(
        {
            "demo": BenchmarkSpec(
                name="Demo",
                dataset="demo/ds",
                revision="0" * 40,
                splits=["test"],
                fields=fields,
            )
        }
    )


def _write_fake_index(
    native: Path, *, items_type: FileType = FileType.ITEM_POSTINGS_V2
) -> None:
    """Placeholder files for every required path; `.items` gets a real header."""
    for path in index_cache.native_index_required_paths(native, require_items=True):
        path.write_text("x", encoding="utf-8")
    with native_sidecar(native, ".items").open("wb") as f:
        write_header(f, file_type=items_type, ngram_n=13, record_count=0)


def test_cache_with_v1_item_postings_is_not_ready(tmp_path: Path) -> None:
    """A cache built before field masks (`.items` v1) is rebuilt."""
    native = tmp_path / "index.native"
    _write_fake_index(native)
    assert native_index_ready(native)
    _write_fake_index(native, items_type=FileType.ITEM_POSTINGS)
    assert not native_index_ready(native)
    # The item table is required too.
    _write_fake_index(native)
    native_sidecar(native, ".items.meta.json").unlink()
    assert not native_index_ready(native)


def test_cache_without_the_item_text_sidecar_is_not_ready(tmp_path: Path) -> None:
    """A cache built before `.items.text.jsonl` is rebuilt (exact channel)."""
    native = tmp_path / "index.native"
    _write_fake_index(native)
    assert native_index_ready(native)
    native_sidecar(native, ".items.text.jsonl").unlink()
    assert not native_index_ready(native)


def test_cache_key_changes_when_a_registry_entry_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same config text, different registry content -> a different cached index."""
    config = tmp_path / "config.yaml"
    config.write_text("benchmarks: [demo]\n", encoding="utf-8")

    monkeypatch.setattr(config_mod, "REGISTRY", _registry(["question"]))
    before = index_cache_path(config, tmp_path)
    assert index_cache_path(config, tmp_path) == before  # deterministic

    monkeypatch.setattr(config_mod, "REGISTRY", _registry(["question", "answer"]))
    after = index_cache_path(config, tmp_path)
    assert after != before
    assert after.parent == before.parent and after.suffix == ".native"


def test_benchmark_specs_hash_is_order_sensitive_and_stable() -> None:
    a = BenchmarkSpec(name="A", dataset="d/a", splits=["test"], fields=["q"])
    b = BenchmarkSpec(name="B", dataset="d/b", splits=["test"], fields=["q"])
    assert benchmark_specs_hash([a, b]) == benchmark_specs_hash([a.model_copy(), b])
    assert benchmark_specs_hash([a, b]) != benchmark_specs_hash([b, a])
    assert benchmark_specs_hash([a]) != benchmark_specs_hash(
        [a.model_copy(update={"revision": "1" * 40})]
    )


def test_ensure_native_index_does_not_reuse_a_stale_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("benchmarks: [demo]\n", encoding="utf-8")
    cache = tmp_path / "cache"
    built: list[str] = []

    def fake_build(cfg) -> None:
        built.append(cfg.out_path)
        _write_fake_index(Path(cfg.out_path))

    monkeypatch.setattr(index_cache, "build_index_hf", fake_build)

    monkeypatch.setattr(config_mod, "REGISTRY", _registry(["question"]))
    first = ensure_native_index(config, cache)
    assert ensure_native_index(config, cache) == first  # cache hit
    assert len(built) == 1

    monkeypatch.setattr(config_mod, "REGISTRY", _registry(["question", "answer"]))
    second = ensure_native_index(config, cache)
    assert second != first
    assert built == [str(first), str(second)]
