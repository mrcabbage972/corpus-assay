"""Test helper: write a native index (and its sidecars) from explicit hashes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from corpus_assay.attribution import AttributionIndex
from corpus_assay.indexing import (
    BuildIndexResult,
    BuildIndexStats,
    write_native_index_artifacts,
)


def write_native_index(
    path: str | Path,
    *,
    ngram: int,
    hashes: Iterable[int],
    sources: Mapping[int, str] | None = None,
) -> str:
    """Write ``path`` (+ ``.meta.json``, ``.attr``, ``.sources.json``) for ``hashes``.

    ``sources`` maps a hash to its benchmark name for attribution. No item postings
    are registered, so no ``.items`` sidecar is written.
    """
    attribution = AttributionIndex()
    for hash_value, source in (sources or {}).items():
        attribution.add_hash_for_source(hash_value, source)
    result = BuildIndexResult(
        ngram=ngram,
        hashes=set(hashes),
        attribution=attribution,
        stats=BuildIndexStats(),
        benchmark_specs=[],
        dataset_fingerprints=[],
    )
    write_native_index_artifacts(result, str(path))
    return str(path)
