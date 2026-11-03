from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# v2 adds ``benchmark_stats``. v1 files (no stats) still load.
INDEX_META_SCHEMA_VERSION = 2


class HashFunctionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class BenchmarkStats(BaseModel):
    """Per-benchmark counts recorded at build time (schema v2)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # Protected items (rows of the indexed splits).
    items: int
    # Distinct n-gram hashes attributed to the benchmark (after subtraction).
    ngrams: int
    # Items the n-gram rule cannot flag: no n-grams at all, and fewer than
    # ``min_hits`` n-grams (a superset of the first).
    items_zero_ngrams: int = 0
    items_below_min_hits: int = 0
    # The min_hits the previous count was measured against.
    min_hits: int | None = None


class IndexMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=INDEX_META_SCHEMA_VERSION)
    created_at: str | None = None
    library_version: str | None = None
    rust_extension_version: str | None = None
    ngram: int
    ngram_config: dict[str, Any] | None = None
    hash_function: HashFunctionMetadata
    normalization_config_sha: str
    index_format_version: int
    ngram_config_version: str
    ngram_config_hash: str
    benchmarks: list[dict[str, Any]] = Field(default_factory=list)
    benchmark_specs: list[dict[str, Any]] = Field(default_factory=list)
    hf_datasets: list[dict[str, Any]] = Field(default_factory=list)
    files: dict[str, Any] | None = None
    benchmark_stats: list[BenchmarkStats] = Field(default_factory=list)
