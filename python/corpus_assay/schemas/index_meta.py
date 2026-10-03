from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

INDEX_META_SCHEMA_VERSION = 1


class HashFunctionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


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
