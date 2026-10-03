from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

STOPGRAMS_META_SCHEMA_VERSION = 1


class StopGramsMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=STOPGRAMS_META_SCHEMA_VERSION)
    format_version: int
    created_at: str | None = None
    library_version: str | None = None
    rust_extension_version: str | None = None
    ngram: int
    hash_fn_id: str
    doc_count_B: int
    tau: float
    df_threshold: int
    stopgram_count: int
    normalization_config_sha: str | None = None
    ngram_config_hash: str | None = None
    built_from: dict[str, Any] | None = None
    files: dict[str, str] | None = None
