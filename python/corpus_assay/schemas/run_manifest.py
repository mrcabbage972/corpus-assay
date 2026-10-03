from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

RUN_MANIFEST_SCHEMA_VERSION = 1


class RunIndexInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    meta_path: str | None = None
    index_format_version: int
    meta_sha256: str | None = None


class RunScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_key: str
    id_key: str | None
    n: int
    min_hits: int
    min_coverage: float
    packed_doc_sep: str
    packed_doc_sep_typo: str | None
    stopgrams_path: str | None = None


class RunHFDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_repo: str
    config_name: str | None = None
    split: str | None = None
    revision: str | None = None
    dataset_fingerprint: str | None = None
    data_files: dict[str, Any] | list[Any] | str | None = None


class RunParquetShard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str | None = None
    bytes: int | None = None


class RunInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_paths: list[str] = Field(default_factory=list)
    hf_datasets: list[RunHFDatasetInput] = Field(default_factory=list)
    parquet_shards: list[RunParquetShard] = Field(default_factory=list)


class RunOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_ndjson: str
    hits_bin: str
    final_summary_json: str


class RunStopGramsInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    meta_path: str | None = None
    format_version: int | None = None
    meta_sha256: str | None = None
    native_sha256: str | None = None
    tau: float | None = None
    df_threshold: int | None = None
    stopgram_count: int | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=RUN_MANIFEST_SCHEMA_VERSION)
    run_id: UUID
    started_at: str
    library_version: str
    rust_extension_version: str
    index: RunIndexInfo
    scan_config: RunScanConfig
    inputs: RunInputs
    outputs: RunOutputs
    config: dict[str, Any] | None = None
    stopgrams: RunStopGramsInfo | None = None
