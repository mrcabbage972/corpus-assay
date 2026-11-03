from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from corpus_assay.schemas.index_meta import BenchmarkStats

# Fields added after version 1 are optional, so the version stays 1 and older
# manifests still load.
RUN_MANIFEST_SCHEMA_VERSION = 1


class RunBenchmarkInfo(BaseModel):
    """One indexed benchmark split, as recorded in the index metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    hf_repo: str | None = None
    config_name: str | None = None
    split: str | None = None
    # The revision the config asked for, and the commit it resolved to.
    revision: str | None = None
    resolved_revision: str | None = None
    dataset_fingerprint: str | None = None


class RunIndexInfo(BaseModel):
    """The index a run scanned against.

    Everything after ``meta_sha256`` is absent (None / empty) in manifests
    written before it was recorded.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    meta_path: str | None = None
    index_format_version: int
    meta_sha256: str | None = None
    # SHA-256 of each index file (None when the file does not exist).
    native_sha256: str | None = None
    attr_sha256: str | None = None
    sources_sha256: str | None = None
    items_sha256: str | None = None
    items_meta_sha256: str | None = None
    items_text_sha256: str | None = None
    item_stopgrams_sha256: str | None = None
    # From the index metadata: the normalization config the index was built
    # under, and what it indexed.
    normalization_config_sha: str | None = None
    ngram_config_hash: str | None = None
    benchmarks: list[RunBenchmarkInfo] = Field(default_factory=list)
    benchmark_stats: list[BenchmarkStats] = Field(default_factory=list)


class RunScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # First text key, kept for readers of older manifests; `text_keys` is the
    # full list (empty in manifests written before it existed).
    text_key: str | None = None
    text_keys: list[str] = Field(default_factory=list)
    messages_key: str | None = None
    id_key: str | None
    n: int
    min_hits: int
    min_coverage: float
    # Per-item coverage threshold (0: off; absent in older manifests).
    min_item_coverage: float = 0.0
    # Gate settings (None in manifests written before they were recorded).
    gate_mode: str | None = None
    min_longest_run: int | None = None
    boilerplate_threshold: int | None = None
    # Exact-match channel and --confirm (absent in older manifests: off).
    exact_channel: str = "off"
    exact_min_chars: int = 12
    confirm: bool = False
    # None: packed-document splitting was off (--no-split).
    packed_doc_sep: str | None
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
