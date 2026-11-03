"""``_FILTER_MANIFEST.json``: what a ``filter`` run removed, and against what."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from corpus_assay.schemas.run_manifest import RunIndexInfo, RunScanConfig

FILTER_MANIFEST_SCHEMA_VERSION = 1

FilterGranularity = Literal["record", "subdoc"]


class FilterShard(BaseModel):
    """One input shard and the cleaned shard written for it."""

    model_config = ConfigDict(extra="forbid")

    input_path: str
    input_sha256: str | None = None
    # Relative to the filter output directory (the input's path relative to
    # the common directory of all inputs).
    output_path: str
    output_sha256: str | None = None
    # parquet, jsonl, jsonl.gz or jsonl.zst (the output has the input's format).
    format: str
    # Rows (JSONL: non-blank lines) before and after filtering.
    rows_before: int
    rows_after: int
    rows_removed: int
    # subdoc granularity: rows kept with some sub-document dropped or blanked.
    rows_modified: int = 0
    # Flagged sub-documents in the shard.
    flagged_subdocs: int = 0


class FilterCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows_removed: int = 0
    rows_modified: int = 0
    flagged_subdocs: int = 0


class FilterRemovals(FilterCounts):
    """Totals over every shard.

    A row or sub-document counts for each benchmark its evidence names (the
    benchmarks with n-gram hits, and the benchmarks of the matched items), so
    the per-benchmark counts can add up to more than the totals.
    """

    rows_before: int = 0
    rows_after: int = 0
    by_benchmark: dict[str, FilterCounts] = Field(default_factory=dict)
    # Flagged sub-documents per channel (ngram, spaced, exact).
    by_channel: dict[str, int] = Field(default_factory=dict)


class FilterManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=FILTER_MANIFEST_SCHEMA_VERSION)
    created_at: str
    library_version: str
    rust_extension_version: str
    granularity: FilterGranularity
    # The YAML config the thresholds came from, as given on the command line.
    config_path: str | None = None
    index: RunIndexInfo
    scan_config: RunScanConfig
    input_paths: list[str] = Field(default_factory=list)
    shards: list[FilterShard] = Field(default_factory=list)
    removals: FilterRemovals
    # Relative to the filter output directory.
    scan_dir: str
    removed_ids: str
    dataset_card_snippet: str
    # The full scan config (``ScanConfig.model_dump()``).
    config: dict[str, Any] | None = None
