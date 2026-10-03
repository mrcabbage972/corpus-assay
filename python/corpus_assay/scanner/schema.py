from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class PerSourceStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Unique leaked hash count attributed to this source within the shard.
    unique_hashes: int = 0


class PerSourceAggregate(BaseModel):
    model_config = ConfigDict(frozen=True)
    by_source: dict[str, PerSourceStats] = Field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, dict[str, int]]:
        return {sid: stats.model_dump() for sid, stats in self.by_source.items()}


class ScanSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file: str
    scanned: int = 0
    contaminated: int = 0
    contam_rate: float = 0.0
    empty_after_filter: int = 0
    per_source: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    error: str | None = None
    run_fingerprint: dict[str, Any] | None = None


class RunSummary(BaseModel):
    """
    Returned by do_scan(). Caller decides what to write/print.
    """

    model_config = ConfigDict(frozen=True)

    total_files: int
    ok_files: int
    failed_files: int
    total_docs_scanned: int
    total_contaminated: int
    contamination_rate: float
    total_empty_after_filter: int

    # Useful for downstream reporting (CSV, debugging, etc.)
    per_file: list[ScanSummary] = Field(default_factory=list)


class SourceHit(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_id: int = Field(ge=0)
    hits: int = Field(ge=0)


class ItemHit(BaseModel):
    """Per-item attribution evidence for one flagged document. ``hits`` is the
    number of distinct index n-grams matching the protected item; it partitions
    into ``unique_hits`` (n-gram belongs to exactly one item), ``shared_hits`` and
    ``boilerplate_hits`` (n-gram in ``>= boilerplate_threshold`` items). Mirrors the
    Rust ``ItemHit``."""

    model_config = ConfigDict(frozen=True)
    item_id: int = Field(ge=0)
    hits: int = Field(ge=0)
    longest_run_tokens: int = Field(default=0, ge=0)
    unique_hits: int = Field(default=0, ge=0)
    shared_hits: int = Field(default=0, ge=0)
    boilerplate_hits: int = Field(default=0, ge=0)


class ContaminationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: int | None = None
    doc_id: Any
    match_count: int | None = None
    src_hits: list[SourceHit] = Field(default_factory=list)
    # Per-item attribution evidence (item gate mode, or union mode with an
    # `.items` sidecar). Empty when no single item cleared the gate.
    item_hits: list[ItemHit] = Field(default_factory=list)
    # Protected item id the spaced channel verified against (spaced-only flags).
    spaced_item: int | None = None
