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
    # Flagged sub-documents per enabled channel ("ngram", "spaced", "exact"); a
    # sub-document flagged by several channels counts in each. Empty in
    # summaries written before channels were counted.
    contaminated_by_channel: Dict[str, int] = Field(default_factory=dict)
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
    # Flagged sub-documents per channel, summed over the ok shards.
    total_contaminated_by_channel: dict[str, int] = Field(default_factory=dict)

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
    # v3: the item's indexed n-grams and the share of them the document holds
    # (hits / item_ngrams). Absent in older records.
    item_ngrams: int | None = Field(default=None, ge=0)
    item_coverage: float | None = Field(default=None, ge=0.0)
    # v3, with an `.items` v2 sidecar: `[field_idx, hits]` for each of the item's
    # spec fields with a hit, and those fields' names (when the index lists them).
    field_hits: list[tuple[int, int]] = Field(default_factory=list)
    fields_matched: list[str] = Field(default_factory=list)
    # v3, with the exact channel: the item's other fields found near the
    # question (see ExactHit).
    choices_found: int | None = Field(default=None, ge=0)
    choices_total: int | None = Field(default=None, ge=0)


class ExactHit(BaseModel):
    """An exact-channel match: the item's question (``field`` 0) occurs verbatim
    in the sub-document after light normalization, on word boundaries.

    ``choices_found`` / ``choices_total``: the item's other fields (each option
    of a list field separately) found within 1,000 characters of the matched
    question.
    """

    model_config = ConfigDict(frozen=True)
    item_id: int = Field(ge=0)
    field: int = Field(default=0, ge=0)
    choices_found: int | None = Field(default=None, ge=0)
    choices_total: int | None = Field(default=None, ge=0)


class ItemBest(BaseModel):
    """Best single-document hold on one item in a shard (``per_item_best/``)."""

    model_config = ConfigDict(frozen=True)
    item_id: int = Field(ge=0)
    hits: int = Field(ge=0)
    item_ngrams: int = Field(ge=0)
    coverage: float = Field(ge=0.0)
    doc_id: str


class ItemBestFile(BaseModel):
    """``per_item_best/<shard>.json``: every item with a hit in the shard."""

    schema_version: int = 1
    items: list[ItemBest] = Field(default_factory=list)


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
    # v3: source column (text key or messages key); role and turn index for
    # chat turns. Absent in v2 records.
    field: str | None = None
    role: str | None = None
    turn: int | None = None
    # v3, exact channel: the items whose question occurs verbatim.
    exact_hits: list[ExactHit] = Field(default_factory=list)
    # v3: channels that flagged the sub-document ("ngram", "spaced", "exact").
    # Empty in older records.
    channels: list[str] = Field(default_factory=list)
