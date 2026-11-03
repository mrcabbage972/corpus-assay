from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# v2 adds per-item attribution evidence (item_hits) and spaced_item, emitted by
# the scanner in item gate mode / union mode with an `.items` sidecar.
# v3 adds where in the record the sub-document came from: `field` (the text or
# messages column), and for chat turns `role` and `turn`. Mirrors
# DOC_RESULT_SCHEMA_VERSION in src/scan.rs.
DOC_RESULT_SCHEMA_VERSION = 3


class SourceHitOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: int = Field(ge=0)
    hits: int = Field(ge=0)


class ItemHitOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: int = Field(ge=0)
    hits: int = Field(ge=0)
    longest_run_tokens: int = Field(default=0, ge=0)
    unique_hits: int = Field(default=0, ge=0)
    shared_hits: int = Field(default=0, ge=0)
    boilerplate_hits: int = Field(default=0, ge=0)


class DocResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=DOC_RESULT_SCHEMA_VERSION)
    doc_id: str | int
    match_count: int = Field(ge=0)
    src_hits: list[tuple[int, int]] | list[SourceHitOut] = Field(default_factory=list)
    # Per-item attribution evidence; empty when no single item cleared the gate.
    # Accepts the v2 object form and the legacy pre-v2 tuple form
    # `[item_id, distinct_hits, longest_run_tokens]` so old NDJSON still validates.
    item_hits: list[tuple[int, int, int]] | list[ItemHitOut] = Field(
        default_factory=list
    )
    # Protected item id the spaced channel verified against (spaced-only flags).
    spaced_item: int | None = None
    # v3: the column the sub-document came from (a text key or the messages
    # key) and, for chat turns, the turn's role and its index in the list.
    field: str | None = None
    role: str | None = None
    turn: int | None = Field(default=None, ge=0)

    def normalized_doc_id(self) -> str:
        return str(self.doc_id)
