from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCAN_SUMMARY_SCHEMA_VERSION = 1
FINAL_SUMMARY_SCHEMA_VERSION = 1


class ScanSummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCAN_SUMMARY_SCHEMA_VERSION)
    file: str
    scanned: int = 0
    contaminated: int = 0
    contam_rate: float = 0.0
    empty_after_filter: int = 0
    per_source: dict[str, dict[str, int]] = Field(default_factory=dict)
    error: str | None = None
    run_fingerprint: dict[str, Any] | None = None


class FinalSummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=FINAL_SUMMARY_SCHEMA_VERSION)
    total_files_scanned: int
    total_docs_scanned: int
    total_contaminated: int
    contamination_rate: float
    total_empty_after_filter: int = 0
    failed_files: int
    total_files: int
