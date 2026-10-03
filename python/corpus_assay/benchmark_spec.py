from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BenchmarkSpec(BaseModel):
    """
    Normalized benchmark spec.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: str
    # "Positive" splits: These go into the index
    splits: list[str] = Field(default_factory=list)

    # "Negative" splits: These are subtracted (e.g., train, auxiliary)
    subtraction_splits: list[str] | None = None

    revision: str | None = None

    # Optional data_files override to pin subsets or local files.
    data_files: dict[str, Any] | list[Any] | str | None = None

    # Explicit single config name (alternative to configs list).
    config_name: str | None = None

    fields: list[str] = Field(default_factory=list)

    # Configs to drop
    drop_configs: list[str] | None = None

    # If configs contains "*", we dynamically fetch all available configs
    configs: list[str] = Field(default_factory=list)

    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _validate_config_fields(self) -> "BenchmarkSpec":
        if self.config_name is not None and not self.config_name.strip():
            raise ValueError("config_name must be a non-empty string when provided")
        if self.config_name and self.configs:
            raise ValueError("Use either config_name or configs, not both")
        return self

    @field_validator("splits")
    @classmethod
    def _validate_splits(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("splits must include at least one split")
        if any(not split.strip() for split in v):
            raise ValueError("splits entries must be non-empty strings")
        return v

    @field_validator("fields")
    @classmethod
    def _validate_fields(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("fields must include at least one field")
        if any(not field.strip() for field in v):
            raise ValueError("fields entries must be non-empty strings")
        return v
