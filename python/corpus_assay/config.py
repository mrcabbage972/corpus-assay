from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from corpus_assay.benchmark_registry import REGISTRY
from corpus_assay.benchmark_spec import BenchmarkSpec


class DecontaminationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmarks: list[BenchmarkSpec] = Field(..., min_length=1)
    ngram: int = Field(default=13, gt=0)
    min_hits: int = Field(default=3, gt=0)
    min_coverage: float = Field(default=0.001, ge=0.0, le=1.0)
    background: "BackgroundConfig | None" = None
    stopgrams: "StopGramsConfig | None" = None
    spaced: "SpacedSeedsConfig | None" = None

    @field_validator("benchmarks", mode="before")
    @classmethod
    def _parse_benchmarks(cls, v: Any) -> list[BenchmarkSpec]:
        if not isinstance(v, list):
            raise TypeError("benchmarks must be a list of benchmark entries")
        specs: list[BenchmarkSpec] = []
        for item in v:
            if isinstance(item, str):
                spec = REGISTRY.resolve_ref(item)
                specs.append(spec)
                continue

            if isinstance(item, dict) and "ref" in item:
                base = REGISTRY.resolve_ref(str(item["ref"]))
                overrides = {key: value for key, value in item.items() if key != "ref"}
                merged = {**base.model_dump(), **overrides}
                specs.append(BenchmarkSpec.model_validate(merged))
                continue

            specs.append(BenchmarkSpec.model_validate(item))
        return specs

    @field_validator("benchmarks")
    @classmethod
    def _dedupe_benchmarks(cls, v: list[BenchmarkSpec]) -> list[BenchmarkSpec]:
        names = [b.name for b in v]
        if len(set(names)) != len(names):
            raise ValueError("benchmarks must not contain duplicates")
        return v


class ScanConfig(BaseModel):
    inputs: list[str]
    text_key: str
    id_key: str | None
    index_path: str  # Path to .native file
    out_dir: str
    min_hits: int = 3
    ngram: int = 13
    min_coverage: float = Field(default=0.001, ge=0.0, le=1.0)
    workers: int = os.cpu_count() or 1
    index_backend: str = "auto"
    packed_doc_sep: str = "<|endoftext|>"
    packed_doc_sep_typo: str | None = "<|endoftext}>"
    max_return_records: int | None = 500
    target_hf_dataset: str | None = None
    target_hf_split: str | None = None
    target_hf_revision: str | None = None
    target_hf_fingerprint: str | None = None
    stopgrams_path: str | None = None
    # Experimental spaced-seed recall channel (optional). When set, the Rust
    # scanner ORs a token-level spaced-verified gate into the exact gate (union
    # recall). The verifier defaults match ``corpus_assay.spaced_seeds``; see
    # ``docs/experimental-spaced-seeds.md``.
    spaced_path: str | None = None
    spaced_min_loci: int = Field(default=3, gt=0)
    spaced_ver_min_span: int = Field(default=17, gt=0)
    spaced_ver_identity: float = Field(default=0.85, ge=0.0, le=1.0)
    # Contamination gate. "union" (default) flags on the index-wide hit total,
    # maximising recall; per-item attribution is still emitted when an
    # `<index>.items` sidecar is present. "item" is the stricter opt-in variant
    # that requires a single protected item to clear min_hits/min_coverage.
    # "auto" is accepted as an alias for "union".
    gate_mode: str = "union"
    # Optional locality clause for item gate: require a contiguous shared run of
    # at least this many normalized tokens with some item (0 disables it).
    min_longest_run: int = Field(default=0, ge=0)
    # Item fan-out at/above which a matched n-gram is counted as boilerplate in the
    # per-item attribution split (unique / shared / boilerplate) emitted for
    # flagged docs.
    boilerplate_threshold: int = Field(default=50, ge=2)

    @field_validator("index_backend")
    @classmethod
    def _validate_index_backend(cls, v: str) -> str:
        mode = v.lower()
        if mode not in {"auto", "memory", "mmap"}:
            raise ValueError("index_backend must be one of: auto, memory, mmap")
        return mode

    @field_validator("gate_mode")
    @classmethod
    def _validate_gate_mode(cls, v: str) -> str:
        mode = v.lower()
        if mode not in {"auto", "item", "union"}:
            raise ValueError("gate_mode must be one of: auto, item, union")
        return mode

    @model_validator(mode="after")
    def _validate_stopgrams(self) -> "ScanConfig":
        if self.stopgrams_path is not None and not self.stopgrams_path:
            raise ValueError("stopgrams_path must be non-empty if set")
        return self

    @model_validator(mode="after")
    def _validate_spaced(self) -> "ScanConfig":
        if self.spaced_path is not None and not self.spaced_path:
            raise ValueError("spaced_path must be non-empty if set")
        return self


class BuildIndexConfig(BaseModel):
    out_path: str
    ngram: int
    benchmarks: list[BenchmarkSpec] = Field(..., min_length=1)
    allow_unpinned: bool = False

    @field_validator("benchmarks")
    @classmethod
    def _validate_benchmarks(
        cls, benchmarks: list[BenchmarkSpec]
    ) -> list[BenchmarkSpec]:
        names = [b.name for b in benchmarks]
        if len(set(names)) != len(names):
            raise ValueError("benchmarks must not contain duplicates")
        return benchmarks


class BackgroundCorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str | None = None
    local_path: str | None = None
    config_name: str | None = None
    split: str = "train"
    fields: list[str] = Field(default_factory=list)
    revision: str | None = None
    packed_doc_sep: str = "<|endoftext|>"
    packed_doc_sep_typo: str | None = "<|endoftext}>"

    @model_validator(mode="after")
    def _require_exactly_one_source(self) -> "BackgroundCorpusConfig":
        has_hf = self.dataset is not None
        has_local = self.local_path is not None
        if has_hf == has_local:
            raise ValueError(
                "Exactly one of 'dataset' (HuggingFace) or 'local_path' (local JSONL) must be set."
            )
        if has_local and not self.fields:
            self.fields = ["text"]
        if has_hf and not self.fields:
            raise ValueError(
                "'fields' must be a non-empty list when 'dataset' (HuggingFace) is set."
            )
        return self


class BackgroundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    corpus: BackgroundCorpusConfig | None = None


class StopGramsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str | None = None

    @model_validator(mode="after")
    def _validate_settings(self) -> "StopGramsConfig":
        if self.enabled and not self.path:
            raise ValueError("stopgrams.path is required when stopgrams is enabled")
        return self


class SpacedSeedsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str | None = None
    min_loci: int = Field(default=3, gt=0)
    ver_min_span: int = Field(default=17, gt=0)
    ver_identity: float = Field(default=0.85, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_settings(self) -> "SpacedSeedsConfig":
        if self.enabled and not self.path:
            raise ValueError("spaced.path is required when spaced is enabled")
        return self


DecontaminationConfig.model_rebuild()

_ConfigAdapter = TypeAdapter(DecontaminationConfig)


def load_decontamination_config(path: str | Path) -> DecontaminationConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/object")

    # extra="forbid" handles unknown keys; required fields handled by pydantic too.
    return _ConfigAdapter.validate_python(raw)
