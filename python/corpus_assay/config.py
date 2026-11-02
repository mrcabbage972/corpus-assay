from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

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

DEFAULT_PACKED_DOC_SEP = "<|endoftext|>"
DEFAULT_PACKED_DOC_SEP_TYPO = "<|endoftext}>"

# Which items the exact-match channel matches: none (off), the items with fewer
# than min_hits n-grams (short), or every item (all).
ExactChannel = Literal["off", "short", "all"]
DEFAULT_EXACT_MIN_CHARS = 12


class DecontaminationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmarks: list[BenchmarkSpec] = Field(..., min_length=1)
    ngram: int = Field(default=13, gt=0)
    min_hits: int = Field(default=3, gt=0)
    min_coverage: float = Field(default=0.001, ge=0.0, le=1.0)
    # Share of one item's n-grams a document must hold (with >= min_hits hits) to
    # be flagged whatever its document coverage; 0 turns it off. See ScanConfig.
    min_item_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    # Exact-match channel; see ScanConfig.
    exact_channel: ExactChannel = "off"
    exact_min_chars: int = Field(default=DEFAULT_EXACT_MIN_CHARS, ge=1)
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
    # Text columns to scan; each is scanned as its own sub-document(s).
    # Default: ["text"], or none when `messages_key` is set.
    text_keys: list[str] = Field(default_factory=list)
    # The single text column of 0.1 configs. Still accepted as input (a name or
    # a list) and folded into `text_keys`; afterwards it reads back the first
    # text key. Not serialized: dumps carry `text_keys` only.
    text_key: str | None = Field(default=None, exclude=True)
    # Optional chat column, a list of ``{role, content}`` structs; each turn is
    # scanned as its own sub-document(s).
    messages_key: str | None = None
    id_key: str | None
    index_path: str  # Path to .native file
    out_dir: str
    min_hits: int = 3
    ngram: int = 13
    min_coverage: float = Field(default=0.001, ge=0.0, le=1.0)
    workers: int = os.cpu_count() or 1
    index_backend: str = "auto"
    # Records are split on this separator and each part is scored on its own.
    # `None` turns splitting off (and the typo replacement with it): a record is
    # one document, and its doc id gets no `-part-<k>` suffix.
    packed_doc_sep: str | None = DEFAULT_PACKED_DOC_SEP
    packed_doc_sep_typo: str | None = DEFAULT_PACKED_DOC_SEP_TYPO
    # Cap on the flagged sub-document records returned per shard (the
    # `contaminated_docs/` sample: the first N flagged, in scan order). `None` (or
    # 0, normalized to `None`) returns every flagged record; hits.bin is always
    # exhaustive regardless.
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
    # Per-item coverage threshold: the share of one protected item's indexed
    # n-grams a document holds (item_coverage = hits / item n-grams). 0 disables.
    # In union mode an item with >= min_hits hits at this coverage flags the
    # document whatever its document coverage (an OR path to the union rule); in
    # item mode it is an extra requirement on the flagging item. Needs `.items`.
    min_item_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    # Exact-match channel: match each eligible item's question (its first field)
    # verbatim, after light normalization (NFKC, lowercase, whitespace runs
    # collapsed; numbers, operators, punctuation and stop words kept), on word
    # boundaries. A match flags the sub-document whatever the n-gram gate says.
    # "short": items with fewer than min_hits n-grams (the n-gram rule cannot
    # flag them); "all": every item; "off" (default). Questions shorter than
    # exact_min_chars characters (after light normalization) are not matched.
    # Needs the `.items.text.jsonl` sidecar.
    exact_channel: ExactChannel = "off"
    exact_min_chars: int = Field(default=DEFAULT_EXACT_MIN_CHARS, ge=1)

    @field_validator("index_backend")
    @classmethod
    def _validate_index_backend(cls, v: str) -> str:
        mode = v.lower()
        if mode not in {"auto", "memory", "mmap"}:
            raise ValueError("index_backend must be one of: auto, memory, mmap")
        return mode

    @model_validator(mode="before")
    @classmethod
    def _accept_text_key(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "text_key" not in data:
            return data
        data = dict(data)
        legacy = data.pop("text_key")
        if legacy is None:
            return data
        keys = [legacy] if isinstance(legacy, str) else list(legacy)
        given = list(data.get("text_keys") or [])
        if given and keys not in (given, given[:1]):
            raise ValueError("text_key and text_keys disagree; pass text_keys only")
        data["text_keys"] = given or keys
        return data

    @model_validator(mode="after")
    def _validate_text_keys(self) -> "ScanConfig":
        if not self.text_keys and self.messages_key is None:
            # Neither given: the historical default column.
            self.text_keys = ["text"]
        keys = [*self.text_keys, *([self.messages_key] if self.messages_key else [])]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError(
                "text_keys and messages_key must be non-empty and distinct"
            )
        self.text_key = self.text_keys[0] if self.text_keys else None
        return self

    @field_validator("max_return_records")
    @classmethod
    def _validate_max_return_records(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v < 0:
            raise ValueError("max_return_records must be >= 0 (0 means no cap)")
        return v or None

    @field_validator("gate_mode")
    @classmethod
    def _validate_gate_mode(cls, v: str) -> str:
        mode = v.lower()
        if mode not in {"auto", "item", "union"}:
            raise ValueError("gate_mode must be one of: auto, item, union")
        return mode

    @model_validator(mode="after")
    def _validate_packed_doc_sep(self) -> "ScanConfig":
        if self.packed_doc_sep == "":
            raise ValueError(
                "packed_doc_sep must be non-empty; use None to disable splitting"
            )
        if self.packed_doc_sep is None:
            self.packed_doc_sep_typo = None
        return self

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
    # Build even when a benchmark loads no rows or indexes no n-grams.
    allow_empty_benchmarks: bool = False
    # Only used to count the items with fewer than min_hits n-grams.
    min_hits: int = Field(default=3, gt=0)
    # Drop n-grams held by many of the indexed items (None: off).
    item_stopgrams: ItemStopGramsConfig | None = None

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
    # `null` keeps every background document whole (no packed-document split).
    packed_doc_sep: str | None = DEFAULT_PACKED_DOC_SEP
    packed_doc_sep_typo: str | None = DEFAULT_PACKED_DOC_SEP_TYPO

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
        if self.packed_doc_sep == "":
            raise ValueError(
                "packed_doc_sep must be non-empty; use null to disable splitting"
            )
        if self.packed_doc_sep is None:
            self.packed_doc_sep_typo = None
        return self


class BackgroundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    corpus: BackgroundCorpusConfig | None = None


class ItemStopGramsConfig(BaseModel):
    """Stop-grams learned from the indexed items themselves (``stopgrams.items``).

    At index time, an n-gram held by more than ``max_item_share`` of all indexed
    items, and by at least ``min_count`` items, is dropped from the index (and
    written to ``<index>.item_stopgrams``). This removes text repeated across
    items, such as a shared instruction line, before matching.
    """

    model_config = ConfigDict(extra="forbid")

    max_item_share: float = Field(default=0.01, gt=0.0, le=1.0)
    min_count: int = Field(default=10, ge=2)


class StopGramsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str | None = None
    # Index-time stop-grams learned from the items (off when absent). Independent
    # of `enabled`/`path`, which apply background stop-grams at scan time.
    items: ItemStopGramsConfig | None = None

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
