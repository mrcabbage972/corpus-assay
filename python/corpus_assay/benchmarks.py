import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from datasets import get_dataset_config_names

from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.constants import FIELD_SEPARATOR
from corpus_assay.utils import _load_dataset_optional_revision

logger = logging.getLogger(__name__)


def _extract_resolved_revision(ds: Any) -> str | None:
    # Best-effort: depends on datasets version and backend
    # Do NOT guess.
    info = getattr(ds, "info", None)
    if info is not None:
        # Some versions expose commit via info
        rev = getattr(info, "revision", None)
        if isinstance(rev, str):
            return rev

    # Fallbacks: known-but-fragile internals (optional)
    rev = getattr(ds, "_revision", None)
    if isinstance(rev, str):
        return rev

    return None


def _to_text_chunks(x: Any) -> list[str]:
    """Recursively flatten arbitrary nested structures into a list of text chunks."""
    if x is None:
        return []

    # Strings are sequences, so handle them before Sequence
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []

    # Common scalar types
    if isinstance(x, (int, float, bool)):
        return [str(x)]

    # Mappings (dict-like)
    if isinstance(x, Mapping):
        # Prefer common HF patterns
        if "text" in x:
            return _to_text_chunks(x["text"])
        if "labels" in x:
            return _to_text_chunks(x["labels"])

        # Otherwise flatten values in a stable key order
        chunks: list[str] = []
        for k in sorted(x.keys(), key=lambda k: str(k)):
            chunks.extend(_to_text_chunks(x[k]))
        return chunks

    # Sequences (list/tuple) but not bytes
    if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray)):
        chunks: list[str] = []
        for item in x:
            chunks.extend(_to_text_chunks(item))
        return chunks

    # Fallback: string conversion
    s = str(x).strip()
    return [s] if s else []


def _concat_fields(obj: Any, keys: list[str]) -> str:
    """
    Deterministically concatenate selected fields into one string with field separators.
    Keeps 0/False, drops None/empty strings, flattens nested structures.
    """
    # tolerate non-dict rows (but mapping-like)
    if not isinstance(obj, Mapping):
        try:
            obj = dict(obj)  # type: ignore[arg-type]
        except Exception:
            # last resort: stringify whole object
            return str(obj).strip()

    chunks: list[str] = []
    for k in keys:
        field_chunks = _to_text_chunks(obj.get(k))
        if not field_chunks:
            continue
        chunks.append(" ".join(field_chunks))
    return f" {FIELD_SEPARATOR} ".join(chunks)


@dataclass
class BenchmarkLoadStatus:
    """What loading a benchmark's positive splits produced, for build-time checks."""

    rows: int = 0
    # "config=... split=..." entries that loaded.
    loaded: list[str] = field(default_factory=list)
    # Why a config or split did not load.
    failures: list[str] = field(default_factory=list)
    # Warnings about the loaded rows (e.g. hidden labels).
    warnings: list[str] = field(default_factory=list)


class BenchmarkLoader:
    """
    Encapsulates logic for loading texts from a benchmark spec.
    Supports:
      - configs = None/[]  -> load dataset without config (config_name=None)
      - configs containing "*" -> resolve all available configs
      - drop_configs -> removed from resolved configs
      - positive vs negative split targeting
    """

    def __init__(self, spec: BenchmarkSpec):
        self.spec = spec
        self.dataset_fingerprints: list[dict[str, Any]] = []
        self._fingerprint_keys: set[tuple[str, str | None, str]] = set()
        # Status of the last load_positive() call (rows, failures, warnings).
        self.load_status = BenchmarkLoadStatus()
        self._config_error: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    def load_positive(self) -> Iterator[str]:
        """Yields texts from spec.splits (data to index)."""
        self.load_status = BenchmarkLoadStatus()
        if not self.spec.splits:
            return
        yield from self._load_splits(self.spec.splits, status=self.load_status)

    def load_negative(self) -> Iterator[str]:
        """Yields texts from spec.subtraction_splits (data to subtract)."""
        if not self.spec.subtraction_splits:
            return
        yield from self._load_splits(
            self.spec.subtraction_splits, status=BenchmarkLoadStatus()
        )

    def resolve_config_names(self) -> Sequence[str | None]:
        return self._resolve_config_names()

    def _resolve_config_names(self) -> Sequence[str | None]:
        self._config_error = None
        if self.spec.config_name is not None:
            return [self.spec.config_name]
        raw_configs = list(self.spec.configs or [])

        # No configs specified => load dataset without config
        if not raw_configs:
            return [None]

        drop_set = set(self.spec.drop_configs or [])

        wants_all_explicitly = "all" in raw_configs
        has_wildcard = "*" in raw_configs

        # -------------------------
        # Explicit list (no wildcard)
        # -------------------------
        if not has_wildcard:
            resolved = [c for c in raw_configs if c not in drop_set]
            if not resolved:
                self._config_error = (
                    "config list resolved to empty after drop_configs "
                    f"(configs={raw_configs!r} drop_configs={sorted(drop_set)!r})"
                )
                logger.warning(f"  [{self.name}] WARNING: {self._config_error}")
            return resolved

        # -------------------------
        # Wildcard expansion
        # -------------------------
        try:
            try:
                available = get_dataset_config_names(
                    self.spec.dataset,
                    trust_remote_code=self.spec.trust_remote_code,
                    revision=self.spec.revision,
                )
            except TypeError:
                available = get_dataset_config_names(self.spec.dataset)
        except Exception as exc:
            self._config_error = (
                f"failed to resolve wildcard configs for "
                f"dataset={self.spec.dataset!r}: {type(exc).__name__}: {exc}"
            )
            logger.error(f"  [{self.name}] ERROR: {self._config_error}")
            return []

        available_set = set(available)

        # Auto-drop "all" only for wildcard expansion, unless user explicitly wants it.
        auto_drop_all = ("all" in available_set) and (not wants_all_explicitly)

        resolved = [
            c
            for c in available
            if c not in drop_set and (c != "all" or not auto_drop_all)
        ]

        # If we auto-dropped "all" and got nothing, but "all" exists, keep it to avoid indexing nothing.
        if (
            not resolved
            and auto_drop_all
            and "all" in available_set
            and "all" not in drop_set
        ):
            resolved = ["all"]

        if not resolved:
            self._config_error = (
                "wildcard matched 0 configs after filtering "
                f"(dataset={self.spec.dataset!r} drop_configs={sorted(drop_set)!r}, "
                f"available={len(available)})"
            )
            logger.warning(f"  [{self.name}] WARNING: {self._config_error}")

        return resolved

    def _load_splits(
        self,
        target_splits: list[str],
        *,
        status: BenchmarkLoadStatus,
    ) -> Iterator[str]:
        """Iterate resolved configs -> load dataset -> yield concatenated text for requested splits."""
        config_names = self._resolve_config_names()
        if not config_names:
            status.failures.append(self._config_error or "no configs resolved")
            return
        found_any = False
        last_config_name: str | None = None

        for config_name in config_names:
            last_config_name = config_name
            for split in target_splits:
                where = f"config={config_name!r} split={split!r}"
                try:
                    load_kwargs = {
                        "split": split,
                        "trust_remote_code": self.spec.trust_remote_code,
                        "revision": self.spec.revision,
                        "data_files": self.spec.data_files,
                    }
                    if config_name is None:
                        ds = _load_dataset_optional_revision(
                            self.spec.dataset,
                            **load_kwargs,
                        )
                    else:
                        ds = _load_dataset_optional_revision(
                            self.spec.dataset,
                            name=config_name,
                            **load_kwargs,
                        )
                except Exception as exc:
                    status.failures.append(f"{where}: {type(exc).__name__}: {exc}")
                    logger.warning(
                        f"  [{self.name}] WARNING: Failed to load "
                        f"dataset={self.spec.dataset!r} {where}: {exc}",
                    )
                    continue

                if isinstance(ds, Mapping):
                    if split not in ds:
                        status.failures.append(f"{where}: split not found")
                        logger.warning(
                            f"  [{self.name}] WARNING: Split '{split}' not found for "
                            f"config={config_name!r}.",
                        )
                        continue
                    rows = ds[split]
                else:
                    rows = ds

                found_any = True
                status.loaded.append(where)
                self._record_dataset_fingerprint(ds, rows, config_name, split)
                for row in rows:
                    status.rows += 1
                    yield _concat_fields(row, self.spec.fields)

        if not found_any:
            logger.warning(
                f"  [{self.name}] WARNING: None of the requested splits were found"
                f" for config={last_config_name!r}.",
            )

    def _record_dataset_fingerprint(
        self, ds: Any, rows: Any, config_name: str | None, split: str
    ) -> None:
        key = (self.spec.dataset, config_name, split)
        if key in self._fingerprint_keys:
            return
        self._fingerprint_keys.add(key)
        fingerprint = getattr(rows, "_fingerprint", None)
        resolved_revision = _extract_resolved_revision(ds)
        self.dataset_fingerprints.append(
            {
                "benchmark": self.spec.name,
                "dataset": self.spec.dataset,
                "config": config_name,
                "split": split,
                "requested_revision": self.spec.revision,
                "resolved_revision": resolved_revision,
                "hf_dataset_fingerprint": fingerprint,
            }
        )
