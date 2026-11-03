from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import typer

from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import BenchmarkLoader, hidden_label_warning
from corpus_assay.config import (
    DecontaminationConfig,
    load_decontamination_config,
)


class _DatasetInfo(Protocol):
    @property
    def splits(self) -> Mapping[str, object] | None: ...


class _DatasetBuilder(Protocol):
    @property
    def info(self) -> _DatasetInfo | None: ...


DatasetBuilderLoader = Callable[..., _DatasetBuilder]

# (spec, config_name, split) -> label values of the first rows of the split, or
# None when the split has no such column.
LabelProbe = Callable[[BenchmarkSpec, str | None, str], list[Any] | None]

# Rows sampled per split by the default label probe.
LABEL_PROBE_ROWS = 200


def stream_label_sample(
    spec: BenchmarkSpec, config_name: str | None, split: str
) -> list[Any] | None:
    """Stream the first rows of a split and return their ``label_field`` values."""
    from itertools import islice

    from datasets import load_dataset

    assert spec.label_field is not None
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
        "revision": spec.revision,
        "data_files": spec.data_files,
    }
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if config_name is not None:
        kwargs["name"] = config_name
    ds = load_dataset(spec.dataset, **kwargs)
    values: list[Any] = []
    for row in islice(ds, LABEL_PROBE_ROWS):
        if spec.label_field not in row:
            return None
        values.append(row[spec.label_field])
    return values


def _splits_from_data_files(data_files: object) -> set[str]:
    """Split names implied by a ``data_files`` value (as `datasets` reads it)."""
    if isinstance(data_files, Mapping):
        return {str(key) for key in data_files}
    if isinstance(data_files, (str, list, tuple)) and data_files:
        return {"train"}
    return set()


def _available_splits(spec: BenchmarkSpec, builder: _DatasetBuilder) -> set[str]:
    info = builder.info
    if info is not None and info.splits:
        return set(info.splits.keys())
    # Some datasets (e.g. opencompass/AIME2025) have no split metadata in their
    # card even though their data_files load fine; fall back to the data_files
    # keys, from the spec or as resolved by the builder.
    splits = _splits_from_data_files(spec.data_files)
    if not splits:
        builder_config = getattr(builder, "config", None)
        splits = _splits_from_data_files(getattr(builder_config, "data_files", None))
    return splits


class BenchmarkLoaderProtocol(Protocol):
    def resolve_config_names(self) -> Sequence[str | None]: ...


@dataclass(frozen=True, slots=True)
class ConfigValidationReport:
    config: DecontaminationConfig
    benchmark_count: int
    ngram: int
    min_hits: int
    min_coverage: float
    remote_checked: int = 0
    remote_skipped: int = 0
    warnings: tuple[str, ...] = ()


class ConfigValidator:
    def __init__(
        self,
        *,
        dataset_builder_loader: DatasetBuilderLoader | None = None,
        benchmark_loader_factory: Callable[[BenchmarkSpec], BenchmarkLoaderProtocol] = (
            BenchmarkLoader
        ),
        config_loader: Callable[[str | Path], DecontaminationConfig] = (
            load_decontamination_config
        ),
        label_probe: LabelProbe = stream_label_sample,
    ) -> None:
        if dataset_builder_loader is None:
            from datasets import load_dataset_builder

            dataset_builder_loader = load_dataset_builder

        assert dataset_builder_loader is not None
        self._load_dataset_builder = dataset_builder_loader
        self._benchmark_loader_factory = benchmark_loader_factory
        self._config_loader = config_loader
        self._label_probe = label_probe

    def validate(
        self,
        config_path: str | Path,
        *,
        check_remote: bool = False,
        allow_unpinned: bool = False,
    ) -> ConfigValidationReport:
        cfg = self._config_loader(config_path)
        self._validate_local(cfg, allow_unpinned=allow_unpinned)

        remote_checked = 0
        remote_skipped = 0
        warnings: list[str] = []

        if check_remote:
            remote_checked, remote_skipped, warnings = self._validate_remote(cfg)

        return ConfigValidationReport(
            config=cfg,
            benchmark_count=len(cfg.benchmarks),
            ngram=cfg.ngram,
            min_hits=cfg.min_hits,
            min_coverage=cfg.min_coverage,
            remote_checked=remote_checked,
            remote_skipped=remote_skipped,
            warnings=tuple(warnings),
        )

    def _validate_local(
        self, cfg: DecontaminationConfig, *, allow_unpinned: bool
    ) -> None:
        if len(cfg.benchmarks) > 64:
            raise typer.BadParameter("At most 64 benchmarks are supported per index.")
        if not allow_unpinned:
            missing = [spec.name for spec in cfg.benchmarks if not spec.revision]
            if missing:
                raise typer.BadParameter(
                    "Benchmarks missing revision pin: "
                    f"{', '.join(missing)}. Use --allow-unpinned to override."
                )

    def _validate_remote(
        self, cfg: DecontaminationConfig
    ) -> tuple[int, int, list[str]]:
        checked = 0
        skipped = 0
        warnings: list[str] = []

        for spec in cfg.benchmarks:
            spec_ok = True
            loader = self._benchmark_loader_factory(spec)
            configs = loader.resolve_config_names()
            if not configs:
                warnings.append(
                    f"[warn] {spec.name}: could not resolve configs; skipping remote validation."
                )
                skipped += 1
                continue

            for config_name in configs:
                try:
                    builder = self._build_dataset_builder(spec, config_name)
                    splits = _available_splits(spec, builder)
                except Exception as exc:
                    warnings.append(
                        f"[warn] {spec.name}: could not validate remotely: {exc}"
                    )
                    spec_ok = False
                    break

                missing = self._find_missing_splits(spec, splits)
                if missing:
                    available = sorted(splits)
                    raise typer.BadParameter(
                        f"{spec.name}: splits not found for config={config_name}: "
                        f"{missing}. Available: {available}"
                    )

                warnings.extend(self._check_labels(spec, config_name))

            if spec_ok:
                checked += 1
            else:
                skipped += 1

        return checked, skipped, warnings

    def _check_labels(self, spec: BenchmarkSpec, config_name: str | None) -> list[str]:
        """Hidden-label warnings for the indexed splits (needs ``label_field``)."""
        if spec.label_field is None:
            return []
        found: list[str] = []
        for split in spec.splits:
            try:
                values = self._label_probe(spec, config_name, split)
            except Exception as exc:
                found.append(
                    f"[warn] {spec.name}: could not read {spec.label_field!r} from "
                    f"split {split!r} to check for hidden labels: {exc}"
                )
                continue
            warning = hidden_label_warning(spec, split, config_name, values)
            if warning is not None:
                found.append(f"[warn] {warning}")
        return found

    def _build_dataset_builder(
        self, spec: BenchmarkSpec, config_name: str | None
    ) -> _DatasetBuilder:
        kwargs: dict[str, Any] = {
            "revision": spec.revision,
            "data_files": spec.data_files,
        }
        if spec.trust_remote_code:
            # Only for script-based datasets: `datasets` 4.x builders reject the
            # key outright, which would make every remote check fail.
            kwargs["trust_remote_code"] = True
        try:
            if config_name is None:
                return self._load_dataset_builder(spec.dataset, **kwargs)
            return self._load_dataset_builder(spec.dataset, config_name, **kwargs)
        except TypeError:
            if config_name is None:
                return self._load_dataset_builder(spec.dataset)
            return self._load_dataset_builder(spec.dataset, config_name)

    @staticmethod
    def _find_missing_splits(spec: BenchmarkSpec, splits: set[str]) -> list[str]:
        to_check = list(spec.splits) + list(spec.subtraction_splits or [])
        return [split for split in to_check if split not in splits]
