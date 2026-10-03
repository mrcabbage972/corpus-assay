from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import typer

from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import BenchmarkLoader
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
    ) -> None:
        if dataset_builder_loader is None:
            from datasets import load_dataset_builder

            dataset_builder_loader = load_dataset_builder

        assert dataset_builder_loader is not None
        self._load_dataset_builder = dataset_builder_loader
        self._benchmark_loader_factory = benchmark_loader_factory
        self._config_loader = config_loader

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
                    splits = (
                        set(builder.info.splits.keys())
                        if builder.info and builder.info.splits
                        else set()
                    )
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

            if spec_ok:
                checked += 1
            else:
                skipped += 1

        return checked, skipped, warnings

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
