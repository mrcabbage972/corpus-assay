from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path

from corpus_assay._version import dist_version
from corpus_assay.attribution import ITEMS_FORMAT_VERSION, AttributionIndex
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.config import BuildIndexConfig, load_decontamination_config
from corpus_assay.constants import ITEMS_TEXT_SUFFIX
from corpus_assay.indexing import build_index_hf
from corpus_assay.normalization import active_ngram_config_path
from corpus_assay.scanner.fingerprint import _file_sha256

logger = logging.getLogger(__name__)


def normalization_config_path() -> Path:
    # Active config (bundled default or a --ngram-config override): an index built
    # under one normalization config must not be reused under another.
    return Path(active_ngram_config_path())


def native_sidecar(native_path: Path, suffix: str) -> Path:
    # e.g., index.native.meta.json (not index.meta.json)
    return native_path.with_name(f"{native_path.name}{suffix}")


def _library_identifier() -> str:
    return dist_version()


def native_index_required_paths(
    native_path: Path, *, require_items: bool = False
) -> list[Path]:
    paths = [
        native_path,
        native_sidecar(native_path, ".meta.json"),
        native_sidecar(native_path, ".attr"),
        native_sidecar(native_path, ".sources.json"),
    ]
    if require_items:
        # Require item postings so a managed (run/run-hf) cache always supports
        # item-level gating: a legacy cache built before item postings lacks this
        # file and is rebuilt, rather than silently falling back to legacy union
        # gating under gate_mode="auto". (The standalone `scan` command does not
        # require it — there, auto degrades to union for legacy indexes.)
        paths.append(native_sidecar(native_path, ".items"))
        paths.append(native_sidecar(native_path, ".items.meta.json"))
        # The raw item text, for the exact-match channel; caches
        # built before it existed are rebuilt.
        paths.append(native_sidecar(native_path, ITEMS_TEXT_SUFFIX))
    return paths


def native_index_ready(native_path: Path) -> bool:
    """Whether a cached index is complete and current.

    A cache whose ``.items`` predates field masks (format v1), or that lacks
    the item text sidecar, is rebuilt, so ``run`` / ``run-hf`` records report
    which item fields matched and the exact channel works.
    """
    required = native_index_required_paths(native_path, require_items=True)
    if not all(path.exists() for path in required):
        return False
    try:
        version = AttributionIndex.items_sidecar_version(
            str(native_sidecar(native_path, ".items"))
        )
    except (OSError, ValueError):
        return False
    return version >= ITEMS_FORMAT_VERSION


def benchmark_specs_hash(benchmarks: Iterable[BenchmarkSpec]) -> str:
    """SHA-256 of the resolved benchmark specs (sorted-keys JSON, in config order)."""
    payload = json.dumps(
        [spec.model_dump(mode="json") for spec in benchmarks],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_cache_path(
    config_path: Path,
    cache_dir: Path,
    *,
    benchmarks: Iterable[BenchmarkSpec] | None = None,
) -> Path:
    """Cache path for the index a config builds.

    Keyed on the config file (which holds ``stopgrams.items``), the resolved
    benchmark specs, the normalization config and the library version. The specs are hashed after registry refs are resolved,
    so a registry change (same config text, different specs) builds a new index
    instead of reusing a stale one. ``benchmarks`` defaults to loading the config.
    """
    config_hash = _file_sha256(config_path)
    if benchmarks is None:
        benchmarks = load_decontamination_config(config_path).benchmarks
    specs_hash = benchmark_specs_hash(benchmarks)[:16]
    norm_path = normalization_config_path()
    norm_hash = _file_sha256(norm_path)
    if norm_hash is None:
        raise FileNotFoundError(f"Could not read normalization config at {norm_path}")
    return cache_dir / (
        f"index_{config_hash}_{specs_hash}_{norm_hash}_{_library_identifier()}.native"
    )


def ensure_native_index(
    cfg_path: Path,
    cache_dir: Path,
    *,
    allow_unpinned: bool = False,
    allow_empty_benchmarks: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_decontamination_config(cfg_path)
    native_path = index_cache_path(cfg_path, cache_dir, benchmarks=cfg.benchmarks)
    if not allow_unpinned:
        missing = [spec.name for spec in cfg.benchmarks if not spec.revision]
        if missing:
            raise ValueError(
                "Benchmarks missing revision pin: "
                f"{', '.join(missing)}. Use --allow-unpinned to override."
            )

    if native_index_ready(native_path):
        logger.info("Using cached native index: %s", native_path)
        return native_path

    logger.info("Building native index for config: %s", cfg_path)
    build_index_hf(
        BuildIndexConfig(
            out_path=str(native_path),
            ngram=cfg.ngram,
            benchmarks=cfg.benchmarks,
            allow_unpinned=allow_unpinned,
            allow_empty_benchmarks=allow_empty_benchmarks,
            min_hits=cfg.min_hits,
            item_stopgrams=cfg.stopgrams.items if cfg.stopgrams else None,
        )
    )
    logger.info("Wrote native index to: %s", native_path)
    return native_path
