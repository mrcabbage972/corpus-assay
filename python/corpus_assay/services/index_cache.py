from __future__ import annotations

import logging
from pathlib import Path

from corpus_assay._version import dist_version
from corpus_assay.config import BuildIndexConfig, load_decontamination_config
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
    return paths


def native_index_ready(native_path: Path) -> bool:
    required = native_index_required_paths(native_path, require_items=True)
    return all(path.exists() for path in required)


def index_cache_path(config_path: Path, cache_dir: Path) -> Path:
    config_hash = _file_sha256(config_path)
    norm_path = normalization_config_path()
    norm_hash = _file_sha256(norm_path)
    if norm_hash is None:
        raise FileNotFoundError(f"Could not read normalization config at {norm_path}")
    return cache_dir / f"index_{config_hash}_{norm_hash}_{_library_identifier()}.native"


def ensure_native_index(
    cfg_path: Path, cache_dir: Path, *, allow_unpinned: bool = False
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    native_path = index_cache_path(cfg_path, cache_dir)
    cfg = load_decontamination_config(cfg_path)
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
        )
    )
    logger.info("Wrote native index to: %s", native_path)
    return native_path
