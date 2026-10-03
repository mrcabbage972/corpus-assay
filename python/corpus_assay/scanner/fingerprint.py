from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from corpus_assay import _native
from corpus_assay._version import dist_version
from corpus_assay.config import ScanConfig
from corpus_assay.constants import INDEX_FORMAT_VERSION
from corpus_assay.normalization import active_ngram_config_path


def _file_sha256(path: Path) -> str | None:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def library_version() -> str:
    return dist_version()


def rust_extension_version() -> str:
    # Version baked into the compiled extension at build time; differs from
    # library_version() when an editable install has a stale native build.
    return str(getattr(_native, "__version__", "unknown"))


def build_run_fingerprint(cfg: ScanConfig) -> dict[str, Any]:
    index_path = Path(cfg.index_path)
    stopgrams_path = Path(cfg.stopgrams_path).resolve() if cfg.stopgrams_path else None
    spaced_path = Path(cfg.spaced_path).resolve() if cfg.spaced_path else None
    payload = {
        "index_path": str(index_path.resolve()),
        "index_sha256": _file_sha256(index_path),
        "index_attr_sha256": _file_sha256(
            index_path.with_name(f"{index_path.name}.attr")
        ),
        "index_meta_sha256": _file_sha256(
            index_path.with_name(f"{index_path.name}.meta.json")
        ),
        "config_sha256": hashlib.sha256(
            json.dumps(cfg.model_dump(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "ngram_config_sha256": _file_sha256(Path(active_ngram_config_path())),
        "library_version": library_version(),
        "rust_index_format_version": INDEX_FORMAT_VERSION,
    }
    if stopgrams_path:
        payload["stopgrams_path"] = str(stopgrams_path)
        payload["stopgrams_sha256"] = _file_sha256(stopgrams_path)
    if spaced_path:
        # The .spaced artifact is a result-affecting scan input (its postings + item
        # tokens drive the spaced gate), so hash its bytes -- not just the path in
        # config_sha256 -- or rebuilding it in place would let the per-file resume
        # cache serve stale summaries.
        payload["spaced_path"] = str(spaced_path)
        payload["spaced_sha256"] = _file_sha256(spaced_path)
        payload["spaced_meta_sha256"] = _file_sha256(
            spaced_path.with_name(f"{spaced_path.name}.meta.json")
        )
    return payload
