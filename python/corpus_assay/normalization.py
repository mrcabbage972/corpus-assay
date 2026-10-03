"""Process-wide ngram normalization config management.

The Rust extension freezes the active config on first normalization use, and
spawned worker processes start fresh: they re-import the package and only see
the bundled default staged in ``__init__``. To carry a parent-process override
into workers, callers need to know which path is currently in effect. This
module wraps ``corpus_assay._native.set_ngram_config_path``
so the active path is also tracked on the Python side, and exposes
``active_ngram_config_path()`` for code that needs to propagate it (e.g. the
stop-gram multiprocessing pool).
"""

from __future__ import annotations

from corpus_assay._native import (
    set_ngram_config_path as _rust_set_ngram_config_path,
)
from corpus_assay.constants import DEFAULT_NGRAM_CONFIG_PATH

_active_ngram_config_path: str = str(DEFAULT_NGRAM_CONFIG_PATH)


def set_ngram_config_path(path: str) -> None:
    """Stage the ngram normalization config at ``path`` for this process.

    Records the path so it can be re-applied in spawned worker processes via
    :func:`active_ngram_config_path`. Raises ``ValueError`` if the file cannot
    be read or does not parse as a valid config, and ``RuntimeError`` if the
    config has already been frozen by an earlier normalization call.
    """
    global _active_ngram_config_path
    _rust_set_ngram_config_path(path)
    _active_ngram_config_path = path


def active_ngram_config_path() -> str:
    """Return the ngram config path currently staged in this process."""
    return _active_ngram_config_path
