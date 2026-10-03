from corpus_assay._version import dist_version
from corpus_assay.constants import (
    DEFAULT_NGRAM_CONFIG_PATH,
    FIELD_SEPARATOR,
    INDEX_FORMAT_VERSION,
)
from corpus_assay.normalization import set_ngram_config_path

# Stage the bundled normalization config into the Rust extension before any
# normalization runs. This is the only place the default config is wired in; a
# CLI may call set_ngram_config_path() again to override it, up until the first
# normalization call freezes the config for the process. Going through the
# Python wrapper (rather than the raw Rust function) records the active path so
# spawned worker processes can re-stage it -- see corpus_assay.normalization.
set_ngram_config_path(str(DEFAULT_NGRAM_CONFIG_PATH))


__version__ = dist_version()

# ----------------------------
# Lazy convenience wrappers
# ----------------------------


def build_index_hf(*args, **kwargs):
    """
    Lazy import wrapper for corpus_assay.indexing.build_index_hf
    """
    from corpus_assay.indexing import build_index_hf as _impl

    return _impl(*args, **kwargs)


def do_scan(*args, **kwargs):
    """
    Lazy import wrapper for corpus_assay.scanner.runner.do_scan
    """
    from corpus_assay.scanner.runner import do_scan as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "__version__",
    "FIELD_SEPARATOR",
    "INDEX_FORMAT_VERSION",
    "build_index_hf",
    "do_scan",
]
