import os
from pathlib import Path

FIELD_SEPARATOR = "<|field_sep|>"
INDEX_FORMAT_VERSION = 2

# Single source of truth for the normalization config path. The Rust extension
# never hardcodes a config path: this path is resolved here and handed to it via
# `set_ngram_config_path` (see corpus_assay/__init__.py). A CLI may pass a
# different file to override it before normalization runs.
DEFAULT_NGRAM_CONFIG_PATH = Path(__file__).resolve().parent / "ngram_config.json"

# Where `run` / `run-hf` cache built indexes by default (honors XDG_CACHE_HOME).
DEFAULT_INDEX_CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "corpus-assay"
    / "indexes"
)
