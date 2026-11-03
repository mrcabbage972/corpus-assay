"""Type stubs for the compiled Rust extension (``src/lib.rs``)."""

from typing import IO, Any

__version__: str

def scan_stream_rust(
    input: IO[bytes],
    text_key: str | list[str],
    id_key: str | None,
    index_path: str,
    n: int,
    min_hits: int,
    min_coverage: float,
    index_backend: str = "auto",
    out_hits_path: str | None = None,
    packed_doc_sep: str | None = "<|endoftext|>",
    packed_doc_sep_typo: str | None = "<|endoftext}>",
    max_return_records: int | None = 500,
    stopgrams_path: str | None = None,
    gate_mode: str = "auto",
    min_longest_run: int = 0,
    spaced_path: str | None = None,
    spaced_min_loci: int = 3,
    spaced_ver_min_span: int = 17,
    spaced_ver_identity: float = 0.85,
    boilerplate_threshold: int = 50,
    messages_key: str | None = None,
    min_item_coverage: float = 0.0,
    out_item_best_path: str | None = None,
) -> tuple[int, int, list[str], int]:
    """Scan a Parquet stream; returns ``(scanned, contaminated, records, empty)``.

    ``text_key`` is one text column or a list of them; ``messages_key`` names an
    optional ``list<struct{role, content}>`` chat column. Each text, and each chat
    turn, is scanned as its own sub-document(s). ``packed_doc_sep=None`` turns
    packed-document splitting off; ``max_return_records`` of ``None`` or ``0``
    returns every flagged record. ``min_item_coverage`` > 0 also flags a document
    holding that share of one item's n-grams (union mode) or requires it of the
    flagging item (item mode). ``out_item_best_path`` receives, as JSON, the best
    single-document coverage of each item with a hit (needs item postings).
    """

def set_ngram_config_path(path: str) -> None: ...
def normalize_text(s: str) -> list[str]: ...
def ngram_allowed(ngram: str) -> bool: ...
def hash_ngram(ngram: str) -> int: ...
def ngram_filter(ngram: str) -> tuple[bool, int | None]: ...
def ngram_config_version() -> str: ...
def ngram_config_hash() -> str: ...
def normalization_config_sha() -> str: ...
def hash_function_metadata() -> dict[str, Any]: ...
def spaced_seed_hashes_rust(
    tokens: list[str], patterns: list[str]
) -> list[tuple[int, int, int]]: ...
def spaced_verified_item_rust(
    spaced_path: str,
    tokens: list[str],
    min_loci: int,
    ver_min_span: int,
    ver_identity: float,
) -> int | None: ...
