from __future__ import annotations

from typing import Iterator, Sequence

# Re-exported (``X as X``) so other modules can import them from here.
from corpus_assay._native import (
    hash_ngram as hash_ngram,
    ngram_allowed as ngram_allowed,
    ngram_filter as ngram_filter,
    normalize_text as normalize_text,
)
from corpus_assay.constants import FIELD_SEPARATOR


def _iter_ngrams_segment(words: Sequence[str], n: int) -> Iterator[str]:
    length = len(words)
    if length < n:
        return
    for i in range(length - n + 1):
        yield " ".join(words[i : i + n])


def _iter_ngrams_segment_with_positions(
    words: Sequence[str], n: int, offset: int
) -> Iterator[tuple[str, int]]:
    length = len(words)
    if length < n:
        return
    for i in range(length - n + 1):
        yield " ".join(words[i : i + n]), offset + i


def iter_ngrams(words: Sequence[str], n: int) -> Iterator[str]:
    """
    Yield contiguous word n-grams, but do NOT cross FIELD_SEPARATOR boundaries.
    """
    start = 0
    for idx, word in enumerate(words):
        if word != FIELD_SEPARATOR:
            continue
        yield from _iter_ngrams_segment(words[start:idx], n)
        start = idx + 1
    yield from _iter_ngrams_segment(words[start:], n)


def iter_ngrams_with_positions(
    words: Sequence[str], n: int
) -> Iterator[tuple[str, int]]:
    """
    Yield contiguous word n-grams with their start positions, but do NOT cross
    FIELD_SEPARATOR boundaries.
    """
    start = 0
    for idx, word in enumerate(words):
        if word != FIELD_SEPARATOR:
            continue
        yield from _iter_ngrams_segment_with_positions(words[start:idx], n, start)
        start = idx + 1
    yield from _iter_ngrams_segment_with_positions(words[start:], n, start)


def iter_ngrams_with_filter(
    words: Sequence[str], n: int
) -> Iterator[tuple[str, bool, int | None]]:
    """
    Yield (ngram, allowed, reject_pattern_idx) using the canonical Rust filter rules.

    - Fast path: ngram_allowed(ng)
    - If disallowed: call ngram_filter(ng) to retrieve the reject pattern index (when available)
    """
    for ng in iter_ngrams(words, n):
        allowed = ngram_allowed(ng)
        reject_idx: int | None = None
        if not allowed:
            allowed, reject_idx = ngram_filter(ng)
        yield ng, bool(allowed), reject_idx


def iter_allowed_ngrams(words: Sequence[str], n: int) -> Iterator[str]:
    """
    Yield allowed n-gram strings from already-normalized tokens.
    """
    for ng, allowed, _reject_idx in iter_ngrams_with_filter(words, n):
        if allowed:
            yield ng


def iter_allowed_ngrams_from_text(text: str, n: int) -> Iterator[str]:
    """
    Normalize text with Rust normalize_text(), then yield allowed n-gram strings.
    """
    toks = normalize_text(text)
    yield from iter_allowed_ngrams(toks, n)


def iter_hashed_ngrams(words: Sequence[str], n: int) -> Iterator[int]:
    """
    Yield allowed n-grams as Rust hash values (u64 exposed to Python as int).
    """
    for ng in iter_allowed_ngrams(words, n):
        yield int(hash_ngram(ng))


def iter_hashed_ngrams_with_positions(
    words: Sequence[str], n: int
) -> Iterator[tuple[int, int]]:
    """
    Yield allowed n-gram hashes and their start positions (token offsets).
    """
    for ng, pos in iter_ngrams_with_positions(words, n):
        allowed = ngram_allowed(ng)
        if not allowed:
            allowed, _reject_idx = ngram_filter(ng)
        if allowed:
            yield int(hash_ngram(ng)), pos


def iter_hashed_ngrams_from_text(text: str, n: int) -> Iterator[int]:
    """
    Normalize text with Rust normalize_text(), then yield allowed n-gram hashes.
    """
    toks = normalize_text(text)
    yield from iter_hashed_ngrams(toks, n)


def iter_hashed_ngrams_with_positions_from_text(
    text: str, n: int
) -> Iterator[tuple[int, int]]:
    """
    Normalize text with Rust normalize_text(), then yield allowed n-gram hashes and
    their start positions.
    """
    toks = normalize_text(text)
    yield from iter_hashed_ngrams_with_positions(toks, n)


def collect_hashed_ngrams_from_text(text: str, n: int) -> set[int]:
    """
    Convenience: return unique allowed n-gram hashes for `text`.
    """
    return set(iter_hashed_ngrams_from_text(text, n))
