import corpus_assay._native as rust_module
from corpus_assay._native import (
    hash_ngram,
    ngram_allowed,
    normalize_text,
)
from corpus_assay.constants import FIELD_SEPARATOR
from corpus_assay.text_utils import iter_ngrams


def test_hash_ngram():
    assert hash_ngram("hello world") == 5814608031911216775


def test_normalize_text_uses_rust() -> None:
    assert normalize_text is rust_module.normalize_text


def test_normalize_text_filters_and_normalizes() -> None:
    assert normalize_text("The ＣＡＴ and DOG!") == ["cat", "dog"]


def test_ngram_allowed_uses_rust_rules() -> None:
    assert ngram_allowed("option a") is False
    assert ngram_allowed("unique phrase about cats") is True


def test_normalize_text_empty_and_stop_words() -> None:
    assert normalize_text("") == []
    assert normalize_text("   ") == []
    assert normalize_text("the and a") == []


def test_normalize_text_preserves_contractions() -> None:
    assert normalize_text("Don't stop believing") == [
        "don't",
        "stop",
        "believing",
    ]


def test_normalize_text_numbers_and_case() -> None:
    assert normalize_text("Version 2.0 of THE thing") == [
        "version",
        "2",
        "0",
        "thing",
    ]


def test_normalize_text_field_separator_blocks_ngrams() -> None:
    tokens = normalize_text(f"alpha {FIELD_SEPARATOR} beta gamma")
    assert FIELD_SEPARATOR in tokens

    ngrams = [ng for ng in iter_ngrams(tokens, 2) if ngram_allowed(ng)]
    assert ngrams == ["beta gamma"]


def test_ngram_allowed_rejects_empty_or_whitespace() -> None:
    assert ngram_allowed("") is False
    assert ngram_allowed("   ") is False


def test_ngram_allowed_rejects_pattern() -> None:
    assert ngram_allowed("Which of the following") is False
