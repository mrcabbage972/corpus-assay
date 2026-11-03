"""Planting helpers of ``selftest`` (#12); the end-to-end runs are in tests/e2e."""

from __future__ import annotations

import random

import pytest

from corpus_assay._native import light_normalize
from corpus_assay.config import ScanConfig
from corpus_assay.selftest import (
    SelftestError,
    _classify,
    _Item,
    _plant_into,
    apply_variant,
    item_text,
    parse_variants,
)


def test_item_text_puts_fields_and_choices_on_their_own_lines() -> None:
    fields = ["What is 2+3?", ["4", "5", "", None], "", "Explain."]
    assert item_text(fields) == "What is 2+3?\n4\n5\nExplain."


def test_variants_keep_the_light_normalized_text() -> None:
    text = "Which planet is known as the Red Planet?\nMars\nVenus"
    rng = random.Random(0)
    for variant in ("verbatim", "case", "whitespace"):
        changed, label = apply_variant(text, variant, rng)
        assert label.startswith(variant)
        # Case and whitespace edits vanish under the exact channel's normalization.
        assert light_normalize(changed) == light_normalize(text)
    wrapped, _ = apply_variant(text, "wrapper", rng)
    lines = wrapped.splitlines()
    assert lines[1:-1] == text.splitlines() and len(lines) == 5
    with pytest.raises(SelftestError):
        apply_variant(text, "typo", rng)


def test_parse_variants() -> None:
    assert parse_variants("verbatim, case,case") == ["verbatim", "case"]
    with pytest.raises(SelftestError):
        parse_variants("verbatim,typo")
    with pytest.raises(SelftestError):
        parse_variants(" , ")


def _cfg(**kwargs) -> ScanConfig:
    return ScanConfig(inputs=[], id_key=None, index_path="i", out_dir="o", **kwargs)


def test_plant_into_text_and_chat_columns() -> None:
    row = {"text": "alpha beta gamma delta"}
    _plant_into(row, "PLANT", _cfg())
    assert row["text"] == "alpha beta gamma\n\nPLANT\n\ndelta"

    cfg = _cfg(messages_key="messages")
    row = {"messages": [{"role": "user", "content": "hi there"}, {"role": "x"}]}
    _plant_into(row, "PLANT", cfg)
    assert row["messages"][0]["content"] == "hi\n\nPLANT\n\nthere"
    row = {"messages": None}
    _plant_into(row, "PLANT", cfg)
    assert row["messages"] == [{"role": "user", "content": "PLANT"}]


def test_classify_mirrors_the_exact_channel_eligibility() -> None:
    items = [
        _Item(0, "a", "b", ["What is the capital city of France?"]),
        _Item(1, "b", "b", ["Short?"]),
        _Item(2, "c", "b", ["A long question with many n-grams in it, surely."]),
    ]
    counts = [0, 0, 9]
    off = _classify(items, _cfg(min_hits=3), counts)
    assert [c.ngram_matchable for c in off.values()] == [False, False, True]
    assert not any(c.exact_matchable for c in off.values())
    short = _classify(items, _cfg(min_hits=3, exact_channel="short"), counts)
    assert [c.exact_matchable for c in short.values()] == [True, False, False]
    every = _classify(items, _cfg(min_hits=3, exact_channel="all"), counts)
    assert [c.exact_matchable for c in every.values()] == [True, False, True]
    # Without .items.meta.json the n-gram count is unknown: counted as matchable.
    unknown = _classify(items, _cfg(min_hits=3), None)
    assert all(c.ngram_matchable and c.ngrams is None for c in unknown.values())
