"""Unit tests for the spaced-seed matching channel (corpus_assay.spaced_seeds).

These operate on raw token lists (no synthetic-corpus dependency); they pin the
properties that make the channel safe: span/weight invariants, the counting-trap
collapse (multiple patterns at one locus count once), the verifier identity
threshold, and the two-channel union predicate.
"""

from __future__ import annotations

import pytest

from corpus_assay.constants import FIELD_SEPARATOR
from corpus_assay.spaced_seeds import (
    DEFAULT_PATTERNS,
    SpacedSeedConfig,
    SpacedSeedDetector,
    SpacedSeedFamily,
    build_exact_index,
    build_spaced_index,
    class_map_num_id,
    class_map_numbers,
    exact_flag,
    iter_exact_grams,
    iter_spaced_seeds,
    spaced_candidates,
    verify,
)

SPAN = 17


def _tok(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{j}" for j in range(n)]


# --------------------------------------------------------------------------- #
# Seed family invariants
# --------------------------------------------------------------------------- #
def test_family_parses_span_weight_and_offsets() -> None:
    fam = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
    assert fam.span == 17
    assert fam.weight == 13
    assert len(fam) == 4
    # offsets are exactly the positions of '1' in each pattern
    for pat, offs in zip(fam.patterns, fam.offsets):
        assert list(offs) == [i for i, c in enumerate(pat) if c == "1"]
        assert len(offs) == fam.weight


def test_family_rejects_mismatched_span_or_weight() -> None:
    with pytest.raises(ValueError):
        SpacedSeedFamily(patterns=("11101011101110111", "1110101110111011"))  # short
    with pytest.raises(ValueError):
        SpacedSeedFamily(patterns=("11101011101110111", "11111011101110111"))  # weight
    with pytest.raises(ValueError):
        SpacedSeedFamily(patterns=("1110101110111021x",))  # non-binary


def test_family_subset() -> None:
    fam = SpacedSeedFamily(patterns=DEFAULT_PATTERNS).subset(2)
    assert len(fam) == 2
    assert fam.patterns == DEFAULT_PATTERNS[:2]
    assert fam.exact_n == 13  # subset preserves exact_n


def test_noncontainment_invariant() -> None:
    """No exact-n window may contain all selected positions (default patterns ok)."""
    fam = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)  # must not raise
    # worst-case window holds < weight for every pattern
    for off in fam.offsets:
        worst = max(
            sum(1 for o in off if s <= o < s + fam.exact_n)
            for s in range(fam.span - fam.exact_n + 1)
        )
        assert worst < fam.weight

    # a contiguous weight-13 run inside a span-17 window packs all 13 ones into a
    # 13-window -> a lone exact 13-gram would trip it -> rejected.
    with pytest.raises(ValueError):
        SpacedSeedFamily(patterns=("11111111111110000",))  # 13 ones, all contiguous
    # span not exceeding exact_n is rejected (every seed is one exact n-gram)
    with pytest.raises(ValueError):
        SpacedSeedFamily(patterns=("1111111111111",), exact_n=13)  # span 13 == exact_n


# --------------------------------------------------------------------------- #
# Counting trap: many patterns at one locus must collapse to ONE distinct locus
# --------------------------------------------------------------------------- #
def test_single_overlap_collapses_to_one_locus() -> None:
    """A lone span-17 overlap fires every pattern but is one (item, diagonal) locus."""
    fam = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)  # 4 patterns
    item = _tok("t", SPAN)  # exactly one 17-token window
    index = build_spaced_index([item], fam)
    # embed the item once, surrounded by unrelated tokens
    doc = _tok("a", 20) + item + _tok("b", 20)

    loci = spaced_candidates(doc, index, fam)
    # exactly one (item, diagonal) candidate, and it covers a single document locus
    assert len(loci) == 1
    (only_loci,) = loci.values()
    assert len(only_loci) == 1  # NOT 4 -- multi-pattern hits collapsed

    det = SpacedSeedDetector([item], fam, SpacedSeedConfig(min_loci=2, ver_min_span=15))
    assert det.spaced_flag(doc, verified=False) is False  # one locus < min_loci=2
    det1 = SpacedSeedDetector(
        [item], fam, SpacedSeedConfig(min_loci=1, ver_min_span=15)
    )
    assert det1.spaced_flag(doc, verified=False) is True


def test_two_loci_on_same_diagonal_verify() -> None:
    """A longer exact overlap yields >=2 loci on one diagonal and verifies."""
    fam = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
    item = _tok("t", 19)  # windows start at 0 and 1 -> same diagonal once embedded
    doc = _tok("a", 10) + item + _tok("b", 10)
    index = build_spaced_index([item], fam)

    loci = spaced_candidates(doc, index, fam)
    # one diagonal carrying at least two distinct document loci
    assert any(len(s) >= 2 for s in loci.values())

    det = SpacedSeedDetector(
        [item], fam, SpacedSeedConfig(min_loci=2, ver_min_span=15, ver_identity=0.9)
    )
    assert det.spaced_flag(doc, verified=True) is True


# --------------------------------------------------------------------------- #
# Verifier identity threshold
# --------------------------------------------------------------------------- #
def test_verify_respects_identity_and_min_span() -> None:
    item = _tok("t", 30)
    doc = list(item)  # perfect copy, diagonal 0
    loci = set(range(0, 30 - SPAN + 1))
    assert verify(doc, item, 0, loci, SPAN, ver_min_span=20, ver_identity=0.95) is True

    # corrupt 40% of the aligned tokens -> identity 0.6 < 0.75 fails
    corrupted = list(item)
    for j in range(0, 30, 2):  # every other token
        corrupted[j] = f"x{j}"
    assert (
        verify(corrupted, item, 0, set(range(0, 30 - SPAN + 1)), SPAN, 20, 0.75)
        is False
    )

    # too-short comparable span fails regardless of identity
    assert verify(doc, item, 0, {0}, SPAN, ver_min_span=40, ver_identity=0.5) is False


# --------------------------------------------------------------------------- #
# Exact channel + union predicate
# --------------------------------------------------------------------------- #
def test_exact_flag_and_union() -> None:
    item = _tok("t", 40)
    det = SpacedSeedDetector(
        [item],
        SpacedSeedFamily(patterns=DEFAULT_PATTERNS),
        SpacedSeedConfig(min_loci=2, ver_min_span=15, ver_identity=0.9),
        n=13,
        exact_min_hits=2,
        exact_min_coverage=0.0,
    )
    # exact copy -> exact channel fires, union fires
    exact_doc = list(item)
    assert det.exact_flag(exact_doc) is True
    assert det.union_flag(exact_doc) is True

    # unrelated document -> neither channel fires
    clean = _tok("z", 60)
    assert det.exact_flag(clean) is False
    assert det.spaced_flag(clean, verified=True) is False
    assert det.union_flag(clean) is False


def test_union_recovers_via_spaced_when_exact_misses() -> None:
    """A document below the exact hit threshold but verified by spaced still flags."""
    item = _tok("t", 40)
    # A single 17-token window yields only 5 exact 13-grams; with min_hits=6 the
    # exact gate misses it, while the spaced verifier confirms the locus. This
    # isolates the union predicate without hand-constructing a near-duplicate.
    det = SpacedSeedDetector(
        [item],
        SpacedSeedFamily(patterns=DEFAULT_PATTERNS),
        SpacedSeedConfig(min_loci=1, ver_min_span=15, ver_identity=0.9),
        n=13,
        exact_min_hits=6,
        exact_min_coverage=0.0,
    )
    span_only = _tok("a", 8) + item[5:22] + _tok("b", 8)
    assert det.exact_flag(span_only) is False
    assert det.spaced_flag(span_only, verified=True) is True
    assert det.union_flag(span_only) is True


# --------------------------------------------------------------------------- #
# Class-aware seeds (candidate generation only)
# --------------------------------------------------------------------------- #
def test_class_maps_token_level() -> None:
    assert class_map_numbers("42") == "<num>"
    assert class_map_numbers("var") == "var"
    assert class_map_numbers("x1") == "x1"  # mixed -> not a pure number
    assert class_map_num_id("42") == "<num>"
    assert class_map_num_id("var0") == "<id>"  # mixed alpha+digit
    assert class_map_num_id("foo") == "foo"  # pure alpha is NOT an id (FP-safe)
    assert class_map_num_id("the") == "the"


def test_class_aware_seed_hash_bridges_number_edit() -> None:
    """A window differing only in a number hashes identically under <num> seeds.

    Offset 0 is a '1' in every default pattern, so a changed token there breaks
    all plain seeds; the number class collapses it and the seeds match.
    """
    fam = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
    w1 = ["42"] + [f"t{i}" for i in range(1, SPAN)]
    w2 = ["99"] + [f"t{i}" for i in range(1, SPAN)]  # only the number differs

    plain1 = {h for _, _, h in iter_spaced_seeds(w1, fam)}
    plain2 = {h for _, _, h in iter_spaced_seeds(w2, fam)}
    assert plain1.isdisjoint(plain2)  # every pattern broke on the changed number

    ca1 = {h for _, _, h in iter_spaced_seeds(w1, fam, class_map_numbers)}
    ca2 = {h for _, _, h in iter_spaced_seeds(w2, fam, class_map_numbers)}
    assert ca1 == ca2  # number class bridges the edit


# --------------------------------------------------------------------------- #
# Exact channel fidelity to the production gate
# --------------------------------------------------------------------------- #
def test_iter_exact_grams_applies_reject_filter() -> None:
    """Rejected n-grams (e.g. MCQ forms) are not indexed, matching production."""
    # "option a" is rejected by the default ngram config; "unique phrase" is allowed.
    assert list(iter_exact_grams(["option", "a"], 2)) == []
    assert len(list(iter_exact_grams(["unique", "phrase"], 2))) == 1


def test_exact_flag_distinct_shingle_denominator() -> None:
    """Coverage divides by distinct shingles, not positional n-grams (so duplicates
    do not dilute it the way they would under a positional denominator)."""
    index = build_exact_index([["x1", "x2"]], n=2)  # the bigram x1 x2
    doc = ["x1", "x2", "x1", "x2", "x1", "x2"]  # 5 positional bigrams, 2 distinct
    # distinct denominator: 1 hit / 2 distinct shingles = 0.5 >= 0.3 -> flagged.
    # a positional denominator (1/5 = 0.2) would NOT clear 0.3.
    assert exact_flag(doc, index, n=2, min_hits=1, min_coverage=0.3) is True
    assert exact_flag(doc, index, n=2, min_hits=1, min_coverage=0.6) is False


# --------------------------------------------------------------------------- #
# Field-separator handling
# --------------------------------------------------------------------------- #
def test_seed_windows_skip_field_separator() -> None:
    fam = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
    item = _tok("t", SPAN)
    index = build_spaced_index([item], fam)
    # the only overlapping window is broken by a field separator -> no candidates
    doc = _tok("a", 5) + item[:8] + [FIELD_SEPARATOR] + item[8:] + _tok("b", 5)
    assert spaced_candidates(doc, index, fam) == {}
