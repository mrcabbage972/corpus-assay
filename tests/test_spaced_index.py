"""Round-trip tests for the ``.spaced`` artifact (experimental spaced-seed channel).

The artifact must faithfully serialize the in-memory spaced index: its postings must
equal ``build_spaced_index``, its token blob must reconstruct the protected items
exactly, and a detector rebuilt from the artifact's tokens must agree with the original
on the union gate. See ``docs/experimental-spaced-seeds.md``.
"""

from __future__ import annotations

import pytest

from corpus_assay.spaced_index import read_spaced_index, write_spaced_index
from corpus_assay.spaced_seeds import (
    CLASS_MAPS,
    DEFAULT_PATTERNS,
    SpacedSeedConfig,
    SpacedSeedDetector,
    SpacedSeedFamily,
    build_spaced_index,
)

ITEM0 = [f"a{i}" for i in range(30)]
ITEM1 = [f"b{i}" for i in range(30)]
# A unicode token + a token shared across items, to stress interning.
ITEM2 = ["shared", "café", "naïve"] + [f"c{i}" for i in range(20)] + ["shared"]
PROTECTED = [ITEM0, ITEM1, ITEM2]

CONFIG = SpacedSeedConfig(min_loci=3, ver_min_span=17, ver_identity=0.85)


def _family() -> SpacedSeedFamily:
    return SpacedSeedFamily(patterns=DEFAULT_PATTERNS)


def test_artifact_roundtrip_postings_and_tokens(tmp_path):
    fam = _family()
    out = tmp_path / "idx.spaced"
    write_spaced_index(out, PROTECTED, fam, CONFIG, class_map=None, n=13)

    art = read_spaced_index(out)

    assert art.patterns == DEFAULT_PATTERNS
    assert art.span == fam.span == 17
    assert art.weight == fam.weight == 13
    assert art.exact_n == 13
    # token blob reconstructs the protected items exactly (interning is lossless)
    assert art.item_tokens == PROTECTED
    # postings match the in-memory reference exactly
    expected = build_spaced_index(PROTECTED, fam, None)
    assert art.postings == {h: list(v) for h, v in expected.items()}


def test_meta_sidecar_written(tmp_path):
    out = tmp_path / "idx.spaced"
    meta = write_spaced_index(out, PROTECTED, _family(), CONFIG, n=13)
    assert (tmp_path / "idx.spaced.meta.json").exists()
    assert meta["format"] == "CASPACE1"
    assert meta["min_loci"] == 3 and meta["ver_identity"] == 0.85
    assert meta["span"] == 17 and meta["n_items"] == len(PROTECTED)


def test_detector_from_artifact_matches_original(tmp_path):
    """A detector rebuilt from the artifact's tokens agrees with the original on the
    union gate across exact-only / spaced-only / both / clean docs."""
    fam = _family()
    out = tmp_path / "idx.spaced"
    write_spaced_index(out, PROTECTED, fam, CONFIG, n=13)
    art = read_spaced_index(out)

    original = SpacedSeedDetector(
        PROTECTED, fam, CONFIG, n=13, exact_min_hits=2, exact_min_coverage=0.01
    )
    rebuilt = SpacedSeedDetector(
        art.item_tokens,
        art.to_family(),
        CONFIG,
        n=13,
        exact_min_hits=2,
        exact_min_coverage=0.01,
    )

    bg = [f"z{i}" for i in range(10)]
    docs = [
        bg + ITEM0[:24] + bg,  # both
        bg + ITEM0[:16] + bg,  # exact-only (< span-17)
        bg + ["X"] + ITEM0[1:24] + bg,  # near-dup
        [f"q{i}" for i in range(40)],  # clean
        bg + ITEM2[:24] + bg,  # interned/unicode item
    ]
    for doc in docs:
        assert original.union_flag(doc) == rebuilt.union_flag(doc)
        assert original.exact_flag(doc) == rebuilt.exact_flag(doc)
        assert original.spaced_flag(doc) == rebuilt.spaced_flag(doc)


def test_bad_magic_rejected(tmp_path):
    out = tmp_path / "idx.spaced"
    write_spaced_index(out, PROTECTED, _family(), CONFIG, n=13)
    raw = bytearray(out.read_bytes())
    raw[0:8] = b"NOTSPACE"
    out.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="bad magic"):
        read_spaced_index(out)


def test_class_aware_resolves_class_map(tmp_path):
    """class_aware (without an explicit class_map) drives the postings, so the serialized
    index matches the declared class_aware rather than silently using no mapping."""
    fam = _family()
    # An item with pure-digit tokens (positions 5-13) so the `num` mapping actually fires.
    num_item = (
        ["q", "w", "e", "r", "t"] + [str(d) for d in range(1, 10)] + ["z", "x", "c"]
    )
    items = [num_item, ITEM0]
    out = tmp_path / "idx.spaced"
    meta = write_spaced_index(out, items, fam, CONFIG, class_aware="num", n=13)
    assert meta["class_aware"] == "num"
    art = read_spaced_index(out)
    expected = build_spaced_index(items, fam, CLASS_MAPS["num"])
    assert art.postings == {h: list(v) for h, v in expected.items()}
    # and it differs from the no-mapping build (sanity: the knob actually took effect)
    plain = build_spaced_index(items, fam, None)
    assert art.postings != {h: list(v) for h, v in plain.items()}


def test_unknown_class_aware_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown class_aware"):
        write_spaced_index(
            tmp_path / "x.spaced",
            PROTECTED,
            _family(),
            CONFIG,
            class_aware="bogus",
            n=13,
        )
