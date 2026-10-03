"""Rust↔Python parity for the experimental spaced-seed channel.

The Rust port must reproduce the Python reference exactly: the same seed stream and the
same verified-flag verdict. We check this two ways:

  1. against the committed golden vectors (``tests/fixtures/spaced_golden.json``), and
  2. against the live Python ``SpacedSeedDetector`` on many randomized documents,

both via the ``.spaced`` artifact and the Rust parity entry points
(``spaced_seed_hashes_rust`` / ``spaced_verified_item_rust``). See
``docs/experimental-spaced-seeds.md``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from corpus_assay._native import (
    spaced_seed_hashes_rust,
    spaced_verified_item_rust,
)
from corpus_assay.spaced_index import write_spaced_index
from corpus_assay.spaced_seeds import (
    DEFAULT_PATTERNS,
    SpacedSeedConfig,
    SpacedSeedDetector,
    SpacedSeedFamily,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "spaced_golden.json").read_text(
        encoding="utf-8"
    )
)
PATTERNS = list(DEFAULT_PATTERNS)
CFG = SpacedSeedConfig(min_loci=3, ver_min_span=17, ver_identity=0.85)


def _write_index(tmp_path, items) -> str:
    out = tmp_path / "golden.spaced"
    write_spaced_index(
        out, items, SpacedSeedFamily(patterns=DEFAULT_PATTERNS), CFG, n=13
    )
    return str(out)


def test_rust_seed_stream_matches_golden():
    """The Rust seed (pos, pid, hash) stream equals the reference's, per golden case."""
    for case in GOLDEN["cases"]:
        rust = [
            [pos, pid, str(h)]
            for pos, pid, h in spaced_seed_hashes_rust(case["doc_tokens"], PATTERNS)
        ]
        assert rust == case["seeds"], f"seed stream mismatch on case {case['name']}"


def test_rust_verified_flag_matches_golden(tmp_path):
    """The Rust verified verdict matches the reference's spaced_verified per golden case."""
    spaced_path = _write_index(tmp_path, GOLDEN["protected_items"])
    for case in GOLDEN["cases"]:
        item = spaced_verified_item_rust(spaced_path, case["doc_tokens"], 3, 17, 0.85)
        assert (item is not None) == case["spaced_verified"], (
            f"verified mismatch on case {case['name']}: rust={item}, "
            f"expected={case['spaced_verified']}"
        )


def test_rust_matches_python_detector_randomized(tmp_path):
    """Randomized fuzz: Rust verified-flag agrees with the live Python detector."""
    rng = random.Random(1234)
    vocab = [f"w{i}" for i in range(120)]
    # protected items of varied length (some long enough for spaced, some not)
    items = [[rng.choice(vocab) for _ in range(rng.randint(10, 45))] for _ in range(25)]
    det = SpacedSeedDetector(
        items,
        SpacedSeedFamily(patterns=DEFAULT_PATTERNS),
        CFG,
        n=13,
        exact_min_hits=2,
        exact_min_coverage=0.01,
    )
    spaced_path = _write_index(tmp_path, items)

    def bg(k):
        return [rng.choice(vocab) for _ in range(k)]

    docs: list[list[str]] = []
    for _ in range(60):
        kind = rng.random()
        it = rng.choice(items)
        if len(it) < 20 or kind < 0.25:
            docs.append(bg(rng.randint(5, 40)))  # likely clean
            continue
        start = rng.randint(0, len(it) - 18)
        end = min(len(it), start + rng.randint(18, 28))
        span = list(it[start:end])
        # inject 0-3 substitutions to straddle the verify threshold
        for _ in range(rng.randint(0, 3)):
            span[rng.randrange(len(span))] = f"EDIT{rng.randrange(99)}"
        docs.append(bg(rng.randint(0, 10)) + span + bg(rng.randint(0, 10)))

    mism = []
    for doc in docs:
        py = det.spaced_flag(doc, verified=True)
        rust = spaced_verified_item_rust(spaced_path, doc, 3, 17, 0.85) is not None
        if py != rust:
            mism.append((doc, py, rust))
    assert not mism, f"{len(mism)}/{len(docs)} spaced-flag mismatches, e.g. {mism[0]}"


def test_missing_index_raises(tmp_path):
    with pytest.raises(Exception):
        spaced_verified_item_rust(
            str(tmp_path / "nope.spaced"), ["a", "b"], 3, 17, 0.85
        )
