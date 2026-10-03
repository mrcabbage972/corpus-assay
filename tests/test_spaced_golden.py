"""Golden parity vectors for the experimental spaced-seed channel.

These vectors freeze the Python reference's *intermediate* outputs --- the per-document
seed stream, the ``(item_id, diagonal) -> doc loci`` collapse, and the
exact/spaced/union verdicts --- on a set of hand-built fixtures that exercise every
regime (clean, exact-only, spaced-only, both, field-separator, counting-trap). The Rust
implementation (``src/spaced.rs``) loads the *same* JSON and must reproduce these
exactly; see ``docs/experimental-spaced-seeds.md``.

Hashes are blake2b-64-le over the seed string ``"s{pid}|tok ... tok"`` --- identical on
both sides --- and are stored as decimal strings so any JSON parser round-trips them
without float loss.

Regenerate after an *intentional* reference change with::

    python tests/test_spaced_golden.py    # rewrites tests/fixtures/spaced_golden.json
"""

from __future__ import annotations

import json
from pathlib import Path

from corpus_assay.spaced_seeds import (
    DEFAULT_PATTERNS,
    SpacedSeedConfig,
    SpacedSeedDetector,
    SpacedSeedFamily,
    iter_spaced_seeds,
    spaced_candidates,
)

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "spaced_golden.json"

# Default operating point: 4 patterns, m=3, L_min=17, theta=0.85, class_aware=none;
# exact n=13, h=2, cov=0.01.
N = 13
MIN_HITS = 2
MIN_COVERAGE = 0.01
MIN_LOCI = 3
VER_MIN_SPAN = 17
VER_IDENTITY = 0.85

# Distinct synthetic tokens so seeds are unique and never trip the reject filter.
ITEM0 = [f"a{i}" for i in range(30)]
ITEM1 = [f"b{i}" for i in range(30)]
PROTECTED = [ITEM0, ITEM1]

FIELD_SEP = "<|field_sep|>"


def _subst(span: list[str], positions: dict[int, str]) -> list[str]:
    out = list(span)
    for p, tok in positions.items():
        out[p] = tok
    return out


def _bg(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _cases() -> list[tuple[str, list[str]]]:
    """(name, doc_tokens). Constructions chosen to hit distinct gate regimes; the
    actual verdicts are whatever the reference produces (recorded, not asserted here)."""
    cases: list[tuple[str, list[str]]] = []

    # 1. Clean: no overlap with any protected item.
    cases.append(("clean", _bg("z", 40)))

    # 2. Exact-only: a 16-token verbatim span (< span-17, so no full seed window) gives
    #    >=2 exact 13-grams but cannot raise a spaced locus.
    cases.append(("exact_only", _bg("z", 12) + ITEM0[0:16] + _bg("y", 12)))

    # 3. Spaced-only: a 24-token near-duplicate of item0 with substitutions at positions
    #    6 and 18 so every contiguous 13-gram is broken (exact -> 0 hits) while several
    #    span-17 windows still align on one diagonal and verify at ~22/24 identity.
    span = _subst(ITEM0[0:24], {6: "EDIT6", 18: "EDIT18"})
    cases.append(("spaced_only", _bg("z", 8) + span + _bg("y", 8)))

    # 4. Both: a 24-token verbatim span of item0 (exact fires AND spaced verifies).
    cases.append(("both", _bg("z", 8) + ITEM0[0:24] + _bg("y", 8)))

    # 5. Field separator inside the overlap: a verbatim span of item1 split by a field
    #    separator, so the windows spanning it are skipped.
    cases.append(("field_sep", _bg("z", 6) + ITEM1[0:12] + [FIELD_SEP] + ITEM1[12:24]))

    # 6. Counting-trap: a single span-17 verbatim window of item0 (one doc locus only),
    #    which fires multiple patterns but collapses to one locus < min_loci.
    cases.append(("counting_trap", _bg("z", 5) + ITEM0[0:17] + _bg("y", 5)))

    return cases


def build_golden() -> dict:
    family = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
    config = SpacedSeedConfig(
        min_loci=MIN_LOCI, ver_min_span=VER_MIN_SPAN, ver_identity=VER_IDENTITY
    )
    det = SpacedSeedDetector(
        PROTECTED,
        family,
        config,
        n=N,
        exact_min_hits=MIN_HITS,
        exact_min_coverage=MIN_COVERAGE,
        class_map=None,
    )

    cases_out = []
    for name, doc in _cases():
        seeds = [
            [pos, pid, str(h)] for pos, pid, h in iter_spaced_seeds(doc, family, None)
        ]
        loci_map = spaced_candidates(doc, det.spaced_index, family, None)
        loci = sorted(
            (
                {
                    "item_id": item_id,
                    "diag": diag,
                    "positions": sorted(positions),
                }
                for (item_id, diag), positions in loci_map.items()
            ),
            key=lambda d: (d["item_id"], d["diag"]),
        )
        cases_out.append(
            {
                "name": name,
                "doc_tokens": doc,
                "seeds": seeds,
                "loci": loci,
                "exact_flag": det.exact_flag(doc),
                "spaced_verified": det.spaced_flag(doc, verified=True),
                "union_flag": det.union_flag(doc),
            }
        )

    return {
        "config": {
            "patterns": list(DEFAULT_PATTERNS),
            "span": family.span,
            "weight": family.weight,
            "exact_n": N,
            "min_hits": MIN_HITS,
            "min_coverage": MIN_COVERAGE,
            "min_loci": MIN_LOCI,
            "ver_min_span": VER_MIN_SPAN,
            "ver_identity": VER_IDENTITY,
            "class_aware": "none",
            "hash": "blake2b-64-le",
            "field_separator": FIELD_SEP,
        },
        "protected_items": PROTECTED,
        "cases": cases_out,
    }


def test_golden_matches_committed():
    """The Python reference must reproduce the committed golden vectors byte-for-byte."""
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} missing; regenerate with `python {Path(__file__).name}`"
    )
    committed = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert build_golden() == committed


def test_golden_covers_every_regime():
    """The fixture set must exercise exact-only, spaced-only, both, and neither."""
    by_name = {c["name"]: c for c in build_golden()["cases"]}
    assert by_name["clean"]["union_flag"] is False
    assert by_name["exact_only"]["exact_flag"] is True
    assert by_name["exact_only"]["spaced_verified"] is False
    assert by_name["spaced_only"]["exact_flag"] is False
    assert by_name["spaced_only"]["spaced_verified"] is True
    assert by_name["both"]["exact_flag"] is True
    assert by_name["both"]["spaced_verified"] is True
    # Counting-trap: one span-17 window is a single locus, below min_loci=3.
    assert by_name["counting_trap"]["spaced_verified"] is False


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(build_golden(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {GOLDEN_PATH}")
