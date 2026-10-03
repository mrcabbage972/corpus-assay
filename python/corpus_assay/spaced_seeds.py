#!/usr/bin/env python3
"""Spaced-seed (PatternHunter-style) matching as an optional recall channel.

Exact 13-grams are precise but brittle: a single substitution in a contaminated
span destroys every 13-gram that covers it. A *spaced seed* of span ``L`` and
weight ``w`` hashes only the ``w`` selected positions of an ``L``-token window
(``L - w`` "don't care" gaps), so it tolerates edits in the gap positions while
preserving ``w`` tokens of evidence. This module implements span-17/weight-13
seeds as a hit-and-verify channel on top of the exact-13 gate:

    flag(d) = exact13_gate(d)  OR  spaced_verified_gate(d)

The exact channel stays the high-precision baseline; the spaced channel recovers
edited/near-duplicate contamination it misses.

Counting trap (handled deliberately): with several seed patterns, one incidental
overlap can fire multiple patterns at the same position. We collapse hits by
``(item_id, diagonal = doc_pos - item_pos)`` and count DISTINCT document loci,
never raw pattern matches. A candidate locus is then confirmed by same-item local
alignment (token identity >= theta over a span of >= ``ver_min_span`` tokens).

This module is the pure-Python reference implementation. It depends only on the
canonical normalisation/hashing (``corpus_assay.text_utils``) so its hashes are
identical to the production exact index. The scanner runs the Rust port in
``src/spaced.rs``, which ``tests/test_spaced_rust_parity.py`` keeps bit-identical
to this module.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

from corpus_assay.constants import FIELD_SEPARATOR
from corpus_assay.text_utils import hash_ngram, ngram_allowed

# Default span-17 / weight-13 seed family. Not claimed optimal; tune it on your
# own validation data under the same false-positive constraints as the exact gate.
DEFAULT_PATTERNS: tuple[str, ...] = (
    "11101011101110111",
    "11101101011011111",
    "11011101110101111",
    "11110110101101111",
)

# Optional hook: map a single token to a class symbol for candidate generation
# ONLY (e.g. numbers -> "<num>", renamed identifiers -> "<id>"). Used when
# hashing seeds, never during verification. None = exact-token seeds.
TokenClassMap = Callable[[str], str]

# Tokens are already normalised (lowercased, word-regex [a-z0-9]+(?:'[a-z0-9]+)?),
# so a "number" is a pure-digit token and an "identifier-like" token mixes letters
# and digits (e.g. var0, x1, item3 -- renamed identifiers and indexed names).
_NUM_RE = re.compile(r"^[0-9]+$")
_ID_RE = re.compile(r"^(?=.*[a-z])(?=.*[0-9])[a-z0-9]+$")


def class_map_numbers(tok: str) -> str:
    """Collapse pure-digit tokens to ``<num>`` for seed candidate generation.

    Targets the number-perturbation edit in template/near-dup contamination, which
    otherwise breaks the selected seed positions. Low false-positive risk: numbers
    are an unambiguous class, and verification still runs on raw tokens.
    """
    return "<num>" if _NUM_RE.match(tok) else tok


def class_map_num_id(tok: str) -> str:
    """Collapse numbers to ``<num>`` and identifier-like tokens to ``<id>``.

    Adds identifier renaming (e.g. ``foo`` -> ``var0``) to the number class. Higher
    recall on template edits but a larger incidental-collision surface, so its
    marginal false-positive rate must be measured (it is, in spaced_seed_selection).
    """
    if _NUM_RE.match(tok):
        return "<num>"
    if _ID_RE.match(tok):
        return "<id>"
    return tok


CLASS_MAPS: dict[str, TokenClassMap | None] = {
    "none": None,
    "num": class_map_numbers,
    "num_id": class_map_num_id,
}


@dataclass(frozen=True)
class SpacedSeedFamily:
    """A set of equal-span, equal-weight spaced-seed bit patterns.

    Each pattern is a bitstring of length ``span``; a ``'1'`` selects the token
    at that offset, a ``'0'`` is a don't-care gap. All patterns must share the
    same span and weight (number of ones) so loci are comparable across patterns.
    """

    patterns: tuple[str, ...] = DEFAULT_PATTERNS
    # per-pattern list of selected offsets (positions of '1'), derived in __post_init__
    offsets: tuple[tuple[int, ...], ...] = field(default_factory=tuple, compare=False)
    # exact n-gram detection size the spaced channel sits beside; used only to enforce
    # the non-containment invariant below (a lone exact n-gram must not trip a seed).
    exact_n: int = 13

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("seed family must contain at least one pattern")
        span = len(self.patterns[0])
        weight = self.patterns[0].count("1")
        offs: list[tuple[int, ...]] = []
        for p in self.patterns:
            if set(p) - {"0", "1"}:
                raise ValueError(f"pattern {p!r} must be a 0/1 bitstring")
            if len(p) != span:
                raise ValueError(f"pattern {p!r} span {len(p)} != family span {span}")
            if p.count("1") != weight:
                raise ValueError(
                    f"pattern {p!r} weight {p.count('1')} != family weight {weight}"
                )
            offs.append(tuple(j for j, c in enumerate(p) if c == "1"))
        # frozen dataclass: set derived field via object.__setattr__
        object.__setattr__(self, "offsets", tuple(offs))

        # Non-containment invariant: no contiguous exact_n-token window of the span may
        # contain ALL of a pattern's selected positions. If it did, a single exact
        # exact_n-token incidental overlap would match every selected token and trip the
        # spaced channel on its own -- collapsing the spaced policy back to exact-n and
        # defeating the point of the gaps. (For span-17/weight-13/exact_n-13 the worst
        # window holds 10 of 13.)
        if span <= self.exact_n:
            raise ValueError(
                f"seed span {span} must exceed exact_n {self.exact_n} "
                f"(otherwise every seed is contained in one exact n-gram)"
            )
        for p, off in zip(self.patterns, offs):
            worst = max(
                sum(1 for o in off if start <= o < start + self.exact_n)
                for start in range(span - self.exact_n + 1)
            )
            if worst >= weight:
                raise ValueError(
                    f"pattern {p!r} packs all {weight} selected positions into a "
                    f"{self.exact_n}-token window; a lone exact {self.exact_n}-gram "
                    f"could trip the spaced channel (non-containment invariant)"
                )

    @property
    def span(self) -> int:
        return len(self.patterns[0])

    @property
    def weight(self) -> int:
        return self.patterns[0].count("1")

    def __len__(self) -> int:
        return len(self.patterns)

    def subset(self, num_patterns: int) -> "SpacedSeedFamily":
        """Return a family using only the first ``num_patterns`` patterns."""
        return SpacedSeedFamily(
            patterns=tuple(self.patterns[:num_patterns]), exact_n=self.exact_n
        )


@dataclass(frozen=True)
class SpacedSeedConfig:
    """Verifier + gate thresholds for the spaced channel (selected on validation)."""

    min_loci: int = 2  # distinct spaced loci required on one (item, diagonal)
    ver_min_span: int = 25  # verified aligned span length, in tokens
    ver_identity: float = 0.75  # token identity over the verified span


def _seed_string(
    toks: Sequence[str],
    i: int,
    offsets: Sequence[int],
    pid: int,
    class_map: TokenClassMap | None,
) -> str:
    """Build the hashable string for pattern ``pid`` at window start ``i``.

    The pattern id is mixed into the string so identical token tuples under
    different patterns hash distinctly (a single overlap firing several patterns
    is collapsed later by locus, not by hash).
    """
    if class_map is None:
        sel = (toks[i + o] for o in offsets)
    else:
        sel = (class_map(toks[i + o]) for o in offsets)
    return f"s{pid}|" + " ".join(sel)


def iter_exact_grams(toks: Sequence[str], n: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start_pos, hash)`` for contiguous ``n``-grams the exact gate accepts.

    Matches the production exact-$n$ scanner: windows crossing a field separator are
    skipped, and each window is passed through the active n-gram reject filter
    (``ngram_allowed``) before hashing, so the exact index/gate built here contain
    exactly the n-grams the deployed detector would---never the rejected
    MCQ/boilerplate forms. A field separator jumps the window past it.
    """
    n_toks = len(toks)
    i = 0
    while i <= n_toks - n:
        sep = -1
        for off in range(n - 1, -1, -1):
            if toks[i + off] == FIELD_SEPARATOR:
                sep = i + off
                break
        if sep != -1:
            i = sep + 1
            continue
        ng = " ".join(toks[i : i + n])
        if ngram_allowed(ng):
            yield i, hash_ngram(ng)
        i += 1


def iter_spaced_seeds(
    toks: Sequence[str],
    family: SpacedSeedFamily,
    class_map: TokenClassMap | None = None,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(start_pos, pattern_id, seed_hash)`` for every seed/window.

    Windows that contain a field separator anywhere in the span are skipped (a
    seed must not bridge two fields); a separator jumps the window past it. Seeds are
    a non-contiguous matcher distinct from the exact gate, so the contiguous-phrase
    reject filter is not applied here---the same-item local verification on raw tokens
    is the spaced channel's false-positive guard.
    """
    span = family.span
    n_toks = len(toks)
    i = 0
    while i <= n_toks - span:
        sep = -1
        for off in range(span - 1, -1, -1):
            if toks[i + off] == FIELD_SEPARATOR:
                sep = i + off
                break
        if sep != -1:
            i = sep + 1
            continue
        for pid, offsets in enumerate(family.offsets):
            yield i, pid, hash_ngram(_seed_string(toks, i, offsets, pid, class_map))
        i += 1


def build_exact_index(eval_tokens: Iterable[Sequence[str]], n: int) -> set[int]:
    """Union set of exact ``n``-gram hashes over all protected items."""
    index: set[int] = set()
    for toks in eval_tokens:
        for _, h in iter_exact_grams(toks, n):
            index.add(h)
    return index


def build_spaced_index(
    eval_tokens: Sequence[Sequence[str]],
    family: SpacedSeedFamily,
    class_map: TokenClassMap | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """Map ``seed_hash -> [(item_id, item_pos), ...]`` over all protected items."""
    index: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for item_id, toks in enumerate(eval_tokens):
        for i, _pid, h in iter_spaced_seeds(toks, family, class_map):
            index[h].append((item_id, i))
    return dict(index)


def exact_flag(
    toks: Sequence[str], index: set[int], n: int, min_hits: int, min_coverage: float
) -> bool:
    """Exact-``n`` union gate matching the production scanner.

    Coverage is distinct index hits over the number of *distinct* document
    shingles (not positional n-grams), so repeated boilerplate or long documents with
    duplicate n-grams are scored the same way the deployed gate scores them.
    """
    shingles = {h for _, h in iter_exact_grams(toks, n)}
    if not shingles:
        return False
    hits = len(shingles & index)
    cov = hits / len(shingles)
    return hits >= min_hits and cov >= min_coverage


def spaced_candidates(
    toks: Sequence[str],
    index: dict[int, list[tuple[int, int]]],
    family: SpacedSeedFamily,
    class_map: TokenClassMap | None = None,
) -> dict[tuple[int, int], set[int]]:
    """Group seed hits into ``(item_id, diagonal) -> {distinct doc loci}``.

    The diagonal ``doc_pos - item_pos`` pins a candidate to one alignment offset
    against one item; collecting the *set* of document start positions there is
    what makes multi-pattern hits at a single locus count once.
    """
    loci: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i, _pid, h in iter_spaced_seeds(toks, family, class_map):
        for item_id, ipos in index.get(h, ()):
            loci[(item_id, i - ipos)].add(i)
    return loci


def verify(
    toks: Sequence[str],
    item_toks: Sequence[str],
    diag: int,
    doc_loci: set[int],
    span: int,
    ver_min_span: int,
    ver_identity: float,
) -> bool:
    """Confirm a candidate locus by same-item local alignment on RAW tokens.

    Walk the document interval spanned by the candidate loci against the item at
    the fixed diagonal; require at least ``ver_min_span`` comparable tokens and
    token identity ``>= ver_identity``. Verification uses raw tokens even when the
    candidate was generated with a class map, so class-aware seeds cannot inflate
    the false-positive rate past this gate.
    """
    if not doc_loci:
        return False
    lo = min(doc_loci)
    hi = max(doc_loci) + span
    # Intersect the candidate interval with the two arrays' valid ranges once, so the
    # loop has no per-token boundary checks and we can bail before it when too short.
    overlap_lo = max(lo, 0, diag)
    overlap_hi = min(hi, len(toks), len(item_toks) + diag)
    total = overlap_hi - overlap_lo
    if total < ver_min_span or total <= 0:
        return False
    matched = sum(
        1 for dp in range(overlap_lo, overlap_hi) if toks[dp] == item_toks[dp - diag]
    )
    return matched / total >= ver_identity


class SpacedSeedDetector:
    """Holds built indexes + thresholds and exposes the channel gates.

    Build once from the protected eval items, then call :meth:`union_flag` (or
    the individual channels) per scanned document.
    """

    def __init__(
        self,
        eval_tokens: Sequence[Sequence[str]],
        family: SpacedSeedFamily = SpacedSeedFamily(),
        config: SpacedSeedConfig = SpacedSeedConfig(),
        *,
        n: int = 13,
        exact_min_hits: int = 2,
        exact_min_coverage: float = 0.01,
        class_map: TokenClassMap | None = None,
    ) -> None:
        self.eval_tokens = list(eval_tokens)
        self.family = family
        self.config = config
        self.n = n
        self.exact_min_hits = exact_min_hits
        self.exact_min_coverage = exact_min_coverage
        self.class_map = class_map
        self.exact_index = build_exact_index(self.eval_tokens, n)
        self.spaced_index = build_spaced_index(self.eval_tokens, family, class_map)

    def exact_flag(self, toks: Sequence[str]) -> bool:
        return exact_flag(
            toks, self.exact_index, self.n, self.exact_min_hits, self.exact_min_coverage
        )

    def spaced_flag(self, toks: Sequence[str], *, verified: bool = True) -> bool:
        loci = spaced_candidates(toks, self.spaced_index, self.family, self.class_map)
        cfg = self.config
        if not verified:
            # naive channel: distinct document loci across all candidates
            allloci: set[int] = set()
            for s in loci.values():
                allloci |= s
            return len(allloci) >= cfg.min_loci
        span = self.family.span
        for (item_id, diag), s in loci.items():
            if len(s) >= cfg.min_loci and verify(
                toks,
                self.eval_tokens[item_id],
                diag,
                s,
                span,
                cfg.ver_min_span,
                cfg.ver_identity,
            ):
                return True
        return False

    def union_flag(self, toks: Sequence[str]) -> bool:
        return self.exact_flag(toks) or self.spaced_flag(toks, verified=True)
