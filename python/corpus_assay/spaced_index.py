"""Serializable spaced-seed index artifact (`.spaced`) for the Rust scanner.

The exact channel ships only n-gram *hashes* (`.native`). The spaced verifier is
token-level --- it aligns a document's raw tokens against a protected item's raw tokens
--- so a spaced index must additionally carry the protected-item token sequences. This
module builds a single little-endian, Rust-loadable binary holding:

  * the seed patterns,
  * an interned token table + per-item token-id arrays (for verification),
  * the ``seed_hash -> [(item_id, item_pos), ...]`` postings, sorted by hash.

See ``docs/experimental-spaced-seeds.md`` for the on-disk layout. The format mirrors
the stop-grams (`src/stopgrams.rs`) and items-postings (`src/items_index.rs`)
artifacts. The Rust scanner (`src/spaced.rs`) reads the same bytes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from corpus_assay.spaced_seeds import (
    CLASS_MAPS,
    SpacedSeedConfig,
    SpacedSeedFamily,
    TokenClassMap,
    build_spaced_index,
)

MAGIC = b"CASPACE1"
FORMAT_VERSION = 1
# magic, version, span, weight, n_patterns, exact_n, n_items, n_postings
_HEADER = struct.Struct("<8sIIIIIIQ")


@dataclass(frozen=True)
class SpacedIndexArtifact:
    """In-memory view of a parsed ``.spaced`` file."""

    patterns: tuple[str, ...]
    span: int
    weight: int
    exact_n: int
    item_tokens: list[list[str]]
    postings: dict[int, list[tuple[int, int]]]

    def to_family(self) -> SpacedSeedFamily:
        return SpacedSeedFamily(patterns=self.patterns, exact_n=self.exact_n)


def _u32(n: int) -> bytes:
    return struct.pack("<I", n)


def write_spaced_index(
    out_path: str | Path,
    eval_tokens: Sequence[Sequence[str]],
    family: SpacedSeedFamily,
    config: SpacedSeedConfig,
    *,
    class_map: TokenClassMap | None = None,
    class_aware: str = "none",
    n: int = 13,
    built_from: dict | None = None,
    created_at: str | None = None,
) -> dict:
    """Build and write a ``.spaced`` artifact (+ ``.spaced.meta.json``).

    Postings are built with ``build_spaced_index`` (the reference), so the artifact is
    a faithful serialization of the in-memory index. Returns the metadata dict written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if class_aware not in CLASS_MAPS:
        raise ValueError(
            f"unknown class_aware {class_aware!r}; expected one of {sorted(CLASS_MAPS)}"
        )
    # Keep the serialized index and the declared class_aware in sync: when no explicit
    # class_map is passed, derive it from class_aware (the canonical knob) so the meta
    # never claims a mapping the postings were not built with.
    if class_map is None:
        class_map = CLASS_MAPS[class_aware]
    items = [list(toks) for toks in eval_tokens]
    postings = build_spaced_index(items, family, class_map)

    # --- intern tokens (sorted for deterministic ids) ------------------------
    vocab = sorted({tok for toks in items for tok in toks})
    tok_id = {tok: i for i, tok in enumerate(vocab)}

    buf = bytearray()
    buf += _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        family.span,
        family.weight,
        len(family.patterns),
        family.exact_n,
        len(items),
        len(postings),
    )

    # patterns block: each pattern as `span` ASCII bytes
    for pat in family.patterns:
        buf += pat.encode("ascii")

    # token blob: intern table, then per-item id arrays
    buf += _u32(len(vocab))
    for tok in vocab:
        b = tok.encode("utf-8")
        buf += _u32(len(b))
        buf += b
    for toks in items:
        buf += _u32(len(toks))
        for tok in toks:
            buf += _u32(tok_id[tok])

    # postings: sorted by hash; entries sorted by (item_id, item_pos)
    for h in sorted(postings):
        entries = sorted(postings[h])
        buf += struct.pack("<QI", h, len(entries))
        for item_id, item_pos in entries:
            buf += struct.pack("<II", item_id, item_pos)

    out_path.write_bytes(bytes(buf))

    meta = {
        "schema_version": 1,
        "format": "CASPACE1",
        "format_version": FORMAT_VERSION,
        "created_at": created_at,
        "ngram": n,
        "span": family.span,
        "weight": family.weight,
        "exact_n": family.exact_n,
        "patterns": list(family.patterns),
        "min_loci": config.min_loci,
        "ver_min_span": config.ver_min_span,
        "ver_identity": config.ver_identity,
        "class_aware": class_aware,
        "hash_fn": "blake2b-64-le",
        "n_items": len(items),
        "n_postings": len(postings),
        "built_from": built_from,
    }
    try:  # best-effort compat fingerprints (require the staged ngram config)
        from corpus_assay._native import (
            ngram_config_hash,
            normalization_config_sha,
        )

        meta["ngram_config_hash"] = ngram_config_hash()
        meta["normalization_config_sha"] = normalization_config_sha()
    except Exception:  # pragma: no cover - metadata best-effort only
        pass

    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def read_spaced_index(path: str | Path) -> SpacedIndexArtifact:
    """Parse a ``.spaced`` file back into a :class:`SpacedIndexArtifact`."""
    data = memoryview(Path(path).read_bytes())
    off = 0
    magic, version, span, weight, n_patterns, exact_n, n_items, n_postings = (
        _HEADER.unpack_from(data, off)
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}; not a .spaced file")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported .spaced format version {version}")
    off += _HEADER.size

    patterns = []
    for _ in range(n_patterns):
        patterns.append(bytes(data[off : off + span]).decode("ascii"))
        off += span

    (n_tokens,) = struct.unpack_from("<I", data, off)
    off += 4
    vocab: list[str] = []
    for _ in range(n_tokens):
        (blen,) = struct.unpack_from("<I", data, off)
        off += 4
        vocab.append(bytes(data[off : off + blen]).decode("utf-8"))
        off += blen

    item_tokens: list[list[str]] = []
    for _ in range(n_items):
        (cnt,) = struct.unpack_from("<I", data, off)
        off += 4
        ids = struct.unpack_from(f"<{cnt}I", data, off)
        off += 4 * cnt
        item_tokens.append([vocab[i] for i in ids])

    postings: dict[int, list[tuple[int, int]]] = {}
    for _ in range(n_postings):
        h, cnt = struct.unpack_from("<QI", data, off)
        off += 12
        flat = struct.unpack_from(f"<{2 * cnt}I", data, off)
        off += 8 * cnt
        postings[h] = [(flat[2 * j], flat[2 * j + 1]) for j in range(cnt)]

    return SpacedIndexArtifact(
        patterns=tuple(patterns),
        span=span,
        weight=weight,
        exact_n=exact_n,
        item_tokens=item_tokens,
        postings=postings,
    )


def spaced_meta_path(spaced_path: str | Path) -> Path:
    """Sidecar meta path for a ``.spaced`` artifact (mirrors ``stopgrams_meta_path``)."""
    return Path(str(spaced_path) + ".meta.json")


def load_spaced_meta(spaced_path: str | Path) -> dict:
    """Load and parse the ``.spaced.meta.json`` sidecar."""
    return json.loads(spaced_meta_path(spaced_path).read_text(encoding="utf-8"))


def validate_spaced_compatibility(
    *,
    spaced_meta: dict,
    index_meta: dict,
    expected_ngram: int,
) -> None:
    """Assert a ``.spaced`` artifact is compatible with the exact ``.native`` index.

    The spaced verifier reuses the same tokenizer/normalization and n-gram config as the
    exact channel; a mismatch means the spaced item tokens were produced under different
    rules than the scanned documents. Mirrors ``validate_stopgrams_compatibility``.
    """
    spaced_ngram = spaced_meta.get("ngram")
    if spaced_ngram != expected_ngram:
        raise ValueError(
            f"Spaced ngram {spaced_ngram} does not match scan ngram {expected_ngram}."
        )
    index_ngram = index_meta.get("ngram", expected_ngram)
    if spaced_ngram != index_ngram:
        raise ValueError(
            f"Spaced ngram does not match index ngram ({spaced_ngram} vs {index_ngram})."
        )
    if spaced_meta.get("normalization_config_sha") != index_meta.get(
        "normalization_config_sha"
    ):
        raise ValueError(
            "Spaced normalization config does not match index normalization config."
        )
    if spaced_meta.get("ngram_config_hash") != index_meta.get("ngram_config_hash"):
        raise ValueError(
            "Spaced ngram config hash does not match index ngram config hash."
        )
