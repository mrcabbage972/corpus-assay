"""End-to-end: the experimental spaced channel in the Rust scanner.

Builds an exact ``.native`` index and a ``.spaced`` index from the same protected item,
then scans a parquet shard with three docs (clean / exact-only / edited spaced-only).
Without the spaced channel only the exact-only doc is flagged; enabling it additionally
recovers the edited doc the coverage/min_hits exact gate misses, and tags it with the
verified item id. See ``docs/experimental-spaced-seeds.md``.
"""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq

from _index_helpers import write_native_index
from corpus_assay._native import scan_stream_rust
from corpus_assay.spaced_index import write_spaced_index
from corpus_assay.spaced_seeds import (
    DEFAULT_PATTERNS,
    SpacedSeedConfig,
    SpacedSeedFamily,
    iter_exact_grams,
)

N = 13
# Protected item: 30 non-stopword tokens that survive normalization unchanged.
PROT = [f"w{i}" for i in range(30)]
FAM = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
CFG = SpacedSeedConfig(min_loci=3, ver_min_span=17, ver_identity=0.85)


def _build_indexes(tmp_path):
    exact_path = tmp_path / "exact.native"
    hashes = {h for _, h in iter_exact_grams(PROT, N)}
    write_native_index(exact_path, ngram=N, hashes=hashes)

    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, CFG, n=N)
    return str(exact_path), str(spaced_path)


def _parquet(texts, ids) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pydict({"text": texts, "doc_id": ids}), buf)
    return buf.getvalue()


def _docs():
    bg = " ".join(f"z{i}" for i in range(12))
    exact_doc = bg + " " + " ".join(PROT[:24]) + " " + bg  # verbatim -> exact fires
    # 24-token span with substitutions at positions 6 and 18: every contiguous 13-gram is
    # broken (exact misses), but >=3 span-17 seed windows still align and verify.
    span = list(PROT[:24])
    span[6], span[18] = "editsix", "editeighteen"
    spaced_doc = bg + " " + " ".join(span) + " " + bg
    clean_doc = " ".join(f"q{i}" for i in range(40))
    return (
        [clean_doc, exact_doc, spaced_doc],
        ["clean", "exact", "spaced"],
    )


def _scan(parquet_bytes, exact_path, spaced_path):
    return scan_stream_rust(
        input=io.BytesIO(parquet_bytes),
        text_key="text",
        id_key="doc_id",
        index_path=exact_path,
        n=N,
        min_hits=2,
        min_coverage=0.01,
        spaced_path=spaced_path,
        spaced_min_loci=3,
        spaced_ver_min_span=17,
        spaced_ver_identity=0.85,
    )


def test_spaced_disabled_misses_edited_doc(tmp_path):
    exact_path, _ = _build_indexes(tmp_path)
    texts, ids = _docs()
    scanned, contaminated, results, _ = scan_stream_rust(
        input=io.BytesIO(_parquet(texts, ids)),
        text_key="text",
        id_key="doc_id",
        index_path=exact_path,
        n=N,
        min_hits=2,
        min_coverage=0.01,
    )
    assert scanned == 3
    # Only the verbatim (exact) doc is flagged when the spaced channel is off.
    flagged_ids = {json.loads(r)["doc_id"] for r in results}
    assert contaminated == 1
    assert flagged_ids == {"exact-part-0"}


def test_spaced_enabled_recovers_edited_doc(tmp_path):
    exact_path, spaced_path = _build_indexes(tmp_path)
    texts, ids = _docs()
    scanned, contaminated, results, _ = _scan(
        _parquet(texts, ids), exact_path, spaced_path
    )
    assert scanned == 3
    recs = {json.loads(r)["doc_id"]: json.loads(r) for r in results}
    # exact-only AND the edited spaced-only doc are now flagged; clean is not.
    assert contaminated == 2
    assert set(recs) == {"exact-part-0", "spaced-part-0"}
    # the spaced-only flag carries the verified protected item id; the exact flag does not
    assert recs["spaced-part-0"].get("spaced_item") == 0
    assert "spaced_item" not in recs["exact-part-0"]
