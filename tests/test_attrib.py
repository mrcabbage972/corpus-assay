# test_attribution_index.py
import json
import struct
from pathlib import Path

import pytest

from corpus_assay.attribution import AttributionIndex, AttributionLegend


def test_get_source_id_assigns_stable_ids():
    idx = AttributionIndex()

    a0 = idx.get_source_id("a")
    b0 = idx.get_source_id("b")
    a1 = idx.get_source_id("a")

    assert a0 == 0
    assert b0 == 1
    assert a1 == 0
    assert idx.id_to_name == ["a", "b"]
    assert idx.name_to_id == {"a": 0, "b": 1}


def test_add_hash_for_source_and_mask_merge():
    idx = AttributionIndex()
    h = 123

    idx.add_hash_for_source(h, "a")  # sid 0 -> 1<<0 = 1
    assert idx.hash2mask[h] == 1

    idx.add_hash_for_source(h, "b")  # sid 1 -> 1<<1 = 2, merged => 3
    assert idx.hash2mask[h] == 3

    # Merging a raw mask also ORs
    idx.add_hash_mask(h, 1 << 5)
    assert idx.hash2mask[h] == (3 | (1 << 5))


def test_iter_mask_source_ids():
    mask = (1 << 0) | (1 << 3) | (1 << 9)
    sids = list(AttributionIndex.iter_mask_source_ids(mask))
    assert sids == [0, 3, 9]


def test_per_source_totals_counts_unique_hashes_per_source():
    idx = AttributionIndex()
    # Sources: a=0, b=1, c=2
    idx.add_hash_for_source(1, "a")  # a
    idx.add_hash_for_source(2, "a")  # a
    idx.add_hash_for_source(2, "b")  # a,b
    idx.add_hash_for_source(3, "b")  # b
    idx.add_hash_for_source(4, "c")  # c
    idx.add_hash_for_source(4, "a")  # a,c

    totals = idx.per_source_totals()
    assert totals == {
        0: 3,  # hashes 1,2,4 include a
        1: 2,  # hashes 2,3 include b
        2: 1,  # hash 4 includes c
    }


def test_write_and_read_sidecar_roundtrip(tmp_path: Path):
    idx = AttributionIndex()
    idx.add_hash_for_source(10, "a")
    idx.add_hash_for_source(20, "b")
    idx.add_hash_for_source(20, "a")  # merged mask for hash 20
    base = str(tmp_path / "sample")

    attr_path = idx.write_sidecar(base)
    assert Path(attr_path).exists()

    pairs = list(AttributionIndex.read_hash_mask_pairs(attr_path))
    # Order in file is insertion order of dict; Python 3.7+ preserves it.
    assert pairs == [
        (10, 1),  # a
        (20, 3),  # a|b
    ]


def test_read_hash_mask_pairs_raises_on_truncated_record(tmp_path: Path):
    p = tmp_path / "bad.attr"
    rec_size = struct.calcsize("<QQ")

    # Write one complete record + one truncated record
    with open(p, "wb") as f:
        f.write(struct.pack("<QQ", 1, 2))
        f.write(b"\x00" * (rec_size - 3))

    with pytest.raises(ValueError, match="Truncated record"):
        list(AttributionIndex.read_hash_mask_pairs(str(p)))


def test_write_and_load_legend_roundtrip(tmp_path: Path):
    idx = AttributionIndex()
    idx.add_hash_for_source(1, "a")
    idx.add_hash_for_source(2, "b")
    totals = idx.per_source_totals()

    base = str(tmp_path / "run1")
    legend_path = idx.write_legend(base, per_source_total=totals)
    assert Path(legend_path).exists()

    legend = AttributionIndex.load_legend(legend_path)
    assert isinstance(legend, AttributionLegend)
    assert legend.id_to_name == ["a", "b"]
    assert legend.name_to_id == {"a": 0, "b": 1}
    assert legend.per_source_total == totals


def test_load_legend_rejects_invalid_schema(tmp_path: Path):
    p = tmp_path / "bad.sources.json"

    # name_to_id should map to ints; put a string to trigger validation error.
    p.write_text(json.dumps({"name_to_id": {"a": "nope"}, "id_to_name": ["a"]}))

    with pytest.raises(Exception):
        _ = AttributionIndex.load_legend(str(p))


def test_attribution_rejects_more_than_64_sources():
    idx = AttributionIndex()
    for i in range(64):
        idx.get_source_id(f"s{i}")

    with pytest.raises(ValueError, match="at most 64 sources"):
        idx.get_source_id("overflow")
