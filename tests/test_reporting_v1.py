from __future__ import annotations

import json
import struct
from pathlib import Path

from corpus_assay.attribution import AttributionIndex
from corpus_assay.reporting import (
    _build_leak_by_benchmark,
    _collect_hits,
    generate_report_from_scan_dir,
)
from corpus_assay.scanner.layout import OutputLayout


def _write_hits(path: Path, records: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for h, mask in records:
            f.write(struct.pack("<QQ", h, mask))


def test_generate_report_from_scan_dir(tmp_path: Path) -> None:
    index_path = tmp_path / "index.native"
    index_path.write_bytes(b"")

    attribution = AttributionIndex()
    attribution.add_hash_for_source(1, "GSM8K")
    attribution.add_hash_for_source(2, "MMLU")
    attribution.add_hash_for_source(3, "GSM8K")
    attribution.add_hash_for_source(3, "MMLU")
    attribution.write_sidecar(str(index_path))
    attribution.write_legend(str(index_path), attribution.per_source_totals())

    meta = {"ngram": 13, "benchmarks": []}
    (tmp_path / "index.native.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    scan_dir = tmp_path / "scan_results"
    scan_dir.mkdir()

    run_manifest = {
        "config": {
            "inputs": ["sample.parquet"],
            "text_key": "text",
            "id_key": None,
            "index_path": str(index_path),
            "out_dir": str(scan_dir),
            "ngram": 13,
            "min_hits": 3,
            "min_coverage": 0.001,
        }
    }
    (scan_dir / "_RUN_MANIFEST.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )

    hits_dir = scan_dir / "per_source_hits"
    _write_hits(
        hits_dir / "shard-a.hits.bin",
        [(1, 1), (3, 1 | 2)],
    )
    _write_hits(
        hits_dir / "shard-b.hits.bin",
        [(2, 2)],
    )

    summaries_dir = scan_dir / "summaries"
    summaries_dir.mkdir()
    (summaries_dir / "shard-a.summary.json").write_text(
        json.dumps(
            {
                "file": "shard-a",
                "scanned": 100,
                "contaminated": 10,
                "contam_rate": 0.1,
            }
        ),
        encoding="utf-8",
    )
    (summaries_dir / "shard-b.summary.json").write_text(
        json.dumps(
            {
                "file": "shard-b",
                "scanned": 50,
                "contaminated": 5,
                "contam_rate": 0.1,
            }
        ),
        encoding="utf-8",
    )

    final_summary = {
        "total_files_scanned": 2,
        "total_docs_scanned": 150,
        "total_contaminated": 15,
        "contamination_rate": 0.1,
        "failed_files": 0,
        "total_files": 2,
    }
    (scan_dir / "_FINAL_SUMMARY.json").write_text(
        json.dumps(final_summary), encoding="utf-8"
    )

    samples_dir = scan_dir / "contaminated_docs"
    samples_dir.mkdir()
    (samples_dir / "shard-a.decontam.sample.ndjson").write_text(
        "{}\n", encoding="utf-8"
    )

    report = generate_report_from_scan_dir(str(scan_dir))

    report_dir = scan_dir / "report"
    assert (report_dir / "report.json").exists()
    assert (report_dir / "report.md").exists()
    assert (report_dir / "leak_by_benchmark.csv").exists()
    assert (report_dir / "leak_by_shard.csv").exists()

    leak_map = {row.benchmark: row for row in report.leak_by_benchmark}
    assert leak_map["GSM8K"].leaked_hashes == 2
    assert leak_map["MMLU"].leaked_hashes == 2
    # Exact bitmap count: fraction is leaked / total_hashes_in_index, no estimator error.
    gsm8k = leak_map["GSM8K"]
    assert gsm8k.leak_fraction == gsm8k.leaked_hashes / gsm8k.total_hashes_in_index

    top_shard = report.bad_shards.top_overall[0]
    assert top_shard.shard == "shard-a"
    assert report.examples.example_files == [
        "contaminated_docs/shard-a.decontam.sample.ndjson"
    ]


def test_leak_fraction_is_exact_and_merges_across_shards(tmp_path: Path) -> None:
    """The index-position bitmap counts distinct matched hashes exactly, and the same
    hash appearing in two shards is OR-merged to a single distinct match."""
    scan_dir = tmp_path / "scan_results"
    layout = OutputLayout(scan_dir)

    # Index universe: four hashes, all owned by source 0 (denominator = 4).
    hash_to_pos = {10: 0, 20: 1, 30: 2, 40: 3}

    # Hash 20 is reported by BOTH shards; the distinct matched set is {10, 20, 30}.
    # Hash 99 is NOT in the index universe and must be ignored (defensive guard).
    _write_hits(layout.hits_dir / "shard-a.hits.bin", [(10, 1), (20, 1)])
    _write_hits(layout.hits_dir / "shard-b.hits.bin", [(20, 1), (30, 1), (99, 1)])

    _shard_rows, bitmap_by_source, _leaked = _collect_hits(
        layout, scan_dir=scan_dir, hash_to_pos=hash_to_pos
    )
    rows = _build_leak_by_benchmark(
        bitmap_by_source=bitmap_by_source,
        total_per_src={0: 4},
        id_to_name=["GSM8K"],
    )

    row = {r.benchmark: r for r in rows}["GSM8K"]
    # 3 distinct hashes matched (20 counted once), not 4 raw hits; exact fraction 3/4.
    assert row.leaked_hashes == 3
    assert row.total_hashes_in_index == 4
    assert row.leak_fraction == 3 / 4


def _scan_scaffold(tmp_path: Path, index_path: Path, extra_config: dict) -> Path:
    """Write the minimal scan-dir scaffold (manifest, summaries, final summary,
    empty sample) that generate_report_from_scan_dir requires. Returns scan_dir."""
    (tmp_path / "index.native.meta.json").write_text(
        json.dumps({"ngram": 13, "benchmarks": []}), encoding="utf-8"
    )
    scan_dir = tmp_path / "scan_results"
    scan_dir.mkdir()
    config = {
        "inputs": ["sample.parquet"],
        "text_key": "text",
        "id_key": None,
        "index_path": str(index_path),
        "out_dir": str(scan_dir),
        "ngram": 13,
        "min_hits": 3,
        "min_coverage": 0.001,
        **extra_config,
    }
    (scan_dir / "_RUN_MANIFEST.json").write_text(
        json.dumps({"config": config}), encoding="utf-8"
    )
    summaries_dir = scan_dir / "summaries"
    summaries_dir.mkdir()
    (summaries_dir / "shard-a.summary.json").write_text(
        json.dumps(
            {"file": "shard-a", "scanned": 100, "contaminated": 10, "contam_rate": 0.1}
        ),
        encoding="utf-8",
    )
    (scan_dir / "_FINAL_SUMMARY.json").write_text(
        json.dumps(
            {
                "total_files_scanned": 1,
                "total_docs_scanned": 100,
                "total_contaminated": 10,
                "contamination_rate": 0.1,
                "failed_files": 0,
                "total_files": 1,
            }
        ),
        encoding="utf-8",
    )
    samples_dir = scan_dir / "contaminated_docs"
    samples_dir.mkdir()
    # Deliberately tiny/empty sample: the leak_by_item rollup must be exhaustive
    # (from hits.bin), independent of this contaminated-docs sample.
    (samples_dir / "shard-a.decontam.sample.ndjson").write_text(
        "{}\n", encoding="utf-8"
    )
    return scan_dir


def test_leak_by_item_exact_split(tmp_path: Path) -> None:
    """Exhaustive per-item leak: exact leaked/protected per item, with each leaked
    hash bucketed unique/shared/boilerplate by its global item fan-out."""
    index_path = tmp_path / "index.native"
    index_path.write_bytes(b"")

    attribution = AttributionIndex()
    i0 = attribution.register_item("GSM8K")
    i1 = attribution.register_item("GSM8K")
    i2 = attribution.register_item("GSM8K")
    # hash -> containing items (fan-out drives the split): 10 unique to i0, 20 shared
    # by i0/i1 (fan-out 2), 30 across i0/i1/i2 (fan-out 3 -> boilerplate at t=3),
    # 40 unique to i2.
    membership = {10: [i0], 20: [i0, i1], 30: [i0, i1, i2], 40: [i2]}
    for h, items in membership.items():
        attribution.add_hash_for_source(h, "GSM8K")
        for it in items:
            attribution.add_hash_for_item(h, it)
    attribution.write_sidecar(str(index_path))
    attribution.write_legend(str(index_path), attribution.per_source_totals())
    attribution.write_items_sidecar(str(index_path), ngram=13)

    scan_dir = _scan_scaffold(
        tmp_path, index_path, extra_config={"boilerplate_threshold": 3}
    )
    # Leaked set = {10, 20, 30} (hash 40 never leaked). Same hash in two shards is
    # de-duplicated by the exhaustive leaked-hash set.
    _write_hits(scan_dir / "per_source_hits" / "shard-a.hits.bin", [(10, 1), (20, 1)])
    _write_hits(scan_dir / "per_source_hits" / "shard-b.hits.bin", [(20, 1), (30, 1)])

    report = generate_report_from_scan_dir(str(scan_dir))

    assert (scan_dir / "report" / "leak_by_item.csv").exists()
    assert (scan_dir / "report" / "leak_by_item.json").exists()

    rows = {r.item_idx: r for r in report.leak_by_item}
    # Only items with >=1 leaked hash are emitted; item 2 leaks only via hash 30.
    assert set(rows) == {i0, i1, i2}

    r0 = rows[i0]
    assert (r0.leaked_hashes, r0.n_protected_hashes) == (3, 3)
    assert r0.leak_fraction == 1.0
    assert (r0.leaked_unique, r0.leaked_shared, r0.leaked_boilerplate) == (1, 1, 1)

    r1 = rows[i1]
    assert (r1.leaked_hashes, r1.n_protected_hashes) == (2, 2)
    assert (r1.leaked_unique, r1.leaked_shared, r1.leaked_boilerplate) == (0, 1, 1)

    r2 = rows[i2]
    # item 2 has 2 protected hashes (30, 40) but only 30 leaked.
    assert (r2.leaked_hashes, r2.n_protected_hashes) == (1, 2)
    assert r2.leak_fraction == 0.5
    assert (r2.leaked_unique, r2.leaked_shared, r2.leaked_boilerplate) == (0, 0, 1)


def test_leak_by_item_absent_without_items_sidecar(tmp_path: Path) -> None:
    """Indexes without an `.items` sidecar simply omit the per-item rollup."""
    index_path = tmp_path / "index.native"
    index_path.write_bytes(b"")
    attribution = AttributionIndex()
    attribution.add_hash_for_source(1, "GSM8K")
    attribution.write_sidecar(str(index_path))
    attribution.write_legend(str(index_path), attribution.per_source_totals())

    scan_dir = _scan_scaffold(tmp_path, index_path, extra_config={})
    _write_hits(scan_dir / "per_source_hits" / "shard-a.hits.bin", [(1, 1)])

    report = generate_report_from_scan_dir(str(scan_dir))
    assert report.leak_by_item == []
