"""Splitting Parquet shards across workers by row group.

A shard scanned in row-group ranges must produce the same outputs as a scan of
the whole shard: summary counts, hit pairs, per-item best, and the record sample
with file-level row indexes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from corpus_assay._native import scan_stream_rust
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import _field_texts
from corpus_assay.config import ScanConfig
from corpus_assay.indexing import build_index, write_native_index_artifacts
from corpus_assay.scanner.fingerprint import build_run_fingerprint
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.runner import do_scan
from corpus_assay.scanner.schema import ScanSummary
from corpus_assay.scanner.tasks import (
    ScanTask,
    merge_part_outputs,
    plan_scan_tasks,
    split_row_groups,
)
from corpus_assay.scanner.worker import scan_one_file_worker

NGRAM = 3
ITEM = (
    "a cyclist accelerates uniformly from rest reaching eighteen metres per second "
    "after twelve seconds compute the displacement covered"
)
OTHER = "photosynthesis converts sunlight carbon dioxide water into glucose oxygen"
HALF_ITEM = " ".join(ITEM.split()[:10])
ROWS = 40
ROW_GROUP_SIZE = 5
# Rows holding benchmark text, spread over several row groups.
PLANTED = {3: ITEM, 11: HALF_ITEM, 17: ITEM, 18: HALF_ITEM, 25: OTHER, 33: ITEM}


class _Loader:
    dataset_fingerprints: list = []

    def __init__(self, spec: BenchmarkSpec, rows: list[dict]) -> None:
        self.spec, self.rows = spec, rows

    def load_positive_fields(self):
        for row in self.rows:
            yield _field_texts(row, self.spec.fields)

    def load_negative(self):
        return iter(())


@pytest.fixture
def index(tmp_path: Path) -> str:
    spec = BenchmarkSpec(name="Q", dataset="x", fields=["question"])
    rows = [{"question": ITEM}, {"question": OTHER}]
    result = build_index(
        benchmarks=[spec],
        ngram=NGRAM,
        loader_factory=lambda s: _Loader(s, rows),  # type: ignore[arg-type,return-value]
    )
    path = tmp_path / "q.native"
    write_native_index_artifacts(result, str(path))
    return str(path)


def _write_shard(path: Path, rows: int = ROWS) -> Path:
    texts = [
        PLANTED.get(i, f"clean filler sentence number {i} about nothing much")
        for i in range(rows)
    ]
    pq.write_table(
        pa.table({"text": texts, "id": [f"doc{i}" for i in range(rows)]}),
        path,
        row_group_size=ROW_GROUP_SIZE,
    )
    return path


def _cfg(index: str, shard: Path, out: Path, **kwargs: Any) -> ScanConfig:
    base: dict[str, Any] = {
        "inputs": [str(shard)],
        "id_key": "id",
        "index_path": index,
        "out_dir": str(out),
        "ngram": NGRAM,
        "min_hits": 1,
        "min_coverage": 0.0,
        "workers": 2,
        "max_return_records": None,
    }
    return ScanConfig(**{**base, **kwargs})


def _outputs(cfg: ScanConfig, shard: Path) -> dict[str, Any]:
    layout = OutputLayout(Path(cfg.out_dir))
    stem = layout.stem_from_path(shard)
    summary = json.loads(layout.summary_path(stem).read_text(encoding="utf-8"))
    summary.pop("run_fingerprint")  # names out_dir
    docs = layout.contaminated_docs_path(stem)
    return {
        "summary": summary,
        "hits": layout.per_hits_path(stem).read_bytes(),
        "item_best": json.loads(
            layout.per_item_best_path(stem).read_text(encoding="utf-8")
        ),
        "records": docs.read_text(encoding="utf-8").splitlines()
        if docs.exists()
        else [],
    }


@pytest.mark.parametrize("max_records", [None, 2])
def test_split_scan_matches_whole_scan(
    index: str, tmp_path: Path, max_records: int | None
) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    whole = _cfg(
        index,
        shard,
        tmp_path / "whole",
        rows_per_task=0,
        max_return_records=max_records,
    )
    split = _cfg(
        index,
        shard,
        tmp_path / "split",
        rows_per_task=7,
        max_return_records=max_records,
    )
    plan = plan_scan_tasks([shard], split, build_run_fingerprint(split))
    assert len(plan.tasks) > 1  # really split

    do_scan(whole, print_progress=False)
    run = do_scan(split, print_progress=False)

    assert run.ok_files == 1 and run.failed_files == 0
    assert _outputs(split, shard) == _outputs(whole, shard)
    assert not OutputLayout(Path(split.out_dir)).parts_dir.exists()


def test_split_records_keep_file_row_indexes(index: str, tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    cfg = _cfg(index, shard, tmp_path / "out", rows_per_task=7)
    do_scan(cfg, print_progress=False)

    records = [json.loads(line) for line in _outputs(cfg, shard)["records"]]
    flagged = sorted(r["row_index"] for r in records)
    assert flagged == sorted(PLANTED)
    for r in records:
        assert r["doc_id"].startswith(f"doc{r['row_index']}-")


def test_rust_reads_only_the_requested_row_groups(index: str, tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    scanned, _, records, _ = scan_stream_rust(
        input=str(shard),
        text_key="text",
        id_key="id",
        index_path=index,
        n=NGRAM,
        min_hits=1,
        min_coverage=0.0,
        max_return_records=None,
        row_groups=(3, 5),  # rows 15..24
    )
    assert scanned == 10
    assert sorted(json.loads(r)["row_index"] for r in records) == [17, 18]

    with pytest.raises(ValueError, match="out of bounds"):
        scan_stream_rust(
            input=str(shard),
            text_key="text",
            id_key="id",
            index_path=index,
            n=NGRAM,
            min_hits=1,
            min_coverage=0.0,
            row_groups=(0, 99),
        )


def test_failed_range_keeps_parts_and_resume_completes(
    index: str, tmp_path: Path
) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    whole = _cfg(index, shard, tmp_path / "whole", rows_per_task=0)
    cfg = _cfg(index, shard, tmp_path / "split", rows_per_task=7)
    fingerprint = build_run_fingerprint(cfg)
    plan = plan_scan_tasks([shard], cfg, fingerprint)
    ranges = plan.parts[shard]

    # Every range but the last is scanned; the last one failed.
    summaries = {
        r: scan_one_file_worker(shard, cfg, fingerprint, row_groups=r)
        for r in ranges[:-1]
    }
    summaries[ranges[-1]] = ScanSummary(file=shard.name, error="worker died")
    merged = merge_part_outputs(shard, ranges, summaries, cfg, fingerprint)

    layout = OutputLayout(Path(cfg.out_dir))
    stem = layout.stem_from_path(shard)
    assert merged.error is not None and "worker died" in merged.error
    assert not layout.summary_path(stem).exists()
    assert (layout.parts_dir / stem).is_dir()

    do_scan(whole, print_progress=False)
    run = do_scan(cfg, print_progress=False)
    assert run.ok_files == 1
    assert _outputs(cfg, shard) == _outputs(whole, shard)
    assert not layout.parts_dir.exists()

    # A finished shard is not planned again.
    again = plan_scan_tasks([shard], cfg, fingerprint)
    assert again.tasks == [] and len(again.done) == 1


def test_split_row_groups() -> None:
    assert split_row_groups([5] * 8, 10) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    # A small tail joins the range before it.
    assert split_row_groups([5, 5, 5, 5, 1], 10) == [(0, 2), (2, 5)]
    assert split_row_groups([5] * 5, 10) == [(0, 2), (2, 4), (4, 5)]
    assert split_row_groups([100], 10) == [(0, 1)]
    assert split_row_groups([], 10) == []


def test_auto_splits_only_when_workers_would_idle(index: str, tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    out = tmp_path / "out"

    cfg = _cfg(index, shard, out, workers=4)
    plan = plan_scan_tasks([shard], cfg, build_run_fingerprint(cfg))
    # 40 rows over 4 x 4 target tasks: 3 rows per task, i.e. one row group each.
    assert [t.row_groups for t in plan.tasks] == [(i, i + 1) for i in range(8)]

    off = _cfg(index, shard, out, workers=4, rows_per_task=0)
    assert plan_scan_tasks([shard], off, {}).tasks == [ScanTask(shard)]
    one = _cfg(index, shard, out, workers=1)
    assert plan_scan_tasks([shard], one, {}).tasks == [ScanTask(shard)]

    many = [_write_shard(tmp_path / f"s{i}.parquet") for i in range(8)]
    enough = _cfg(index, shard, out, workers=2)
    assert plan_scan_tasks(many, enough, {}).tasks == [ScanTask(f) for f in many]


def test_single_row_group_shard_is_reported(
    index: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    shard = tmp_path / "big.parquet"
    pq.write_table(pa.table({"text": ["x"] * 100, "id": ["i"] * 100}), shard)
    cfg = _cfg(index, shard, tmp_path / "out", rows_per_task=10)
    with caplog.at_level(logging.WARNING):
        plan = plan_scan_tasks([shard], cfg, {})
    assert plan.tasks == [ScanTask(shard)]
    assert "single row group" in caplog.text


def test_jsonl_shards_are_scanned_whole(index: str, tmp_path: Path) -> None:
    shard = tmp_path / "docs.jsonl"
    shard.write_text(
        "".join(json.dumps({"text": ITEM, "id": f"d{i}"}) + "\n" for i in range(20)),
        encoding="utf-8",
    )
    cfg = _cfg(index, shard, tmp_path / "out", rows_per_task=2)
    assert plan_scan_tasks([shard], cfg, {}).tasks == [ScanTask(shard)]


def test_fingerprint_ignores_parallelism_settings(index: str, tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    a = _cfg(index, shard, tmp_path / "out", workers=2, rows_per_task=None)
    b = _cfg(index, shard, tmp_path / "out", workers=16, rows_per_task=3)
    c = _cfg(index, shard, tmp_path / "out", workers=2, min_hits=2)
    assert build_run_fingerprint(a) == build_run_fingerprint(b)
    assert build_run_fingerprint(a) != build_run_fingerprint(c)
