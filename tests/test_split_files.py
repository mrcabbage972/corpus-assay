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
from corpus_assay.scanner import runner as scan_runner, worker as scan_worker
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


@pytest.fixture
def scanned_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int] | None]:
    """Observe real scans, including cache misses, with deterministic completion order.

    Only the pool is in-process: workers, native scanning and merging are real.
    Reverse completion order exercises record ordering and earliest-best ties.
    """
    scanned: list[tuple[int, int] | None] = []
    original = scan_worker.scan_file

    def observe(file_path, cfg, layout, **kwargs):
        scanned.append(kwargs.get("row_groups"))
        return original(file_path, cfg, layout, **kwargs)

    class InlinePool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def imap_unordered(self, func, tasks):
            for task in reversed(list(tasks)):
                yield func(task)

    class InlineContext:
        def Pool(self, processes, initializer=None, initargs=()):
            if initializer is not None:
                initializer(*initargs)
            return InlinePool()

    monkeypatch.setattr(scan_worker, "scan_file", observe)
    monkeypatch.setattr(scan_runner, "get_context", lambda _: InlineContext())
    return scanned


@pytest.mark.parametrize("max_records", [None, 2])
@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param({"workers": 2}, {"workers": 4}, id="more-workers"),
        pytest.param({"workers": 4}, {"workers": 2}, id="fewer-workers"),
        pytest.param({"rows_per_task": 20}, {"rows_per_task": 10}, id="smaller-tasks"),
        pytest.param({"rows_per_task": 10}, {"rows_per_task": 20}, id="larger-tasks"),
        pytest.param(
            {"rows_per_task": 10}, {"rows_per_task": 0}, id="splitting-disabled"
        ),
        pytest.param({"workers": 4}, {"workers": 1}, id="single-worker"),
        pytest.param(
            {"rows_per_task": 10}, {"rows_per_task": 1000}, id="one-large-task"
        ),
    ],
)
def test_resume_after_parallelism_change_reuses_completed_rows(
    index: str,
    tmp_path: Path,
    scanned_ranges: list[tuple[int, int] | None],
    before: dict[str, int],
    after: dict[str, int],
    max_records: int | None,
) -> None:
    # 32 row groups ensure that both auto worker counts produce different plans.
    rows = 160
    shard = _write_shard(tmp_path / "shard.parquet", rows=rows)
    cfg = _cfg(
        index, shard, tmp_path / "resume", max_return_records=max_records, **before
    )
    resumed = _cfg(
        index, shard, tmp_path / "resume", max_return_records=max_records, **after
    )
    fingerprint = build_run_fingerprint(cfg)
    assert build_run_fingerprint(resumed) == fingerprint
    ranges = plan_scan_tasks([shard], cfg, fingerprint).parts[shard]
    new_plan = plan_scan_tasks([shard], resumed, fingerprint)
    assert new_plan.tasks != [ScanTask(shard, r) for r in ranges]

    # Completed work is scattered, as it is with unordered pool results.
    completed = ranges[1::2]
    summaries = {
        r: scan_one_file_worker(shard, cfg, fingerprint, row_groups=r)
        for r in reversed(completed)
    }
    assert all(s.error is None for s in summaries.values())
    summaries.update(
        {
            r: ScanSummary(file=shard.name, error="worker died")
            for r in ranges
            if r not in summaries
        }
    )
    assert merge_part_outputs(shard, ranges, summaries, cfg, fingerprint).error
    layout = OutputLayout(Path(cfg.out_dir))
    assert not layout.summary_path(layout.stem_from_path(shard)).exists()

    whole = _cfg(
        index,
        shard,
        tmp_path / "whole",
        rows_per_task=0,
        max_return_records=max_records,
    )
    baseline = do_scan(whole, print_progress=False)
    scanned_ranges.clear()
    run = do_scan(resumed, print_progress=False)

    assert run.ok_files == 1 and run.failed_files == 0
    assert run.total_docs_scanned == baseline.total_docs_scanned == rows
    assert run.total_contaminated == baseline.total_contaminated
    assert run.total_contaminated_by_channel == baseline.total_contaminated_by_channel
    assert _outputs(resumed, shard) == _outputs(whole, shard)

    all_groups = set(range(rows // ROW_GROUP_SIZE))
    completed_groups = {i for start, end in completed for i in range(start, end)}
    actual_groups = sorted(
        i
        for bounds in scanned_ranges
        for i in (range(*bounds) if bounds is not None else sorted(all_groups))
    )
    # A worker call which reuses its summary is not a scan. Every uncovered row
    # group must actually be scanned once, and no finished row group again.
    assert (actual_groups, layout.parts_dir.exists()) == (
        sorted(all_groups - completed_groups),
        False,
    )

    scanned_ranges.clear()
    assert do_scan(resumed, print_progress=False).ok_files == 1
    assert scanned_ranges == []
    assert not layout.parts_dir.exists()


def test_resume_with_overlapping_old_parts_does_not_double_count(
    index: str,
    tmp_path: Path,
    scanned_ranges: list[tuple[int, int] | None],
) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    cfg = _cfg(index, shard, tmp_path / "resume", rows_per_task=5)
    fingerprint = build_run_fingerprint(cfg)
    # Older versions can leave overlapping completed ranges after interrupting
    # two runs that used different task sizes. Never merge overlapping totals.
    # Keeping (0, 3) greedily would throw away reusable work in (2, 4).
    for bounds in [(0, 3), (0, 2), (2, 4), (4, 6)]:
        summary = scan_one_file_worker(shard, cfg, fingerprint, row_groups=bounds)
        assert summary.error is None

    whole = _cfg(index, shard, tmp_path / "whole", rows_per_task=0)
    do_scan(whole, print_progress=False)
    scanned_ranges.clear()
    run = do_scan(cfg, print_progress=False)
    assert run.ok_files == 1 and run.total_docs_scanned == ROWS
    assert _outputs(cfg, shard) == _outputs(whole, shard)
    assert None not in scanned_ranges
    assert sorted(i for bounds in scanned_ranges if bounds for i in range(*bounds)) == [
        6,
        7,
    ]
    assert not OutputLayout(Path(cfg.out_dir)).parts_dir.exists()


@pytest.mark.parametrize("rows_per_task", [0, 5])
def test_finished_shard_cleans_leftover_parts_without_rescanning(
    index: str,
    tmp_path: Path,
    scanned_ranges: list[tuple[int, int] | None],
    rows_per_task: int,
) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    cfg = _cfg(index, shard, tmp_path / "out", rows_per_task=rows_per_task)
    fingerprint = build_run_fingerprint(cfg)
    assert scan_one_file_worker(shard, cfg, fingerprint).error is None
    expected = _outputs(cfg, shard)
    # Model a crash after the final summary was written but before cleanup.
    assert (
        scan_one_file_worker(shard, cfg, fingerprint, row_groups=(0, 2)).error is None
    )
    layout = OutputLayout(Path(cfg.out_dir))
    assert layout.parts_dir.exists()

    scanned_ranges.clear()
    assert do_scan(cfg, print_progress=False).ok_files == 1
    assert scanned_ranges == []
    assert _outputs(cfg, shard) == expected
    assert not layout.parts_dir.exists()


def test_auto_resume_reuses_parts_after_another_shard_finishes(
    index: str,
    tmp_path: Path,
    scanned_ranges: list[tuple[int, int] | None],
) -> None:
    rows = 160
    pending = _write_shard(tmp_path / "pending.parquet", rows=rows)
    finished = _write_shard(tmp_path / "finished.parquet", rows=rows)
    inputs = [str(pending), str(finished)]
    cfg = _cfg(index, pending, tmp_path / "resume", inputs=inputs, workers=2)
    fingerprint = build_run_fingerprint(cfg)
    ranges = plan_scan_tasks([pending, finished], cfg, fingerprint).parts[pending]
    completed = ranges[1::2]
    for bounds in completed:
        assert (
            scan_one_file_worker(pending, cfg, fingerprint, row_groups=bounds).error
            is None
        )
    assert scan_one_file_worker(finished, cfg, fingerprint).error is None

    # Same settings, but half the rows now have a final summary. Auto sizing
    # must not make the pending shard's valid, wider ranges disappear.
    whole = _cfg(index, pending, tmp_path / "whole", inputs=inputs, rows_per_task=0)
    do_scan(whole, print_progress=False)
    scanned_ranges.clear()
    run = do_scan(cfg, print_progress=False)
    assert run.ok_files == 2 and run.failed_files == 0
    assert run.total_docs_scanned == 2 * rows
    for shard in (pending, finished):
        assert _outputs(cfg, shard) == _outputs(whole, shard)
    expected = set(range(rows // ROW_GROUP_SIZE)) - {
        i for start, end in completed for i in range(start, end)
    }
    assert None not in scanned_ranges
    assert sorted(
        i for bounds in scanned_ranges if bounds for i in range(*bounds)
    ) == sorted(expected)
    assert not OutputLayout(Path(cfg.out_dir)).parts_dir.exists()


@pytest.mark.parametrize("rows_per_task", [0, 5])
def test_resume_does_not_reuse_parts_from_a_different_fingerprint(
    index: str,
    tmp_path: Path,
    scanned_ranges: list[tuple[int, int] | None],
    rows_per_task: int,
) -> None:
    shard = _write_shard(tmp_path / "shard.parquet")
    old = _cfg(index, shard, tmp_path / "resume", rows_per_task=10, min_hits=1000)
    fingerprint = build_run_fingerprint(old)
    ranges = plan_scan_tasks([shard], old, fingerprint).parts[shard]
    for bounds in ranges[:-1]:
        assert (
            scan_one_file_worker(shard, old, fingerprint, row_groups=bounds).error
            is None
        )

    cfg = _cfg(index, shard, tmp_path / "resume", rows_per_task=rows_per_task)
    assert build_run_fingerprint(cfg) != fingerprint
    whole = _cfg(index, shard, tmp_path / "whole", rows_per_task=0)
    do_scan(whole, print_progress=False)
    scanned_ranges.clear()
    assert do_scan(cfg, print_progress=False).ok_files == 1
    assert _outputs(cfg, shard) == _outputs(whole, shard)
    assert sorted(
        i
        for bounds in scanned_ranges
        for i in range(*(bounds if bounds is not None else (0, ROWS // ROW_GROUP_SIZE)))
    ) == list(range(ROWS // ROW_GROUP_SIZE))
    assert not OutputLayout(Path(cfg.out_dir)).parts_dir.exists()


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
