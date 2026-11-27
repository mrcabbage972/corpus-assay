"""Split Parquet shards into row-group ranges for the worker pool, and merge back.

A scan runs one task per shard unless that leaves workers idle. Then a Parquet
shard with several row groups is cut into contiguous row-group ranges of about
``rows_per_task`` rows, each scanned by its own task into ``_parts/``. When every
range of a shard is done, ``merge_part_outputs`` writes the shard's usual outputs
(summary, hits, item best, record sample), the same as a scan of the whole shard,
and removes its parts. Readers of a run (``report``, ``filter``, ``inspect-run``)
never see the parts.
"""

from __future__ import annotations

import logging
import math
import shutil
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_assay import formats
from corpus_assay.attribution import AttributionIndex
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.io import (
    maybe_write_contaminated_docs,
    read_item_best,
    try_read_existing_summary,
    write_summary,
)
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.schema import ItemBest, ItemBestFile, ScanSummary
from corpus_assay.scanner.worker import build_scan_summary

logger = logging.getLogger(__name__)

# With rows_per_task unset, split shards when there are fewer than this many
# tasks per worker, aiming at this many: a few tasks per worker keeps every
# worker busy to the end when tasks differ in size.
AUTO_TASKS_PER_WORKER = 4


@dataclass(frozen=True)
class ScanTask:
    """One unit of work: a whole shard, or a row-group range ``[start, end)``."""

    file: Path
    row_groups: tuple[int, int] | None = None


@dataclass(frozen=True)
class ScanPlan:
    tasks: list[ScanTask]
    # Row-group ranges of each split shard, in order.
    parts: dict[Path, list[tuple[int, int]]]
    # Shards whose summary from an earlier run is still valid (only looked up
    # when splitting is considered; otherwise each task checks its own).
    done: list[ScanSummary]


def split_row_groups(rows: Sequence[int], target_rows: int) -> list[tuple[int, int]]:
    """Cut row groups into contiguous ranges of at least ``target_rows`` rows.

    A last range under half the target joins the one before it.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    acc = 0
    for i, n in enumerate(rows):
        acc += n
        if acc >= target_rows:
            ranges.append((start, i + 1))
            start, acc = i + 1, 0
    if start < len(rows):
        if ranges and acc < target_rows / 2:
            ranges[-1] = (ranges[-1][0], len(rows))
        else:
            ranges.append((start, len(rows)))
    return ranges


def _row_group_rows(path: Path) -> list[int] | None:
    """Rows per row group of a Parquet shard; None when unreadable (the scan
    of the whole shard then reports the error)."""
    import pyarrow.parquet as pq

    try:
        meta = pq.read_metadata(path)
    except Exception:  # noqa: BLE001 - any unreadable footer: scan it whole
        return None
    return [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]


def plan_scan_tasks(
    files: list[Path], cfg: ScanConfig, run_fingerprint: dict[str, Any]
) -> ScanPlan:
    """Decide which shards to split and into which row-group ranges."""
    whole = ScanPlan(tasks=[ScanTask(f) for f in files], parts={}, done=[])
    rows_per_task = cfg.rows_per_task
    target_tasks = cfg.workers * AUTO_TASKS_PER_WORKER
    if rows_per_task == 0:
        return whole
    if rows_per_task is None and (cfg.workers <= 1 or len(files) >= target_tasks):
        return whole

    layout = OutputLayout(Path(cfg.out_dir))
    row_counts: dict[Path, list[int]] = {}
    done: list[ScanSummary] = []
    finished: set[Path] = set()
    for f in files:
        if not f.name.endswith(".parquet"):
            continue
        # A shard already scanned under this fingerprint needs no task.
        summary = try_read_existing_summary(
            layout.summary_path(layout.stem_from_path(f)), run_fingerprint
        )
        if summary is not None:
            done.append(summary)
            finished.add(f)
            continue
        rows = _row_group_rows(f)
        if rows is not None:
            row_counts[f] = rows

    if rows_per_task is None:
        total = sum(sum(rows) for rows in row_counts.values())
        rows_per_task = max(1, math.ceil(total / target_tasks))

    tasks: list[ScanTask] = []
    parts: dict[Path, list[tuple[int, int]]] = {}
    unsplittable: list[str] = []
    for f in files:
        if f in finished:
            continue
        rows = row_counts.get(f)
        ranges = split_row_groups(rows, rows_per_task) if rows else []
        if len(ranges) > 1:
            parts[f] = ranges
            tasks.extend(ScanTask(f, r) for r in ranges)
            continue
        if rows and len(rows) == 1 and rows[0] >= 2 * rows_per_task:
            unsplittable.append(f.name)
        tasks.append(ScanTask(f))

    if unsplittable:
        shown = ", ".join(unsplittable[:5])
        more = f" and {len(unsplittable) - 5} more" if len(unsplittable) > 5 else ""
        logger.warning(
            "%d Parquet shard(s) hold a single row group and are scanned by one "
            "worker each: %s%s. Rewrite them with smaller row groups (pyarrow "
            "write_table(..., row_group_size=...)) to split them across workers.",
            len(unsplittable),
            shown,
            more,
        )
    jsonl = sum(1 for f in files if not f.name.endswith(".parquet"))
    if jsonl and (parts or cfg.rows_per_task):
        logger.info(
            "%d JSONL shard(s) are scanned one task each; convert them to Parquet "
            "to split them across workers.",
            jsonl,
        )
    return ScanPlan(tasks=tasks, parts=parts, done=done)


def _write_hits(path: Path, pairs: dict[int, int], ngram: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        formats.write_header(
            f,
            file_type=formats.FileType.HITS_PAIRS,
            ngram_n=ngram,
            record_count=len(pairs),
        )
        for h in sorted(pairs):
            f.write(struct.pack("<QQ", h, pairs[h]))


def merge_part_outputs(
    file_path: Path,
    ranges: list[tuple[int, int]],
    summaries: dict[tuple[int, int], ScanSummary],
    cfg: ScanConfig,
    run_fingerprint: dict[str, Any],
) -> ScanSummary:
    """Combine the outputs of a shard's row-group ranges into the shard's own.

    The result matches a scan of the whole shard: counts add up, hit pairs are
    unioned, each item keeps its best sub-document (the earliest on ties), and
    the record sample is the first ``max_return_records`` records in row order.
    When a range failed, nothing is written and its parts stay for a resume.
    """
    errors = [
        f"row groups {start}-{end}: {summaries[(start, end)].error}"
        for start, end in ranges
        if summaries[(start, end)].error is not None
    ]
    if errors:
        return ScanSummary(file=file_path.name, error="; ".join(errors))

    layout = OutputLayout(Path(cfg.out_dir))
    stem = layout.stem_from_path(file_path)
    layout.ensure_dirs()
    part_layouts = [layout.part_layout(stem, r) for r in ranges]

    pairs: dict[int, int] = {}
    best: dict[int, ItemBest] = {}
    any_item_best = False
    records: list[str] = []
    cap = cfg.max_return_records
    channels: dict[str, int] = {}
    for part in part_layouts:
        hits_path = part.per_hits_path(stem)
        if hits_path.exists():
            for h, mask in AttributionIndex.read_hash_mask_pairs(
                str(hits_path), expected_file_type=AttributionIndex.HITS_FILE_TYPE
            ):
                pairs[h] = pairs.get(h, 0) | mask
        best_path = part.per_item_best_path(stem)
        if best_path.exists():
            any_item_best = True
            for item in read_item_best(best_path).items:
                current = best.get(item.item_id)
                if current is None or item.hits > current.hits:
                    best[item.item_id] = item
        docs_path = part.contaminated_docs_path(stem)
        if docs_path.exists() and (cap is None or len(records) < cap):
            with docs_path.open(encoding="utf-8") as f:
                records.extend(line.rstrip("\n") for line in f if line.strip())
    if cap is not None:
        records = records[:cap]

    part_summaries = [summaries[r] for r in ranges]
    for s in part_summaries:
        for channel, count in s.contaminated_by_channel.items():
            channels[channel] = channels.get(channel, 0) + count
    scanned = sum(s.scanned for s in part_summaries)
    contaminated = sum(s.contaminated for s in part_summaries)
    empty_after_filter = sum(s.empty_after_filter for s in part_summaries)

    hits_path = layout.per_hits_path(stem)
    _write_hits(hits_path, pairs, cfg.ngram)
    best_path = layout.per_item_best_path(stem)
    if any_item_best:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(
            ItemBestFile(items=[best[i] for i in sorted(best)]).model_dump_json(),
            encoding="utf-8",
        )
    else:
        best_path.unlink(missing_ok=True)
    maybe_write_contaminated_docs(
        layout.contaminated_docs_path(stem), contaminated, records
    )

    summary = build_scan_summary(
        file_path.name,
        scanned,
        contaminated,
        empty_after_filter,
        hits_path,
        run_fingerprint,
        channel_counts=channels,
    )
    write_summary(layout.summary_path(stem), summary)

    shutil.rmtree(layout.parts_dir / stem, ignore_errors=True)
    try:
        layout.parts_dir.rmdir()  # only once every split shard is merged
    except OSError:
        pass
    return summary
