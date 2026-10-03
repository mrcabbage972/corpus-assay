from __future__ import annotations

import glob
import sys
from functools import partial
from multiprocessing import get_context
from pathlib import Path

from corpus_assay.config import ScanConfig
from corpus_assay.normalization import (
    active_ngram_config_path,
    set_ngram_config_path,
)
from corpus_assay.scanner.fingerprint import build_run_fingerprint
from corpus_assay.scanner.schema import RunSummary, ScanSummary
from corpus_assay.scanner.worker import scan_one_file_worker

SUPPORTED_SCAN_EXTENSIONS = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst")


def collect_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    missing_inputs: list[str] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            matches = [
                Path(match)
                for match in glob.glob(str(path / "**" / "*"), recursive=True)
            ]
            files.extend(match for match in matches if match.is_file())
            continue

        if path.is_file():
            files.append(path)
            continue

        matches = [
            Path(match)
            for match in glob.glob(item, recursive=True)
            if Path(match).is_file()
        ]
        if matches:
            files.extend(matches)
        else:
            missing_inputs.append(item)

    unique_files = sorted({file.resolve() for file in files}, key=str)
    filtered_files = [file for file in unique_files if _is_supported_scan_file(file)]
    if not filtered_files:
        hint = ", ".join(missing_inputs) if missing_inputs else ", ".join(inputs)
        raise FileNotFoundError(
            f"No shard files matched supported extensions {SUPPORTED_SCAN_EXTENSIONS}: {hint}"
        )
    return filtered_files


def _is_supported_scan_file(path: Path) -> bool:
    return path.name.endswith(SUPPORTED_SCAN_EXTENSIONS)


def do_scan(cfg: ScanConfig, *, print_progress: bool = True) -> RunSummary:
    files = collect_files(cfg.inputs)

    run_fingerprint = build_run_fingerprint(cfg)
    scan_func = partial(scan_one_file_worker, cfg=cfg, run_fingerprint=run_fingerprint)

    total_scanned = 0
    total_contaminated = 0
    total_empty_after_filter = 0
    per_file: list[ScanSummary] = []

    ok = 0
    failed = 0

    # Spawned workers re-import the package and see only the bundled default
    # config; re-stage the parent's active config (e.g. a --ngram-config
    # override) in each worker before it normalizes anything.
    with get_context("spawn").Pool(
        processes=cfg.workers,
        initializer=set_ngram_config_path,
        initargs=(active_ngram_config_path(),),
    ) as pool:
        for summary in pool.imap_unordered(scan_func, files):
            per_file.append(summary)

            if summary.error is None:
                ok += 1
                total_scanned += summary.scanned
                total_contaminated += summary.contaminated
                total_empty_after_filter += summary.empty_after_filter
            else:
                failed += 1

            if print_progress and ok > 0 and ok % 100 == 0:
                rate = (total_contaminated / total_scanned) if total_scanned else 0.0
                print(
                    f"[scan:PROGRESS] Files ok: {ok:,}/{len(files):,} | "
                    f"Failed: {failed:,} | "
                    f"Docs: ~{total_scanned:,} | "
                    f"Contam: {total_contaminated:,} ({rate:.4%})",
                    file=sys.stderr,
                )

    rate = (total_contaminated / total_scanned) if total_scanned else 0.0
    return RunSummary(
        total_files=len(files),
        ok_files=ok,
        failed_files=failed,
        total_docs_scanned=total_scanned,
        total_contaminated=total_contaminated,
        contamination_rate=rate,
        total_empty_after_filter=total_empty_after_filter,
        per_file=per_file,
    )
