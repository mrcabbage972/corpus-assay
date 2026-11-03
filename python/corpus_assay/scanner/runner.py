from __future__ import annotations

import glob
import multiprocessing
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from multiprocessing import get_context
from pathlib import Path

from corpus_assay.config import ScanConfig
from corpus_assay.constants import ITEMS_TEXT_SUFFIX
from corpus_assay.normalization import (
    active_ngram_config_path,
    set_ngram_config_path,
)
from corpus_assay.scanner.fingerprint import build_run_fingerprint
from corpus_assay.scanner.schema import RunSummary, ScanSummary
from corpus_assay.scanner.worker import scan_one_file_worker

SUPPORTED_SCAN_EXTENSIONS = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst")

# Spawned workers re-import the caller's main script. When that script calls
# do_scan() at module level, each worker starts a scan of its own while it is
# still starting up; multiprocessing kills it and the parent's Pool respawns it
# forever, so the run hangs. The worker writes this message to the file named by
# _SPAWN_GUARD_ENV (set by the parent around the pool) and the parent raises it.
_SPAWN_GUARD_ENV = "CORPUS_ASSAY_SPAWN_GUARD_FILE"
_POOL_POLL_SECONDS = 0.5
MISSING_MAIN_GUARD_MESSAGE = (
    "corpus_assay do_scan() was called while a scan worker process was starting. "
    "The scanner runs its workers with the 'spawn' start method, which re-imports "
    "the main script in every worker, so a script that calls do_scan() at module "
    "level runs its setup and the scan again in each worker. Put the code that "
    'starts the scan under an `if __name__ == "__main__":` guard.'
)


def _in_spawn_bootstrap() -> bool:
    """True while a spawned child is still importing the parent's main module.

    This is the state multiprocessing itself checks before refusing to start a
    process ("An attempt has been made to start a new process before the current
    process has finished its bootstrapping phase").
    """
    return bool(getattr(multiprocessing.current_process(), "_inheriting", False))


def _refuse_scan_during_worker_bootstrap() -> None:
    if not _in_spawn_bootstrap():
        return
    marker = os.environ.get(_SPAWN_GUARD_ENV)
    if marker:
        try:
            Path(marker).write_text(MISSING_MAIN_GUARD_MESSAGE, encoding="utf-8")
        except OSError:
            pass
    raise RuntimeError(MISSING_MAIN_GUARD_MESSAGE)


@contextmanager
def _spawn_guard_marker() -> Iterator[Path]:
    """Expose a marker file to the spawned workers through the environment."""
    previous = os.environ.get(_SPAWN_GUARD_ENV)
    with tempfile.TemporaryDirectory(prefix="corpus-assay-") as tmp:
        marker = Path(tmp) / "unguarded-main"
        os.environ[_SPAWN_GUARD_ENV] = str(marker)
        try:
            yield marker
        finally:
            if previous is None:
                os.environ.pop(_SPAWN_GUARD_ENV, None)
            else:
                os.environ[_SPAWN_GUARD_ENV] = previous


def _iter_pool_results(
    results: Iterator[ScanSummary], marker: Path
) -> Iterator[ScanSummary]:
    """Yield pool results, raising if a worker reported an unguarded main script."""
    next_result = getattr(results, "next", None)  # multiprocessing IMapIterator
    if next_result is None:
        yield from results
        return
    while True:
        try:
            yield next_result(timeout=_POOL_POLL_SECONDS)
        except StopIteration:
            return
        except multiprocessing.TimeoutError:
            if marker.exists():
                raise RuntimeError(MISSING_MAIN_GUARD_MESSAGE) from None


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


def check_item_text_sidecar(cfg: ScanConfig) -> None:
    """Raise ``ValueError`` when the scan needs the item text sidecar but the
    index has none (it was built before the sidecar existed)."""
    features = []
    if cfg.exact_channel != "off":
        features.append(f"--exact-channel {cfg.exact_channel}")
    if cfg.confirm:
        features.append("--confirm")
    if not features:
        return
    index_path = Path(cfg.index_path)
    sidecar = index_path.with_name(f"{index_path.name}{ITEMS_TEXT_SUFFIX}")
    if not sidecar.exists():
        raise ValueError(
            f"{' and '.join(features)} {'need' if len(features) > 1 else 'needs'} "
            f"the item text sidecar {sidecar}, "
            "which this index lacks (it was built before the sidecar existed). "
            "Rebuild the index with build-index; run and run-hf rebuild their "
            "cached index."
        )


def do_scan(cfg: ScanConfig, *, print_progress: bool = True) -> RunSummary:
    """Scan every shard in ``cfg.inputs`` with a pool of ``cfg.workers`` processes.

    Workers are started with ``spawn``, so a script that calls this must do so
    under ``if __name__ == "__main__":``; otherwise a ``RuntimeError`` says so.
    """
    _refuse_scan_during_worker_bootstrap()
    check_item_text_sidecar(cfg)
    files = collect_files(cfg.inputs)

    run_fingerprint = build_run_fingerprint(cfg)
    scan_func = partial(scan_one_file_worker, cfg=cfg, run_fingerprint=run_fingerprint)

    total_scanned = 0
    total_contaminated = 0
    total_empty_after_filter = 0
    total_by_channel: dict[str, int] = {}
    per_file: list[ScanSummary] = []

    ok = 0
    failed = 0

    # Spawned workers re-import the package and see only the bundled default
    # config; re-stage the parent's active config (e.g. a --ngram-config
    # override) in each worker before it normalizes anything.
    with (
        _spawn_guard_marker() as guard_marker,
        get_context("spawn").Pool(
            processes=cfg.workers,
            initializer=set_ngram_config_path,
            initargs=(active_ngram_config_path(),),
        ) as pool,
    ):
        results = pool.imap_unordered(scan_func, files)
        for summary in _iter_pool_results(results, guard_marker):
            per_file.append(summary)

            if summary.error is None:
                ok += 1
                total_scanned += summary.scanned
                total_contaminated += summary.contaminated
                total_empty_after_filter += summary.empty_after_filter
                for channel, count in summary.contaminated_by_channel.items():
                    total_by_channel[channel] = total_by_channel.get(channel, 0) + count
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
        total_contaminated_by_channel=dict(sorted(total_by_channel.items())),
        per_file=per_file,
    )
