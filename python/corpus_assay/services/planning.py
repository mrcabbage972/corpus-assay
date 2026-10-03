from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from corpus_assay.scanner.runner import collect_files
from corpus_assay.services.index_cache import native_index_required_paths


@dataclass(frozen=True, slots=True)
class ScanPlan:
    files: list[Path]
    out_dir: Path
    index: Path
    required_index_files: list[Path]
    missing_index_files: list[Path]

    @property
    def index_ok(self) -> bool:
        return not self.missing_index_files


def plan_scan(
    inputs: list[str],
    index: str | Path,
    out_dir: Path,
    *,
    check_index: bool = True,
) -> ScanPlan:
    files = collect_files(inputs)
    index_path = Path(index)

    if check_index:
        required = native_index_required_paths(index_path)
        missing = [path for path in required if not path.exists()]
    else:
        required = []
        missing = []

    return ScanPlan(
        files=files,
        out_dir=out_dir,
        index=index_path,
        required_index_files=required,
        missing_index_files=missing,
    )
