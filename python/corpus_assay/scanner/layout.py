from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# -----------------------------
# Pydantic domain models
# -----------------------------


# -----------------------------
# Output layout (paths only)
# -----------------------------


@dataclass(frozen=True, slots=True)
class OutputLayout:
    out_dir: Path

    @staticmethod
    def stem_from_path(path: Path) -> str:
        return re.sub(r"\.(parquet|jsonl|jsonl\.gz|jsonl\.zst)$", "", path.name)

    @property
    def hits_dir(self) -> Path:
        return self.out_dir / "per_source_hits"

    @property
    def summaries_dir(self) -> Path:
        return self.out_dir / "summaries"

    @property
    def contaminated_dir(self) -> Path:
        return self.out_dir / "contaminated_docs"

    def per_hits_path(self, stem: str) -> Path:
        return self.hits_dir / f"{stem}.hits.bin"

    def summary_path(self, stem: str) -> Path:
        return self.summaries_dir / f"{stem}.summary.json"

    def contaminated_docs_path(self, stem: str) -> Path:
        return self.contaminated_dir / f"{stem}.decontam.sample.ndjson"

    @property
    def final_summary_json(self) -> Path:
        return self.out_dir / "_FINAL_SUMMARY.json"

    @property
    def final_summary_csv(self) -> Path:
        return self.out_dir / "_FINAL_SUMMARY.csv"

    @property
    def run_manifest_json(self) -> Path:
        return self.out_dir / "_RUN_MANIFEST.json"

    def validate_layout(self) -> None:
        missing = []
        required_paths = [
            self.out_dir,
            self.hits_dir,
            self.summaries_dir,
            self.run_manifest_json,
        ]
        for path in required_paths:
            if not path.exists():
                missing.append(path)
        if missing:
            missing_display = "\n  ".join(str(path) for path in missing)
            raise FileNotFoundError("Missing scan output paths:\n  " + missing_display)

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.hits_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.contaminated_dir.mkdir(parents=True, exist_ok=True)
