import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from corpus_assay.benchmark_registry import REGISTRY
from corpus_assay.config import DecontaminationConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_registry_dataset.py"


def test_export_writes_every_registry_entry_and_a_card(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    table = pq.read_table(tmp_path / "data" / "registry.parquet")
    rows = table.to_pylist()
    assert [r["key"] for r in rows] == REGISTRY.list_registry()
    assert all(r["revision"] for r in rows)

    hellaswag = next(r for r in rows if r["key"] == "hellaswag")
    assert hellaswag["splits"] == ["validation"]
    assert hellaswag["scored_split"] == "validation"

    # Each row's spec_yaml is a valid corpus-assay config.
    for r in rows:
        DecontaminationConfig.model_validate(yaml.safe_load(r["spec_yaml"]))

    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(card.split("---")[1])
    assert front["configs"][0]["data_files"][0]["path"] == "data/registry.parquet"
    assert "benchmark decontamination of\nLLM pretraining data" in card
    assert card.count("\n| `") >= len(rows)
