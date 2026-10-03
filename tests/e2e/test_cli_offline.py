"""Offline end-to-end runs of the ``corpus-assay`` CLI.

Each step runs the real CLI in a subprocess (``python -m corpus_assay``), with the
Hugging Face hub and datasets libraries forced offline. The protected benchmark is
a local JSONL file loaded through ``dataset: json`` + ``data_files``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from corpus_assay.constants import DEFAULT_NGRAM_CONFIG_PATH

pytestmark = pytest.mark.e2e

# Three protected items: long, content-rich, disjoint vocabularies.
ITEMS = [
    {
        "question": (
            "Which enzyme unwinds the double helix ahead of the replication fork "
            "while topoisomerase relieves torsional strain downstream and primase "
            "lays short primers for polymerase extension"
        ),
        "answer": "helicase separates complementary strands using hydrolysis energy",
    },
    {
        "question": (
            "A cyclist accelerates uniformly from rest reaching eighteen metres per "
            "second after twelve seconds compute the displacement covered during "
            "that interval assuming negligible drag forces"
        ),
        "answer": "one hundred eight metres follows from average velocity times duration",
    },
    {
        # Plain synthetic tokens so the spaced-seed edit below is easy to reason about.
        "question": " ".join(f"w{i}" for i in range(30)),
        "answer": "synthetic item for the spaced channel",
    },
]
FILLER = (
    "harbour weather reports mention gentle breezes over the marina while local "
    "bakeries open early and commuters queue patiently for the morning ferry"
)


def _cli(
    *args: str, env: dict[str, str], ok: bool = True
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, "-m", "corpus_assay", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    if ok:
        assert proc.returncode == 0, (
            f"{args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HOME=str(tmp_path / "hf_home"),
        HF_DATASETS_CACHE=str(tmp_path / "hf_datasets"),
    )
    return env


@pytest.fixture
def config(tmp_path: Path) -> Path:
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(i) + "\n" for i in ITEMS), encoding="utf-8")
    cfg = {
        "benchmarks": [
            {
                "name": "TinyQA",
                "dataset": "json",
                "data_files": {"test": items.as_posix()},
                "splits": ["test"],
                "fields": ["question", "answer"],
            }
        ],
        "ngram": 13,
        "min_hits": 3,
        "min_coverage": 0.001,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def index(tmp_path: Path, config: Path, env: dict[str, str]) -> Path:
    out = tmp_path / "idx" / "index.native"
    out.parent.mkdir()
    _cli(
        "build-index",
        "--config",
        str(config),
        "--out-index",
        str(out),
        "--allow-unpinned",
        env=env,
    )
    return out


def _shards(root: Path) -> Path:
    shards = root / "shards"
    shards.mkdir()
    clean = [f"{FILLER} number {i}" for i in range(9)]
    leaked = f"{FILLER} {ITEMS[0]['question']} {ITEMS[0]['answer']} {FILLER}"
    pq.write_table(
        pa.table({"text": [*clean, leaked], "id": [f"p{i}" for i in range(10)]}),
        shards / "a.parquet",
    )
    rows = [
        {"text": FILLER, "id": "j0"},
        {"text": f"{FILLER} {ITEMS[1]['question']} {FILLER}", "id": "j1"},
    ]
    (shards / "b.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return shards


def _summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "_FINAL_SUMMARY.json").read_text(encoding="utf-8"))


def test_validate_build_scan_report(
    tmp_path: Path, config: Path, index: Path, env: dict[str, str]
) -> None:
    _cli("validate-config", "--config", str(config), "--allow-unpinned", env=env)
    for suffix in (".meta.json", ".attr", ".sources.json", ".items"):
        assert index.with_name(index.name + suffix).exists(), suffix
    _cli("inspect-index", "--index", str(index), env=env)

    shards = _shards(tmp_path)
    out = tmp_path / "out"
    _cli(
        "scan", "--input", str(shards), "--index", str(index), "--out-dir", str(out),
        "--config", str(config), "--id-key", "id", "--workers", "2",
        env=env,
    )  # fmt: skip
    summary = _summary(out)
    assert summary["failed_files"] == 0
    assert summary["total_docs_scanned"] == 12
    assert summary["total_contaminated"] == 2

    _cli("inspect-run", "--scan-dir", str(out), env=env)
    _cli("report", "--scan-dir", str(out), env=env)
    with (out / "report" / "leak_by_benchmark.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any("TinyQA" in json.dumps(row) for row in rows)


def test_run_caches_index_per_ngram_config(
    tmp_path: Path, config: Path, env: dict[str, str]
) -> None:
    shards = _shards(tmp_path)
    cache = tmp_path / "cache"
    common = [
        "--config", str(config), "--input", str(shards), "--index-cache-dir", str(cache),
        "--allow-unpinned", "--workers", "2",
    ]  # fmt: skip

    _cli("run", *common, "--out-dir", str(tmp_path / "out1"), env=env)
    assert _summary(tmp_path / "out1")["total_contaminated"] == 2
    assert len(list(cache.glob("*.native"))) == 1

    # A different normalization config must build (not reuse) its own index.
    custom = json.loads(DEFAULT_NGRAM_CONFIG_PATH.read_text(encoding="utf-8"))
    custom["stop_words"] = []
    custom_path = tmp_path / "ngram_config.json"
    custom_path.write_text(json.dumps(custom), encoding="utf-8")
    _cli(
        "--ngram-config", str(custom_path), "run", *common,
        "--out-dir", str(tmp_path / "out2"),
        env=env,
    )  # fmt: skip
    assert len(list(cache.glob("*.native"))) == 2
    assert _summary(tmp_path / "out2")["total_contaminated"] == 2

    # Scanning that index without the override is refused.
    custom_index = max(cache.glob("*.native"), key=lambda p: p.stat().st_mtime)
    proc = _cli(
        "scan", "--input", str(shards), "--index", str(custom_index),
        "--out-dir", str(tmp_path / "out3"), "--config", str(config),
        env=env, ok=False,
    )  # fmt: skip
    assert proc.returncode != 0
    assert "different ngram normalization config" in proc.stdout + proc.stderr


def test_spaced_seeds_recover_an_edited_copy(
    tmp_path: Path, config: Path, index: Path, env: dict[str, str]
) -> None:
    spaced = index.with_name("index.spaced")
    _cli(
        "build-spaced-index", "--config", str(config), "--out", str(spaced),
        "--allow-unpinned",
        env=env,
    )  # fmt: skip

    # 24 tokens of the synthetic item with substitutions at positions 6 and 18:
    # every contiguous 13-gram is broken, but span-17 seeds still align and verify.
    tokens = [f"w{i}" for i in range(24)]
    tokens[6], tokens[18] = "edited", "changed"
    background = " ".join(f"z{i}" for i in range(12))
    shards = tmp_path / "spaced_shards"
    shards.mkdir()
    pq.write_table(
        pa.table({"text": [f"{background} {' '.join(tokens)} {background}"]}),
        shards / "s.parquet",
    )

    base = [
        "scan", "--input", str(shards), "--index", str(index), "--config", str(config),
        "--workers", "1",
    ]  # fmt: skip
    _cli(*base, "--out-dir", str(tmp_path / "exact"), env=env)
    _cli(
        *base,
        "--out-dir",
        str(tmp_path / "spaced"),
        "--spaced-seeds",
        str(spaced),
        env=env,
    )
    assert _summary(tmp_path / "exact")["total_contaminated"] == 0
    assert _summary(tmp_path / "spaced")["total_contaminated"] == 1


def test_stopgrams_from_local_background(
    tmp_path: Path, config: Path, index: Path, env: dict[str, str]
) -> None:
    background = tmp_path / "background.jsonl"
    background.write_text(
        "".join(json.dumps({"text": f"{FILLER} {i}"}) + "\n" for i in range(20)),
        encoding="utf-8",
    )
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["background"] = {
        "enabled": True,
        "corpus": {"local_path": background.as_posix(), "fields": ["text"]},
    }
    bg_config = tmp_path / "bg_config.yaml"
    bg_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    stopgrams = tmp_path / "stopgrams.native"
    _cli(
        "build-stopgrams-direct", "--config", str(bg_config), "--tau", "0.5",
        "--out", str(stopgrams), "--workers", "1",
        env=env,
    )  # fmt: skip
    assert stopgrams.exists()

    shards = _shards(tmp_path)
    out = tmp_path / "out"
    _cli(
        "scan", "--input", str(shards), "--index", str(index), "--out-dir", str(out),
        "--config", str(config), "--stopgrams", str(stopgrams), "--workers", "1",
        env=env,
    )  # fmt: skip
    summary = _summary(out)
    assert summary["failed_files"] == 0
    assert summary["total_contaminated"] == 2
