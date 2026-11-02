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

# Three protected items: long, content-rich, disjoint vocabularies. The first two
# have n-grams in both fields (question and answer).
ITEMS = [
    {
        "question": (
            "Which enzyme unwinds the double helix ahead of the replication fork "
            "while topoisomerase relieves torsional strain downstream and primase "
            "lays short primers for polymerase extension"
        ),
        "answer": (
            "helicase separates complementary strands using energy from atp "
            "hydrolysis while single strand binding proteins stabilise the exposed "
            "template bases"
        ),
    },
    {
        "question": (
            "A cyclist accelerates uniformly from rest reaching eighteen metres per "
            "second after twelve seconds compute the displacement covered during "
            "that interval assuming negligible drag forces"
        ),
        "answer": (
            "one hundred eight metres follows from the average velocity of nine "
            "metres per second multiplied by the twelve second duration of motion"
        ),
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

    # Shard a holds item 0's question and answer, shard b item 1's question only.
    hits = {r["doc_id"]: r["item_hits"][0] for r in _records(out) if r.get("item_hits")}
    assert hits["p9-part-0"]["fields_matched"] == ["question", "answer"]
    assert hits["j1-part-0"]["fields_matched"] == ["question"]

    _cli("inspect-run", "--scan-dir", str(out), env=env)
    _cli("report", "--scan-dir", str(out), env=env)
    with (out / "report" / "leak_by_benchmark.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any("TinyQA" in json.dumps(row) for row in rows)
    report = json.loads((out / "report" / "report.json").read_text("utf-8"))
    assert report["field_matches"] == {
        "records": 2,
        "item_matches": 2,
        "full": 1,
        "question_only": 1,
        "partial": 0,
        "single_field": 0,
    }
    assert "| 1 | 1 | 0 | 0 |" in (out / "report" / "report.md").read_text("utf-8")


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


def _benchmark_config(tmp_path: Path, name: str, benchmarks: list[dict]) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump({"benchmarks": benchmarks, "ngram": 13, "min_hits": 3}),
        encoding="utf-8",
    )
    return path


def test_build_index_fails_on_a_benchmark_that_loads_nothing(
    tmp_path: Path, config: Path, env: dict[str, str]
) -> None:
    good = yaml.safe_load(config.read_text(encoding="utf-8"))["benchmarks"][0]
    missing = {
        "name": "Gone",
        "dataset": "json",
        "data_files": {"test": (tmp_path / "missing.jsonl").as_posix()},
        "splits": ["test"],
        "fields": ["question"],
    }
    cfg = _benchmark_config(tmp_path, "missing", [good, missing])
    out = tmp_path / "idx" / "index.native"
    args = ["build-index", "--config", str(cfg), "--out-index", str(out)]

    proc = _cli(*args, "--allow-unpinned", env=env, ok=False)
    assert proc.returncode == 1
    assert "Gone (json): loaded no rows" in proc.stderr
    assert "--allow-empty-benchmarks" in proc.stderr
    assert not list(out.parent.glob("index.native*")), "nothing must be written"

    proc = _cli(*args, "--allow-unpinned", "--allow-empty-benchmarks", env=env)
    assert out.exists()
    assert "WARNING (--allow-empty-benchmarks): Gone (json)" in proc.stdout
    meta = json.loads(out.with_name(out.name + ".meta.json").read_text("utf-8"))
    assert meta["schema_version"] == 2
    stats = {s["name"]: s for s in meta["benchmark_stats"]}
    assert stats["Gone"]["items"] == 0
    assert stats["TinyQA"]["items"] == 3
    assert stats["TinyQA"]["ngrams"] > 0


def test_build_index_fails_on_a_benchmark_with_no_ngrams(
    tmp_path: Path, env: dict[str, str]
) -> None:
    items = tmp_path / "short.jsonl"
    items.write_text(json.dumps({"q": "two words"}) + "\n", encoding="utf-8")
    cfg = _benchmark_config(
        tmp_path,
        "short",
        [
            {
                "name": "Tiny",
                "dataset": "json",
                "data_files": {"test": items.as_posix()},
                "splits": ["test"],
                "fields": ["q"],
            }
        ],
    )
    out = tmp_path / "index.native"
    proc = _cli(
        "build-index", "--config", str(cfg), "--out-index", str(out),
        "--allow-unpinned",
        env=env, ok=False,
    )  # fmt: skip
    assert proc.returncode == 1
    assert "Tiny (json): loaded 1 items but indexed no n-grams" in proc.stderr
    assert not out.exists()


def test_inspect_index_lists_benchmark_counts(
    tmp_path: Path, config: Path, index: Path, env: dict[str, str]
) -> None:
    proc = _cli("inspect-index", "--index", str(index), env=env)
    assert "meta schema_version: 2" in proc.stdout
    row = next(
        line.split()
        for line in proc.stdout.splitlines()
        if line.strip().startswith("TinyQA")
    )
    # benchmark, items, n-grams, 0 n-grams, < min_hits
    assert row[1] == "3" and int(row[2].replace(",", "")) > 0
    assert row[3:] == ["0", "0"]
    assert "unflaggable items: 0 with 0 n-grams" in proc.stdout


def test_build_index_warns_about_hidden_labels(
    tmp_path: Path, env: dict[str, str]
) -> None:
    items = tmp_path / "hidden.jsonl"
    rows = [{"question": f"{ITEMS[i]['question']}", "label": -1} for i in range(2)]
    items.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = _benchmark_config(
        tmp_path,
        "hidden",
        [
            {
                "name": "Hidden",
                "dataset": "json",
                "data_files": {"test": items.as_posix()},
                "splits": ["test"],
                "fields": ["question"],
                "label_field": "label",
            }
        ],
    )
    out = tmp_path / "index.native"
    proc = _cli(
        "build-index", "--config", str(cfg), "--out-index", str(out),
        "--allow-unpinned",
        env=env,
    )  # fmt: skip
    assert "every 'label' value in split 'test' is missing, empty or -1" in (
        proc.stdout
    )
    assert out.exists()


def _records(out_dir: Path) -> list[dict]:
    lines = [
        line
        for path in sorted((out_dir / "contaminated_docs").glob("*.ndjson"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return sorted((json.loads(line) for line in lines), key=lambda r: r["doc_id"])


def test_scan_preference_and_chat_fields(
    tmp_path: Path, config: Path, index: Path, env: dict[str, str]
) -> None:
    leak = f"{ITEMS[0]['question']} {ITEMS[0]['answer']}"
    shards = tmp_path / "shards"
    shards.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3],
                "prompt": [FILLER, leak, FILLER],
                "chosen": [leak, FILLER, FILLER],
                "rejected": [FILLER, FILLER, f"{FILLER} {leak}"],
                "source": ["a", "b", "c"],
            }
        ),
        shards / "pref.parquet",
    )
    chat = [
        {"id": "c1", "messages": [
            {"role": "user", "content": FILLER},
            {"role": "assistant", "content": f"{FILLER}<|endoftext|>{leak}"},
        ]},
        {"id": "c2", "messages": [{"role": "user", "content": FILLER}]},
    ]  # fmt: skip
    (shards / "chat.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in chat), encoding="utf-8"
    )
    common = [
        "--index", str(index), "--config", str(config), "--id-key", "id",
        "--workers", "1", "--all-records",
    ]  # fmt: skip

    pref_out = tmp_path / "pref_out"
    proc = _cli(
        "scan", "--input", str(shards / "pref.parquet"), "--out-dir", str(pref_out),
        "--text-key", "prompt", "--text-key", "chosen", "--text-key", "rejected",
        *common, env=env,
    )  # fmt: skip
    assert "not scanning string column(s) source" in proc.stderr
    assert [(r["doc_id"], r["field"]) for r in _records(pref_out)] == [
        ("1/chosen-part-0", "chosen"),
        ("2/prompt-part-0", "prompt"),
        ("3/rejected-part-0", "rejected"),
    ]
    manifest = json.loads((pref_out / "_RUN_MANIFEST.json").read_text("utf-8"))
    assert manifest["scan_config"]["text_keys"] == ["prompt", "chosen", "rejected"]

    for no_split, expected_id in [
        (False, "c1/messages/1-part-1"),
        (True, "c1/messages/1"),
    ]:
        chat_out = tmp_path / f"chat_out_{no_split}"
        _cli(
            "scan", "--input", str(shards / "chat.jsonl"), "--out-dir", str(chat_out),
            "--messages-key", "messages", *(["--no-split"] if no_split else []),
            *common, env=env,
        )  # fmt: skip
        records = _records(chat_out)
        assert [(r["doc_id"], r["field"], r["role"], r["turn"]) for r in records] == [
            (expected_id, "messages", "assistant", 1)
        ]
        _cli("inspect-run", "--scan-dir", str(chat_out), env=env)
        _cli("report", "--scan-dir", str(chat_out), env=env)
