"""The ``filter`` command (#8): cleaned shards, the filter manifest, removed ids.

Runs the CLI in-process against a tiny 3-gram index with the real Rust scanner.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

import corpus_assay.cli as cli
from _index_helpers import write_native_index
from corpus_assay._native import hash_ngram, ngram_allowed, normalize_text
from corpus_assay.config import ScanConfig
from corpus_assay.filtering import (
    DATASET_CARD_SNIPPET_NAME,
    FILTER_MANIFEST_NAME,
    REMOVED_IDS_NAME,
    _RowFlags,
    clean_row,
)
from corpus_assay.scanner.io import parse_contamination_record
from corpus_assay.schemas import FilterManifest
from corpus_assay.text_utils import iter_ngrams

NGRAM = 3
LEAK = "photosynthesis converts light energy into chemical energy stored in glucose"
CLEAN = "completely unrelated sentence about sailing boats and harbour weather"
CLEAN2 = "another quiet paragraph on garden hedges and the old mill"
SEP = "<|endoftext|>"

runner = CliRunner()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    grams = [g for g in iter_ngrams(normalize_text(LEAK), NGRAM) if ngram_allowed(g)]
    hashes = {hash_ngram(g) for g in grams}
    path = tmp_path / "idx" / "idx.native"
    path.parent.mkdir()
    write_native_index(
        path, ngram=NGRAM, hashes=hashes, sources={h: "bench" for h in hashes}
    )
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    cfg = {
        "benchmarks": [
            {"name": "bench", "dataset": "json", "splits": ["test"], "fields": ["q"]}
        ],
        "ngram": NGRAM,
        "min_hits": 2,
        "min_coverage": 0.0,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _filter(
    shards: Path | list[Path],
    index: Path,
    config: Path,
    out: Path,
    *extra: str,
    code: int = 0,
) -> Any:
    inputs = shards if isinstance(shards, list) else [shards]
    args = ["filter", "--index", str(index), "--out-dir", str(out)]
    for shard in inputs:
        args += ["--input", str(shard)]
    args += ["--config", str(config), "--workers", "1", *extra]
    result = runner.invoke(
        cli.app, args, env={"COLUMNS": "400", "TERMINAL_WIDTH": "400"}
    )
    assert result.exit_code == code, result.output + repr(result.exception)
    return result


def _manifest(out: Path) -> FilterManifest:
    return FilterManifest.model_validate_json(
        (out / FILTER_MANIFEST_NAME).read_text(encoding="utf-8")
    )


def _removed(out: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (out / REMOVED_IDS_NAME).read_text(encoding="utf-8").splitlines()
    ]


def test_record_granularity_drops_flagged_rows(
    tmp_path: Path, index_path: Path, config: Path
) -> None:
    shards = tmp_path / "data"
    (shards / "sub").mkdir(parents=True)
    schema = pa.schema(
        [("id", pa.string()), ("text", pa.string()), ("score", pa.float64())],
        metadata={b"origin": b"test"},
    )
    pq.write_table(
        pa.table(
            {
                "id": ["a0", "a1", "a2", "a3"],
                "text": [CLEAN, LEAK, CLEAN2, f"{CLEAN}{SEP}{LEAK}"],
                "score": [0.1, 0.2, 0.3, 0.4],
            },
            schema=schema,
        ),
        shards / "a.parquet",
        compression="zstd",
    )
    # Blank and whitespace-only lines are not rows; kept lines stay byte-identical.
    jsonl_lines = [
        json.dumps({"id": "b0", "text": CLEAN, "extra": [1, 2]}) + "\n",
        "\n",
        "   \n",
        json.dumps({"id": "b1", "text": LEAK}) + "\n",
        '{"id":   "b2", "text": "' + CLEAN2 + '"}\r\n',
        json.dumps({"id": "b3", "text": f"{LEAK} again"}),
    ]
    (shards / "sub" / "b.jsonl").write_text("".join(jsonl_lines), encoding="utf-8")
    with gzip.open(shards / "sub" / "c.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "c0", "text": LEAK}) + "\n")
        fh.write(json.dumps({"id": "c1", "text": CLEAN}) + "\n")
    inputs_before = {p: _sha(p) for p in shards.rglob("*") if p.is_file()}

    out = tmp_path / "clean"
    _filter(shards, index_path, config, out, "--id-key", "id")

    # Inputs untouched.
    assert {p: _sha(p) for p in inputs_before} == inputs_before

    table = pq.read_table(out / "a.parquet")
    assert table.schema.equals(schema, check_metadata=True)
    assert table.column("id").to_pylist() == ["a0", "a2"]
    assert table.column("score").to_pylist() == [0.1, 0.3]
    assert (
        pq.ParquetFile(out / "a.parquet").metadata.row_group(0).column(0).compression
        == "ZSTD"
    )

    kept = (out / "sub" / "b.jsonl").read_bytes().decode("utf-8")
    assert kept == jsonl_lines[0] + jsonl_lines[1] + jsonl_lines[2] + jsonl_lines[4]

    with gzip.open(out / "sub" / "c.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert [json.loads(line)["id"] for line in fh] == ["c1"]

    manifest = _manifest(out)
    assert manifest.granularity == "record"
    assert manifest.index.native_sha256 == _sha(index_path)
    assert manifest.index.meta_sha256 == _sha(
        index_path.with_name("idx.native.meta.json")
    )
    assert manifest.scan_config.gate_mode == "union"
    assert manifest.scan_config.min_hits == 2
    by_path = {s.output_path: s for s in manifest.shards}
    assert set(by_path) == {"a.parquet", "sub/b.jsonl", "sub/c.jsonl.gz"}
    assert (by_path["a.parquet"].rows_before, by_path["a.parquet"].rows_after) == (4, 2)
    assert (by_path["sub/b.jsonl"].rows_before, by_path["sub/b.jsonl"].rows_after) == (
        4,
        2,
    )
    assert (
        by_path["sub/b.jsonl"].input_sha256 == inputs_before[shards / "sub" / "b.jsonl"]
    )
    assert by_path["sub/b.jsonl"].output_sha256 == _sha(out / "sub" / "b.jsonl")
    assert manifest.removals.rows_before == 10
    assert manifest.removals.rows_removed == 5
    assert manifest.removals.flagged_subdocs == 5
    assert manifest.removals.by_benchmark["bench"].rows_removed == 5
    assert manifest.removals.by_channel == {"ngram": 5}

    removed = {(r["shard"], r["row_index"]): r for r in _removed(out)}
    assert set(removed) == {
        ("a.parquet", 1),
        ("a.parquet", 3),
        ("sub/b.jsonl", 1),
        ("sub/b.jsonl", 3),
        ("sub/c.jsonl.gz", 0),
    }
    assert removed[("a.parquet", 3)]["doc_id"] == "a3-part-1"
    assert removed[("a.parquet", 3)]["part"] == 1
    assert removed[("a.parquet", 3)]["benchmarks"] == ["bench"]
    assert removed[("a.parquet", 3)]["row_removed"] is True

    snippet = (out / DATASET_CARD_SNIPPET_NAME).read_text(encoding="utf-8")
    assert _sha(index_path) in snippet
    assert "corpus-assay filter" in snippet
    # The scan evidence is kept.
    assert (out / "_scan" / "_RUN_MANIFEST.json").exists()


def test_subdoc_granularity_drops_only_the_flagged_part(
    tmp_path: Path, index_path: Path, config: Path
) -> None:
    shard = tmp_path / "data" / "p.parquet"
    shard.parent.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["r0", "r1", "r2", "r3"],
                "text": [
                    f"{CLEAN}{SEP}{LEAK}{SEP}{CLEAN2}",
                    LEAK,
                    CLEAN,
                    f"{CLEAN}<|endoftext}}>{LEAK}",
                ],
            }
        ),
        shard,
    )
    out = tmp_path / "clean"
    _filter(shard, index_path, config, out, "--id-key", "id", "--granularity", "subdoc")

    table = pq.read_table(out / "p.parquet")
    assert table.column("id").to_pylist() == ["r0", "r2", "r3"]
    assert table.column("text").to_pylist() == [
        f"{CLEAN}{SEP}{CLEAN2}",
        CLEAN,
        CLEAN,
    ]
    manifest = _manifest(out)
    assert manifest.granularity == "subdoc"
    assert manifest.removals.rows_removed == 1
    assert manifest.removals.rows_modified == 2
    assert manifest.removals.by_benchmark["bench"].rows_modified == 2
    rows = {r["row_index"]: r["row_removed"] for r in _removed(out)}
    assert rows == {0: False, 1: True, 3: False}


def test_subdoc_blanks_text_fields_and_removes_chat_turns(
    tmp_path: Path, index_path: Path, config: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {"id": "x0", "prompt": CLEAN, "chosen": LEAK},
        {"id": "x1", "prompt": LEAK, "chosen": f"{LEAK}."},
        {"id": "x2", "prompt": CLEAN, "chosen": CLEAN2},
    ]
    (data / "pref.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out = tmp_path / "clean_pref"
    _filter(
        data, index_path, config, out,
        "--id-key", "id", "--text-key", "prompt", "--text-key", "chosen",
        "--granularity", "subdoc",
    )  # fmt: skip
    kept = [
        json.loads(line)
        for line in (out / "pref.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert kept == [
        {"id": "x0", "prompt": CLEAN, "chosen": ""},
        {"id": "x2", "prompt": CLEAN, "chosen": CLEAN2},
    ]
    fields = sorted((r["row_index"], r["field"]) for r in _removed(out))
    assert fields == [(0, "chosen"), (1, "chosen"), (1, "prompt")]

    chat = tmp_path / "chat"
    chat.mkdir()
    turn = pa.struct([("role", pa.string()), ("content", pa.string())])
    pq.write_table(
        pa.table(
            {
                "id": ["c0", "c1"],
                "messages": [
                    [
                        {"role": "user", "content": CLEAN},
                        {"role": "assistant", "content": LEAK},
                        {"role": "user", "content": CLEAN2},
                    ],
                    [{"role": "user", "content": LEAK}],
                ],
            },
            schema=pa.schema([("id", pa.string()), ("messages", pa.list_(turn))]),
        ),
        chat / "chat.parquet",
    )
    out = tmp_path / "clean_chat"
    _filter(
        chat, index_path, config, out,
        "--id-key", "id", "--messages-key", "messages", "--granularity", "subdoc",
    )  # fmt: skip
    table = pq.read_table(out / "chat.parquet")
    assert table.column("id").to_pylist() == ["c0"]
    assert table.column("messages").to_pylist() == [
        [
            {"role": "user", "content": CLEAN},
            {"role": "user", "content": CLEAN2},
        ]
    ]
    removed = {r["row_index"]: r for r in _removed(out)}
    assert removed[0]["role"] == "assistant" and removed[0]["turn"] == 1
    assert removed[0]["doc_id"] == "c0/messages/1-part-0"

    # Record granularity drops the whole conversation.
    out = tmp_path / "clean_chat_rec"
    _filter(
        chat, index_path, config, out, "--id-key", "id", "--messages-key", "messages"
    )
    assert pq.read_table(out / "chat.parquet").num_rows == 0
    assert pq.read_table(out / "chat.parquet").schema.equals(
        pq.read_schema(chat / "chat.parquet")
    )


def test_refuses_non_empty_out_dir_unless_overwrite(
    tmp_path: Path, index_path: Path, config: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.jsonl").write_text(
        json.dumps({"text": LEAK}) + "\n" + json.dumps({"text": CLEAN}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    result = _filter(data, index_path, config, out, code=2)
    assert "not empty" in result.output
    assert not (out / "a.jsonl").exists()

    _filter(data, index_path, config, out, "--overwrite")
    assert (out / "a.jsonl").read_text(encoding="utf-8") == json.dumps(
        {"text": CLEAN}
    ) + "\n"
    assert (out / "keep.txt").exists()
    # A second overwrite replaces the first run's outputs.
    _filter(data, index_path, config, out, "--overwrite", "--granularity", "subdoc")
    assert _manifest(out).granularity == "subdoc"

    # An output directory inside an input directory is refused.
    result = _filter(data, index_path, config, data / "clean", code=2)
    assert "inside the input directory" in result.output


def test_clean_row_drops_a_row_left_without_text() -> None:
    cfg = ScanConfig(
        inputs=[],
        text_keys=["a", "b"],
        messages_key="m",
        id_key=None,
        index_path="i",
        out_dir="o",
    )
    flags = _RowFlags()
    for rec in (
        {"doc_id": "x", "field": "a", "part": 0},
        {"doc_id": "x", "field": "m", "turn": 0, "part": 0},
    ):
        flags.add(parse_contamination_record(json.dumps(rec)))
    row: dict[str, Any] = {"a": LEAK, "b": None, "m": [{"role": "u", "content": LEAK}]}
    assert clean_row(row, flags, cfg) is False
    row = {
        "a": LEAK,
        "b": "  ",
        "m": [{"role": "u", "content": LEAK}, {"content": "hi"}],
    }
    assert clean_row(row, flags, cfg) is True
    assert row == {"a": "", "b": "  ", "m": [{"content": "hi"}]}


def test_filter_refuses_failed_scans(
    tmp_path: Path, index_path: Path, config: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.jsonl").write_text(json.dumps({"body": LEAK}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    result = _filter(data, index_path, config, out, code=1)
    assert "failed to scan" in result.output
    assert not (out / "a.jsonl").exists()
    assert not (out / FILTER_MANIFEST_NAME).exists()
