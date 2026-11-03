"""Several text keys and chat messages, each scanned as its own document (#7).

Exercises the real Rust scanner on Parquet and JSONL fixtures: preference data
(prompt / chosen / rejected) and chat ``messages`` columns, checking the doc ids
and the ``field`` / ``role`` / ``turn`` of each record.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from _index_helpers import write_native_index
from corpus_assay._native import (
    hash_ngram,
    ngram_allowed,
    normalize_text,
    scan_stream_rust,
)
from corpus_assay.config import ScanConfig
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.worker import scan_file
from corpus_assay.text_utils import iter_ngrams

NGRAM = 3
LEAK = "photosynthesis converts light energy into chemical energy stored in glucose"
CLEAN = "completely unrelated sentence about sailing boats and harbour weather"


@pytest.fixture
def index_path(tmp_path: Path) -> str:
    grams = [g for g in iter_ngrams(normalize_text(LEAK), NGRAM) if ngram_allowed(g)]
    hashes = {hash_ngram(g) for g in grams}
    return write_native_index(
        tmp_path / "idx.native",
        ngram=NGRAM,
        hashes=hashes,
        sources={h: "bench" for h in hashes},
    )


def _write(
    path: Path, rows: list[dict[str, Any]], schema: pa.Schema | None = None
) -> Path:
    if path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    else:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _scan(
    tmp_path: Path, index: str, shard: Path, **kwargs: Any
) -> tuple[int, int, list[dict[str, Any]]]:
    cfg = ScanConfig(
        inputs=[str(shard)],
        id_key="id",
        index_path=index,
        out_dir=str(tmp_path / "out"),
        ngram=NGRAM,
        min_hits=2,
        min_coverage=0.0,
        workers=1,
        max_return_records=None,
        **kwargs,
    )
    layout = OutputLayout(Path(cfg.out_dir))
    layout.ensure_dirs()
    scanned, contaminated, _empty, records = scan_file(shard, cfg, layout)
    parsed = sorted((json.loads(r) for r in records), key=lambda r: r["doc_id"])
    return scanned, contaminated, parsed


def _where(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["doc_id"],
        record.get("field"),
        record.get("role"),
        record.get("turn"),
    )


PREFERENCE_ROWS = [
    {"id": "a", "prompt": CLEAN, "chosen": LEAK, "rejected": CLEAN},
    {"id": "b", "prompt": LEAK, "chosen": CLEAN, "rejected": None},
]


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl"])
def test_preference_fields_are_scanned_separately(
    tmp_path: Path, index_path: str, suffix: str
) -> None:
    shard = _write(tmp_path / f"pref{suffix}", PREFERENCE_ROWS)

    scanned, contaminated, records = _scan(
        tmp_path, index_path, shard, text_keys=["prompt", "chosen", "rejected"]
    )

    # 5 non-null texts; the leak is in a's answer and in b's prompt.
    assert (scanned, contaminated) == (5, 2)
    assert [_where(r) for r in records] == [
        ("a/chosen-part-0", "chosen", None, None),
        ("b/prompt-part-0", "prompt", None, None),
    ]
    assert all(r["schema_version"] == 3 for r in records)


CHAT_ROWS = [
    {
        "id": "c1",
        "messages": [
            {"role": "system", "content": CLEAN},
            {"role": "user", "content": LEAK},
            {"role": "assistant", "content": f"{CLEAN}<|endoftext|>{LEAK}"},
        ],
    },
    {"id": "c2", "messages": [{"role": "user", "content": CLEAN}]},
    {"id": "c3", "messages": None},
]


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl"])
def test_chat_turns_carry_role_and_turn(
    tmp_path: Path, index_path: str, suffix: str
) -> None:
    shard = _write(tmp_path / f"chat{suffix}", CHAT_ROWS)

    scanned, contaminated, records = _scan(
        tmp_path, index_path, shard, messages_key="messages"
    )

    # Turns: system, user, assistant (split in two), c2's user turn.
    assert (scanned, contaminated) == (5, 2)
    assert [_where(r) for r in records] == [
        ("c1/messages/1-part-0", "messages", "user", 1),
        ("c1/messages/2-part-1", "messages", "assistant", 2),
    ]


def test_chat_no_split_keeps_turns_whole(tmp_path: Path, index_path: str) -> None:
    shard = _write(tmp_path / "chat.parquet", CHAT_ROWS)

    scanned, _contaminated, records = _scan(
        tmp_path, index_path, shard, messages_key="messages", packed_doc_sep=None
    )

    assert scanned == 4
    assert [r["doc_id"] for r in records] == ["c1/messages/1", "c1/messages/2"]


def test_text_keys_and_messages_together(tmp_path: Path, index_path: str) -> None:
    rows = [{"id": 5, "system": LEAK, "messages": [{"role": "user", "content": LEAK}]}]
    shard = _write(tmp_path / "mixed.jsonl", rows)

    _scanned, _contaminated, records = _scan(
        tmp_path, index_path, shard, text_keys=["system"], messages_key="messages"
    )

    # Integer ids are cast to strings.
    assert [_where(r) for r in records] == [
        ("5/messages/0-part-0", "messages", "user", 0),
        ("5/system-part-0", "system", None, None),
    ]


def test_single_text_key_keeps_classic_doc_ids(tmp_path: Path, index_path: str) -> None:
    shard = _write(tmp_path / "plain.parquet", [{"id": "x", "text": LEAK}])

    _scanned, _contaminated, records = _scan(tmp_path, index_path, shard)

    assert [_where(r) for r in records] == [("x-part-0", "text", None, None)]


def test_large_list_and_large_string_messages(tmp_path: Path, index_path: str) -> None:
    turn = pa.struct([("role", pa.large_string()), ("content", pa.large_string())])
    schema = pa.schema([("id", pa.int32()), ("messages", pa.large_list(turn))])
    rows = [{"id": 1, "messages": [{"role": None, "content": LEAK}, None]}]
    shard = _write(tmp_path / "large.parquet", rows, schema=schema)

    _scanned, _contaminated, records = _scan(
        tmp_path, index_path, shard, messages_key="messages"
    )

    # A null role is omitted from the record.
    assert [_where(r) for r in records] == [
        ("1/messages/0-part-0", "messages", None, 0)
    ]


def test_messages_column_of_wrong_type_errors(tmp_path: Path, index_path: str) -> None:
    shard = _write(tmp_path / "bad.parquet", [{"id": "x", "messages": LEAK}])

    with pytest.raises(ValueError, match="expected a list of"):
        _scan(tmp_path, index_path, shard, messages_key="messages")


def test_missing_messages_key_errors(tmp_path: Path, index_path: str) -> None:
    for suffix in (".parquet", ".jsonl"):
        shard = _write(tmp_path / f"nomsg{suffix}", [{"id": "x", "text": LEAK}])
        with pytest.raises(ValueError, match="Messages key 'messages' not found"):
            _scan(tmp_path, index_path, shard, messages_key="messages")


def test_rust_accepts_a_list_of_text_keys(tmp_path: Path, index_path: str) -> None:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(PREFERENCE_ROWS), buffer)
    buffer.seek(0)

    scanned, contaminated, results, _ = scan_stream_rust(
        input=buffer,
        text_key=["prompt", "chosen"],
        id_key="id",
        index_path=index_path,
        n=NGRAM,
        min_hits=2,
        min_coverage=0.0,
    )

    assert (scanned, contaminated) == (4, 2)
    assert sorted(json.loads(r)["doc_id"] for r in results) == [
        "a/chosen-part-0",
        "b/prompt-part-0",
    ]


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl"])
def test_warns_about_unscanned_text_columns(
    tmp_path: Path, index_path: str, suffix: str, caplog: pytest.LogCaptureFixture
) -> None:
    rows = [
        {
            "text": LEAK,
            "notes": "a free-text column",
            "conversation": [{"role": "user", "content": CLEAN}],
            "score": 3,
        }
    ]
    shard = _write(tmp_path / f"extra{suffix}", rows)

    with caplog.at_level(logging.WARNING, logger="corpus_assay.scanner.worker"):
        _scan(tmp_path, index_path, shard)

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2, messages
    unscanned, missing_id = messages
    assert "string column(s) notes" in unscanned
    assert "chat column(s) conversation" in unscanned
    assert "score" not in unscanned
    assert "id column 'id' not found" in missing_id


def test_no_warning_when_every_text_column_is_scanned(
    tmp_path: Path, index_path: str, caplog: pytest.LogCaptureFixture
) -> None:
    shard = _write(tmp_path / "pref.parquet", PREFERENCE_ROWS)

    with caplog.at_level(logging.WARNING, logger="corpus_assay.scanner.worker"):
        _scan(tmp_path, index_path, shard, text_keys=["prompt", "chosen", "rejected"])

    assert caplog.records == []
