"""Stop-grams learned from the indexed items (`stopgrams.items`, #11).

N-grams held by many items (a shared instruction line) are dropped at index
time, so they neither link unrelated items to a document nor dilute an item's
coverage: a copy of an item under a different wrapper still matches in full.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import corpus_assay.cli as cli
from corpus_assay._native import hash_ngram, normalize_text, scan_stream_rust
from corpus_assay.attribution import AttributionIndex
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import _field_texts
from corpus_assay.config import ItemStopGramsConfig, load_decontamination_config
from corpus_assay.indexing import build_index, write_native_index_artifacts
from corpus_assay.inspect import inspect_index
from corpus_assay.schemas import IndexMeta
from corpus_assay.stopgrams import read_stopgrams_header
from corpus_assay.text_utils import iter_allowed_ngrams

NGRAM = 3
INSTRUCTION = "write python code to solve the following problem carefully"
TOPICS = [
    "reverse a linked list in place",
    "count vowels inside every sentence",
    "merge overlapping calendar intervals quickly",
    "detect cycles within directed graphs",
    "parse roman numerals into integers",
    "rotate square matrices clockwise ninety degrees",
    "compress repeated characters using counts",
    "validate balanced bracket sequences",
    "find median among two sorted arrays",
    "schedule meeting rooms minimizing overlap",
]
ITEMS = [f"{INSTRUCTION} {topic}" for topic in TOPICS]


def _ngrams(text: str) -> set[int]:
    return {hash_ngram(ng) for ng in iter_allowed_ngrams(normalize_text(text), NGRAM)}


class _Loader:
    dataset_fingerprints: list = []

    def __init__(self, spec: BenchmarkSpec) -> None:
        self.spec = spec

    def load_positive_fields(self):
        for text in ITEMS:
            yield _field_texts({"q": text}, self.spec.fields)

    def load_negative(self):
        return iter(())


def _build(tmp_path: Path, name: str, stop: ItemStopGramsConfig | None) -> str:
    spec = BenchmarkSpec(name="Code", dataset="x", fields=["q"])
    result = build_index(
        benchmarks=[spec],
        ngram=NGRAM,
        loader_factory=_Loader,  # type: ignore[arg-type]
        item_stopgrams=stop,
    )
    path = tmp_path / f"{name}.native"
    write_native_index_artifacts(result, str(path))
    return str(path)


def _scan(index: str, texts: list[str], **kwargs: Any) -> tuple[int, list[dict]]:
    buffer = io.BytesIO()
    pq.write_table(
        pa.table({"text": texts, "id": [f"d{i}" for i in range(len(texts))]}), buffer
    )
    buffer.seek(0)
    args: dict[str, Any] = {"min_hits": 3, "min_coverage": 0.0, **kwargs}
    _, contaminated, records, _ = scan_stream_rust(
        input=buffer, text_key="text", id_key="id", index_path=index, n=NGRAM, **args
    )
    return contaminated, sorted(
        (json.loads(r) for r in records), key=lambda r: r["doc_id"]
    )


STOP = ItemStopGramsConfig(max_item_share=0.5, min_count=5)


def test_repeated_instruction_is_dropped_from_the_index(tmp_path: Path) -> None:
    index = _build(tmp_path, "stop", STOP)
    instruction = _ngrams(INSTRUCTION)
    native = set(AttributionIndex.read_hash_mask_pairs(index + ".attr"))
    kept = {h for h, _mask in native}
    assert not kept & instruction
    # Each topic keeps its own n-grams (and those spanning instruction and topic,
    # which only one item holds).
    for text in ITEMS:
        assert _ngrams(text) - instruction <= kept

    with open(index + ".item_stopgrams", "rb") as f:
        header = read_stopgrams_header(f)
        dropped = set(struct.unpack(f"<{header.record_count}Q", f.read()))
    assert header.ngram == NGRAM
    assert dropped == instruction

    meta = IndexMeta.model_validate_json(Path(index + ".meta.json").read_text("utf-8"))
    assert meta.schema_version == 2
    assert meta.item_stopgrams is not None
    assert meta.item_stopgrams.model_dump() == {
        "max_item_share": 0.5,
        "min_count": 5,
        "n_items": len(ITEMS),
        "dropped": len(instruction),
        "file": "stop.native.item_stopgrams",
    }
    assert meta.files and meta.files["item_stopgrams"] == "stop.native.item_stopgrams"
    # Counts are taken after the drop.
    (stats,) = meta.benchmark_stats
    assert stats.ngrams == len(kept)
    items_meta = json.loads(Path(index + ".items.meta.json").read_text("utf-8"))
    for item, text in zip(items_meta["items"], ITEMS):
        assert item["n_protected_hashes"] == len(_ngrams(text) - instruction)

    info = inspect_index(Path(index))
    stop = info["item_stopgrams"]
    assert isinstance(stop, dict) and stop["dropped"] == len(instruction)
    assert info["items_format"] == 2


def test_thresholds_need_both_share_and_count(tmp_path: Path) -> None:
    # Held by all 10 items: > 50% of them, but fewer than 11.
    index = _build(
        tmp_path, "high", ItemStopGramsConfig(max_item_share=0.5, min_count=11)
    )
    assert Path(index + ".item_stopgrams").stat().st_size == 24  # header only
    # min_count met, but not more than all of the items.
    index = _build(
        tmp_path, "all", ItemStopGramsConfig(max_item_share=1.0, min_count=2)
    )
    meta = json.loads(Path(index + ".meta.json").read_text("utf-8"))
    assert meta["item_stopgrams"]["dropped"] == 0


def test_without_item_stopgrams_nothing_is_written(tmp_path: Path) -> None:
    index = _build(tmp_path, "plain", None)
    assert not Path(index + ".item_stopgrams").exists()
    meta = json.loads(Path(index + ".meta.json").read_text("utf-8"))
    assert "item_stopgrams" not in meta


def test_a_wrapped_copy_still_matches_in_full(tmp_path: Path) -> None:
    plain = _build(tmp_path, "plain", None)
    stop = _build(tmp_path, "stop", STOP)
    wrapped = f"lets think step by step about this {ITEMS[3]} and answer below please"
    instruction_only = f"{INSTRUCTION} then explain the weather forecast for tomorrow"
    topic_only = f"lets think step by step about this {TOPICS[3]} and answer below"

    for gate_mode in ("union", "item"):
        # Without item stop-grams the instruction alone links a document to every
        # item (a false positive), and the wrapped copy is attributed to every
        # item through it.
        contaminated, records = _scan(
            plain, [wrapped, instruction_only], gate_mode=gate_mode
        )
        assert contaminated == 2
        assert len(records[0]["item_hits"]) == len(records[1]["item_hits"]) == 10
        assert records[0]["item_hits"][0]["item_id"] == 3

        # With them, only the copied item matches, in full and through n-grams
        # unique to it; the instruction alone matches nothing.
        contaminated, records = _scan(
            stop, [wrapped, instruction_only], gate_mode=gate_mode
        )
        assert contaminated == 1
        (record,) = records
        assert record["doc_id"] == "d0-part-0"
        (hit,) = record["item_hits"]
        assert hit["item_id"] == 3
        assert hit["item_coverage"] == 1.0
        assert hit["unique_hits"] == hit["hits"]

    # A copy without the instruction holds more of the item once the instruction
    # no longer counts in it.
    _, (before,) = _scan(plain, [topic_only], min_hits=1)
    _, (after,) = _scan(stop, [topic_only], min_hits=1)
    before_cov = next(h for h in before["item_hits"] if h["item_id"] == 3)
    (after_cov,) = after["item_hits"]
    assert after_cov["item_coverage"] > before_cov["item_coverage"]


runner = CliRunner()


def test_build_index_flags_override_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "benchmarks:\n  - {name: B, dataset: x, splits: [test], fields: [q]}\n"
        "stopgrams:\n  items: {max_item_share: 0.2}\n",
        encoding="utf-8",
    )
    assert load_decontamination_config(config).stopgrams.items == ItemStopGramsConfig(  # type: ignore[union-attr]
        max_item_share=0.2, min_count=10
    )
    built: list[Any] = []
    monkeypatch.setattr(cli, "build_index_hf", built.append)
    base = ["build-index", "--config", str(config), "--out-index", str(tmp_path / "i")]

    for extra, expected in [
        ([], ItemStopGramsConfig(max_item_share=0.2, min_count=10)),
        (
            ["--item-stopgrams-min-count", "3"],
            ItemStopGramsConfig(max_item_share=0.2, min_count=3),
        ),
        (
            ["--item-stopgrams-share", "0.05"],
            ItemStopGramsConfig(max_item_share=0.05, min_count=10),
        ),
    ]:
        result = runner.invoke(cli.app, [*base, *extra])
        assert result.exit_code == 0, result.output
        assert built[-1].item_stopgrams == expected

    config.write_text(
        "benchmarks:\n  - {name: B, dataset: x, splits: [test], fields: [q]}\n",
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, base)
    assert result.exit_code == 0 and built[-1].item_stopgrams is None
    result = runner.invoke(cli.app, [*base, "--item-stopgrams-share", "0.01"])
    assert built[-1].item_stopgrams == ItemStopGramsConfig(max_item_share=0.01)
    result = runner.invoke(cli.app, [*base, "--item-stopgrams-share", "0"])
    assert result.exit_code != 0
    assert "Invalid item stop-gram settings" in result.output
