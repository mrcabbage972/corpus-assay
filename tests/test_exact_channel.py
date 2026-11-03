"""The exact-match channel, choices near the question, and ``confirm``.

The exact channel matches each eligible item's question verbatim after light
normalization (NFKC, lowercase, whitespace collapsed; numbers, operators and stop
words kept) on word boundaries, so it can flag items too short for the n-gram
rule. ``confirm`` adds report-only copy-vs-variant evidence.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from corpus_assay._native import (
    light_normalize,
    longest_common_substring,
    scan_stream_rust,
)
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import _field_texts, _field_values
from corpus_assay.config import ScanConfig
from corpus_assay.indexing import build_index, write_native_index_artifacts
from corpus_assay.inspect import inspect_index
from corpus_assay.scanner.fingerprint import build_run_fingerprint
from corpus_assay.scanner.io import parse_contamination_record
from corpus_assay.scanner.runner import check_item_text_sidecar
from corpus_assay.schemas import DocResult

NGRAM = 13
# Short: no 13-gram after stop-word removal, so the n-gram rule cannot flag it.
SHORT = {"question": "What is 2+3 apples in total?", "choices": ["5", "6", "x9", "y4"]}
# Shorter than the default --exact-min-chars (10 characters).
TINY = {"question": "Is it red?", "choices": []}
TEMPLATE = (
    "In the figure a circle is inscribed in a square whose corners touch a larger "
    "circle drawn around it, and the shaded region lies between the two circles "
    "outside the square. If the radius of the small circle is {a} centimetres and "
    "the square has side {b} centimetres, what is the area of the shaded region?"
)
LONG = {"question": TEMPLATE.format(a=7, b=14), "choices": ["49 pi", "98", "x", "y"]}
FILLER = "harbour weather reports mention gentle breezes over the marina " * 30


class _Loader:
    dataset_fingerprints: list = []

    def __init__(self, spec: BenchmarkSpec, rows: list[dict]) -> None:
        self.spec, self.rows = spec, rows

    def load_positive_fields(self):
        for row in self.rows:
            yield _field_values(row, self.spec.fields)

    def load_negative(self):
        return iter(())


@pytest.fixture
def index(tmp_path: Path) -> str:
    spec = BenchmarkSpec(name="MC", dataset="x", fields=["question", "choices"])
    rows = [SHORT, TINY, LONG]
    result = build_index(
        benchmarks=[spec],
        ngram=NGRAM,
        loader_factory=lambda s: _Loader(s, rows),  # type: ignore[arg-type,return-value]
    )
    path = tmp_path / "mc.native"
    write_native_index_artifacts(result, str(path))
    return str(path)


def _scan(
    index: str, texts: list[str], **kwargs
) -> tuple[int, list[dict], dict[str, int]]:
    buffer = io.BytesIO()
    pq.write_table(
        pa.table({"text": texts, "id": [f"d{i}" for i in range(len(texts))]}),
        buffer,
    )
    buffer.seek(0)
    counts: dict[str, int] = {}
    args = {"min_hits": 3, "min_coverage": 0.001, **kwargs}
    _, contaminated, records, _ = scan_stream_rust(
        input=buffer,
        text_key="text",
        id_key="id",
        index_path=index,
        n=NGRAM,
        channel_counts=counts,
        **args,
    )
    parsed = sorted((json.loads(r) for r in records), key=lambda r: r["doc_id"])
    for line in records:
        DocResult.model_validate_json(line)
        parse_contamination_record(line)
    return contaminated, parsed, counts


def test_light_normalize_keeps_numbers_operators_and_stop_words() -> None:
    assert light_normalize("  What   IS\t2+3?\n") == "what is 2+3?"
    assert light_normalize("ＡＢＣ　ｘ²") == "abc x2"  # NFKC
    assert light_normalize("2+3") != light_normalize("2-3")
    assert light_normalize("$\\frac{1}{2}$ of the") == "$\\frac{1}{2}$ of the"


def test_longest_common_substring() -> None:
    assert longest_common_substring("abcdef", "zzcdezz") == 3
    assert longest_common_substring("abc", "xyz") == 0
    assert longest_common_substring("Same Text", "same   text") == 9
    assert longest_common_substring("", "abc") == 0


def test_field_values_keep_list_fields_as_lists() -> None:
    row = {"question": " Q ", "choices": ["a", "b"], "n": None, "one": ["x"]}
    keys = ["question", "choices", "n", "one"]
    assert _field_values(row, keys) == ["Q", ["a", "b"], "", "x"]
    assert _field_texts(row, keys) == ["Q", "a b", "", "x"]
    arc = {"choices": {"text": ["red", "blue"], "label": ["A", "B"]}}
    assert _field_values(arc, ["choices"]) == [["red", "blue"]]


def test_build_index_writes_the_item_text_sidecar(index: str) -> None:
    path = Path(index + ".items.text.jsonl")
    lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert lines[0] == {
        "item_idx": 0,
        "benchmark": "MC",
        "item_id": "MC#0",
        "fields": ["What is 2+3 apples in total?", ["5", "6", "x9", "y4"]],
    }
    assert [line["item_idx"] for line in lines] == [0, 1, 2]
    assert lines[1]["fields"] == ["Is it red?", ""]
    meta = json.loads(Path(index + ".meta.json").read_text("utf-8"))
    assert meta["files"]["items_text"] == "mc.native.items.text.jsonl"
    assert inspect_index(Path(index))["items_text"] is True


def test_short_item_is_flagged_only_with_the_exact_channel(index: str) -> None:
    doc = f"quiz time. {SHORT['question']} answer below."
    contaminated, _, counts = _scan(index, [doc])
    assert contaminated == 0
    assert counts == {"ngram": 0}

    for mode in ("short", "all"):
        contaminated, records, counts = _scan(index, [doc], exact_channel=mode)
        assert contaminated == 1, mode
        (rec,) = records
        assert rec["channels"] == ["exact"]
        assert rec["match_count"] == 0
        assert rec["exact_hits"] == [
            {"item_id": 0, "field": 0, "choices_found": 0, "choices_total": 4}
        ]
        assert counts == {"ngram": 0, "exact": 1}


def test_operators_and_word_boundaries_are_respected(index: str) -> None:
    docs = [
        "quiz: what is 2-3 apples in total? done",  # another problem
        "somewhat is 2+3 apples in total? done",  # starts inside a word
        "quiz:what is 2+3 apples in total?yes",  # punctuation edges are fine
        "QUIZ:\n  WHAT is   2+3 APPLES in total?",  # case and whitespace
    ]
    contaminated, records, _ = _scan(index, docs, exact_channel="short")
    assert contaminated == 2
    assert [r["doc_id"] for r in records] == ["d2-part-0", "d3-part-0"]


def test_exact_min_chars_filters_short_stems(index: str) -> None:
    doc = "a question: is it red? yes"
    contaminated, _, _ = _scan(index, [doc], exact_channel="short")
    assert contaminated == 0
    contaminated, records, _ = _scan(
        index, [doc], exact_channel="short", exact_min_chars=5
    )
    assert contaminated == 1
    assert records[0]["exact_hits"][0]["item_id"] == 1


def test_choices_are_counted_near_the_question_only(index: str) -> None:
    choices = " ".join(f"({c})" for c in SHORT["choices"])
    near = f"{SHORT['question']} options: {choices}"
    far = f"{SHORT['question']} {FILLER} options: {choices}"
    assert len(FILLER) > 1000
    _, records, _ = _scan(index, [near, far], exact_channel="short")
    found = {r["doc_id"]: r["exact_hits"][0]["choices_found"] for r in records}
    assert found == {"d0-part-0": 4, "d1-part-0": 0}


def test_long_items_are_matched_with_all_and_carry_both_channels(index: str) -> None:
    doc = f"{FILLER} {LONG['question']} choices: 98 and x"
    _, records, counts = _scan(index, [doc], exact_channel="short")
    assert records[0]["channels"] == ["ngram"]
    assert counts == {"ngram": 1, "exact": 0}
    # With the text sidecar loaded, n-gram item hits get the choices too.
    hit = records[0]["item_hits"][0]
    assert (hit["item_id"], hit["choices_found"], hit["choices_total"]) == (2, 2, 4)

    _, records, counts = _scan(index, [doc], exact_channel="all")
    assert records[0]["channels"] == ["ngram", "exact"]
    assert records[0]["exact_hits"][0]["item_id"] == 2
    assert counts == {"ngram": 1, "exact": 1}


def test_confirm_tells_a_copy_from_a_template_variant(index: str) -> None:
    copy = f"{FILLER} {LONG['question']}"
    variant = f"{FILLER} {TEMPLATE.format(a=5, b=10)}"
    contaminated, records, _ = _scan(index, [copy, variant], confirm=True)
    assert contaminated == 2  # the n-gram rule flags both
    by_doc = {r["doc_id"]: r["item_hits"][0] for r in records}
    copy_hit, variant_hit = by_doc["d0-part-0"], by_doc["d1-part-0"]
    assert copy_hit["numbers_total"] == copy_hit["numbers_matched"] == 2
    assert copy_hit["lcs_ratio"] == 1.0
    assert copy_hit["lcs_chars"] == len(light_normalize(LONG["question"]))
    assert variant_hit["numbers_total"] == 2
    assert variant_hit["numbers_matched"] == 0
    assert variant_hit["lcs_ratio"] < copy_hit["lcs_ratio"]
    assert "exact_hits" not in records[0]  # confirm alone adds no exact channel


def test_confirm_never_changes_the_flags(index: str) -> None:
    docs = [
        f"{FILLER} {LONG['question']}",
        f"{FILLER} {TEMPLATE.format(a=5, b=10)}",
        FILLER,
        f"quiz. {SHORT['question']}",
        "short",
    ]
    for kwargs in ({}, {"exact_channel": "short"}, {"gate_mode": "item"}):
        plain = _scan(index, docs, **kwargs)
        confirmed = _scan(index, docs, confirm=True, **kwargs)
        assert plain[0] == confirmed[0]
        assert [r["doc_id"] for r in plain[1]] == [r["doc_id"] for r in confirmed[1]]
        assert plain[2] == confirmed[2]


def test_index_without_the_text_sidecar(index: str, tmp_path: Path) -> None:
    Path(index + ".items.text.jsonl").unlink()
    doc = f"{FILLER} {LONG['question']}"
    # Scans that do not ask for item text still work.
    contaminated, records, _ = _scan(index, [doc])
    assert contaminated == 1
    assert "choices_found" not in records[0]["item_hits"][0]
    for kwargs in ({"exact_channel": "short"}, {"confirm": True}):
        with pytest.raises(ValueError, match="items.text.jsonl.*Rebuild"):
            _scan(index, [doc], **kwargs)

    cfg = ScanConfig(
        inputs=[], id_key=None, index_path=index, out_dir=str(tmp_path / "o")
    )
    check_item_text_sidecar(cfg)
    with pytest.raises(ValueError, match="--exact-channel short needs"):
        check_item_text_sidecar(cfg.model_copy(update={"exact_channel": "short"}))
    with pytest.raises(ValueError, match="--confirm needs"):
        check_item_text_sidecar(cfg.model_copy(update={"confirm": True}))


def test_records_without_the_new_fields_still_parse() -> None:
    v2 = {"schema_version": 2, "doc_id": "a", "match_count": 3, "src_hits": [[0, 3]]}
    v3 = {
        **v2,
        "schema_version": 3,
        "item_hits": [{"item_id": 1, "hits": 3, "item_ngrams": 4}],
        "field": "text",
    }
    for raw in (v2, v3):
        DocResult.model_validate(raw)
        record = parse_contamination_record(json.dumps(raw))
        assert record.exact_hits == [] and record.channels == []


def test_fingerprint_and_config_record_the_text_channels(index: str) -> None:
    cfg = ScanConfig(inputs=["x"], id_key=None, index_path=index, out_dir="out")
    before = build_run_fingerprint(cfg)
    assert before["index_items_text_sha256"]
    path = Path(index + ".items.text.jsonl")
    path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
    assert (
        build_run_fingerprint(cfg)["index_items_text_sha256"]
        != (before["index_items_text_sha256"])
    )
    # The settings are part of the config, hence of the fingerprint.
    exact = cfg.model_copy(update={"exact_channel": "short", "confirm": True})
    assert build_run_fingerprint(exact)["config_sha256"] != before["config_sha256"]
    assert cfg.model_dump()["exact_channel"] == "off"
    assert cfg.model_dump()["exact_min_chars"] == 12
    with pytest.raises(ValueError):
        ScanConfig.model_validate({**cfg.model_dump(), "exact_channel": "some"})
