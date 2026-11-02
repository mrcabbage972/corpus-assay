"""Item postings with field masks (`.items` v2): which fields of an item matched.

Builds real indexes through ``build_index`` (each spec field hashed on its own),
scans through the Rust worker, and checks that ``field_hits`` tells a full
multiple-choice copy (question and choices) from a question-only copy, that
`.items` v1 sidecars still load and scan, and that the run fingerprint covers
the item sidecars.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from corpus_assay._native import hash_ngram, normalize_text, scan_stream_rust
from corpus_assay.attribution import ITEMS_FORMAT_VERSION, AttributionIndex
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import _concat_fields, _field_texts
from corpus_assay.config import ScanConfig
from corpus_assay.formats import FileType, write_header
from corpus_assay.indexing import build_index, write_native_index_artifacts
from corpus_assay.scanner.fingerprint import build_run_fingerprint
from corpus_assay.scanner.io import parse_contamination_record
from corpus_assay.schemas import DocResult
from corpus_assay.text_utils import iter_allowed_ngrams

NGRAM = 3
QUESTION = (
    "which planet hosts the tallest volcano olympus mons across our entire solar "
    "system today"
)
CHOICES = [
    "mars rusty fourth planet",
    "venus cloudy second planet",
    "jupiter gaseous fifth giant",
]
OTHER = "photosynthesis converts sunlight carbon dioxide water into glucose oxygen"
FILLER = "harbour weather reports mention gentle breezes over quiet marina docks"


class _FieldsLoader:
    """Loader double yielding each row's field texts, like ``BenchmarkLoader``."""

    dataset_fingerprints: list = []

    def __init__(self, spec: BenchmarkSpec, rows: list[dict]) -> None:
        self.spec = spec
        self.rows = rows

    def load_positive_fields(self):
        for row in self.rows:
            yield _field_texts(row, self.spec.fields)

    def load_negative(self):
        return iter(())


def _build(tmp_path: Path, rows: list[dict], fields: list[str]) -> str:
    spec = BenchmarkSpec(name="MC", dataset="x", fields=fields)
    result = build_index(
        benchmarks=[spec],
        ngram=NGRAM,
        loader_factory=lambda s: _FieldsLoader(s, rows),  # type: ignore[arg-type,return-value]
    )
    index = tmp_path / "mc.native"
    write_native_index_artifacts(result, str(index))
    return str(index)


def _scan(index: str, texts: list[str], **kwargs) -> list[dict]:
    buffer = io.BytesIO()
    pq.write_table(
        pa.table({"text": texts, "id": [f"d{i}" for i in range(len(texts))]}),
        buffer,
    )
    buffer.seek(0)
    args = {"min_hits": 2, "min_coverage": 0.0, **kwargs}
    _, _, records, _ = scan_stream_rust(
        input=buffer, text_key="text", id_key="id", index_path=index, n=NGRAM, **args
    )
    return sorted((json.loads(r) for r in records), key=lambda r: r["doc_id"])


def _ngrams(text: str) -> set[int]:
    return {hash_ngram(ng) for ng in iter_allowed_ngrams(normalize_text(text), NGRAM)}


@pytest.fixture
def mc_index(tmp_path: Path) -> str:
    rows = [
        {"question": QUESTION, "choices": CHOICES},
        {"question": OTHER, "choices": ["a", "b"]},
    ]
    return _build(tmp_path, rows, ["question", "choices"])


def test_field_texts_keep_the_spec_positions() -> None:
    row = {"question": "", "choices": ["x", "y"], "answer": 0}
    assert _field_texts(row, ["question", "choices", "answer"]) == ["", "x y", "0"]
    # The joined text (spaced index, older callers) is unchanged.
    assert _concat_fields(row, ["question", "choices", "answer"]) == (
        "x y <|field_sep|> 0"
    )


def test_items_v2_records_field_masks_and_names(mc_index: str) -> None:
    assert AttributionIndex.items_sidecar_version(mc_index + ".items") == 2
    meta = json.loads(Path(mc_index + ".items.meta.json").read_text("utf-8"))
    assert meta["format_version"] == ITEMS_FORMAT_VERSION
    assert meta["fields"] == {"MC": ["question", "choices"]}

    question, choices = _ngrams(QUESTION), _ngrams(" ".join(CHOICES))
    masks = {
        h: dict(postings)
        for h, postings in AttributionIndex.read_item_postings(mc_index + ".items")
    }
    assert {h for h, m in masks.items() if m.get(0) == 0b01} == question
    assert {h for h, m in masks.items() if m.get(0) == 0b10} == choices
    assert meta["items"][0]["n_protected_hashes"] == len(question | choices)


def test_field_hits_tell_a_full_copy_from_a_question_only_copy(
    mc_index: str,
) -> None:
    full = f"{FILLER} {QUESTION} {' '.join(CHOICES)} {FILLER}"
    question_only = f"{FILLER} {QUESTION} {FILLER}"
    for gate_mode in ("union", "item"):
        records = _scan(mc_index, [full, question_only], gate_mode=gate_mode)
        assert [r["doc_id"] for r in records] == ["d0-part-0", "d1-part-0"]
        full_hit, question_hit = (r["item_hits"][0] for r in records)

        n_question = len(_ngrams(QUESTION))
        n_choices = len(_ngrams(" ".join(CHOICES)))
        assert full_hit["field_hits"] == [[0, n_question], [1, n_choices]]
        assert full_hit["fields_matched"] == ["question", "choices"]
        assert question_hit["field_hits"] == [[0, n_question]]
        assert question_hit["fields_matched"] == ["question"]

        for record in records:
            DocResult.model_validate(record)
            parsed = parse_contamination_record(json.dumps(record))
            assert parsed.item_hits[0].field_hits


def _write_v1_items(index: str) -> None:
    """Rewrite an index's `.items` in the 0.1 layout (no field masks)."""
    postings = list(AttributionIndex.read_items_sidecar(index + ".items"))
    with open(index + ".items", "wb") as f:
        write_header(
            f,
            file_type=FileType.ITEM_POSTINGS,
            ngram_n=NGRAM,
            record_count=len(postings),
        )
        for h, ids in postings:
            f.write(struct.pack("<QI", h, len(ids)))
            f.write(struct.pack(f"<{len(ids)}I", *ids))


def test_v1_items_sidecar_still_loads_and_scans(tmp_path: Path) -> None:
    rows = [{"question": QUESTION, "choices": CHOICES}]
    index = _build(tmp_path, rows, ["question", "choices"])
    _write_v1_items(index)
    assert AttributionIndex.items_sidecar_version(index + ".items") == 1
    assert all(
        mask == 0
        for _h, postings in AttributionIndex.read_item_postings(index + ".items")
        for _item, mask in postings
    )

    full = f"{FILLER} {QUESTION} {' '.join(CHOICES)}"
    for gate_mode in ("union", "item"):
        (record,) = _scan(index, [full], gate_mode=gate_mode)
        (hit,) = record["item_hits"]
        assert hit["hits"] == len(_ngrams(QUESTION) | _ngrams(" ".join(CHOICES)))
        assert "field_hits" not in hit and "fields_matched" not in hit
        assert (
            parse_contamination_record(json.dumps(record)).item_hits[0].field_hits == []
        )


def test_fingerprint_changes_when_the_item_sidecars_change(mc_index: str) -> None:
    cfg = ScanConfig(inputs=["x"], id_key=None, index_path=mc_index, out_dir="out")
    before = build_run_fingerprint(cfg)
    assert before["index_items_sha256"] and before["index_items_meta_sha256"]

    _write_v1_items(mc_index)  # same hashes, different .items bytes
    after = build_run_fingerprint(cfg)
    assert after["index_items_sha256"] != before["index_items_sha256"]
    assert after["index_sha256"] == before["index_sha256"]

    meta_path = Path(mc_index + ".items.meta.json")
    meta_path.write_text(meta_path.read_text("utf-8") + " ", encoding="utf-8")
    assert (
        build_run_fingerprint(cfg)["index_items_meta_sha256"]
        != after["index_items_meta_sha256"]
    )
