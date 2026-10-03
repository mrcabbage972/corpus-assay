from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import corpus_assay.scanner.runner as m
import corpus_assay.scanner.worker as w
from corpus_assay.attribution import AttributionIndex
from corpus_assay.scanner.io import parse_contamination_record, write_run_outputs
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.worker import aggregate_per_source_hits

# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Path().glob() in the module is relative to CWD; keep tests deterministic by chdir.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def cfg(workdir: Path) -> Any:
    """
    Lightweight config object compatible with build_run_fingerprint(), which now calls cfg.model_dump().
    """
    out_dir = workdir / "out"
    index_path = workdir / "dummy.index"
    index_path.write_text("x", encoding="utf-8")

    class Cfg(SimpleNamespace):
        def model_dump(self) -> dict[str, Any]:
            # Mirror Pydantic model_dump() enough for fingerprinting.
            return {
                "inputs": self.inputs,
                "out_dir": self.out_dir,
                "workers": self.workers,
                "text_key": self.text_key,
                "id_key": self.id_key,
                "index_path": self.index_path,
                "ngram": self.ngram,
                "min_hits": self.min_hits,
                "min_coverage": self.min_coverage,
                "packed_doc_sep": self.packed_doc_sep,
                "packed_doc_sep_typo": self.packed_doc_sep_typo,
                "max_return_records": self.max_return_records,
                "target_hf_dataset": self.target_hf_dataset,
                "target_hf_split": self.target_hf_split,
                "target_hf_revision": self.target_hf_revision,
                "target_hf_fingerprint": self.target_hf_fingerprint,
                "index_backend": self.index_backend,  # <--- Added
                "stopgrams_path": self.stopgrams_path,
                "spaced_path": self.spaced_path,
                "spaced_min_loci": self.spaced_min_loci,
                "spaced_ver_min_span": self.spaced_ver_min_span,
                "spaced_ver_identity": self.spaced_ver_identity,
                "gate_mode": self.gate_mode,
                "min_longest_run": self.min_longest_run,
                "boilerplate_threshold": self.boilerplate_threshold,
            }

    return Cfg(
        inputs=["shards/*.jsonl"],
        out_dir=str(out_dir),
        workers=1,
        text_key="text",
        id_key="id",
        index_path=str(index_path),
        ngram=13,
        min_hits=2,
        min_coverage=0.2,
        packed_doc_sep="<<<DOC>>>",
        packed_doc_sep_typo="<<<DOC>>",
        max_return_records=5,
        target_hf_dataset=None,
        target_hf_split=None,
        target_hf_revision=None,
        target_hf_fingerprint=None,
        index_backend="auto",
        stopgrams_path=None,
        spaced_path=None,
        spaced_min_loci=3,
        spaced_ver_min_span=17,
        spaced_ver_identity=0.85,
        gate_mode="auto",
        min_longest_run=0,
        boilerplate_threshold=50,
    )


@pytest.fixture()
def make_shards(workdir: Path) -> callable:  # type: ignore
    def _make(names: list[str]) -> list[Path]:
        shard_dir = workdir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for n in names:
            p = shard_dir / n
            p.write_text('{"id": 1, "text": "hi"}\n', encoding="utf-8")
            paths.append(p)
        return paths

    return _make


@pytest.fixture()
def patch_mp_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    do_scan() uses multiprocessing Pool; make it run inline for tests.
    """

    class DummyPool:
        def __init__(self, processes: int):
            self.processes = processes

        def __enter__(self) -> "DummyPool":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def imap_unordered(self, func, iterable):
            for x in iterable:
                yield func(x)

    class DummyCtx:
        def Pool(self, processes: int, initializer=None, initargs=()):
            # Mirror multiprocessing: run the per-worker initializer once.
            if initializer is not None:
                initializer(*initargs)
            return DummyPool(processes)

    monkeypatch.setattr(m, "get_context", lambda _: DummyCtx())


@pytest.fixture()
def patch_rust_scanner(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch scan_stream_rust so tests don't depend on Rust / index files.

    Behavior:
      - scanned = 10 for every file
      - contaminated depends on filename
      - json_results lines are NDJSON strings expected by parse_contamination_record()
    """

    def fake_scan_stream_rust(
        *,
        input,
        text_key: str,
        id_key: str,
        index_path: str,
        index_backend: str,
        n: int,
        min_hits: int,
        min_coverage: float,
        packed_doc_sep: str,
        packed_doc_sep_typo: str,
        out_hits_path: str,
        max_return_records: int | None,
        stopgrams_path: str | None,
        gate_mode: str = "auto",
        min_longest_run: int = 0,
        boilerplate_threshold: int = 50,
        spaced_path: str | None = None,
        spaced_min_loci: int = 3,
        spaced_ver_min_span: int = 17,
        spaced_ver_identity: float = 0.85,
    ):
        # distinguish shards by stem (the hits path embeds it; JSONL input arrives
        # as an in-memory Parquet buffer without a file name)
        base = Path(out_hits_path).name

        if "bad" in base:
            # simulate error path by raising
            raise RuntimeError("boom")

        if "clean" in base:
            contaminated = 0
            json_results: list[str] = []
        else:
            contaminated = 2
            json_results = [
                json.dumps({"doc_id": "d1", "src_hits": [[1, 3], [2, 1]]}),
                json.dumps(
                    {"doc_id": "d2", "src_hits": [[1, 2], [2, 2]]}
                ),  # tie => both primary
            ]

        return 10, contaminated, json_results, 0

    monkeypatch.setattr(w, "scan_stream_rust", fake_scan_stream_rust)


# ----------------------------
# Unit tests: parsing + aggregation
# ----------------------------


@pytest.mark.parametrize(
    "line, expected_doc_id, expected_hits",
    [
        (
            json.dumps({"doc_id": 7, "src_hits": [[1, 2], ["3", "4"]]}),
            7,
            [(1, 2), (3, 4)],
        ),
        (json.dumps({"doc_id": "x", "src_hits": []}), "x", []),
        (json.dumps({"doc_id": "y"}), "y", []),
        (json.dumps({"doc_id": "z", "src_hits": None}), "z", []),
    ],
)
def test_parse_contamination_record_normalizes(
    line: str, expected_doc_id: Any, expected_hits: list[tuple[int, int]]
):
    rec = parse_contamination_record(line)
    assert rec.doc_id == expected_doc_id
    assert [(h.source_id, h.hits) for h in rec.src_hits] == expected_hits


def test_v2_record_carries_item_evidence_through_both_models():
    """A schema-v2 NDJSON line (item_hits + spaced_item) must round-trip through
    parse_contamination_record and validate through DocResult -- previously the
    evidence was dropped (ContaminationRecord) or rejected (DocResult extra=forbid)."""
    from corpus_assay.schemas import DocResult
    from corpus_assay.schemas.doc_result import ItemHitOut

    line = json.dumps(
        {
            "schema_version": 2,
            "doc_id": "d0",
            "match_count": 5,
            "src_hits": [[0, 5]],
            "item_hits": [
                {
                    "item_id": 3,
                    "hits": 5,
                    "longest_run_tokens": 7,
                    "unique_hits": 4,
                    "shared_hits": 1,
                    "boilerplate_hits": 0,
                }
            ],
            "spaced_item": 3,
        }
    )

    rec = parse_contamination_record(line)
    assert rec.match_count == 5
    assert rec.spaced_item == 3
    assert len(rec.item_hits) == 1
    ih = rec.item_hits[0]
    assert (ih.item_id, ih.hits, ih.unique_hits, ih.shared_hits) == (3, 5, 4, 1)

    doc = DocResult.model_validate(json.loads(line))
    assert doc.schema_version == 2
    assert doc.spaced_item == 3
    doc_hit = doc.item_hits[0]
    assert isinstance(doc_hit, ItemHitOut)
    assert doc_hit.boilerplate_hits == 0

    # Backward compatible: a v1 line (no item fields) still validates, empty item_hits.
    v1 = json.dumps(
        {"schema_version": 1, "doc_id": "d1", "match_count": 2, "src_hits": []}
    )
    assert DocResult.model_validate(json.loads(v1)).item_hits == []

    # Legacy pre-v2 tuple item_hits [[item_id, hits, longest_run]] must not break:
    # parse_contamination_record normalizes it; DocResult accepts the tuple form.
    legacy = json.dumps(
        {
            "schema_version": 1,
            "doc_id": "d2",
            "match_count": 2,
            "item_hits": [[0, 2, 3]],
        }
    )
    rec_legacy = parse_contamination_record(legacy)
    assert (rec_legacy.item_hits[0].item_id, rec_legacy.item_hits[0].hits) == (0, 2)
    assert rec_legacy.item_hits[0].longest_run_tokens == 3
    DocResult.model_validate(json.loads(legacy))  # does not raise


def test_aggregate_per_source_counts_hits_from_hits_bin(tmp_path: Path):
    index = AttributionIndex()
    index.add_hash_mask(101, 0b01)
    index.add_hash_mask(102, 0b11)
    index.add_hash_mask(103, 0b10)

    base_path = tmp_path / "sample"
    # This creates a file with FileType=2 (ATTR_MASKS)
    attr_path = Path(index.write_sidecar(str(base_path)))

    hits_path = tmp_path / "sample.hits.bin"

    # Reuse the attribution sidecar's bytes as a hits file.
    data = bytearray(attr_path.read_bytes())

    # The header format is: MAGIC (8s) | FileType (I) ...
    # We need to change FileType from 2 to 3 (HITS_PAIRS).
    # Offset 8 is the start of the 4-byte FileType integer.
    data[8] = 3

    # Write the patched data as the hits file
    hits_path.write_bytes(data)

    agg = aggregate_per_source_hits(hits_path)
    d = agg.by_source

    assert set(d.keys()) == {"0", "1"}

    # src0: hashes 101 + 102
    assert d["0"].unique_hashes == 2
    # src1: hashes 102 + 103
    assert d["1"].unique_hashes == 2


# ----------------------------
# Unit tests: OutputLayout
# ----------------------------


def test_output_layout_paths(workdir: Path):
    layout = OutputLayout(workdir / "out")

    assert layout.hits_dir == (workdir / "out" / "per_source_hits")
    assert layout.summaries_dir == (workdir / "out" / "summaries")
    assert layout.contaminated_dir == (workdir / "out" / "contaminated_docs")

    assert layout.final_summary_json == (workdir / "out" / "_FINAL_SUMMARY.json")
    assert layout.final_summary_csv == (workdir / "out" / "_FINAL_SUMMARY.csv")
    assert layout.run_manifest_json == (workdir / "out" / "_RUN_MANIFEST.json")

    stem = "abc"
    assert layout.per_hits_path(stem).name == "abc.hits.bin"
    assert layout.summary_path(stem).name == "abc.summary.json"
    assert layout.contaminated_docs_path(stem).name == "abc.decontam.sample.ndjson"


# ----------------------------
# Integration-ish tests: do_scan + write_run_outputs
# ----------------------------


def test_do_scan_returns_run_summary_and_caller_writes_outputs(
    workdir: Path,
    cfg: Any,
    make_shards,
    patch_mp_inline,
    patch_rust_scanner,
):
    # 2 ok files: one contaminated, one clean
    make_shards(["a.jsonl", "clean.jsonl"])

    run = m.do_scan(cfg, print_progress=False)

    assert run.total_files == 2
    assert run.ok_files == 2
    assert run.failed_files == 0
    assert run.total_docs_scanned == 20  # 10 per file
    assert run.total_contaminated == 2  # only a.jsonl contaminated

    # caller writes outputs
    layout = OutputLayout(Path(cfg.out_dir))
    write_run_outputs(layout, run, cfg)

    assert layout.final_summary_json.exists()
    assert layout.final_summary_csv.exists()
    assert layout.run_manifest_json.exists()

    data = json.loads(layout.final_summary_json.read_text(encoding="utf-8"))
    assert data["total_docs_scanned"] == 20
    assert data["total_contaminated"] == 2

    manifest = json.loads(layout.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest["config"]["ngram"] == cfg.ngram
    assert "library_version" in manifest


def test_do_scan_counts_failures(
    workdir: Path,
    cfg: Any,
    make_shards,
    patch_mp_inline,
    patch_rust_scanner,
):
    make_shards(["a.jsonl", "bad.jsonl", "clean.jsonl"])

    run = m.do_scan(cfg, print_progress=False)

    assert run.total_files == 3
    assert run.ok_files == 2
    assert run.failed_files == 1
    assert run.total_docs_scanned == 20
    assert run.total_contaminated == 2
