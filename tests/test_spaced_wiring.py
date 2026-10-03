"""Config / CLI / worker wiring for the experimental spaced-seed channel.

Covers the high-level plumbing that lets the spaced-seed recall channel be selected
from a config or ``--spaced-seeds`` flag and flow through ``ScanConfig`` and the
``scan_file`` worker into the Rust scanner: config models, path resolution,
``.spaced``/index compatibility validation, and worker pass-through. The low-level
parity and hot-loop integration are covered by ``test_spaced_rust_parity.py`` and
``test_spaced_scan_integration.py``.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer

import corpus_assay.indexing as indexing
from _index_helpers import write_native_index
from corpus_assay._native import (
    ngram_config_hash,
    normalization_config_sha,
    normalize_text,
)
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.cli import (
    _resolve_spaced,
    _resolve_spaced_path,
    _validate_spaced_compatibility,
)
from corpus_assay.config import (
    DecontaminationConfig,
    ScanConfig,
    SpacedSeedsConfig,
)
from corpus_assay.scanner.fingerprint import build_run_fingerprint
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.worker import scan_file
from corpus_assay.spaced_index import read_spaced_index, write_spaced_index
from corpus_assay.spaced_seeds import (
    DEFAULT_PATTERNS,
    SpacedSeedConfig,
    SpacedSeedFamily,
    iter_exact_grams,
)

N = 13
PROT = [f"w{i}" for i in range(30)]
FAM = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
SEED_CFG = SpacedSeedConfig(min_loci=3, ver_min_span=17, ver_identity=0.85)


# --------------------------------------------------------------------------- #
# config models
# --------------------------------------------------------------------------- #
def test_spaced_seeds_config_defaults_and_validation():
    cfg = SpacedSeedsConfig()
    assert cfg.enabled is False
    assert cfg.min_loci == 3 and cfg.ver_min_span == 17 and cfg.ver_identity == 0.85
    # enabled requires a path
    with pytest.raises(ValueError, match="spaced.path is required"):
        SpacedSeedsConfig(enabled=True)
    # disabled without a path is fine
    SpacedSeedsConfig(enabled=False)


def test_scan_config_spaced_fields():
    cfg = ScanConfig(
        inputs=["a.parquet"],
        text_key="text",
        id_key=None,
        index_path="idx.native",
        out_dir="out",
        spaced_path="x.spaced",
    )
    assert cfg.spaced_path == "x.spaced"
    assert cfg.spaced_min_loci == 3
    assert cfg.spaced_ver_min_span == 17
    assert cfg.spaced_ver_identity == 0.85
    # empty string is rejected (distinct from None = disabled)
    with pytest.raises(ValueError, match="spaced_path must be non-empty"):
        ScanConfig(
            inputs=["a.parquet"],
            text_key="text",
            id_key=None,
            index_path="idx.native",
            out_dir="out",
            spaced_path="",
        )


def test_decontamination_config_parses_spaced_block():
    cfg = DecontaminationConfig.model_validate(
        {
            "benchmarks": [{"name": "b", "dataset": "d", "splits": ["test"]}],
            "spaced": {"enabled": True, "path": "p.spaced", "min_loci": 4},
        }
    )
    assert cfg.spaced is not None
    assert cfg.spaced.enabled and cfg.spaced.path == "p.spaced"
    assert cfg.spaced.min_loci == 4


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def _cfg_with_spaced(spaced: SpacedSeedsConfig | None) -> DecontaminationConfig:
    return DecontaminationConfig.model_validate(
        {
            "benchmarks": [{"name": "b", "dataset": "d", "splits": ["test"]}],
            "spaced": spaced.model_dump() if spaced else None,
        }
    )


def test_resolve_spaced_path_precedence():
    cfg = _cfg_with_spaced(SpacedSeedsConfig(enabled=True, path="from_cfg.spaced"))
    # explicit flag wins over config
    assert _resolve_spaced_path(cfg, spaced_path="flag.spaced") == "flag.spaced"
    # falls back to the enabled config path
    assert _resolve_spaced_path(cfg, spaced_path=None) == "from_cfg.spaced"
    # disabled config -> None
    disabled = _cfg_with_spaced(SpacedSeedsConfig(enabled=False))
    assert _resolve_spaced_path(disabled, spaced_path=None) is None
    # no config at all -> None
    assert _resolve_spaced_path(_cfg_with_spaced(None), spaced_path=None) is None


def test_resolve_spaced_propagates_config_verifier_params():
    cfg = _cfg_with_spaced(
        SpacedSeedsConfig(
            enabled=True,
            path="from_cfg.spaced",
            min_loci=5,
            ver_min_span=21,
            ver_identity=0.9,
        )
    )
    # config-resolved: carries the configured thresholds (not the defaults)
    res = _resolve_spaced(cfg, spaced_path=None)
    assert res.path == "from_cfg.spaced"
    assert (res.min_loci, res.ver_min_span, res.ver_identity) == (5, 21, 0.9)
    # explicit flag carries only a path -> default verifier params
    flagged = _resolve_spaced(cfg, spaced_path="flag.spaced")
    assert flagged.path == "flag.spaced"
    assert (flagged.min_loci, flagged.ver_min_span, flagged.ver_identity) == (
        3,
        17,
        0.85,
    )


# --------------------------------------------------------------------------- #
# compatibility validation
# --------------------------------------------------------------------------- #
def _write_index_meta(tmp_path, *, ngram=N, match=True):
    index_path = tmp_path / "exact.native"
    hashes = {h for _, h in iter_exact_grams(PROT, N)}
    write_native_index(index_path, ngram=N, hashes=hashes)
    meta = {
        "ngram": ngram,
        "normalization_config_sha": normalization_config_sha(),
        "ngram_config_hash": (ngram_config_hash() if match else "deadbeef-mismatch"),
    }
    (tmp_path / "exact.native.meta.json").write_text(json.dumps(meta))
    return str(index_path)


def _scan_cfg(index_path, spaced_path):
    return ScanConfig(
        inputs=["x.parquet"],
        text_key="text",
        id_key=None,
        index_path=index_path,
        out_dir="out",
        ngram=N,
        spaced_path=spaced_path,
    )


def test_validate_spaced_compatibility_pass(tmp_path):
    index_path = _write_index_meta(tmp_path)
    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, SEED_CFG, n=N)
    # should not raise
    _validate_spaced_compatibility(_scan_cfg(index_path, str(spaced_path)))


def test_validate_spaced_compatibility_ngram_mismatch(tmp_path):
    index_path = _write_index_meta(tmp_path, ngram=7)
    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, SEED_CFG, n=N)
    with pytest.raises(typer.BadParameter, match="ngram"):
        _validate_spaced_compatibility(_scan_cfg(index_path, str(spaced_path)))


def test_validate_spaced_compatibility_config_hash_mismatch(tmp_path):
    index_path = _write_index_meta(tmp_path, match=False)
    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, SEED_CFG, n=N)
    with pytest.raises(typer.BadParameter, match="ngram config hash"):
        _validate_spaced_compatibility(_scan_cfg(index_path, str(spaced_path)))


def test_validate_spaced_missing_meta_guard(tmp_path):
    index_path = _write_index_meta(tmp_path)
    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, SEED_CFG, n=N)
    (tmp_path / "prot.spaced.meta.json").unlink()
    cfg = _scan_cfg(index_path, str(spaced_path))
    # without the escape hatch, a missing meta is an error
    with pytest.raises(typer.BadParameter, match="meta not found"):
        _validate_spaced_compatibility(cfg)
    # with it, validation is skipped
    _validate_spaced_compatibility(cfg, allow_unverified_spaced=True)


def test_validate_spaced_noop_when_disabled(tmp_path):
    index_path = _write_index_meta(tmp_path)
    # spaced_path=None -> nothing to validate, even with a bogus index meta
    _validate_spaced_compatibility(_scan_cfg(index_path, None))


# --------------------------------------------------------------------------- #
# worker pass-through (ScanConfig -> scan_file -> Rust scanner)
# --------------------------------------------------------------------------- #
def _docs_parquet() -> bytes:
    bg = " ".join(f"z{i}" for i in range(12))
    exact_doc = bg + " " + " ".join(PROT[:24]) + " " + bg
    span = list(PROT[:24])
    span[6], span[18] = "editsix", "editeighteen"
    spaced_doc = bg + " " + " ".join(span) + " " + bg
    clean_doc = " ".join(f"q{i}" for i in range(40))
    buf = io.BytesIO()
    pq.write_table(
        pa.Table.from_pydict(
            {
                "text": [clean_doc, exact_doc, spaced_doc],
                "doc_id": ["clean", "exact", "spaced"],
            }
        ),
        buf,
    )
    return buf.getvalue()


def test_scan_file_worker_passes_spaced_through(tmp_path):
    index_path = _write_index_meta(tmp_path)
    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, SEED_CFG, n=N)

    shard = tmp_path / "shard.parquet"
    shard.write_bytes(_docs_parquet())
    out_dir = tmp_path / "out"
    layout = OutputLayout(out_dir)
    layout.ensure_dirs()

    cfg = ScanConfig(
        inputs=[str(shard)],
        text_key="text",
        id_key="doc_id",
        index_path=index_path,
        out_dir=str(out_dir),
        ngram=N,
        min_hits=2,
        min_coverage=0.01,
        spaced_path=str(spaced_path),
    )
    scanned, contaminated, _empty, results = scan_file(shard, cfg, layout)
    assert scanned == 3
    # The worker forwarded the spaced params: the edited spaced-only doc is recovered
    # alongside the verbatim exact doc; the clean doc is not flagged.
    flagged = {json.loads(r)["doc_id"]: json.loads(r) for r in results}
    assert contaminated == 2
    assert set(flagged) == {"exact-part-0", "spaced-part-0"}
    assert flagged["spaced-part-0"].get("spaced_item") == 0


# --------------------------------------------------------------------------- #
# eval-item loader + build-spaced-index CLI
# --------------------------------------------------------------------------- #
def test_build_eval_item_tokens_normalizes_via_loader():
    spec = BenchmarkSpec(name="B", dataset="local", splits=["test"], fields=["text"])

    class _FakeLoader:
        dataset_fingerprints = [{"hf_repo": "local"}]

        def load_positive(self):
            return iter(["w0 w1 w2 alpha beta", "gamma delta"])

    tokens, fingerprints = indexing.build_eval_item_tokens(
        [spec],
        loader_factory=lambda _s: _FakeLoader(),  # type: ignore[arg-type,return-value]
    )
    assert tokens == [
        normalize_text("w0 w1 w2 alpha beta"),
        normalize_text("gamma delta"),
    ]
    assert fingerprints == [{"hf_repo": "local"}]


def test_build_spaced_index_cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from corpus_assay import cli

    spec = BenchmarkSpec(
        name="B", dataset="local", revision="rev1", splits=["test"], fields=["text"]
    )
    monkeypatch.setattr(
        cli,
        "load_decontamination_config",
        lambda _path: SimpleNamespace(benchmarks=[spec], ngram=N),
    )
    monkeypatch.setattr(
        indexing,
        "build_eval_item_tokens",
        lambda _benchmarks: ([list(PROT)], [{"hf_repo": "local"}]),
    )

    out = tmp_path / "out.spaced"
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("benchmarks: []\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli.app,
        ["build-spaced-index", "--config", str(cfg_file), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and (tmp_path / "out.spaced.meta.json").exists()
    art = read_spaced_index(out)
    assert art.item_tokens == [list(PROT)]


def test_build_spaced_index_cli_rejects_unpinned(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from corpus_assay import cli

    spec = BenchmarkSpec(name="B", dataset="local", splits=["test"], fields=["text"])
    monkeypatch.setattr(
        cli,
        "load_decontamination_config",
        lambda _path: SimpleNamespace(benchmarks=[spec], ngram=N),
    )
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("benchmarks: []\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli.app,
        ["build-spaced-index", "--config", str(cfg_file), "--out", str(tmp_path / "o")],
    )
    assert result.exit_code != 0
    assert "missing revision pin" in result.output


def test_build_spaced_index_cli_rejects_class_aware(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from corpus_assay import cli

    spec = BenchmarkSpec(
        name="B", dataset="local", revision="rev1", splits=["test"], fields=["text"]
    )
    monkeypatch.setattr(
        cli,
        "load_decontamination_config",
        lambda _path: SimpleNamespace(benchmarks=[spec], ngram=N),
    )
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("benchmarks: []\n", encoding="utf-8")
    # The Rust scanner hashes raw tokens, so a class-collapsed artifact would never
    # match; the builder must refuse non-'none' class_aware rather than emit a dud.
    result = CliRunner().invoke(
        cli.app,
        [
            "build-spaced-index",
            "--config",
            str(cfg_file),
            "--out",
            str(tmp_path / "o.spaced"),
            "--class-aware",
            "num",
        ],
    )
    assert result.exit_code != 0
    assert "not supported" in result.output
    assert not (tmp_path / "o.spaced").exists()


# --------------------------------------------------------------------------- #
# run fingerprint includes the spaced artifact bytes (resume-cache correctness)
# --------------------------------------------------------------------------- #
def test_run_fingerprint_includes_spaced_bytes(tmp_path):
    index_path = _write_index_meta(tmp_path)
    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, SEED_CFG, n=N)
    cfg = _scan_cfg(index_path, str(spaced_path))

    fp = build_run_fingerprint(cfg)
    assert "spaced_sha256" in fp and fp["spaced_sha256"]
    assert fp["spaced_path"] == str(spaced_path.resolve())

    # Rebuilding the .spaced in place with different content must change the
    # fingerprint, so stale per-file summaries are not reused.
    before = fp["spaced_sha256"]
    write_spaced_index(spaced_path, [PROT, list(reversed(PROT))], FAM, SEED_CFG, n=N)
    assert build_run_fingerprint(cfg)["spaced_sha256"] != before

    # No spaced channel -> no spaced keys in the fingerprint.
    assert "spaced_sha256" not in build_run_fingerprint(_scan_cfg(index_path, None))


def test_run_fingerprint_tolerates_missing_spaced_meta(tmp_path):
    # With --allow-unverified-spaced the .spaced.meta.json may be absent; the
    # fingerprint must not crash (mirrors index_meta/attr handling: _file_sha256
    # returns None on a missing file), recording the data hash and a None meta hash.
    index_path = _write_index_meta(tmp_path)
    spaced_path = tmp_path / "prot.spaced"
    write_spaced_index(spaced_path, [PROT], FAM, SEED_CFG, n=N)
    (tmp_path / "prot.spaced.meta.json").unlink()
    fp = build_run_fingerprint(_scan_cfg(index_path, str(spaced_path)))
    assert fp["spaced_sha256"]
    assert fp["spaced_meta_sha256"] is None
