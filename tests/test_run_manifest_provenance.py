"""The run manifest records the index files, benchmarks and gate settings.

Manifests in the 0.1.0 format (main before these fields) must still load in
``reproduce``, ``inspect-run`` and ``report``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import corpus_assay.cli as cli
from _index_helpers import write_native_index
from corpus_assay._native import hash_ngram, ngram_allowed, normalize_text
from corpus_assay.config import ScanConfig
from corpus_assay.schemas import RunManifest
from corpus_assay.text_utils import iter_ngrams

NGRAM = 3
LEAK = "photosynthesis converts light energy into chemical energy stored in glucose"
CLEAN = "completely unrelated sentence about sailing boats and harbour weather"

runner = CliRunner()

# The ScanConfig fields of 0.1.0, which wrote `config` with these keys only.
V010_CONFIG_KEYS = {
    "inputs", "text_key", "id_key", "index_path", "out_dir", "min_hits", "ngram",
    "min_coverage", "workers", "index_backend", "packed_doc_sep",
    "packed_doc_sep_typo", "max_return_records", "target_hf_dataset",
    "target_hf_split", "target_hf_revision", "target_hf_fingerprint",
    "stopgrams_path", "spaced_path", "spaced_min_loci", "spaced_ver_min_span",
    "spaced_ver_identity", "gate_mode", "min_longest_run", "boilerplate_threshold",
}  # fmt: skip
V010_SCAN_CONFIG_KEYS = {
    "text_key", "id_key", "n", "min_hits", "min_coverage", "packed_doc_sep",
    "packed_doc_sep_typo", "stopgrams_path",
}  # fmt: skip
V010_INDEX_KEYS = {"path", "meta_path", "index_format_version", "meta_sha256"}


@pytest.fixture
def scan_dir(tmp_path: Path) -> Path:
    grams = [g for g in iter_ngrams(normalize_text(LEAK), NGRAM) if ngram_allowed(g)]
    hashes = {hash_ngram(g) for g in grams}
    index = tmp_path / "idx.native"
    write_native_index(
        index, ngram=NGRAM, hashes=hashes, sources={h: "bench" for h in hashes}
    )
    shard = tmp_path / "data" / "a.jsonl"
    shard.parent.mkdir()
    shard.write_text(
        json.dumps({"text": LEAK}) + "\n" + json.dumps({"text": CLEAN}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmarks": [
                    {
                        "name": "bench",
                        "dataset": "json",
                        "splits": ["t"],
                        "fields": ["q"],
                    }
                ],
                "ngram": NGRAM,
                "min_hits": 2,
                "min_coverage": 0.0,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "scan"
    result = runner.invoke(
        cli.app,
        [
            "scan", "--input", str(shard.parent), "--index", str(index),
            "--out-dir", str(out), "--config", str(config), "--workers", "1",
            "--gate-mode", "union", "--min-longest-run", "4",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    return out


def _manifest(scan_dir: Path) -> dict[str, Any]:
    return json.loads((scan_dir / "_RUN_MANIFEST.json").read_text(encoding="utf-8"))


def test_manifest_records_index_files_benchmarks_and_gate(scan_dir: Path) -> None:
    manifest = RunManifest.model_validate(_manifest(scan_dir))
    index = Path(manifest.index.path)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert manifest.index.native_sha256 == sha(index)
    assert manifest.index.attr_sha256 == sha(index.with_name("idx.native.attr"))
    assert manifest.index.sources_sha256 == sha(
        index.with_name("idx.native.sources.json")
    )
    assert manifest.index.items_sha256 is None  # no item sidecar in this index
    meta = json.loads(index.with_name("idx.native.meta.json").read_text("utf-8"))
    assert manifest.index.normalization_config_sha == meta["normalization_config_sha"]
    assert manifest.index.ngram_config_hash == meta["ngram_config_hash"]
    assert manifest.scan_config.gate_mode == "union"
    assert manifest.scan_config.min_longest_run == 4
    assert manifest.scan_config.boilerplate_threshold == 50


def test_index_info_carries_benchmarks_and_stats(tmp_path: Path) -> None:
    from corpus_assay.scanner.io import build_index_info

    index = tmp_path / "i.native"
    index.write_bytes(b"x")
    meta = {
        "normalization_config_sha": "n",
        "ngram_config_hash": "h",
        "benchmarks": [
            {
                "name": "gsm8k",
                "hf_repo": "openai/gsm8k",
                "config_name": "main",
                "split": "test",
                "revision": "abc",
                "resolved_revision": "abcdef",
                "dataset_fingerprint": "f",
            }
        ],
        "benchmark_stats": [{"name": "gsm8k", "items": 1319, "ngrams": 99}],
    }
    index.with_name("i.native.meta.json").write_text(json.dumps(meta), "utf-8")
    info = build_index_info(index)
    assert info.native_sha256 == hashlib.sha256(b"x").hexdigest()
    assert info.benchmarks[0].resolved_revision == "abcdef"
    assert info.benchmarks[0].split == "test"
    assert info.benchmark_stats[0].items == 1319
    assert (info.normalization_config_sha, info.ngram_config_hash) == ("n", "h")


def _downgrade_to_v010(scan_dir: Path) -> None:
    """Rewrite the manifest as 0.1.0 wrote it."""
    payload = _manifest(scan_dir)
    payload["index"] = {k: payload["index"][k] for k in V010_INDEX_KEYS}
    scan_config = {k: payload["scan_config"].get(k) for k in V010_SCAN_CONFIG_KEYS}
    payload["scan_config"] = scan_config
    config = dict(payload["config"])
    config["text_key"] = config["text_keys"][0]
    payload["config"] = {k: config[k] for k in V010_CONFIG_KEYS}
    (scan_dir / "_RUN_MANIFEST.json").write_text(json.dumps(payload), "utf-8")


def test_v010_manifest_still_loads(
    scan_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _downgrade_to_v010(scan_dir)
    manifest = RunManifest.model_validate(_manifest(scan_dir))
    assert manifest.index.native_sha256 is None
    assert manifest.index.benchmarks == []
    assert manifest.scan_config.gate_mode is None

    result = runner.invoke(cli.app, ["inspect-run", "--scan-dir", str(scan_dir)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli.app, ["report", "--scan-dir", str(scan_dir)])
    assert result.exit_code == 0, result.output

    seen: list[ScanConfig] = []
    monkeypatch.setattr(cli, "_run_scan", lambda cfg, out_dir: seen.append(cfg))
    result = runner.invoke(
        cli.app, ["reproduce", "--manifest", str(scan_dir / "_RUN_MANIFEST.json")]
    )
    assert result.exit_code == 0, result.output
    assert seen and seen[0].text_keys == ["text"]
    assert seen[0].min_longest_run == 4


def test_reproduce_checks_the_index_sha(
    scan_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _manifest(scan_dir)
    payload["index"]["native_sha256"] = "0" * 64
    (scan_dir / "_RUN_MANIFEST.json").write_text(json.dumps(payload), "utf-8")
    monkeypatch.setattr(cli, "_run_scan", lambda cfg, out_dir: None)
    result = runner.invoke(
        cli.app,
        ["reproduce", "--manifest", str(scan_dir / "_RUN_MANIFEST.json")],
        env={"COLUMNS": "400", "TERMINAL_WIDTH": "400"},
    )
    assert result.exit_code != 0
    assert "Index sha256 mismatch" in result.output
