# tests/test_cli.py
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import corpus_assay.cli as cli  # noqa: F401
from corpus_assay.benchmark_spec import BenchmarkSpec

runner = CliRunner()


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid noisy root logger setup affecting test output.
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)


@pytest.fixture()
def tmp_cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "\n".join(
            [
                "benchmarks:",
                "  - name: MMLU",
                "    dataset: cais/mmlu",
                "    configs: ['*']",
                "    splits: ['test']",
                "    fields: ['question']",
                "ngram: 13",
                "min_hits: 3",
                "min_coverage: 0.001",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def fake_decon_cfg() -> Any:
    return SimpleNamespace(
        ngram=13,
        min_hits=3,
        min_coverage=0.001,
        background=None,
        stopgrams=None,
        spaced=None,
        benchmarks=[
            BenchmarkSpec(
                name="MMLU",
                dataset="cais/mmlu",
                configs=["*"],
                splits=["test"],
                fields=["question"],
            )
        ],
    )


@pytest.fixture()
def patch_config_loader(monkeypatch: pytest.MonkeyPatch, fake_decon_cfg: Any) -> None:
    monkeypatch.setattr(cli, "load_decontamination_config", lambda _: fake_decon_cfg)


@pytest.fixture()
def patch_scan_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Patch scanning entrypoints so CLI tests don't run multiprocessing or Rust.
    Returns a dict capturing calls for assertions.
    """
    calls: dict[str, Any] = {"do_scan": [], "write_run_outputs": []}

    # Return object shaped enough for write_run_outputs()
    fake_run = SimpleNamespace(
        total_files=1,
        ok_files=1,
        failed_files=0,
        total_docs_scanned=10,
        total_contaminated=2,
        contamination_rate=0.2,
        per_file=[],
        # allow model_dump if caller uses it
        model_dump=lambda: {"ok_files": 1},
    )

    def fake_do_scan(scan_cfg: Any, *args: Any, **kwargs: Any) -> Any:
        calls["do_scan"].append(scan_cfg)
        return fake_run

    def fake_write_run_outputs(layout: Any, run: Any, cfg: Any) -> None:
        calls["write_run_outputs"].append((layout, run, cfg))

    monkeypatch.setattr(cli, "do_scan", fake_do_scan)
    monkeypatch.setattr(cli, "write_run_outputs", fake_write_run_outputs)

    return calls


@pytest.fixture()
def patch_index_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"build_index_hf": []}

    def fake_build_index_hf(idx_cfg: Any) -> None:
        calls["build_index_hf"].append(idx_cfg)

    monkeypatch.setattr(cli, "build_index_hf", fake_build_index_hf)
    return calls


# ----------------------------
# Tests: scan
# ----------------------------


def test_version_flag_prints_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "library_version", lambda: "1.2.3")
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "corpus-assay 1.2.3"


def test_validate_config_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cfg = SimpleNamespace(
        ngram=13,
        min_hits=3,
        min_coverage=0.001,
        benchmarks=[
            BenchmarkSpec(
                name="MMLU",
                dataset="cais/mmlu",
                configs=["*"],
                splits=["test"],
                fields=["question"],
                revision="rev1",  # Satisfy requirement or use --allow-unpinned
            )
        ],
    )
    monkeypatch.setattr(cli, "load_decontamination_config", lambda _: fake_cfg)

    result = runner.invoke(cli.app, ["validate-config", "--config", "cfg.yaml"])
    assert result.exit_code == 0
    assert "Config OK." in result.output


def test_validate_config_rejects_too_many_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cfg = SimpleNamespace(
        ngram=13,
        min_hits=3,
        min_coverage=0.001,
        benchmarks=[BenchmarkSpec(name=f"B{i}", dataset="ds") for i in range(65)],
    )
    monkeypatch.setattr(cli, "load_decontamination_config", lambda _: fake_cfg)

    result = runner.invoke(
        cli.app, ["validate-config", "--config", "cfg.yaml", "--allow-unpinned"]
    )
    assert result.exit_code != 0
    assert "At most 64 benchmarks" in result.output


def test_scan_requires_config(tmp_path: Path):
    out = tmp_path / "out"
    idx = tmp_path / "idx.native"
    idx.write_text("x", encoding="utf-8")

    # missing --config should fail
    result = runner.invoke(
        cli.app,
        [
            "scan",
            "--input",
            str(tmp_path / "*.parquet"),
            "--index",
            str(idx),
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_scan_with_config_uses_config_values(
    tmp_path: Path,
    tmp_cfg_file: Path,
    patch_config_loader: None,
    patch_scan_pipeline: dict[str, Any],
):
    out = tmp_path / "out"
    idx = tmp_path / "idx.native"
    idx.write_text("x", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "scan",
            "--input",
            str(tmp_path / "*.parquet"),
            "--index",
            str(idx),
            "--out-dir",
            str(out),
            "--config",
            str(tmp_cfg_file),
        ],
    )
    assert result.exit_code == 0, result.output

    # do_scan called once with ScanConfig containing config-derived hyperparams
    assert len(patch_scan_pipeline["do_scan"]) == 1
    scan_cfg = patch_scan_pipeline["do_scan"][0]
    assert scan_cfg.ngram == 13
    assert scan_cfg.min_hits == 3
    assert scan_cfg.min_coverage == 0.001


def test_scan_dry_run_plans_only(
    tmp_path: Path,
    tmp_cfg_file: Path,
    patch_scan_pipeline: dict[str, Any],
):
    out = tmp_path / "out"
    idx = tmp_path / "idx.native"
    idx.write_text("x", encoding="utf-8")
    (tmp_path / "idx.native.meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "idx.native.attr").write_text("attr", encoding="utf-8")
    (tmp_path / "idx.native.sources.json").write_text("{}", encoding="utf-8")  # <-- add

    (tmp_path / "a.parquet").write_text("x", encoding="utf-8")
    (tmp_path / "b.parquet").write_text("y", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "scan",
            "--input",
            str(tmp_path / "*.parquet"),
            "--index",
            str(idx),
            "--out-dir",
            str(out),
            "--config",
            str(tmp_cfg_file),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Would scan 2 files" in result.output
    assert len(patch_scan_pipeline["do_scan"]) == 0


def test_scan_uses_config_thresholds(tmp_path: Path, tmp_cfg_file: Path) -> None:
    out = tmp_path / "out"
    idx = tmp_path / "idx.native"
    idx.write_text("x", encoding="utf-8")
    (tmp_path / "idx.native.meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "idx.native.attr").write_text("attr", encoding="utf-8")
    (tmp_path / "idx.native.sources.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a.parquet").write_text("x", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "scan",
            "--input",
            str(tmp_path / "*.parquet"),
            "--index",
            str(idx),
            "--out-dir",
            str(out),
            "--config",
            str(tmp_cfg_file),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Using config: {tmp_cfg_file}" in result.output
    assert "ngram=13 min_hits=3 min_coverage=0.001" in result.output


# ----------------------------
# Tests: run
# ----------------------------


def test_run_builds_or_reuses_index_and_scans(
    tmp_path: Path,
    tmp_cfg_file: Path,
    patch_config_loader: None,
    patch_scan_pipeline: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    # Make _ensure_native_index return a fake path without doing real work
    native = tmp_path / "cache" / "index_abc.native"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text("x", encoding="utf-8")

    # Update: accepts kwargs
    monkeypatch.setattr(
        cli, "_ensure_native_index", lambda cfg_path, cache_dir, **kwargs: native
    )

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--config",
            str(tmp_cfg_file),
            "--input",
            str(tmp_path / "*.parquet"),
            "--out-dir",
            str(tmp_path / "out"),
            "--allow-unpinned",  # Add this to pass ensure_native_index logic inside cli
        ],
    )
    assert result.exit_code == 0, result.output

    assert len(patch_scan_pipeline["do_scan"]) == 1
    scan_cfg = patch_scan_pipeline["do_scan"][0]
    assert scan_cfg.index_path == str(native)
    assert scan_cfg.ngram == 13
    assert scan_cfg.min_hits == 3
    assert scan_cfg.min_coverage == 0.001


def test_run_hf_records_target_revision(
    tmp_path: Path,
    tmp_cfg_file: Path,
    patch_config_loader: None,
    patch_scan_pipeline: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = tmp_path / "cache" / "index_abc.native"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli, "_ensure_native_index", lambda *args, **kwargs: native)

    parquet_path = tmp_path / "cached.parquet"
    parquet_path.write_text("x", encoding="utf-8")
    dataset = SimpleNamespace(cache_files=[{"filename": str(parquet_path)}])

    def fake_load_dataset_optional_revision(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["revision"] == "rev123"
        return dataset

    monkeypatch.setattr(
        cli, "_load_dataset_optional_revision", fake_load_dataset_optional_revision
    )

    result = runner.invoke(
        cli.app,
        [
            "run-hf",
            "--config",
            str(tmp_cfg_file),
            "--hf-dataset",
            "demo/ds",
            "--split",
            "train",
            "--hf-revision",
            "rev123",
            "--out-dir",
            str(tmp_path / "out"),
            "--allow-unpinned",
        ],
    )
    assert result.exit_code == 0, result.output

    scan_cfg = patch_scan_pipeline["do_scan"][0]
    assert scan_cfg.target_hf_dataset == "demo/ds"
    assert scan_cfg.target_hf_split == "train"
    assert scan_cfg.target_hf_revision == "rev123"


def test_run_dry_run_skips_native_build(
    tmp_path: Path,
    tmp_cfg_file: Path,
    patch_config_loader: None,
    patch_scan_pipeline: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    def _fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("_ensure_native_index should not be called in dry-run")

    monkeypatch.setattr(cli, "_ensure_native_index", _fail)

    (tmp_path / "shard.parquet").write_text("x", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--config",
            str(tmp_cfg_file),
            "--input",
            str(tmp_path / "*.parquet"),
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Would scan 1 files" in result.output
    assert len(patch_scan_pipeline["do_scan"]) == 0


# ----------------------------
# Tests: build-index + build-index-native
# ----------------------------


def test_build_index_calls_builder(
    tmp_path: Path,
    tmp_cfg_file: Path,
    patch_config_loader: None,
    patch_index_build: dict[str, Any],
):
    out_index = tmp_path / "index.native"
    result = runner.invoke(
        cli.app,
        [
            "build-index",
            "--out-index",
            str(out_index),
            "--config",
            str(tmp_cfg_file),
            "--allow-unpinned",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(patch_index_build["build_index_hf"]) == 1
    idx_cfg = patch_index_build["build_index_hf"][0]
    assert idx_cfg.out_path == str(out_index)
    assert idx_cfg.ngram == 13


# ----------------------------
# Tests: report
# ----------------------------


def test_report_calls_reporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str | None, int, int]] = []

    def fake_report(
        scan_dir: str,
        *,
        out_dir: str | None,
        topk_overall: int,
        topk_per_benchmark: int,
    ) -> Any:
        calls.append((scan_dir, out_dir, topk_overall, topk_per_benchmark))
        return SimpleNamespace(
            leak_by_benchmark=[],
            bad_shards=SimpleNamespace(top_overall=[]),
        )

    monkeypatch.setattr(cli, "generate_report_from_scan_dir", fake_report)

    result = runner.invoke(
        cli.app,
        [
            "report",
            "--scan-dir",
            str(tmp_path / "scan_results"),
            "--out-dir",
            str(tmp_path / "out"),
            "--topk-overall",
            "10",
            "--topk-per-benchmark",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            str(tmp_path / "scan_results"),
            str(tmp_path / "out"),
            10,
            5,
        )
    ]


def test_scan_dry_run_fails_when_index_missing_sidecars(
    tmp_path: Path,
    tmp_cfg_file: Path,
    patch_scan_pipeline: dict[str, Any],
):
    out = tmp_path / "out"
    idx = tmp_path / "idx.native"
    idx.write_text("x", encoding="utf-8")

    # Intentionally omit required sidecars to make _plan_scan fail.
    # (tmp_path / "idx.native.meta.json").write_text("{}", encoding="utf-8")
    # (tmp_path / "idx.native.attr").write_text("attr", encoding="utf-8")
    # (tmp_path / "idx.native.sources.json").write_text("{}", encoding="utf-8")

    (tmp_path / "a.parquet").write_text("x", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "scan",
            "--input",
            str(tmp_path / "*.parquet"),
            "--index",
            str(idx),
            "--out-dir",
            str(out),
            "--config",
            str(tmp_cfg_file),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0, result.output
    assert "Index missing required files" in result.output

    # Ensure dry-run failure doesn't accidentally run the scan pipeline.
    assert len(patch_scan_pipeline["do_scan"]) == 0
    assert len(patch_scan_pipeline["write_run_outputs"]) == 0
