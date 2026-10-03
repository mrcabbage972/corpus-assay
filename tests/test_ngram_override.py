"""A global ``--ngram-config`` override must reach every stage of a run.

Covers three ways the override used to be dropped: the ``run``/``run-hf`` index
cache key hashed the bundled config, spawned scan workers re-imported the
package and normalized with the bundled default, and nothing stopped a scan
against an index built under a different config.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

import corpus_assay.cli as cli
import corpus_assay.normalization as normalization
from corpus_assay.constants import DEFAULT_NGRAM_CONFIG_PATH
from corpus_assay.services.index_cache import index_cache_path

_REPO_ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


@pytest.fixture
def no_stopwords_config(tmp_path: Path) -> Path:
    """The bundled config with an emptied ``stop_words`` list."""
    cfg = json.loads(DEFAULT_NGRAM_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["stop_words"] = []
    path = tmp_path / "ngram_config_no_stopwords.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def test_index_cache_key_tracks_active_config(
    tmp_path: Path, no_stopwords_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("ngram: 13\n", encoding="utf-8")
    default_key = index_cache_path(config, tmp_path)

    monkeypatch.setattr(
        normalization, "_active_ngram_config_path", str(no_stopwords_config)
    )
    assert index_cache_path(config, tmp_path) != default_key


def test_spawned_scan_workers_use_the_override(
    tmp_path: Path, no_stopwords_config: Path
) -> None:
    """Every 3-gram of the leaked text contains a stop word, so it only matches the
    index when the workers keep stop words, i.e. when they honor the override."""
    code = """
        import sys
        from pathlib import Path

        from corpus_assay.normalization import set_ngram_config_path

        set_ngram_config_path(sys.argv[1])
        tmp = Path(sys.argv[2])
        sys.path.insert(0, "tests")

        import pyarrow as pa
        import pyarrow.parquet as pq
        from _index_helpers import write_native_index

        from corpus_assay._native import hash_ngram, ngram_allowed, normalize_text
        from corpus_assay.config import ScanConfig
        from corpus_assay.scanner.runner import do_scan
        from corpus_assay.text_utils import iter_ngrams

        text = "the alpha of beta the gamma of delta"
        grams = [g for g in iter_ngrams(normalize_text(text), 3) if ngram_allowed(g)]
        hashes = {hash_ngram(g) for g in grams}
        index = tmp / "idx.native"
        write_native_index(
            index, ngram=3, hashes=hashes, sources={h: "bench" for h in hashes}
        )
        (tmp / "in").mkdir()
        pq.write_table(
            pa.table({"text": [text, "completely unrelated words here"]}),
            tmp / "in" / "a.parquet",
        )
        cfg = ScanConfig(
            inputs=[str(tmp / "in")],
            text_key="text",
            id_key=None,
            index_path=str(index),
            out_dir=str(tmp / "out"),
            ngram=3,
            min_hits=1,
            min_coverage=0.0,
            workers=2,
        )
        run = do_scan(cfg, print_progress=False)
        print(run.failed_files, run.total_contaminated)
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code),
            str(no_stopwords_config),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    failed, contaminated = proc.stdout.split()[-2:]
    assert (failed, contaminated) == ("0", "1")


def _index_with_config_hash(tmp_path: Path, config_hash: str) -> Path:
    index = tmp_path / "idx.native"
    index.write_bytes(b"")
    meta = {"ngram": 13, "ngram_config_hash": config_hash}
    index.with_name("idx.native.meta.json").write_text(json.dumps(meta))
    return index


def test_scan_refuses_index_built_with_another_config(tmp_path: Path) -> None:
    index = _index_with_config_hash(tmp_path, "0" * 64)
    config = tmp_path / "config.yaml"
    config.write_text(
        "benchmarks:\n"
        "  - name: x\n"
        "    dataset: x\n"
        "    splits: [test]\n"
        "    fields: [text]\n",
        encoding="utf-8",
    )
    args = [
        "scan",
        "--input",
        str(tmp_path),
        "--index",
        str(index),
        "--out-dir",
        str(tmp_path / "out"),
        "--config",
        str(config),
    ]
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0
    assert "different ngram normalization config" in result.output

    called: list[object] = []
    original = cli._run_scan
    try:
        cli._run_scan = lambda scan_cfg, out_dir: called.append(scan_cfg)  # type: ignore[assignment]
        result = runner.invoke(cli.app, [*args, "--allow-config-mismatch"])
    finally:
        cli._run_scan = original
    assert result.exit_code == 0, result.output
    assert called
