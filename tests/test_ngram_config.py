"""Regression tests for the runtime-configurable ngram normalization config.

The Rust extension no longer hardcodes a config path: ``set_ngram_config_path``
stages a JSON file, and the active config freezes on the first normalization
call. Because the freeze is per-process, every scenario that needs a distinct
config state runs in its own interpreter via :func:`_run`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

import corpus_assay.cli as cli
from corpus_assay.constants import DEFAULT_NGRAM_CONFIG_PATH

_REPO_ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()

# Rich/Typer wraps option names with per-token ANSI color codes when CI
# sets FORCE_COLOR=1, splitting "--ngram-config" into three separately-
# styled chunks. Strip ANSI before substring-matching so the tests are
# robust to that styling.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _run(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter.

    A fresh process is required: the ngram config freezes on first use, so each
    config state must be exercised in its own interpreter. Extra ``args`` are
    passed through and visible to the snippet as ``sys.argv[1:]``.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )


@pytest.fixture
def no_stopwords_config(tmp_path: Path) -> Path:
    """The bundled config with an emptied ``stop_words`` list."""
    cfg = json.loads(DEFAULT_NGRAM_CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["stop_words"], "expected the bundled config to define stop-words"
    cfg["stop_words"] = []
    path = tmp_path / "ngram_config_no_stopwords.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def test_default_config_staged_on_import() -> None:
    """Importing the package stages the bundled config; stop-words are removed."""
    proc = _run(
        """
        import corpus_assay  # stages DEFAULT_NGRAM_CONFIG_PATH on import
        from corpus_assay._native import normalize_text
        print(normalize_text("the a of physics"))
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "['physics']"


def test_override_before_first_use_changes_normalization(
    no_stopwords_config: Path,
) -> None:
    """An override staged before any normalization swaps the active config."""
    proc = _run(
        """
        import sys
        import corpus_assay
        from corpus_assay._native import (
            set_ngram_config_path,
            normalize_text,
            normalization_config_sha,
        )
        set_ngram_config_path(sys.argv[1])
        print(normalize_text("the a of physics"))
        print(normalization_config_sha())
        """,
        str(no_stopwords_config),
    )
    assert proc.returncode == 0, proc.stderr
    tokens, sha = proc.stdout.splitlines()
    # Stop-words are retained now.
    assert tokens == "['the', 'a', 'of', 'physics']"

    # And the fingerprint reflects the overridden config, not the default.
    default = _run(
        """
        import corpus_assay
        from corpus_assay._native import normalization_config_sha
        print(normalization_config_sha())
        """
    )
    assert default.returncode == 0, default.stderr
    assert sha != default.stdout.strip()


def test_override_after_first_use_raises(no_stopwords_config: Path) -> None:
    """Once normalization has run, the config is frozen for the process.

    The override target exists and parses, so the file read and JSON validation
    both succeed -- the failure is specifically the lock, not a read error.
    """
    proc = _run(
        """
        import sys
        import corpus_assay
        from corpus_assay._native import (
            set_ngram_config_path,
            normalize_text,
        )
        normalize_text("freeze the config")
        try:
            set_ngram_config_path(sys.argv[1])
            print("NO_ERROR")
        except RuntimeError:
            print("RUNTIME_ERROR")
        except Exception as exc:  # noqa: BLE001
            print("OTHER:" + type(exc).__name__)
        """,
        str(no_stopwords_config),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "RUNTIME_ERROR"


def test_restage_identical_config_after_freeze_is_noop() -> None:
    """Re-staging the already-frozen config is an idempotent no-op."""
    proc = _run(
        """
        import corpus_assay
        from corpus_assay.constants import DEFAULT_NGRAM_CONFIG_PATH
        from corpus_assay._native import (
            set_ngram_config_path,
            normalize_text,
        )
        normalize_text("freeze the config")
        set_ngram_config_path(str(DEFAULT_NGRAM_CONFIG_PATH))  # same content
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_missing_config_path_raises() -> None:
    """A path that cannot be read raises ValueError."""
    proc = _run(
        """
        import corpus_assay
        from corpus_assay._native import set_ngram_config_path
        try:
            set_ngram_config_path("/tmp/does_not_exist_ngram_config.json")
            print("NO_ERROR")
        except ValueError as exc:
            print("VALUE_ERROR:" + str(exc))
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("VALUE_ERROR:")
    assert "does_not_exist_ngram_config.json" in proc.stdout


def test_invalid_config_json_raises(tmp_path: Path) -> None:
    """A file that does not parse as an NgramConfig raises ValueError."""
    bad = tmp_path / "bad_config.json"
    bad.write_text('{"version": "1.0.0"}', encoding="utf-8")  # missing fields
    proc = _run(
        """
        import sys
        import corpus_assay
        from corpus_assay._native import set_ngram_config_path
        try:
            set_ngram_config_path(sys.argv[1])
            print("NO_ERROR")
        except ValueError as exc:
            print("VALUE_ERROR:" + str(exc))
        """,
        str(bad),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("VALUE_ERROR:")
    assert "invalid ngram config JSON" in proc.stdout


def test_cli_exposes_ngram_config_option() -> None:
    """The global --ngram-config option is wired into the CLI."""
    # Force a wide terminal so Rich doesn't truncate option names mid-string,
    # and strip ANSI because FORCE_COLOR styles each `--`/word with its own
    # escape, which would split the substring.
    result = runner.invoke(cli.app, ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--ngram-config" in _strip_ansi(result.output)


def test_cli_rejects_missing_ngram_config() -> None:
    """--ngram-config validates that the file exists before any command runs."""
    result = runner.invoke(
        cli.app,
        ["--ngram-config", "/tmp/does_not_exist_ngram_config.json", "list-benchmarks"],
    )
    assert result.exit_code != 0
    assert "ngram-config" in _strip_ansi(result.output)
