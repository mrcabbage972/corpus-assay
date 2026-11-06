"""A script that calls do_scan() without a __main__ guard fails fast (#13).

The scanner's spawn pool re-imports the main script in each worker. Without the
guard every worker re-runs the scan while it starts up; multiprocessing kills it,
the pool respawns it, and the run used to hang. The parent now raises.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_SETUP = """
import sys
from pathlib import Path

sys.path.insert(0, {tests_dir!r})

import pyarrow as pa
import pyarrow.parquet as pq

from _index_helpers import write_native_index
from corpus_assay.config import ScanConfig
from corpus_assay.scanner import do_scan

base = Path({base!r})


def scan():
    base.mkdir(parents=True, exist_ok=True)
    idx = write_native_index(base / "idx.native", ngram=3, hashes={{1, 2, 3}})
    shards = []
    for name in ("a", "b"):
        shard = base / f"{{name}}.parquet"
        pq.write_table(pa.table({{"text": ["alpha beta gamma delta epsilon"]}}), shard)
        shards.append(str(shard))
    cfg = ScanConfig(
        inputs=shards,
        text_key="text",
        id_key=None,
        index_path=idx,
        out_dir=str(base / "out"),
        ngram=3,
        workers=2,
    )
    print("ok_files", do_scan(cfg, print_progress=False).ok_files)
"""


def _run(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "script.py"
    script.write_text(
        _SETUP.format(tests_dir=str(TESTS_DIR), base=str(tmp_path / "work"))
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )


def test_unguarded_script_raises_instead_of_hanging(tmp_path: Path) -> None:
    try:
        proc = _run(tmp_path, "\nscan()\n")
    except subprocess.TimeoutExpired:
        pytest.fail("unguarded do_scan() hung instead of raising")

    assert proc.returncode != 0
    last_line = proc.stderr.strip().splitlines()[-1]
    assert last_line.startswith("RuntimeError: corpus_assay do_scan() was called")
    assert 'if __name__ == "__main__":' in last_line


def test_guarded_script_scans(tmp_path: Path) -> None:
    proc = _run(tmp_path, '\nif __name__ == "__main__":\n    scan()\n')

    assert proc.returncode == 0, proc.stderr
    assert "ok_files 2" in proc.stdout
