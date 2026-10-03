from __future__ import annotations

import json
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from corpus_assay._native import hash_ngram, normalize_text
from corpus_assay.constants import DEFAULT_NGRAM_CONFIG_PATH
from corpus_assay.stopgrams import (
    _build_stopgrams_part,
    _merge_parts_to_stopgrams,
    read_stopgrams_header,
)
from corpus_assay.text_utils import iter_ngrams_with_filter

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_stopgrams(path: Path) -> tuple[list[int], int]:
    hashes: list[int] = []
    with path.open("rb") as handle:
        header = read_stopgrams_header(handle)
        for _ in range(header.record_count):
            chunk = handle.read(8)
            hashes.append(struct.unpack("<Q", chunk)[0])
    return hashes, header.record_count


class FakeDataset:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def shard(self, num_shards: int, index: int) -> "FakeDataset":
        return FakeDataset(
            [row for idx, row in enumerate(self._rows) if idx % num_shards == index]
        )


def _hash_token(token: str) -> int:
    tokens = normalize_text(token)
    grams = [ng for ng, allowed, _ in iter_ngrams_with_filter(tokens, n=1) if allowed]
    return hash_ngram(grams[0])


def test_build_stopgrams_direct_merges_parts_per_doc_df(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"text": "alpha alpha beta"},
        {"text": "alpha gamma gamma gamma"},
        {"text": "beta gamma gamma"},
        {"text": "alpha beta"},
    ]

    def fake_loader(*args, **kwargs):
        return FakeDataset(rows)

    monkeypatch.setattr(
        "corpus_assay.stopgrams._load_dataset_optional_revision", fake_loader
    )

    part_paths = []
    stats_paths = []
    for worker_id in range(2):
        part_path = tmp_path / f"part_{worker_id}.bin"
        stats_path = tmp_path / f"part_{worker_id}.json"
        part_paths.append(part_path)
        stats_paths.append(stats_path)
        _build_stopgrams_part(
            dataset="fake",
            local_path=None,
            config_name=None,
            split="train",
            revision=None,
            fields=["text"],
            ngram=1,
            packed_doc_sep="|||",
            packed_doc_sep_typo=None,
            worker_id=worker_id,
            num_workers=2,
            max_docs_per_worker=None,
            sample_prob=None,
            seed=0,
            sketch_size=16,
            out_part_path=part_path,
            out_stats_path=stats_path,
        )

    doc_count_B = 0
    for stats_path in stats_paths:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        doc_count_B += int(payload["doc_count"])

    out_path = tmp_path / "stopgrams.native"
    df_threshold, stop_count = _merge_parts_to_stopgrams(
        part_paths,
        ngram=1,
        doc_count_B=doc_count_B,
        tau=0.75,
        out_path=out_path,
    )

    hashes, count = _read_stopgrams(out_path)
    assert df_threshold == 3
    assert stop_count == count == len(hashes)
    assert hashes == sorted(hashes)
    assert _hash_token("alpha") in hashes
    assert _hash_token("beta") in hashes
    assert _hash_token("gamma") not in hashes


def test_spawn_workers_use_custom_ngram_config(tmp_path: Path) -> None:
    """A custom --ngram-config must reach stop-gram worker processes.

    Regression test: ``build_stopgrams_direct_from_hf_dataset`` spawns its
    workers, which re-import ``corpus_assay`` and only stage the bundled
    default. Without explicit propagation, a parent-process override is silently
    dropped: every worker normalizes with the default config (stripping common
    stop-words), the resulting stop-gram file omits any stop-word the user
    wanted retained, and the meta still records the override's fingerprint --
    silent, hard-to-detect corruption.

    Driven through a subprocess because the config freezes per-process on first
    normalization use, and we need a clean parent that stages a non-default
    config before spawning.
    """
    # Custom config: same as bundled, minus stop-word filtering.
    cfg = json.loads(DEFAULT_NGRAM_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["stop_words"] = []
    custom_cfg = tmp_path / "ngram_config_no_stopwords.json"
    custom_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    # Four-doc corpus: every doc contains "the" so df("the") == doc_count.
    # With workers=2 and tau=0.5, the df threshold is 2; if "the" survives
    # normalization in the workers it should land in the output.
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"text": f"the the {word}"})
            for word in ("alpha", "beta", "gamma", "delta")
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "stopgrams.native"
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            """
            import struct
            import sys

            import corpus_assay  # stages bundled default on import
            from corpus_assay.normalization import set_ngram_config_path
            from corpus_assay._native import (
                hash_ngram,
                normalize_text,
            )
            from corpus_assay.stopgrams import (
                build_stopgrams_direct_from_hf_dataset,
                read_stopgrams_header,
            )
            from corpus_assay.text_utils import iter_ngrams_with_filter


            def main() -> None:
                custom_cfg, corpus_path, out_path = sys.argv[1:4]
                set_ngram_config_path(custom_cfg)
                build_stopgrams_direct_from_hf_dataset(
                    dataset=None,
                    local_path=corpus_path,
                    config_name=None,
                    split="train",
                    revision=None,
                    fields=["text"],
                    ngram=1,
                    packed_doc_sep="<|endoftext|>",
                    packed_doc_sep_typo=None,
                    tau=0.5,
                    out_path=out_path,
                    workers=2,
                    max_docs=None,
                    sample_prob=None,
                    seed=0,
                    tmp_dir=None,
                )

                tokens = normalize_text("the")
                grams = [
                    ng
                    for ng, allowed, _ in iter_ngrams_with_filter(tokens, n=1)
                    if allowed
                ]
                the_hash = hash_ngram(grams[0])

                with open(out_path, "rb") as fh:
                    header = read_stopgrams_header(fh)
                    stored = set()
                    for _ in range(header.record_count):
                        stored.add(struct.unpack("<Q", fh.read(8))[0])

                print("THE_PRESENT:" + str(the_hash in stored))


            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(driver), str(custom_cfg), str(corpus), str(out_path)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "THE_PRESENT:True" in proc.stdout, proc.stdout + "\n" + proc.stderr
