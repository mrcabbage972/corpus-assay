# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `build-index` fails, writing nothing, when a configured benchmark loads no rows or
  indexes no n-grams (#5). The error names each such benchmark and the reason (the load
  error, a missing split, or fields too short for `ngram`). For a gated dataset it says
  to log in with `hf auth login` and accept the dataset terms. `--allow-empty-benchmarks`
  builds the index anyway, on `build-index`, `run` and `run-hf`.
- Per-benchmark counts in the index metadata (`benchmark_stats`: items, n-grams, items
  with no n-grams, items with fewer than `min_hits` n-grams). `build-index` ends with a
  table of them, and `inspect-index` prints them with the number of items the n-gram rule
  cannot flag (`--min-hits` recounts against another threshold). For older indexes,
  `inspect-index` derives the counts from `.sources.json` and `.items.meta.json`.
- Hidden-label check (#6): a new optional spec key, `label_field`, names the label
  column (never indexed). `build-index` and `validate-config --check-remote` warn when
  every value of it in an indexed split is missing, `""` or `-1`, and name the split
  harnesses score when the dataset is in the registry.
- Registry entries (#14): `aime24`, `aime25`, `amc23`, `math500`, `gsm8k_solutions`,
  `jeebench`, `gpqa_diamond`, `humaneval`, `mbpp`, `livecodebench`, `ifeval`,
  `commonsenseqa`, `arc_challenge`, `arc_easy`, `piqa`. Each entry carries metadata: the
  split harnesses score, whether the dataset is gated, the share of items with fewer than
  3 13-grams, and notes. `describe-benchmark` prints it after the spec, as YAML comments.
- `--all-records` on `scan`, `run` and `run-hf` writes every flagged record to
  `contaminated_docs/` (same as `--max-return-records 0`).
- `--no-split` on `scan`, `run` and `run-hf` turns off packed-document splitting
  (both `<|endoftext|>` and its typo variant): each record is scored as one document
  and its doc id is the plain `<id>`, without a `-part-<k>` suffix. In the config,
  `background.corpus.packed_doc_sep: null` does the same for `build-stopgrams-direct`.
  `ScanConfig.packed_doc_sep` and the Rust `scan_stream_rust` accept `None`, and the
  run manifest records `packed_doc_sep: null`.

### Changed

- **Index metadata schema version 2.** `<index>.native.meta.json` gains
  `benchmark_stats`, and its `benchmark_specs` entries gain `label_field`.
  `INDEX_META_SCHEMA_VERSION` goes from 1 to 2. Version 1 files still load. The binary
  index formats and hashes are unchanged, so existing indexes keep their meaning.
- **Registry entries changed in place.** A config that names these keys now indexes
  different text, so rebuild indexes built from them:
  - `mmlu`: the `all` config instead of every subject config, fields `question, choices`
    (`answer` dropped).
  - `hellaswag`: the `validation` split (test labels are hidden) instead of `test`,
    fields `ctx, endings` (`label` dropped).
  - `arc`: still both ARC configs, fields `question, choices` (`answerKey` dropped).
    The new `arc_challenge` and `arc_easy` index them separately.
  - `gsm8k`: the `question` only. The worked solutions moved to `gsm8k_solutions`.
  - `boolq`: `aps/super_glue` (config `boolq`) instead of `google/boolq`.
  - `copa`: fields `premise, choice1, choice2` (`question` dropped).
  - `lambada`: the `default` config, named explicitly.
- Every registry entry is pinned to a Hub commit, so configs written by `init-config`
  pass the revision-pin check without `--allow-unpinned`.

### Fixed

- The `run`/`run-hf` index cache was keyed on the config file's text, so a config
  naming registry keys reused an index built from the old registry entries after the
  registry changed. The key now also hashes the resolved benchmark specs. Existing
  cached indexes are rebuilt once.
- `validate-config --check-remote` rejected datasets whose card lists no splits even
  though their `data_files` load (for example `opencompass/AIME2025`: "splits not found
  ... Available: []"). It now takes the split names from the `data_files` keys.
- `--max-return-records 0` now removes the cap, as its help text said; it used to
  return no records at all. `ScanConfig(max_return_records=0)` and the Rust scanner
  treat 0 as no cap too, and the docs now say that the default output is a sample of
  the first N flagged sub-documents per shard (`hits.bin` is always exhaustive).
- A script that calls `do_scan` without an `if __name__ == "__main__":` guard now
  fails with a `RuntimeError` that names the guard. The spawned workers used to re-run
  the script and the scan at import, die, and be respawned forever, so the run hung.
- No empty `contaminated_docs/<shard>.decontam.sample.ndjson` is written for a shard
  with no records, and a stale one from an earlier run of the shard is removed.

## [0.1.0] - 2026-10-03

Initial public release.

### Added

- `build-index`: hashed word n-gram index from Hugging Face or local benchmark
  splits, with per-benchmark and per-item attribution sidecars, subtraction splits,
  and provenance metadata (dataset revisions, fingerprints, normalization hashes).
- `scan`, `run`, `run-hf`: multi-process Rust scanner over Parquet and JSONL
  (`.gz`/`.zst`) shards, with an index-wide (`union`) or single-item (`item`) gate,
  resumable per-shard outputs, and sampled per-document evidence.
- `report`: leak tables by benchmark, shard, and item (JSON, CSV, Markdown).
- `build-stopgrams-direct`: stop-grams learned from a background corpus, applied
  with `--stopgrams`.
- `init-config`, `list-benchmarks`, `describe-benchmark`, `validate-config`,
  `inspect-index`, `inspect-run`, `reproduce`.
- Global `--ngram-config` to override the bundled normalization config. Scans refuse
  an index built with a different config unless `--allow-config-mismatch` is passed.
- Experimental: spaced-seed recall channel (`build-spaced-index`, `--spaced-seeds`).
- Prebuilt abi3 wheels for CPython 3.12+ on Linux, macOS, and Windows.

[Unreleased]: https://github.com/mrcabbage972/corpus-assay/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mrcabbage972/corpus-assay/releases/tag/v0.1.0
