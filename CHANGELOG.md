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
- Registry entries (#14): `aime24`, `aime25`, `amc23`, `math500`, `gsm8k_solutions`,
  `jeebench`, `gpqa_diamond`, `humaneval`, `mbpp`, `livecodebench`, `ifeval`,
  `commonsenseqa`, `arc_challenge`, `arc_easy`, `piqa`. Each entry carries metadata: the
  split harnesses score, whether the dataset is gated, the share of items with fewer than
  3 13-grams, and notes. `describe-benchmark` prints it after the spec, as YAML comments.

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
