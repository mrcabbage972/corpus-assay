# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Unreleased

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
