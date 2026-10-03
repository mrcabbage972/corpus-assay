# Architecture

corpus-assay is a Python package (`python/corpus_assay/`) with a compiled Rust
extension (`src/`, imported as `corpus_assay._native`), built with
[maturin](https://www.maturin.rs) and [PyO3](https://pyo3.rs). The extension targets
the CPython stable ABI (`abi3-py312`), so one wheel per platform serves every Python
version from 3.12 up.

## Python side

| Module | Role |
| --- | --- |
| `cli.py` | Typer CLI (`corpus-assay`, `python -m corpus_assay`) |
| `config.py`, `benchmark_spec.py`, `benchmark_registry.py` | YAML config models and the built-in benchmark registry |
| `benchmarks.py` | Loads benchmark items from `datasets` and joins their fields |
| `indexing.py`, `attribution.py`, `formats.py` | Build the index and write its sidecars |
| `scanner/` | Multiprocess scan driver (`runner.py`), per-shard worker (`worker.py`), output layout and I/O |
| `reporting.py` | Leak reports from a scan directory |
| `stopgrams.py` | Stop-gram builder and compatibility checks |
| `spaced_seeds.py`, `spaced_index.py` | Experimental spaced-seed reference implementation and `.spaced` format |
| `normalization.py`, `constants.py` | Normalization-config staging, defaults |
| `services/` | `validate-config`, HF-to-Parquet materialization, index cache, dry-run planning |
| `schemas/` | Pydantic models for every JSON artifact |

## Rust side

| Module | Role |
| --- | --- |
| `lib.rs` | Python module definition (`corpus_assay._native`) |
| `text.rs` | Normalization (NFKC, regex tokens, stop words), reject patterns, BLAKE2b-64 n-gram hashing, normalization-config staging |
| `scan.rs` | The scanner: Parquet batches via Arrow, packed-document splitting, gates, per-item evidence, hit and record output |
| `index.rs`, `mmap_index.rs`, `items_index.rs` | Load the `.native`, `.attr` (in memory or memory-mapped) and `.items` artifacts, with per-process caches |
| `stopgrams.rs` | Memory-mapped stop-gram set |
| `spaced.rs` | Experimental spaced-seed matcher and verifier |
| `formats.rs` | The shared binary header |

The exported functions are `scan_stream_rust`, the `text` utilities
(`normalize_text`, `hash_ngram`, `ngram_allowed`, `ngram_filter`, config hashes,
`set_ngram_config_path`), and two spaced-seed parity helpers used by tests. Their
signatures are in `python/corpus_assay/_native.pyi`.

## Normalization config lifecycle

The Rust extension never hardcodes a config path. On import, `corpus_assay`
stages the bundled `ngram_config.json` through
`corpus_assay.normalization.set_ngram_config_path`. The global `--ngram-config` option
re-stages it before any command runs. The first normalization call freezes the config
for the process, and later changes raise an error.

Scan workers are started with `spawn`, so each one re-imports the package and would see
only the bundled default. The pool's initializer therefore re-stages the parent's
active config path in every worker. The stop-gram builder does the same.

## Process model

`do_scan` collects the input shards and maps them over a `spawn` pool, one shard per
task. Each worker:

1. converts JSONL to Parquet in memory if needed;
2. calls `scan_stream_rust`, which releases the GIL for the scan loop;
3. writes `summaries/<shard>.summary.json`, `per_source_hits/<shard>.hits.bin`, and the
   sampled records.

The parent aggregates the per-shard summaries into a `RunSummary`; the CLI then writes
`_FINAL_SUMMARY.*` and `_RUN_MANIFEST.json`.

## Debugging

Set `CORPUS_ASSAY_SCAN_LOG_EMPTY_FILTERED=1` to log documents that become empty after
stop-gram filtering.
