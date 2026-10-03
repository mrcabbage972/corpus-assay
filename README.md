<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mrcabbage972/corpus-assay/main/docs/assets/logo-dark.svg">
    <img alt="corpus-assay: audit benchmark overlap" src="https://raw.githubusercontent.com/mrcabbage972/corpus-assay/main/docs/assets/logo.svg" width="560">
  </picture>
</h1>

<p align="center">
  <a href="https://github.com/mrcabbage972/corpus-assay/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/mrcabbage972/corpus-assay/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/corpus-assay/"><img alt="PyPI" src="https://img.shields.io/pypi/v/corpus-assay.svg?logo=pypi&amp;logoColor=white"></a>
  <a href="https://pypi.org/project/corpus-assay/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/corpus-assay.svg?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/mrcabbage972/corpus-assay/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.23125328"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23125328.svg"></a>
</p>

**Find benchmark contamination in LLM training corpora.** corpus-assay builds a hashed
n-gram index from evaluation benchmarks, streams your Parquet/JSONL shards through a
multi-process Rust scanner, and reports which documents overlap which benchmark, down to
the individual test item.

It answers one question: *does my training data contain text from the benchmarks I am
going to report on?*

- **Fast.** The scanner is written in Rust, uses a memory-mapped index, and runs one
  process per core. That's about 80k short documents/s (28 MB/s of text) on 8 cores; see
  [Performance](#performance).
- **Attributed.** Every flagged document carries per-benchmark hit counts. Reports break
  leaks down by benchmark, by shard, and by test item.
- **Auditable.** Indexes and scan runs record the config, the normalization-config hash,
  dataset revisions, and the tool version. `reproduce` re-runs a scan from its manifest
  after checking that the inputs are unchanged.
- **Tunable.** Built-in reject patterns drop multiple-choice boilerplate. Optional
  stop-grams can be learned from a background corpus. An item-level gate is available
  for stricter flagging.

## What it detects (and what it doesn't)

A document is flagged when it shares at least `min_hits` distinct word 13-grams with the
benchmark index **and** those hits cover at least `min_coverage` of the document's
distinct 13-grams. Matching runs on normalized text (Unicode NFKC, lowercased, stop
words removed), so it survives casing, punctuation, and whitespace changes.

This is a *verbatim-overlap* detector. Paraphrases, translations, and heavily edited
copies will not be flagged. An experimental spaced-seed channel
([docs](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/experimental-spaced-seeds.md))
recovers lightly edited copies.

## Performance

These numbers come from a benchmark run on one GCP `c2d-standard-16` VM (AMD EPYC 7B13,
8 physical / 16 logical cores, 63 GB RAM). It scanned 1.56M short documents (0.5 GB of
text, median 28 tokens) in 256 Parquet shards against a benchmark index of 931k
n-grams (22 MB). Each figure is the median of three runs. Peak memory is the summed PSS
of the worker processes.

| Workers | Index backend | Docs/s | MB/s of text | Peak memory |
| ---: | --- | ---: | ---: | ---: |
| 1 | memory | 12,126 | 4.1 | 0.65 GB |
| 8 | memory | 85,487 | 29.3 | 2.30 GB |
| 8 | mmap | 80,357 | 27.5 | 1.91 GB |
| 8 | mmap, with spaced seeds | 54,732 | 18.7 | 4.53 GB |

- **Scaling.** Throughput grows nearly linearly with physical cores: 7.1× at 8 workers
  (mmap). Documents/s depends on document length, so MB/s of text is the more portable
  figure, about 3.5 MB/s per worker.
- **Large indexes.** Use the mmap backend, the default with more than one worker. It
  shares one copy of the index across workers. With a 1.6 GB index at 8 workers, peak
  memory was 2.2 GB with mmap and 29.9 GB with the in-memory backend.
- **Contamination rate.** Throughput barely changes as contamination rises: from 80.4k
  to 78.8k docs/s between 0% and 10% contaminated documents.
- **Not included.** The memory figures exclude the per-item postings sidecar (`.items`).
  Each worker loads it into memory to attribute flagged documents to individual items.

## Install

```bash
pip install corpus-assay
```

Prebuilt wheels cover CPython 3.12+ on Linux (x86_64, aarch64), macOS (arm64, x86_64),
and Windows (x64). On other platforms pip builds from source, which needs a
[Rust toolchain](https://rustup.rs).

## Quickstart

**1. Write a config.** Start from the built-in benchmark registry (`corpus-assay
list-benchmarks` shows what's in it):

```bash
corpus-assay init-config --benchmarks gsm8k --benchmarks mmlu --out config.yaml
```

This produces a config you can edit:

```yaml
benchmarks:
  - name: GSM8K
    dataset: openai/gsm8k
    configs: [main]
    splits: [test]
    fields: [question, answer]
    revision: null        # pin a Hub commit for reproducible indexes
  # ...
ngram: 13
min_hits: 3
min_coverage: 0.001
```

**2. Build the index.** This downloads the benchmark splits from the Hugging Face Hub:

```bash
corpus-assay build-index --config config.yaml --out-index bench.native --allow-unpinned
```

Without `--allow-unpinned`, benchmarks that have no `revision` are rejected. Pin
revisions in the config to make the index reproducible, then drop the flag.

**3. Scan your corpus.** Shards can be Parquet files or JSONL files, optionally `.gz` or
`.zst` compressed. Pass directories, globs, or files:

```bash
corpus-assay scan --config config.yaml --index bench.native \
  --input "data/**/*.parquet" --id-key id --out-dir scan_out
```

**4. Report.**

```bash
corpus-assay report --scan-dir scan_out
```

```
Top benchmarks by leak fraction:
  GSM8K: 0.11% (85/76252)
Top leaking shards:
  shard-0: 85 unique hashes (85.00/1k docs)
```

Each flagged document is written as a sample record to
`scan_out/contaminated_docs/*.ndjson`:

```json
{"doc_id": "doc500-part-0", "match_count": 85, "src_hits": [[0, 85]],
 "item_hits": [{"item_id": 7, "hits": 85, "longest_run_tokens": 75, "unique_hits": 85, ...}]}
```

### One-command variants

```bash
# Build (or reuse a cached) index, then scan local shards
corpus-assay run --config config.yaml --input data/ --out-dir scan_out --allow-unpinned

# Same, but scan a Hugging Face dataset split (materialized to Parquet first)
corpus-assay run-hf --config config.yaml --hf-dataset allenai/c4 --split validation \
  --hf-revision <commit> --out-dir scan_out --allow-unpinned
```

### Benchmarks from local files

Any `datasets` loader works, including local JSON or Parquet files. Nothing touches the
network:

```yaml
benchmarks:
  - name: InternalQA
    dataset: json
    data_files: {test: /data/internal_qa.jsonl}
    splits: [test]
    fields: [question, answer]
```

## Commands

| Command | Purpose |
| --- | --- |
| `init-config` | Write a starter config from registry benchmarks |
| `list-benchmarks`, `describe-benchmark` | Browse the built-in benchmark registry |
| `validate-config` | Validate a config (`--check-remote` resolves configs and splits on the Hub) |
| `build-index` | Build the n-gram index (`.native` + sidecars) |
| `inspect-index` | Validate an index and its sidecars |
| `scan` | Scan Parquet/JSONL shards against an index |
| `run`, `run-hf` | Build or reuse a cached index, then scan local shards or an HF split |
| `report` | Leak reports (JSON, CSV, Markdown) from a scan directory |
| `inspect-run` | Validate a scan output directory |
| `reproduce` | Re-run a scan from its `_RUN_MANIFEST.json` after checking input hashes |
| `build-stopgrams-direct` | Learn stop-grams (frequent background n-grams) from a corpus |
| `build-spaced-index` | *Experimental:* build the spaced-seed index |

`corpus-assay <command> --help` shows every option. The full reference is in
[docs/cli.md](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/cli.md).

## Outputs

```
scan_out/
├── _FINAL_SUMMARY.json / .csv      # totals: docs scanned, contaminated, failed files
├── _RUN_MANIFEST.json              # full scan config, input hashes, tool versions
├── summaries/<shard>.summary.json  # per-shard counts (also the resume cache)
├── per_source_hits/<shard>.hits.bin
├── contaminated_docs/<shard>.decontam.sample.ndjson   # up to 500 records per shard
└── report/                         # written by `report`
    ├── report.md / report.json
    ├── leak_by_benchmark.csv / .json
    ├── leak_by_shard.csv / .json
    └── leak_by_item.csv / .json
```

Re-running `scan` with the same inputs and config skips shards that are already done.
File formats are documented in
[docs/outputs.md](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/outputs.md).

## Reducing false positives

- **Reject patterns.** The bundled normalization config drops n-grams that are
  multiple-choice scaffolding ("which of the following", "option a", ...). You can supply
  your own with `corpus-assay --ngram-config my_config.json <command>`. The override must
  be used for both `build-index` and `scan`; a mismatch is detected and refused.
- **Stop-grams.** Build a list of n-grams that are frequent in a clean background corpus
  (for example Wikipedia), then pass it with `--stopgrams`. Matches on those n-grams are
  ignored:

  ```yaml
  background:
    enabled: true
    corpus: {dataset: wikimedia/wikipedia, config_name: 20231101.en, split: train, fields: [text]}
  ```

  ```bash
  corpus-assay build-stopgrams-direct --config config.yaml --tau 0.001 \
    --out stopgrams.native --max-docs 1000000
  corpus-assay scan ... --stopgrams stopgrams.native
  ```

- **Item gate.** `--gate-mode item` flags a document only if a *single* benchmark item
  clears the thresholds on its own, rather than the union of all matches.

## Python API

```python
from pathlib import Path

from corpus_assay import build_index_hf, do_scan
from corpus_assay.config import (
    BuildIndexConfig,
    ScanConfig,
    load_decontamination_config,
)
from corpus_assay.scanner import OutputLayout, write_run_outputs

if __name__ == "__main__":  # required: scan workers are spawned processes
    cfg = load_decontamination_config("config.yaml")
    build_index_hf(
        BuildIndexConfig(
            out_path="bench.native",
            ngram=cfg.ngram,
            benchmarks=cfg.benchmarks,
            allow_unpinned=True,
        )
    )
    scan_cfg = ScanConfig(
        inputs=["data/"],
        text_key="text",
        id_key="id",
        index_path="bench.native",
        out_dir="scan_out",
        ngram=cfg.ngram,
        min_hits=cfg.min_hits,
        min_coverage=cfg.min_coverage,
    )
    run = do_scan(scan_cfg)
    write_run_outputs(
        layout=OutputLayout(Path(scan_cfg.out_dir)), run=run, cfg=scan_cfg
    )
    print(run.total_contaminated, run.contamination_rate)
```

## Limitations

- At most 64 benchmarks per index, because attribution is stored as a 64-bit mask. Use
  several indexes for more.
- The text column must be a plain UTF-8 string column. Parquet `large_string` columns are
  rejected; cast them first.
- JSONL shards are loaded fully into memory and converted to Parquet before scanning.
  Use Parquet for very large shards.
- Documents are split on `<|endoftext|>`, and each part is scored separately. Records are
  named `<id>-part-<k>`.
- Only verbatim overlap is detected; see [above](#what-it-detects-and-what-it-doesnt).

## Documentation

- [Configuration reference](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/configuration.md)
- [How matching works](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/how-it-works.md)
- [Output and file formats](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/outputs.md)
- [CLI reference](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/cli.md)
- [Experimental: spaced seeds](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/experimental-spaced-seeds.md)
- [Architecture](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/architecture.md)
- [Relation to the MixtureVitae decontamination](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/lineage.md)

## Background

corpus-assay grew out of the decontamination pipeline used for the 13-gram
contamination analysis of [MixtureVitae](https://arxiv.org/abs/2509.25531) (Appendix D).
It keeps that pipeline's normalization, hashing and flagging thresholds, but some
changes alter results, so default runs do not reproduce the paper's numbers exactly.
[docs/lineage.md](https://github.com/mrcabbage972/corpus-assay/blob/main/docs/lineage.md)
lists the changes and how to get close to the original protocol.

## Citing

If you use corpus-assay in your research, please cite it using
[CITATION.cff](https://github.com/mrcabbage972/corpus-assay/blob/main/CITATION.cff)
(GitHub's "Cite this repository" button). Every release is archived on Zenodo with its
own DOI. The concept DOI [10.5281/zenodo.23125328](https://doi.org/10.5281/zenodo.23125328)
covers all versions.

## Contributing

See [CONTRIBUTING.md](https://github.com/mrcabbage972/corpus-assay/blob/main/CONTRIBUTING.md).
Bug reports and pull requests are welcome.

## License

Apache-2.0. See [LICENSE](https://github.com/mrcabbage972/corpus-assay/blob/main/LICENSE).
