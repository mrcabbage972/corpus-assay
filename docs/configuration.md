# Configuration reference

A config is a YAML file. `init-config` writes a starter file, and `validate-config`
checks one (`--check-remote` also resolves configs and splits against the Hugging Face
Hub). Unknown keys are rejected.

```yaml
benchmarks:            # required, at least one; at most 64 per index
  - ...
ngram: 13              # n-gram length (tokens)
min_hits: 3            # distinct matching n-grams needed to flag a document
min_coverage: 0.001    # matching / distinct n-grams in the document, in [0, 1]
background: ...        # optional: corpus for build-stopgrams-direct
stopgrams: ...         # optional: stop-grams to apply during scans
spaced: ...            # optional, experimental: spaced-seed index to apply during scans
```

## `benchmarks`

Each entry is one of three forms:

```yaml
benchmarks:
  - gsm8k                       # 1. a registry key (see `corpus-assay list-benchmarks`)
  - ref: mmlu                   # 2. a registry key with overrides
    revision: <commit-sha>
  - name: MyBench               # 3. a full spec
    dataset: org/my-bench
    splits: [test]
    fields: [question, answer]
```

Keys of a full spec:

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | (required) | Benchmark label used in attribution and reports. Must be unique. |
| `dataset` | (required) | HF dataset id, or a `datasets` builder such as `json` or `parquet` together with `data_files`. |
| `splits` | (required) | Splits whose items are protected (indexed), e.g. `[test]`. |
| `fields` | (required) | Item fields to index. They are joined with a field separator, and n-grams never cross fields. Lists and dicts are flattened. |
| `configs` | `[]` | Dataset configs to load. `["*"]` expands to all configs (except `all` unless listed). Empty means no config. |
| `config_name` | `null` | A single config; alternative to `configs`. |
| `drop_configs` | `null` | Configs to remove after `*` expansion. |
| `subtraction_splits` | `null` | Splits whose n-grams are *removed* from the index (e.g. `[train]`), so only n-grams specific to the protected split remain. |
| `revision` | `null` | Hub commit hash or tag. `build-index` requires it unless `--allow-unpinned` is passed. |
| `data_files` | `null` | Passed to `datasets.load_dataset`, e.g. `{test: /path/items.jsonl}`. |
| `trust_remote_code` | `false` | Passed to `datasets.load_dataset`. Script-based datasets are not supported by `datasets` 4.x. |

Built-in registry: `arc`, `boolq`, `copa`, `gsm8k`, `hellaswag`, `lambada`, `mmlu`,
`mmlu_pro`, `openbookqa`, `winogrande`. Use `corpus-assay describe-benchmark <key>` to
see each entry's spec.

**Short items.** N-grams never cross fields, so a field shorter than `ngram`
normalized tokens contributes nothing to the index. COPA's fields are one short
sentence each, so a 13-gram index contains none of its items. WinoGrande and OpenBookQA
are only partly covered. `build-index` warns about benchmarks that end up with no
n-grams. For such benchmarks, build a separate index with a smaller `ngram` (for example
8) and validate its false-positive rate on clean text before relying on it.

## `background` and `stopgrams`

Stop-grams are n-grams that are frequent in a clean background corpus. Matches on them
are ignored during scans. Build them once with `build-stopgrams-direct`:

```yaml
background:
  enabled: true
  corpus:
    dataset: wikimedia/wikipedia    # or local_path: /data/background.jsonl
    config_name: 20231101.en
    split: train
    fields: [text]
    revision: null
stopgrams:
  enabled: true                     # apply during scan/run/run-hf
  path: ./stopgrams.native
```

`background.corpus` needs exactly one of `dataset` or `local_path`. `local_path` is a
JSONL file. `build-stopgrams-direct --tau T` keeps the n-grams that occur in at least a
`T` fraction of the background documents (for example `--tau 0.001`). Document
frequency is estimated with a per-worker Space-Saving sketch (`--sketch-size`). The
estimate is a lower bound, so the list errs toward fewer stop-grams. A
`--stopgrams PATH` flag overrides `stopgrams.path`.

Stop-grams must have been built with the same `ngram` and normalization config as the
index. The scan checks this against the `.meta.json` files.

## `spaced` (experimental)

```yaml
spaced:
  enabled: true
  path: ./index.spaced
  min_loci: 3          # distinct seed hits on one (item, diagonal) before verifying
  ver_min_span: 17     # minimum aligned span, in tokens
  ver_identity: 0.85   # minimum token identity over that span
```

See [experimental-spaced-seeds.md](experimental-spaced-seeds.md).

## Normalization config (`--ngram-config`)

The global `--ngram-config PATH` option replaces the bundled
`corpus_assay/ngram_config.json` for one command:

```json
{
  "version": "1.0.0",
  "stop_words": ["a", "an", "and", "the", "..."],
  "word_re": "[a-z0-9]+(?:'[a-z0-9]+)?",
  "ngram_reject_patterns": ["\\boption[s]?\\s*[a-d]\\b", "..."]
}
```

- `word_re` runs on NFKC-normalized, lowercased text.
- `ngram_reject_patterns` are Rust `regex` patterns, matched against each
  space-joined n-gram.

An index records the hash of the config it was built with. `scan`, `run` and `run-hf`
refuse to scan with a different active config unless `--allow-config-mismatch` is
passed. `run`/`run-hf` also key their index cache on that hash.

## Scan options worth knowing

| Option | Default | Meaning |
| --- | --- | --- |
| `--text-key` / `--id-key` | `text` / none | Columns to read. Without an id, records use a hash of the text. |
| `--workers` | CPU count | Worker processes (one shard at a time each). |
| `--index-backend` | `auto` | `memory` or `mmap`; `auto` uses mmap when `workers > 1`. |
| `--gate-mode` | `union` | `union` or `item` (see [how-it-works.md](how-it-works.md)). |
| `--min-longest-run` | `0` | Item gate only: also require a contiguous shared run of N tokens. |
| `--boilerplate-threshold` | `50` | Item fan-out at which a matched n-gram counts as boilerplate in attribution. |
| `--max-return-records` | `500` | Sample records written per shard (`0` disables). |
| `--dry-run` | off | Print what would be scanned and exit. |
