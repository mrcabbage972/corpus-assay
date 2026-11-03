# Configuration reference

A config is a YAML file. `init-config` writes a starter file, and `validate-config`
checks one (`--check-remote` also resolves configs and splits against the Hugging Face
Hub, and checks for hidden labels). Unknown keys are rejected.

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
| `revision` | `null` | Hub commit hash or tag. `build-index` requires it unless `--allow-unpinned` is passed. Registry entries are pinned. |
| `data_files` | `null` | Passed to `datasets.load_dataset`, e.g. `{test: /path/items.jsonl}`. When the dataset card lists no splits, `validate-config --check-remote` takes the split names from these keys (a string or list means `train`). |
| `trust_remote_code` | `false` | Passed to `datasets.load_dataset`. Script-based datasets are not supported by `datasets` 4.x. |
| `label_field` | `null` | The item's label or answer column. Never indexed; used only to check that the indexed splits have public labels (see below). |

### The built-in registry

Each registry entry indexes the split and the text an evaluation harness shows the
model (not the label), pinned to a Hub commit, so `init-config` writes configs that pass
the revision-pin check. `corpus-assay describe-benchmark <key>` prints an entry's spec,
followed by its registry metadata as YAML comments: the split harnesses score, whether
the dataset is gated, the share of short items, and notes.

| Key | Dataset | Config | Split | Fields | Short items |
| --- | --- | --- | --- | --- | ---: |
| `aime24` | `HuggingFaceH4/aime_2024` | | train | problem | 0% |
| `aime25` | `opencompass/AIME2025` | `data_files` I + II | test | question | 3% |
| `amc23` | `math-ai/amc23` | | test | question | 12% |
| `math500` | `HuggingFaceH4/MATH-500` | | test | problem | 32% |
| `gsm8k` | `openai/gsm8k` | main | test | question | 2% |
| `gsm8k_solutions` | `openai/gsm8k` | main | test | question, answer | 0% |
| `jeebench` | `daman1209arora/jeebench` | `data_files` test.json | test | question | 0% |
| `gpqa_diamond` | `Idavidrein/gpqa` (gated) | gpqa_diamond | train | Question, Correct Answer, Incorrect Answer 1–3 | |
| `humaneval` | `openai/openai_humaneval` | | test | prompt | 1% |
| `mbpp` | `google-research-datasets/mbpp` | full | test | text, test_list | 4% |
| `livecodebench` | `livecodebench/code_generation_lite` | `data_files` test.jsonl … test5.jsonl | test | question_content | |
| `ifeval` | `google/IFEval` | | train | prompt | 12% |
| `mmlu` | `cais/mmlu` | all | test | question, choices | 32% |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | | test | question, options | 8% |
| `hellaswag` | `Rowan/hellaswag` | | **validation** | ctx, endings | 0% |
| `commonsenseqa` | `tau/commonsense_qa` | | **validation** | question, choices | 95% |
| `arc` | `allenai/ai2_arc` | ARC-Challenge + ARC-Easy | test | question, choices | 49% |
| `arc_challenge` | `allenai/ai2_arc` | ARC-Challenge | test | question, choices | 37% |
| `arc_easy` | `allenai/ai2_arc` | ARC-Easy | test | question, choices | 55% |
| `piqa` | `baber/piqa` | | **validation** | goal, sol1, sol2 | 68% |
| `boolq` | `aps/super_glue` | boolq | **validation** | passage, question | 1% |
| `winogrande` | `allenai/winogrande` | winogrande_xl | **validation** | sentence, option1, option2 | 85% |
| `openbookqa` | `allenai/openbookqa` | main | test | question_stem, choices | 77% |
| `copa` | `aps/super_glue` | copa | **validation** | premise, choice1, choice2 | 100% |
| `lambada` | `EleutherAI/lambada_openai` | default | test | text | 0% |

Bold splits are the ones lm-eval scores because the test labels are hidden. "Short
items" is the share of items with fewer than 3 13-grams, measured with the bundled
normalization config; blank means not measured. `arc` is both ARC configs as one
benchmark; use `arc_challenge` and `arc_easy` instead to report them apart, not
together with `arc`. `gsm8k` indexes the question only; add `gsm8k_solutions` to also
catch copies of the worked solutions.

**Gated datasets.** `gpqa_diamond` needs a Hub login (`hf auth login`) and the dataset's
terms accepted on its Hub page. Without them it loads nothing, and `build-index` fails
with a message saying so.

### Empty benchmarks

`build-index` (and the index build inside `run` and `run-hf`) fails, writing nothing,
when a benchmark loads no rows or indexes no n-grams. The error names each such
benchmark and the reason: the load error, a missing split, or fields too short for
`ngram`. Pass `--allow-empty-benchmarks` to build the index anyway; the problems are
then printed as warnings. A benchmark that loads only some of its configs or splits is
not an error, but each failure is logged.

The index's `.meta.json` records per-benchmark counts (items, n-grams, items with no
n-grams, items with fewer than `min_hits` n-grams). `build-index` prints them as a
table at the end of the build, and `inspect-index` prints them for an existing index.

**Short items.** N-grams never cross fields, so a field shorter than `ngram`
normalized tokens contributes nothing to the index, and an item with fewer than
`min_hits` n-grams can never be flagged by the n-gram rule. The registry table above
gives the share per benchmark; `build-index` and `inspect-index` count these items for
any index. COPA's fields are one short sentence each, so a 13-gram index contains none
of its items, and `build-index` fails on it unless `--allow-empty-benchmarks` is
passed. For such benchmarks, build a separate index with a smaller `ngram` (for
example 8) and validate its false-positive rate on clean text before relying on it.

### Hidden labels

Several benchmarks hide their test labels (HellaSwag, BoolQ, WinoGrande, COPA,
CommonsenseQA, PIQA among them), so harnesses score their validation split, and an
index built from `test` protects the wrong items. When a spec sets `label_field`,
`build-index` warns if every value of that column in an indexed split is missing, `""`
or `-1`, and names the split harnesses score when the dataset is in the registry.
`validate-config --check-remote` runs the same check on the first 200 rows of each
split. Registry entries set `label_field` where the dataset has one.

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
    packed_doc_sep: "<|endoftext|>" # default; null keeps each document whole
stopgrams:
  enabled: true                     # apply during scan/run/run-hf
  path: ./stopgrams.native
```

`background.corpus` needs exactly one of `dataset` or `local_path`. `local_path` is a
JSONL file. `build-stopgrams-direct --tau T` keeps the n-grams that occur in at least a
`T` fraction of the background documents (for example `--tau 0.001`). Background
records are split into documents on `packed_doc_sep` (and its typo variant
`packed_doc_sep_typo`, default `<|endoftext}>`); set `packed_doc_sep: null` to keep each
record whole. Document
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
| `--text-key` | `text` | Text column; repeat it to scan several (each is its own document). |
| `--messages-key` | none | Chat column of `{role, content}` turns; each turn is its own document. |
| `--id-key` | none | Id column (any type castable to string). Without an id, records use a hash of the text. |
| `--workers` | CPU count | Worker processes (one shard at a time each). |
| `--index-backend` | `auto` | `memory` or `mmap`; `auto` uses mmap when `workers > 1`. |
| `--gate-mode` | `union` | `union` or `item` (see [how-it-works.md](how-it-works.md)). |
| `--min-longest-run` | `0` | Item gate only: also require a contiguous shared run of N tokens. |
| `--boilerplate-threshold` | `50` | Item fan-out at which a matched n-gram counts as boilerplate in attribution. |
| `--max-return-records` | `500` | Flagged records written per shard: a sample of the first N flagged. `0` removes the cap. |
| `--all-records` | off | Write every flagged record (same as `--max-return-records 0`). |
| `--no-split` | off | Do not split records on the packed-document separator: each record is one document, with id `<id>`. |
| `--dry-run` | off | Print what would be scanned and exit. |
