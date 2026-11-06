# Outputs and file formats

All binary files are little-endian. Every JSON sidecar carries a `schema_version`.

## Index artifacts (`build-index`)

| File | Contents |
| --- | --- |
| `<index>.native` | Header (type 1) + sorted `u64` n-gram hashes |
| `<index>.native.attr` | Header (type 2) + `<u64 hash, u64 benchmark_mask>` records, sorted by hash |
| `<index>.native.items` | Item postings, format v2: header (type 5) + per hash, sorted: `<u64 hash, u32 n, n × (u32 item_idx, u32 field_mask)>` |
| `<index>.native.items.meta.json` | Item table: `(benchmark, item_id, n_protected_hashes)` per item index, plus `fields`: each benchmark's field names, in spec order |
| `<index>.native.items.text.jsonl` | Item text: one JSON line per item, `{"item_idx", "benchmark", "item_id", "fields"}`, with the raw field texts (see below). Read by `--exact-channel` and `--confirm` |
| `<index>.native.item_stopgrams` | With `stopgrams.items`: the n-grams dropped as item stop-grams, in the stop-grams format (`CASTOP1` header, sorted `u64` hashes) |
| `<index>.native.sources.json` | Benchmark legend: `id_to_name`, `name_to_id`, per-benchmark hash totals |
| `<index>.native.meta.json` | Provenance: tool versions, `ngram`, normalization config hash/SHA, hash function, benchmark specs, HF dataset fingerprints and resolved revisions, and per-benchmark counts (`benchmark_stats`) |

`.meta.json` is at `schema_version` 2. Version 2 adds `benchmark_stats`, one entry per
benchmark: `name`, `items`, `ngrams` (distinct hashes attributed to it),
`items_zero_ngrams`, `items_below_min_hits` (items with fewer than `min_hits` n-grams,
those with none included) and the `min_hits` used. Version 1 files have no
`benchmark_stats` and still load; `inspect-index` then derives what it can from
`.sources.json` and `.items.meta.json`. An index built with `stopgrams.items` also
records `item_stopgrams`: `max_item_share`, `min_count`, `n_items` (the items the
share was taken of), `dropped` (the number of n-grams removed) and `file`; the
`benchmark_stats` are counted after the drop.

Bit `i` of a benchmark mask is set when the n-gram occurs in benchmark `id_to_name[i]`.

### Item postings (`.items`)

Each posting names an item (its index in `.items.meta.json`) and, in format v2, a
field mask: bit `k` is set when the n-gram occurs in the item's `k`-th spec field
(`fields[<benchmark>][k]`; fields past the 32nd share bit 31). Each field is hashed on
its own at build time, so an n-gram belongs to one field of an item, or to several
when the fields repeat text. An item's `n_protected_hashes` is its number of postings.

The two formats are told apart by the header's file type, not its format version
(which stays 1): type 4 is v1, `<u64 hash, u32 n, n × u32 item_idx>`, written by
0.1.x; type 5 is v2. Both load. A v1 sidecar has no field information, so records
scanned against it carry no `field_hits`, and `report` skips the field-match table;
rebuild the index to get them. `run` / `run-hf` rebuild a cached index whose `.items`
is v1.

### Item text (`.items.text.jsonl`)

One line per indexed item, in item order (`item_idx` is the item's index in
`.items.meta.json`, the `item_id` of the records):

```json
{"item_idx": 0, "benchmark": "mmlu", "item_id": "mmlu#0",
 "fields": ["What is the capital of France?", ["Paris", "Lyon", "Nice", "Lille"]]}
```

`fields` holds one entry per spec field, in spec order, before any normalization: the
field's text (`""` when empty), or a list of strings when the field holds several
texts (a list of multiple-choice options, or a `{"text": [...], "label": [...]}`
mapping), so each option can be checked on its own. The exact channel matches the
first field. Indexes built before this sidecar existed lack it: they still scan, but
`--exact-channel` and `--confirm` stop with an error asking for a rebuild, and `run` /
`run-hf` rebuild a cached index without it. `inspect-index` says whether it is present.

### Common binary header (36 bytes)

Used by `.native`, `.attr`, `.items`, and `hits.bin`.

| Offset | Type | Field |
| --- | --- | --- |
| 0 | `8s` | magic `CAFMTv1\0` |
| 8 | `u32` | file type: 1 hashes, 2 attribution masks, 3 hit pairs, 4 item postings v1, 5 item postings v2 |
| 12 | `u32` | format version (`1`) |
| 16 | `u32` | endianness marker `0x01020304` |
| 20 | `u32` | n-gram length |
| 24 | `u32` | reserved (`0`) |
| 28 | `u64` | record count |

Python readers and writers live in `corpus_assay/formats.py`, and the Rust ones in
`src/formats.rs`.

## Stop-grams (`build-stopgrams-direct`)

- `<stopgrams>.native`: a 24-byte header (`8s` magic `CASTOP1\0`, `u32` version,
  `u32` n-gram length, `u64` count), followed by sorted `u64` hashes.
- `<stopgrams>.native.meta.json`: background corpus, `tau`, document count, threshold,
  and the normalization hashes used for the compatibility check.

## Spaced-seed index (`build-spaced-index`, experimental)

`<index>.spaced` has this layout:

```
header   8s magic "CASPACE1", u32 version, u32 span, u32 weight, u32 n_patterns,
         u32 exact_n, u32 n_items, u64 n_postings
patterns n_patterns × span ASCII bytes ('0'/'1')
tokens   u32 n_tokens, then per token: u32 byte_len + UTF-8 bytes   (intern table)
items    per item: u32 n_item_tokens + n × u32 token_id
postings sorted by hash, per hash: u64 seed_hash, u32 n, n × (u32 item_id, u32 item_pos)
```

`<index>.spaced.meta.json` records the seed family, the verifier parameters, and the
normalization hashes.

## Scan outputs (`scan`, `run`, `run-hf`)

```
<out_dir>/
├── _FINAL_SUMMARY.json / .csv
├── _RUN_MANIFEST.json
├── summaries/<shard>.summary.json
├── per_source_hits/<shard>.hits.bin
├── per_item_best/<shard>.json
└── contaminated_docs/<shard>.decontam.sample.ndjson
```

- **`_FINAL_SUMMARY.json`** holds the run totals: `total_files`, `failed_files`,
  `total_docs_scanned`, `total_contaminated`, `contamination_rate`,
  `total_empty_after_filter` and `total_contaminated_by_channel`, the flagged
  sub-documents per enabled channel (`{"ngram": 120, "exact": 14}`; `spaced` with
  `--spaced-seeds`). A sub-document flagged by several channels counts in each, so
  the channel counts can add up to more than `total_contaminated`. Documents are
  counted per packed sub-document (of each scanned field or chat turn).
- **`_RUN_MANIFEST.json`** describes the run:
  - `index`: the index path, and the SHA-256 of each index file (`native_sha256`,
    `meta_sha256`, `attr_sha256`, `sources_sha256`, `items_sha256`,
    `items_meta_sha256`, `items_text_sha256`, `item_stopgrams_sha256`; `null` for a
    file the index lacks); from the index metadata, `normalization_config_sha`,
    `ngram_config_hash`, the indexed `benchmarks` (`name`, `hf_repo`,
    `config_name`, `split`, the requested `revision`, the `resolved_revision` and
    the `dataset_fingerprint`) and the per-benchmark counts (`benchmark_stats`).
  - `scan_config`: the settings that decide what is flagged: `text_keys`,
    `messages_key`, `id_key`, `n`, `min_hits`, `min_coverage`,
    `min_item_coverage`, `gate_mode`, `min_longest_run`, `boilerplate_threshold`,
    `exact_channel`, `exact_min_chars`, `confirm`, `packed_doc_sep` (`null` under
    `--no-split`) and `stopgrams_path`.
  - `config` (the full scan config), `inputs` (the input paths, and the SHA-256 of
    each Parquet input), tool versions, and stop-gram provenance.

  `reproduce` reads it, re-checks the recorded Parquet input, index metadata and
  (when recorded) `.native` hashes, and re-runs the scan. Manifests written before
  the index file hashes, benchmarks and gate settings were recorded lack them
  (`null` or empty) and still load in `reproduce`, `inspect-run` and `report`.
- **`summaries/<shard>.summary.json`** holds per-shard counts (including
  `contaminated_by_channel`, as in the final summary), an `error` field when the
  shard failed, and a run fingerprint. A re-run with the same fingerprint skips the
  shard. The fingerprint covers the index (including `.items`, `.items.meta.json`,
  `.items.text.jsonl` and `.item_stopgrams`), config, and normalization config.
  Summaries written before `contaminated_by_channel` existed still load.
- **`per_source_hits/<shard>.hits.bin`** has a header of type 3, then
  `<u64 hash, u64 mask>` for **every** distinct matched hash in the shard. `report`
  aggregates these files. They hold n-gram matches only: exact-channel matches
  appear in the records and the per-channel counts, not here.
- **`per_item_best/<shard>.json`** (indexes with `.items`) lists, for every item with
  at least one hit in the shard, the sub-document holding the largest share of its
  n-grams, over every scanned sub-document (flagged or not):
  `{"schema_version": 1, "items": [{"item_id": 7, "hits": 85, "item_ngrams": 90,
  "coverage": 0.944, "doc_id": "doc500-part-0"}]}`. Ties keep the first
  sub-document. `report` merges the shards into `best_doc_coverage`.
- **`contaminated_docs/<shard>.decontam.sample.ndjson`** holds one JSON object per
  flagged sub-document. **By default this is a sample:** the first
  `--max-return-records` (500) flagged sub-documents of each shard, in scan order.
  Pass `--max-return-records 0` or `--all-records` to write every flagged record (for
  an audit, or to remove the flagged documents). No file is written for a shard with no
  records. The counts in the summaries and the `hits.bin` files are exhaustive
  either way.

  ```json
  {"schema_version": 3, "doc_id": "doc500-part-0", "match_count": 85,
   "src_hits": [[0, 85]],
   "item_hits": [{"item_id": 7, "hits": 85, "longest_run_tokens": 75,
                  "unique_hits": 85, "shared_hits": 0, "boilerplate_hits": 0,
                  "item_ngrams": 90, "item_coverage": 0.944,
                  "field_hits": [[0, 60], [1, 25]],
                  "fields_matched": ["question", "choices"]}],
   "channels": ["ngram"], "field": "text", "part": 0, "row_index": 417}
  ```

  A short item flagged by the exact channel, with `--confirm`:

  ```json
  {"schema_version": 3, "doc_id": "doc9-part-0", "match_count": 0, "src_hits": [],
   "exact_hits": [{"item_id": 3, "field": 0, "choices_found": 4, "choices_total": 4,
                   "numbers_total": 2, "numbers_matched": 2, "lcs_chars": 31,
                   "lcs_ratio": 1.0}],
   "channels": ["exact"], "field": "text"}
  ```

  A flagged turn of a chat column (`--messages-key messages`):

  ```json
  {"schema_version": 3, "doc_id": "conv42/messages/3-part-0", "match_count": 40,
   "src_hits": [[0, 40]], "field": "messages", "role": "assistant", "turn": 3}
  ```

  - `doc_id` names the sub-document:

    | Scan | `doc_id` |
    | --- | --- |
    | one `--text-key`, no `--messages-key` | `<id>-part-<k>` |
    | several `--text-key`s, or `--messages-key` | `<id>/<field>-part-<k>`, or `<id>/<field>/<turn>-part-<k>` for a chat turn |
    | `--no-split` | as above, without `-part-<k>` |
    | no `--id-key` (or an empty id) | a hash of the sub-document text |

    `<k>` counts the packed parts of one text from 0. Non-string ids (integers, ...)
    are cast to strings.
  - `src_hits` contains `[benchmark_id, hits]` pairs.
  - `item_hits` is present when the index has item postings: one entry per item
    that clears the gate on its own, or that passes the `--min-item-coverage` path
    (`item_id` indexes `.items.meta.json`). `item_ngrams` is the item's number of
    indexed n-grams and `item_coverage = hits / item_ngrams`, the share of the item
    the document holds (as opposed to `min_coverage`, a share of the document).
    `field_hits` lists `[field_idx, hits]` for each of the item's spec fields with a
    hit (a hash in two fields of the item counts in both), and `fields_matched`
    names them, so a copy of a whole multiple-choice item (question and choices)
    reads differently from one of the question alone. Both are absent with a v1
    `.items` sidecar.
  - `spaced_item` appears when the experimental spaced-seed channel flagged the
    document.
  - `channels` lists the channels that flagged the sub-document: `ngram` (the
    n-gram gate), `spaced`, `exact`.
  - `exact_hits` (with `--exact-channel`) lists the items whose question (`field`
    0) occurs verbatim in the sub-document, after light normalization and on word
    boundaries. `match_count` and `src_hits` stay n-gram counts, so a record flagged
    by the exact channel alone can have `match_count` 0.
  - Text evidence, on `item_hits` and `exact_hits` entries, needs the item text
    sidecar and is computed for the written records only. With `--exact-channel` or
    `--confirm`: `choices_found` / `choices_total`, the item's other fields (each
    option of a list field on its own) found within 1,000 characters of the
    matched question (anywhere in the sub-document when the question does not occur
    verbatim). With `--confirm`: `numbers_total` / `numbers_matched`, the distinct
    numbers of the item's question and how many occur in the sub-document, and
    `lcs_chars` / `lcs_ratio`, the longest common substring of the question and the
    sub-document, in characters and as a share of the question.
  - `field` is the column the sub-document came from (a text key, or the messages
    key). For a chat turn, `role` is the turn's role (omitted when it has none) and
    `turn` its index in the messages list, counted from 0.
  - `row_index` is the record's row in the shard, counted from 0 (for JSONL, the
    index among the non-blank lines: blank and whitespace-only lines are not rows),
    and `part` the index of the packed part within the field or turn (the `<k>` of
    `-part-<k>`; absent under `--no-split`). Together with `field` and `turn` they
    locate the sub-document without parsing the doc id; `filter` uses them.
  - Schema history: v3 added `field`, `role` and `turn`, and later the optional
    `item_hits` fields `item_ngrams`, `item_coverage`, `field_hits` and
    `fields_matched`, then `channels`, `exact_hits` and the text evidence fields,
    then `row_index` and `part`; v2 added `item_hits` and
    `spaced_item`. Readers (`inspect-run`, `report`) still accept v2 records.

## Filtered data (`filter`)

```
<out_dir>/
├── <shards>                  # cleaned copies, at the inputs' relative paths
├── _FILTER_MANIFEST.json
├── removed_ids.ndjson
├── DATASET_CARD_SNIPPET.md
└── _scan/                    # the scan, as written by `scan` (every flagged record)
```

- **Cleaned shards** keep the input's format and its path relative to the common
  directory of all input shards (`data/a/x.parquet` and `data/b/y.jsonl` become
  `<out_dir>/a/x.parquet` and `<out_dir>/b/y.jsonl`). Parquet keeps the schema,
  its metadata, every column and the compression codec of the first column; JSONL
  (`.gz`, `.zst`) keeps every line it does not remove byte for byte, blank lines
  included, and the compression. With `--granularity subdoc`, a cleaned JSONL line is
  re-serialized (`json.dumps`, non-ASCII kept), and flagged packed parts are dropped
  and the rest rejoined with the separator (a typo separator `<|endoftext}>` comes
  back as `<|endoftext|>`); a text field left with nothing is set to `""`, a chat
  turn left with nothing is removed from the messages list, and a row left with no
  text in any scanned column is dropped. Input files are never modified.
- **`_FILTER_MANIFEST.json`** (`schema_version` 1):
  - `index` and `scan_config`, as in `_RUN_MANIFEST.json`: the index files'
    SHA-256, the benchmarks with their revisions, resolved revisions and splits,
    the per-benchmark item and n-gram counts, the normalization and n-gram config
    hashes, the gate, thresholds and channels;
  - `granularity`, `config_path`, `created_at`, `library_version`,
    `rust_extension_version`, `input_paths` and `config` (the full scan config);
  - `shards`: per input shard, `input_path`, `input_sha256`, `output_path`
    (relative to `<out_dir>`), `output_sha256`, `format`, `rows_before`,
    `rows_after`, `rows_removed`, `rows_modified` (subdoc) and `flagged_subdocs`;
  - `removals`: the totals (`rows_before`, `rows_after`, `rows_removed`,
    `rows_modified`, `flagged_subdocs`), `by_benchmark` (the same counts per
    benchmark) and `by_channel` (flagged sub-documents per channel). A row or
    sub-document counts for every benchmark its evidence names: the benchmarks of
    its `src_hits` (named through `.sources.json`) and of its `item_hits`,
    `exact_hits` and `spaced_item` items. So the per-benchmark counts can add up to
    more than the totals.
- **`removed_ids.ndjson`**: one line per flagged sub-document: `shard` (the output
  path), `row_index`, `doc_id`, `field`, `role`, `turn` and `part` when present,
  `benchmarks`, `channels`, and `row_removed` (false when `subdoc` kept the row).
- **`DATASET_CARD_SNIPPET.md`**: a Markdown section for the dataset card, generated
  from the manifest: the benchmarks with revisions and splits, the matching
  settings, the tool version, the index SHA-256, the removal counts, the command
  to reproduce the run and how to cite it.

## Self-test (`selftest`)

```
<out_dir>/
├── planted.parquet   # control hosts, then one row per sampled item and variant
├── truth.jsonl       # per row: row_index, kind (control / plant), doc_id with --id-key,
│                     #   item_idx, item_id, benchmark, variant, perturbation
├── selftest.json
└── scans/<run>/      # each scan of planted.parquet, as written by `scan`
```

`selftest.json` (`schema_version` 1) holds the `index` and `scan_config` (as in the
run manifest), the sample (`n_items`, `seed`, `variants`, `host`), each sampled item
with its indexed n-gram count, question length and whether the n-gram rule and the
exact channel can match it, the `unmatchable` items (no enabled channel can match
them; left out of every recall figure), and per run (`union`, `item`, and
`<gate>+exact` with `--exact-channel`): `overall` and `by_variant` recall
(`matchable`, `found`, `recall`), `by_channel` (the run with the exact channel:
`ngram` and `exact`), `attributed` (found plants whose record names the planted
item), `missed` plants, and `false_positives` among the `controls`. A plant counts
as found when the run flags any sub-document of its row. `configured_run`,
`gated_recall`, `min_recall` and `passed` give the verdict the exit code follows.

## Reports (`report`)

`<out_dir>/report/` (or `--out-dir`) contains:

- **`report.md` / `report.json`:** run overview, scan summary, and the top tables.
  With a v2 `.items` sidecar, `field_matches` counts the item matches in the sampled
  records (`contaminated_docs/`) by which fields matched: `full` (every field of the
  item that has n-grams, e.g. question and choices), `question_only` (only the first
  spec field), `partial` (another subset) and `single_field` (the item has n-grams in
  one field only). The scan summary includes the flagged documents per channel.
  `exact_matches` (when a sampled record has an exact hit) counts the exact
  matches, the matched items, the records flagged by the exact channel alone, and,
  for items with choices, the matches with all, some or none of the choices near the
  question. `confirmed_pairs` is the number of sampled (document, item) pairs with
  `--confirm` evidence, and `template_variants` lists the likely template variants
  among them.
- **`template_variants.csv` / `.json`:** the flagged (document, item) pairs of the
  sampled records that look like other problems from the item's template: some of
  the item's numbers are missing from the document (`numbers_matched <
  numbers_total`) and its longest common substring with the question is below 90%
  of the question (`lcs_ratio < 0.9`). Columns: `shard`, `doc_id`, `item_idx`,
  `item_id`, `benchmark`, `channel`, `hits`, `item_coverage`, `numbers_matched`,
  `numbers_total`, `lcs_chars`, `lcs_ratio`. Empty without `--confirm`.
- **`leak_by_benchmark.csv` / `.json`:** `benchmark`, `total_hashes_in_index`,
  `leaked_hashes`, `leak_fraction`.
- **`leak_by_shard.csv` / `.json`:** unique leaked hashes per shard, per 1k documents.
- **`leak_by_item.csv` / `.json`:** per item, `n_protected_hashes`, `leaked_hashes`,
  `leak_fraction`, and the unique, shared and boilerplate split. Requires the `.items`
  sidecar. `best_doc_coverage` is the largest share of the item's n-grams any single
  scanned sub-document held, with `best_doc_hits`, `best_doc_id` and
  `best_doc_shard`; `leak_fraction` pools the whole corpus, so an item whose
  n-grams are scattered over many documents has a high `leak_fraction` and a low
  `best_doc_coverage`. These are empty for scans without `per_item_best/`.
