# Outputs and file formats

All binary files are little-endian. Every JSON sidecar carries a `schema_version`.

## Index artifacts (`build-index`)

| File | Contents |
| --- | --- |
| `<index>.native` | Header (type 1) + sorted `u64` n-gram hashes |
| `<index>.native.attr` | Header (type 2) + `<u64 hash, u64 benchmark_mask>` records, sorted by hash |
| `<index>.native.items` | Item postings, format v2: header (type 5) + per hash, sorted: `<u64 hash, u32 n, n × (u32 item_idx, u32 field_mask)>` |
| `<index>.native.items.meta.json` | Item table: `(benchmark, item_id, n_protected_hashes)` per item index, plus `fields`: each benchmark's field names, in spec order |
| `<index>.native.sources.json` | Benchmark legend: `id_to_name`, `name_to_id`, per-benchmark hash totals |
| `<index>.native.meta.json` | Provenance: tool versions, `ngram`, normalization config hash/SHA, hash function, benchmark specs, HF dataset fingerprints and resolved revisions, and per-benchmark counts (`benchmark_stats`) |

`.meta.json` is at `schema_version` 2. Version 2 adds `benchmark_stats`, one entry per
benchmark: `name`, `items`, `ngrams` (distinct hashes attributed to it),
`items_zero_ngrams`, `items_below_min_hits` (items with fewer than `min_hits` n-grams,
those with none included) and the `min_hits` used. Version 1 files have no
`benchmark_stats` and still load; `inspect-index` then derives what it can from
`.sources.json` and `.items.meta.json`.

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
  `total_docs_scanned`, `total_contaminated`, `contamination_rate` and
  `total_empty_after_filter`. Documents are counted per packed sub-document (of each
  scanned field or chat turn).
- **`_RUN_MANIFEST.json`** holds the full scan config (including `text_keys`,
  `messages_key`, and `packed_doc_sep`, which is `null` under `--no-split`), the index
  paths and SHA-256s, the input file list, tool versions, and stop-gram provenance. `reproduce` reads it,
  re-checks the recorded hashes, and re-runs the scan.
- **`summaries/<shard>.summary.json`** holds per-shard counts, an `error` field when the
  shard failed, and a run fingerprint. A re-run with the same fingerprint skips the
  shard. The fingerprint covers the index (including `.items` and
  `.items.meta.json`), config, and normalization config.
- **`per_source_hits/<shard>.hits.bin`** has a header of type 3, then
  `<u64 hash, u64 mask>` for **every** distinct matched hash in the shard. `report`
  aggregates these files.
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
   "field": "text"}
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
  - `field` is the column the sub-document came from (a text key, or the messages
    key). For a chat turn, `role` is the turn's role (omitted when it has none) and
    `turn` its index in the messages list, counted from 0.
  - Schema history: v3 added `field`, `role` and `turn`, and later the optional
    `item_hits` fields `item_ngrams`, `item_coverage`, `field_hits` and
    `fields_matched`; v2 added `item_hits` and
    `spaced_item`. Readers (`inspect-run`, `report`) still accept v2 records.

## Reports (`report`)

`<out_dir>/report/` (or `--out-dir`) contains:

- **`report.md` / `report.json`:** run overview, scan summary, and the top tables.
  With a v2 `.items` sidecar, `field_matches` counts the item matches in the sampled
  records (`contaminated_docs/`) by which fields matched: `full` (every field of the
  item that has n-grams, e.g. question and choices), `question_only` (only the first
  spec field), `partial` (another subset) and `single_field` (the item has n-grams in
  one field only).
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
