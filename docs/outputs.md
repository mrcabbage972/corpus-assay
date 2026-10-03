# Outputs and file formats

All binary files are little-endian. Every JSON sidecar carries a `schema_version`.

## Index artifacts (`build-index`)

| File | Contents |
| --- | --- |
| `<index>.native` | Header (type 1) + sorted `u64` n-gram hashes |
| `<index>.native.attr` | Header (type 2) + `<u64 hash, u64 benchmark_mask>` records, sorted by hash |
| `<index>.native.items` | Header (type 4) + per hash: `<u64 hash, u32 n, n × u32 item_idx>` |
| `<index>.native.items.meta.json` | Item table: `(benchmark, item_id, n_protected_hashes)` per item index |
| `<index>.native.sources.json` | Benchmark legend: `id_to_name`, `name_to_id`, per-benchmark hash totals |
| `<index>.native.meta.json` | Provenance: tool versions, `ngram`, normalization config hash/SHA, hash function, benchmark specs, HF dataset fingerprints and resolved revisions |

Bit `i` of a benchmark mask is set when the n-gram occurs in benchmark `id_to_name[i]`.

### Common binary header (36 bytes)

Used by `.native`, `.attr`, `.items`, and `hits.bin`.

| Offset | Type | Field |
| --- | --- | --- |
| 0 | `8s` | magic `CAFMTv1\0` |
| 8 | `u32` | file type: 1 hashes, 2 attribution masks, 3 hit pairs, 4 item postings |
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
└── contaminated_docs/<shard>.decontam.sample.ndjson
```

- **`_FINAL_SUMMARY.json`** holds the run totals: `total_files`, `failed_files`,
  `total_docs_scanned`, `total_contaminated`, `contamination_rate` and
  `total_empty_after_filter`. Documents are counted per packed sub-document.
- **`_RUN_MANIFEST.json`** holds the full scan config, the index paths and SHA-256s,
  the input file list, tool versions, and stop-gram provenance. `reproduce` reads it,
  re-checks the recorded hashes, and re-runs the scan.
- **`summaries/<shard>.summary.json`** holds per-shard counts, an `error` field when the
  shard failed, and a run fingerprint. A re-run with the same fingerprint skips the
  shard. The fingerprint covers the index, config, and normalization config.
- **`per_source_hits/<shard>.hits.bin`** has a header of type 3, then
  `<u64 hash, u64 mask>` for **every** distinct matched hash in the shard. `report`
  aggregates these files.
- **`contaminated_docs/<shard>.decontam.sample.ndjson`** holds one JSON object per
  flagged sub-document, capped at `--max-return-records`:

  ```json
  {"schema_version": 2, "doc_id": "doc500-part-0", "match_count": 85,
   "src_hits": [[0, 85]],
   "item_hits": [{"item_id": 7, "hits": 85, "longest_run_tokens": 75,
                  "unique_hits": 85, "shared_hits": 0, "boilerplate_hits": 0}]}
  ```

  - `src_hits` contains `[benchmark_id, hits]` pairs.
  - `item_hits` is present when the index has item postings.
  - `spaced_item` appears when the experimental spaced-seed channel flagged the
    document.

## Reports (`report`)

`<out_dir>/report/` (or `--out-dir`) contains:

- **`report.md` / `report.json`:** run overview, scan summary, and the top tables.
- **`leak_by_benchmark.csv` / `.json`:** `benchmark`, `total_hashes_in_index`,
  `leaked_hashes`, `leak_fraction`.
- **`leak_by_shard.csv` / `.json`:** unique leaked hashes per shard, per 1k documents.
- **`leak_by_item.csv` / `.json`:** per item, `n_protected_hashes`, `leaked_hashes`,
  `leak_fraction`, and the unique, shared and boilerplate split. Requires the `.items`
  sidecar.
