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
- Several text fields and chat messages per record. `--text-key` is repeatable
  (`--text-key prompt --text-key chosen --text-key rejected`), and `--messages-key`
  scans a `list<struct{role, content}>` chat column. Each field, and each chat turn,
  is scanned as its own document (packed splitting then applies within it), and the
  records say where the match is: `field`, plus `role` and `turn` for chat turns.
  This works for Parquet and JSONL; the JSONL conversion keeps every requested column.
- A warning, once per shard, names the string columns (and chat-like columns) that
  the scan does not read, and a missing `--id-key` column.
- `large_string`, string-view and string-dictionary text columns are read; they used to
  be rejected.
- `--all-records` on `scan`, `run` and `run-hf` writes every flagged record to
  `contaminated_docs/` (same as `--max-return-records 0`).
- `--no-split` on `scan`, `run` and `run-hf` turns off packed-document splitting
  (both `<|endoftext|>` and its typo variant): each record is scored as one document
  and its doc id is the plain `<id>`, without a `-part-<k>` suffix. In the config,
  `background.corpus.packed_doc_sep: null` does the same for `build-stopgrams-direct`.
  `ScanConfig.packed_doc_sep` and the Rust `scan_stream_rust` accept `None`, and the
  run manifest records `packed_doc_sep: null`.

- Stop-grams learned from the indexed items (#11). `stopgrams.items: {max_item_share:
  0.01, min_count: 10}` (off by default; `build-index --item-stopgrams-share` and
  `--item-stopgrams-min-count` override it, and either turns it on) drops at index time
  the n-grams held by more than that share of all items and by at least `min_count`
  items, such as an instruction repeated across a training set's questions. They leave
  `.native`, `.attr` and `.items`, items' `n_protected_hashes` and the per-benchmark
  counts are taken after the drop, the dropped hashes go to `<index>.item_stopgrams`
  (stop-grams format), and the index meta records `item_stopgrams` (thresholds, item
  count, number dropped). `build-index` prints the count, and `inspect-index` prints it
  with the `.items` format version. The meta field is optional, so the meta schema
  stays at version 2.
- Per-item coverage (#2). Records' `item_hits` entries gain `item_ngrams` (the
  item's indexed n-grams) and `item_coverage = hits / item_ngrams`, the share of the
  item one document holds. `--min-item-coverage C` (on `scan`, `run`, `run-hf`;
  config key `min_item_coverage`; default 0, off) flags a document when one item has
  at least `min_hits` hits and `item_coverage >= C`, whatever the document coverage,
  in `union` mode (so a long document holding a whole item is caught); in `item` mode
  it is an extra requirement on the flagging item. It needs the `.items` sidecar.
  Union-mode evidence lists the items that pass this path too.
- `per_item_best/<shard>.json`: for every item hit in a shard, the sub-document
  holding the largest share of its n-grams, over every scanned sub-document. `report`
  adds `best_doc_coverage` (with `best_doc_hits`, `best_doc_id`, `best_doc_shard`) to
  `leak_by_item` next to `leak_fraction`, and `report.md` shows the gate mode and
  `min_item_coverage`. The run manifest's `scan_config` records `min_item_coverage`.
  `scan_stream_rust` takes `min_item_coverage` and `out_item_best_path`.
- Which fields of an item matched (#3). Records' `item_hits` entries gain
  `field_hits` (`[field_idx, hits]` per matched spec field) and `fields_matched` (the
  field names), so a copy of a whole multiple-choice item (question and choices) is
  told apart from one of the question alone. `report` counts the item matches of the
  sampled records as full-item, question-only, partial and single-field
  (`field_matches` in `report.json`, a table in `report.md`).
- **Item text sidecar** `<index>.native.items.text.jsonl`, written by `build-index`:
  one JSON line per item, `{item_idx, benchmark, item_id, fields}`, with each spec
  field's raw text (a list of strings for a list-valued field, such as
  multiple-choice options). The exact channel and `--confirm` read it. The index
  meta lists it under `files.items_text`, the run fingerprint hashes it, and
  `inspect-index` says whether it is present. **Existing indexes lack it and must be
  rebuilt to use `--exact-channel` or `--confirm`** (they still scan without them, and
  those flags stop with an error that says so); `run` / `run-hf` rebuild a cached index
  without it.
- Exact-match channel (#1). `--exact-channel short|all` (on `scan`, `run`, `run-hf`;
  config key `exact_channel`; default `off`, so existing results do not change)
  matches each eligible item's question (its first field) verbatim, after a light
  normalization (NFKC, lowercase, whitespace runs collapsed; numbers, operators,
  punctuation, LaTeX and stop words kept, so `2+3` and `2-3` stay different), on word
  boundaries, with one Aho-Corasick automaton per worker. `short` covers the items
  with fewer than `min_hits` n-grams, which the n-gram rule can never flag; `all`
  covers every item. `--exact-min-chars` (default 12; config `exact_min_chars`) leaves
  out shorter questions. A match flags the sub-document whatever the n-gram gate
  decided. `describe-benchmark`, `build-index` and `inspect-index` point to it when a
  benchmark has short items. `aho-corasick` becomes a direct Rust dependency (it was
  already in the lockfile).
- Records (schema v3, optional fields) gain `channels` (`ngram`, `spaced`, `exact`:
  what flagged the sub-document) and, with the exact channel, `exact_hits`
  (`[{item_id, field: 0, choices_found, choices_total}]`). Shard summaries gain
  `contaminated_by_channel` and `_FINAL_SUMMARY.json` gains
  `total_contaminated_by_channel`: flagged sub-documents per enabled channel; older
  summaries still load. `report` shows them, and counts the exact matches of the
  sampled records (`exact_matches`). `hits.bin` stays n-gram only. `scan_stream_rust`
  takes `exact_channel`, `exact_min_chars`, `confirm` and a `channel_counts` dict it
  fills.
- Choices near the question (rest of #3). With the exact channel or `--confirm`,
  every `exact_hits` and `item_hits` entry of a written record carries
  `choices_found` / `choices_total`: how many of the item's other fields (each option
  of a list field on its own) occur within 1,000 characters of the matched question
  (for an n-gram hit, of the question's first verbatim occurrence, or anywhere in the
  sub-document when it has none).
- Copy-vs-variant confirmation (#4). `--confirm` (config `confirm`; default off)
  adds to every flagged (document, item) pair of the written records
  `numbers_total` / `numbers_matched` (the distinct numbers of the item's question,
  and those found in the sub-document as whole tokens) and `lcs_chars` / `lcs_ratio`
  (the longest common substring of the question and the sub-document, computed with a
  suffix automaton, linear in the document). It is report only: the flagged documents
  are the same with and without it. `report` lists the likely template variants
  (some numbers missing and `lcs_ratio < 0.9`) in `report.md`, `report.json`
  (`confirmed_pairs`, `template_variants`) and `template_variants.csv` / `.json`.
- The run manifest's `scan_config` records `exact_channel`, `exact_min_chars` and
  `confirm` (absent in older manifests: off).
- Records (schema v3, optional fields) gain `row_index`, the record's row in its
  shard counted from 0 (for JSONL, among the non-blank lines, as the scan reads
  them), and `part`, the packed part's index within its field or turn (the `<k>` of
  `-part-<k>`; absent under `--no-split`), so a flagged sub-document can be located
  without parsing its doc id. Older records still parse.
- The run manifest records what the index is and how the scan gated (#8):
  `index` gains the SHA-256 of every index file (`native_sha256`, `attr_sha256`,
  `sources_sha256`, `items_sha256`, `items_meta_sha256`, `items_text_sha256`,
  `item_stopgrams_sha256`), `normalization_config_sha`, `ngram_config_hash`, the
  indexed `benchmarks` (with requested and resolved revisions, splits and dataset
  fingerprints) and `benchmark_stats`; `scan_config` gains `gate_mode`,
  `min_longest_run` and `boilerplate_threshold`. All optional: older manifests
  still load in `reproduce`, `inspect-run` and `report`. `reproduce` also checks
  the `.native` SHA-256 when the manifest has it.

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
- **Item postings format v2 (binary format change).** `<index>.native.items` records
  are now `<u64 hash, u32 n, n × (u32 item_idx, u32 field_mask)>`: bit `k` of the
  mask says the n-gram occurs in the item's `k`-th spec field. The header is the same
  CAFMTv1 header; v2 is told apart by its file type (5, where v1 is 4), so 0.1.x
  readers reject a v2 file instead of misreading it. `build-index` hashes each field
  on its own (the n-grams are the same as before, since they never crossed fields),
  and `.items.meta.json` gains `format_version` and `fields`, each benchmark's field
  names. **Old indexes still load and scan**, but have no field information: their
  records carry no `field_hits`, and `report` skips the field table. Rebuild to get
  them; `run` / `run-hf` rebuild a cached index whose `.items` is v1 (or that lacks
  `.items.meta.json`).
- Every registry entry is pinned to a Hub commit, so configs written by `init-config`
  pass the revision-pin check without `--allow-unpinned`.
- **Record schema v3.** `contaminated_docs/*.ndjson` records now have
  `schema_version: 3` and an optional `field` (always written by this version), `role`
  and `turn`. v2 records still parse (`DocResult`, `parse_contamination_record`,
  `inspect-run`). `DOC_RESULT_SCHEMA_VERSION` is 3.
- **Doc ids of multi-field scans.** With several `--text-key`s or a `--messages-key`,
  doc ids name the field: `<id>/<field>-part-<k>`, or `<id>/<field>/<turn>-part-<k>`
  for a chat turn. A scan of one text key keeps `<id>-part-<k>`.
- Non-string id columns (integers and so on) are cast to strings in Parquet input too,
  instead of being silently ignored in favour of text hashes; an id column that cannot
  be cast is an error.
- `ScanConfig` takes `text_keys: list[str]` and `messages_key`; the old
  `text_key="..."` argument (and old manifests' `config.text_key`) is still accepted.
  The run manifest's `scan_config` records `text_keys` and `messages_key`.
  `scan_stream_rust` accepts a list for `text_key` and a `messages_key` argument.
- The resume fingerprint covers the new config fields, so shards scanned by an
  earlier version are re-scanned.
- The order of the options in `scan`/`run`/`run-hf --help` changed slightly: the
  shared scan options are defined once and listed together.
- `ScanConfig` gains `exact_channel`, `exact_min_chars` and `confirm`, so the resume
  fingerprint of every run changes and shards scanned by an earlier version are
  re-scanned once.
- Benchmark loaders' `load_positive_fields` yields a list of strings for a
  list-valued field (`benchmarks._field_values`); the hashed text is unchanged.

### Fixed

- A resumed scan reused shard summaries computed against an earlier `.items` or
  `.items.meta.json` when the item sidecars were rebuilt in place (only `.native`,
  `.attr` and `.meta.json` were fingerprinted). The run fingerprint now hashes both,
  and `.item_stopgrams`. Shards scanned by an earlier version are re-scanned once.
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
