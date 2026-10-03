# How matching works

corpus-assay is a hashed word n-gram overlap detector. Indexing and scanning share
one normalization and hashing pipeline, implemented once in Rust (`src/text.rs`), so a
benchmark n-gram and a corpus n-gram hash identically.

## 1. Normalization

Every text (benchmark item or corpus document) goes through these steps:

1. **Unicode NFKC**, then lowercasing.
2. **Tokenization** with the regex `word_re` (default `[a-z0-9]+(?:'[a-z0-9]+)?`).
   Punctuation and whitespace disappear.
3. **Stop-word removal** using `stop_words` (40 common English function words by
   default).

All three settings come from the normalization config, a JSON file bundled as
`corpus_assay/ngram_config.json`. You can override it for a whole command with
`corpus-assay --ngram-config path.json <command>`. The config is frozen on first use
within a process, and its SHA-256 is recorded in every index and scan manifest.

## 2. N-grams and reject patterns

Word n-grams (default `n = 13`) are formed over the normalized tokens, with two
exceptions:

- **Field boundaries.** A benchmark item is built by joining its configured `fields`
  (for example `question`, `choices`, `answer`) with a field separator. N-grams never
  cross a field boundary.
- **Reject patterns.** N-grams that match any regex in `ngram_reject_patterns` are
  dropped. The defaults target multiple-choice scaffolding ("which of the following",
  "option a", "true false true", "correct answer is", ...). Such text is shared across
  thousands of unrelated items, so it would cause false positives.

Each surviving n-gram (tokens joined by single spaces) is hashed with **BLAKE2b,
8-byte digest, read as a little-endian `u64`**.

## 3. Index

`build-index` loads each benchmark's `splits` (the protected items, usually `test`)
from the Hugging Face Hub or from local `data_files`. It writes:

- `index.native`: the sorted set of protected n-gram hashes.
- `index.native.attr`: for each hash, a 64-bit mask of the benchmarks it occurs in.
  This mask is why an index holds at most 64 benchmarks.
- `index.native.items` (+ `.items.meta.json`): for each hash, the ids of the individual
  items that contain it. Used for per-item attribution and the item gate.
- `index.native.sources.json` and `index.native.meta.json`: the benchmark legend and
  provenance (configs, dataset fingerprints, revisions, normalization hashes).

**Subtraction splits.** If a benchmark sets `subtraction_splits` (for example
`[train]`), n-grams that also occur in those splits are removed from the index. What
remains is specific to the protected split, so a corpus that legitimately contains the
*training* split is not flagged as test contamination.

## 4. Scanning

Each shard is processed by one worker process, in this order:

1. The text column is read in batches. JSONL shards are first converted to Parquet in
   memory.
2. Each row is split into **sub-documents** on `<|endoftext|>`, the packed-document
   separator; the variant typo `<|endoftext}>` is treated the same way. Sub-documents
   shorter than `2·n` bytes or `n` tokens are skipped. Counts and records refer to
   sub-documents, with ids `<id>-part-<k>`.
3. The sub-document is normalized and its **distinct** allowed n-gram hashes are
   collected.
4. If stop-grams are enabled, hashes in the stop-gram set are removed. The coverage
   denominator is the distinct-n-gram count *before* this removal.
5. **Gate.**
   - `union` (default): flag if `hits >= min_hits` and `hits / distinct_ngrams >=
     min_coverage`. Here `hits` is the number of distinct document n-grams found
     anywhere in the index.
   - `item`: the same test, but `hits` counts the n-grams shared with a *single*
     protected item. Optionally, `--min-longest-run N` also requires a contiguous run of
     `N` tokens shared with that item.
   - Spaced seeds (experimental, opt-in): when the exact gate does not fire, the
     spaced-seed verifier can still flag the document. See
     [experimental-spaced-seeds.md](experimental-spaced-seeds.md).
6. For flagged documents, a sample record is emitted (up to `--max-return-records` per
   shard, default 500). It carries per-benchmark hit counts and, when the index has item
   postings, per-item evidence. Each item's hits are split into:
   - **unique**: the n-gram belongs to just this item;
   - **shared**: it belongs to a few items;
   - **boilerplate**: it belongs to `>= --boilerplate-threshold` items (default 50).
7. Every matched hash with its benchmark mask is written to `<shard>.hits.bin`. This is
   exhaustive, not sampled, and it is what `report` aggregates.

Workers are started with the `spawn` method. With more than one worker the index is
memory-mapped (`--index-backend auto`), so the hash set is shared through the page cache
rather than copied into every process.

## 5. Reports

`report` aggregates the `hits.bin` files into leak tables:

- **By benchmark:** leaked distinct hashes / hashes in the index.
- **By shard.**
- **By item:** exact per-item leak fractions, split into unique, shared and boilerplate
  hashes.

These counts cover every matched hash, not just the sampled records.

## Choosing thresholds

The defaults (`ngram: 13`, `min_hits: 3`, `min_coverage: 0.001`) are conservative
settings for web-scale pretraining text. Raising `min_hits` or `min_coverage` trades
recall for precision. For short documents (forum posts, Q&A pages), coverage is easy to
reach, and `min_hits` is the binding constraint. Validate your thresholds on a sample:
plant known benchmark items into clean text and check that they are flagged, then
inspect a sample of flagged documents for false positives.
