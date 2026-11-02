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

**Light normalization.** The exact-match channel (section 4) uses a
second, lighter normalization instead: Unicode NFKC, lowercasing, every whitespace run
collapsed to one space, and trimming. Nothing else changes: numbers, operators,
punctuation, LaTeX and stop words are kept, so `What is 2+3?` and `What is 2-3?` stay
different. It does not depend on the normalization config.

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
  items that contain it, and which of the item's fields it occurs in (each field is
  hashed on its own). Used for per-item attribution, the item gate, and reporting
  which fields of an item a document matched.
- `index.native.items.text.jsonl`: the raw text of every item, field by field (a
  list-valued field such as multiple-choice options keeps each option apart). Read by
  the exact-match channel; nothing else needs it.
- `index.native.sources.json` and `index.native.meta.json`: the benchmark legend and
  provenance (configs, dataset fingerprints, revisions, normalization hashes), plus
  per-benchmark item and n-gram counts.

**Empty benchmarks fail the build.** If a benchmark loads no rows (a dataset removed
from the Hub, a gated dataset without a login, a wrong split or path) or contributes no
n-grams, `build-index` stops before writing anything and names the benchmark and the
reason. `--allow-empty-benchmarks` (also on `run` and `run-hf`) builds the index anyway.

**Item stop-grams.** With `stopgrams.items` (off by default), n-grams held by more
than a share of all indexed items (and at least a minimum number of them), such as an
instruction repeated across the questions of a training set, are dropped from the index
after every benchmark is loaded. See [configuration.md](configuration.md).

**Subtraction splits.** If a benchmark sets `subtraction_splits` (for example
`[train]`), n-grams that also occur in those splits are removed from the index. What
remains is specific to the protected split, so a corpus that legitimately contains the
*training* split is not flagged as test contamination.

## 4. Scanning

Each shard is processed by one worker process, in this order:

1. The text columns (`--text-key`, repeatable) and the chat column (`--messages-key`)
   are read in batches. JSONL shards are first converted to Parquet in memory. Each
   text, and each chat turn, is handled as its own text from here on, so a match never
   spans two fields or two turns. String columns that are not scanned are reported in
   a warning.
2. Each row is split into **sub-documents** on `<|endoftext|>`, the packed-document
   separator; the variant typo `<|endoftext}>` is treated the same way. Sub-documents
   shorter than `2·n` bytes or `n` tokens are skipped. Counts and records refer to
   sub-documents, with ids `<id>-part-<k>` (see [outputs.md](outputs.md) for the ids
   of multi-field scans). `--no-split` turns this off (for chat data
   or decoded text that may legitimately contain the token): each row is then one
   document, with id `<id>`.
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
   - **Item coverage** (`--min-item-coverage C`, off by default). `min_coverage` is a
     share of the *document's* n-grams, so a long document holding a whole item can
     stay below it, and a short document can clear it with a fragment of an item.
     Item coverage is a share of the *item's* n-grams: `item_coverage = hits /
     item_ngrams` for the n-grams one document shares with one item. In `union` mode
     a document is also flagged when one item has at least `min_hits` hits and
     `item_coverage >= C`, whatever the document coverage (an OR path). In `item`
     mode, `item_coverage >= C` is an extra requirement on the flagging item.
     Stop-grams removed at scan time still count in `item_ngrams`.
   - Spaced seeds (experimental, opt-in): when the n-gram gate does not fire, the
     spaced-seed verifier can still flag the document. See
     [experimental-spaced-seeds.md](experimental-spaced-seeds.md).
   - **Exact match** (`--exact-channel short|all`, off by default); see below. A
     match flags the sub-document whatever the n-gram gate decided.
6. For flagged documents, a record is emitted. By default this is a sample: the first
   `--max-return-records` flagged sub-documents per shard (default 500); `--all-records`
   or `--max-return-records 0` emits every one. It carries per-benchmark hit counts
   and, when the index has item postings, per-item evidence. Each item's hits are
   split into:
   - **unique**: the n-gram belongs to just this item;
   - **shared**: it belongs to a few items;
   - **boilerplate**: it belongs to `>= --boilerplate-threshold` items (default 50).

   and counted per field of the item (`field_hits`, named in `fields_matched`), which
   tells a copy of the whole item (a question with its choices) from one of the
   question alone.
   Each record lists the channels that flagged it (`channels`: `ngram`, `spaced`,
   `exact`), and the shard summary counts the flagged sub-documents per channel.
7. Every matched hash with its benchmark mask is written to `<shard>.hits.bin`. This is
   exhaustive, not sampled, and it is what `report` aggregates. With item postings, the
   best single-document coverage of each item hit in the shard goes to
   `per_item_best/<shard>.json`. `hits.bin` holds n-gram matches only; exact-channel
   matches are in the records and the per-channel counts.

### The exact-match channel

The n-gram rule needs `min_hits` distinct shared 13-grams. After stop-word removal,
many short items (short math problems, most commonsense multiple-choice questions)
have fewer than that, and some have none, so the n-gram rule can never flag them,
however exact the copy. `build-index` and `inspect-index` count these items per
benchmark, and `describe-benchmark` gives the share for registry entries.

`--exact-channel short` matches the **question** (the first spec field) of every item
with fewer than `min_hits` n-grams verbatim; `--exact-channel all` matches every item.
Both sides are lightly normalized (section 1), the questions go into one Aho-Corasick
automaton (built once per worker, so the corpus is still read once), and a match must
sit on word boundaries: an alphanumeric character at the edge of the match may not
touch an alphanumeric character outside it, so `3 apples` does not match inside
`23 apples`. Questions shorter than `--exact-min-chars` characters (default 12, after
light normalization) are left out, since short stems ("Is it red?") occur everywhere.
A sub-document shorter than `2·n` bytes, which the n-gram rule skips, is still matched.

What it can and cannot catch:

- It catches verbatim copies, including copies that differ only in case, Unicode
  form (full-width characters, ligatures) or whitespace and line breaks.
- It misses any other edit inside the question: a changed word or number, a
  reordered clause, a paraphrase, different punctuation, a typo, a translation, or
  LaTeX written another way (`\frac12` vs `\frac{1}{2}`). Only the question is
  matched, so an item whose question is generic and whose substance is in the
  choices is not caught either.
- It matches short questions, which may also occur in unrelated text; review exact
  matches, especially with `all` and a low `--exact-min-chars`.

### Choices near the question

When the exact channel is on, every written record gets, for each item
it matched (exact or n-gram), `choices_found` / `choices_total`: how many of the item's
other fields occur, on word boundaries, within 1,000 characters of the matched question
in the lightly normalized sub-document. Each option of a list-valued field (MMLU's
`choices`) counts on its own. For an exact match, the location is the match. For an
n-gram match it is the first verbatim occurrence of the question, if any; otherwise
the choices are looked for in the whole sub-document. Short choices (a single number
or letter) are easy to find by chance; read the counts as supporting evidence.

## 5. Reports

`report` aggregates the `hits.bin` files into leak tables:

- **By benchmark:** leaked distinct hashes / hashes in the index.
- **By shard.**
- **By item:** exact per-item leak fractions, split into unique, shared and boilerplate
  hashes, next to the best single-document coverage of the item (`best_doc_coverage`).
  The leak fraction pools every document; the best document coverage says whether one
  document holds the item.

These counts cover every matched hash, not just the sampled records. From the sampled
records, `report` also counts which fields of the matched items matched, and the exact
matches and how many of their choices were found.

## Choosing thresholds

The defaults (`ngram: 13`, `min_hits: 3`, `min_coverage: 0.001`) are conservative
settings for web-scale pretraining text. Raising `min_hits` or `min_coverage` trades
recall for precision. For short documents (forum posts, Q&A pages), coverage is easy to
reach, and `min_hits` is the binding constraint. Validate your thresholds on a sample:
plant known benchmark items into clean text and check that they are flagged, then
inspect a sample of flagged documents for false positives.
