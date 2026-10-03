# Relation to the MixtureVitae decontamination

corpus-assay grew out of the decontamination pipeline built for **MixtureVitae** (Nguyen
et al., [arXiv:2509.25531](https://arxiv.org/abs/2509.25531)). That pipeline produced
the paper's 13-gram contamination analysis, described in Section 3.4 and Appendix D:
it scanned 345,697,271 documents against an index built from 19 benchmark sources.
Its code is in the
[`decontaminate/`](https://github.com/ontocord/mixturevitae/tree/main/decontaminate)
folder of the MixtureVitae repository.

The core method is unchanged: the same normalization, the same 13-grams and hash, and
the same flagging thresholds. A few changes, however, mean that **corpus-assay with
default settings does not reproduce the paper's numbers exactly**. This page lists
what is shared, what changed, and what that means for reproducing the paper.

## Timeline

| When | What |
| --- | --- |
| Sep–Oct 2025 | Pipeline written for MixtureVitae (`decontam_hf.py` plus a Rust scanner, `fast_decont`). The analysis appears in every arXiv version from v1 (29 Sep 2025) to v5 (12 Jan 2026). |
| Jan 2026 | Code moved into a standalone tool (working name *open-decontaminator*) and developed through 2026. Changes include modular code, YAML configs, a benchmark registry, binary index formats, a memory-mapped backend, provenance metadata, stop-grams, per-item attribution, and a test suite. |
| Oct 2026 | Released as corpus-assay 0.1.0. |

## What is the same

| Aspect | Both implementations |
| --- | --- |
| Normalization | Unicode NFKC, then lowercasing. Tokens match `[a-z0-9]+(?:'[a-z0-9]+)?`. The same 40 English stop words are removed. |
| Boilerplate filter | The same 26 reject patterns, in the same order. They drop multiple-choice scaffolding (for example "which of the following" or "option a"). |
| N-grams | Word 13-grams over the normalized tokens. |
| Hash | BLAKE2b with an 8-byte digest, read as a little-endian `u64`, over the n-gram's tokens joined by single spaces. The same n-gram gets the same hash in both. |
| Flagging rule | A document is flagged when it has at least 3 distinct matching n-grams **and** coverage of at least 0.1%. This is the union over the whole index, the default `--gate-mode union`. |
| Documents | Rows are split on `<|endoftext|>`, and the variant `<|endoftext}>` is treated the same way. Parts shorter than 26 bytes or 13 tokens are skipped. Records are named `<id>-part-<k>`. |
| Leak % | For each benchmark, the number of its distinct n-grams found in the corpus, divided by its distinct n-grams in the index. |

You can check the equivalence yourself: LAMBADA has a single text field, so none of the
index-side changes below apply to it. Its test split yields **156,841** indexed
n-grams in both implementations.

## What changed and affects results

### 1. N-grams no longer cross field boundaries

MixtureVitae joined a benchmark item's fields with plain spaces, so its n-grams could
span the question, the choices and the answer. corpus-assay places a field separator
between fields and never forms an n-gram across it.

As a result, multi-field benchmarks index fewer n-grams. Text that straddles fields, such
as the end of a question running into its first choice, is no longer matched. Fields
shorter than 13 normalized tokens contribute nothing; COPA is the extreme case.

Index sizes per benchmark show the effect. The MixtureVitae numbers come from its
published index legend. The corpus-assay numbers come from the built-in registry,
built in October 2026 from the current Hub revisions.

| Benchmark | MixtureVitae: split, fields | n-grams | corpus-assay registry: split, fields | n-grams |
| --- | --- | ---: | --- | ---: |
| LAMBADA | test; text | 156,841 | test; text | 156,841 |
| MMLU | test; question, choices, answer; minus train | 453,576 | test; question, choices, answer | 343,827 |
| HellaSwag | test; ctx, endings, label | 804,889 | test; ctx, endings, label | 690,222 |
| GSM8K | test; question, answer | 92,071 | test; question, answer | 76,252 |
| ARC | test; question, choices, answerKey | 46,359 | test; question, choices, answerKey | 21,970 |
| BoolQ | super_glue test; passage, question, label | 175,345 | google/boolq validation; question, passage | 146,305 |
| WinoGrande | test; sentence, option1, option2, answer | 4,115 | validation; sentence, option1, option2 | 1,209 |
| OpenBookQA | test; question_stem, choices, answerKey, fact1 | 2,272 | test; question_stem, choices | 806 |
| COPA | test; premise, choice1, choice2, question, label | 762 | validation; premise, choice1, choice2, question | 0 |

The last four rows also differ in split or fields, so their gap is not only caused by
field boundaries.

### 2. Coverage is computed over allowed n-grams

MixtureVitae applied the reject patterns only when building the index. When scanning, it
counted every distinct 13-gram of a document in the coverage denominator. corpus-assay
applies the same filter to documents too. Its denominator therefore counts only
n-grams that could ever match: windows matching a reject pattern, and windows that cross
a field separator, are left out. The numerator is unaffected, because rejected n-grams
are never in the index.

This only matters for long documents. With `min_hits = 3` and `min_coverage = 0.001`,
any document with at most 3,000 distinct n-grams that has 3 hits also clears the
coverage threshold. Above 3,000, corpus-assay's coverage is greater than or equal to
MixtureVitae's. So on the same index it flags at least the same documents, and possibly
a few more long ones.

### 3. Train-split subtraction is opt-in

The paper describes subtracting train-split n-grams from the test index (Appendix
D.1.1). The MixtureVitae code did this only for MMLU, by removing n-grams from its
`train` split. MMLU's subject configurations have no `train` split, so this most likely
removed nothing.

In corpus-assay, subtraction is a per-benchmark option, `subtraction_splits`. It is
off for every entry in the built-in registry.

### 4. Benchmarks are configured, not hard-coded

MixtureVitae indexed a fixed list of 19 sources:
- MMLU, IFEval (google/IFEval plus Multi-IF), ARC, COPA, LAMBADA, OpenBookQA, WinoGrande
- BoolQ, HellaSwag, PIQA, GSM8K, ALERT, GPQA, MATH, MBPP, HumanEval
- SimpleQA, CommonsenseQA, DoNotAnswer

MATH contributed no n-grams, because the dataset could not be loaded at the time.

corpus-assay has no built-in default; you list the benchmarks in a YAML config. Its
registry covers 9 of those sources and adds MMLU-Pro. For COPA, WinoGrande, BoolQ and
OpenBookQA it uses different splits or fields (see the table above). Several of the
dataset ids MixtureVitae used (`hendrycks_test`, `piqa`, `super_glue`, `winogrande`)
were script-based. The `datasets` 4.x releases that corpus-assay requires can no
longer load script-based datasets, so those benchmarks need their current Hub
equivalents.

### 5. Smaller differences

- **Flattening fields.** When turning list and dict fields into text, corpus-assay
  keeps `0` and `False` values, which MixtureVitae dropped. Dicts without `text`/`labels`
  become their values in sorted key order, rather than a JSON dump.
- **Flagged-document records.** MixtureVitae wrote every flagged document id and had a
  `filter` command that removed them from the dataset. corpus-assay writes a sample of
  records, by default up to 500 per shard; raise `--max-return-records` to get them all.
  It has no `filter` command. The exhaustive `per_source_hits/*.hits.bin` files and the
  leak reports are unaffected by the sample cap.

## Engineering changes (results unaffected)

- **Inputs.** Parquet as before, plus JSONL (`.jsonl`, `.jsonl.gz`, `.jsonl.zst`).
  Inputs can be directories, recursive globs or file lists. `run-hf` scans a Hugging
  Face split, converting it to Parquet first.
- **Index files.** A sorted, headered binary format (`.native`, `.attr`, `.items`),
  plus a `.meta.json` with provenance. MixtureVitae used a pickle and headerless sidecars;
  corpus-assay cannot read those indexes, so rebuild them.
- **Scaling.** With several workers, every worker shares one memory-mapped copy of the
  index. MixtureVitae loaded a full copy per worker. See
  [Performance](https://github.com/mrcabbage972/corpus-assay#performance).
- **Reproducibility.**
  - Benchmark revisions must be pinned unless `--allow-unpinned` is passed.
  - Indexes record dataset fingerprints and the normalization-config hash. A scan
    refuses an index built under a different normalization config.
  - Runs write a manifest that `reproduce` can replay.
  - Resumed scans reuse a shard's result only if its inputs and config are unchanged.
- **New optional features, all off by default:**
  - stop-grams learned from a background corpus;
  - an item-level gate;
  - per-item leak reports;
  - an experimental spaced-seed channel for lightly edited copies.
- **Quality and packaging.** A test suite (unit, Rust parity, offline end-to-end), CI on
  Linux, macOS and Windows, and prebuilt wheels on PyPI.

## Reproducing the paper's protocol

To get as close to Appendix D as corpus-assay allows:

1. In the YAML config, list the MixtureVitae benchmarks with their splits and fields.
   The full list is in `load_*_texts` in the MixtureVitae
   [`decontam_hf.py`](https://github.com/ontocord/mixturevitae/blob/main/decontaminate/decontam_hf.py).
   Use current Hub ids where the original ones were script-based.
2. Keep the defaults `ngram: 13`, `min_hits: 3`, `min_coverage: 0.001` and
   `--gate-mode union`.
3. Pass a large `--max-return-records` if you need every flagged document id.

Two differences remain, because they are built into the matching code: n-grams never
cross fields (§1), and coverage is computed over allowed n-grams (§2). Expect smaller
indexes for multi-field benchmarks and slightly different flags among very long
documents. The net effect on the paper's contamination rates has not been measured.
