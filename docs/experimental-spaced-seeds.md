# Experimental: spaced-seed recall channel

> **Status: experimental.** The CLI flags, config keys, and the `.spaced` file format
> may change in a minor release. The exact n-gram gate remains the default and
> recommended detector.

Exact 13-grams are precise but brittle. A single substituted word destroys every
13-gram that covers it, so a benchmark item with a few edits (a changed number, a
renamed entity) can slip past the exact gate. The spaced-seed channel adds a
gap-tolerant, *verified* second chance:

```
flag(d) = exact_gate(d)  OR  spaced_verified_gate(d)
```

## How it works

- **Seeds.** A spaced seed of span 17 and weight 13 hashes only the 13 "care"
  positions of a 17-token window. The four default patterns are in
  `corpus_assay.spaced_seeds.DEFAULT_PATTERNS`. A substitution that lands on a gap
  position leaves the seed intact. No contiguous 13-token window contains all care
  positions of a pattern, so a lone exact 13-gram never triggers this channel.
- **Postings.** `build-spaced-index` stores, for every protected item, its normalized
  tokens and the `seed_hash -> (item_id, item_pos)` postings.
- **Locus collapse.** Seed hits on a document are grouped by
  `(item_id, diagonal = doc_pos - item_pos)`, counting *distinct document positions*,
  not raw pattern matches. Several patterns firing at one spot count once.
- **Verification.** A candidate `(item, diagonal)` with at least `min_loci` loci (default
  3) is aligned token-by-token against the item's raw tokens. It is confirmed when the
  comparable span is at least `ver_min_span` tokens (default 17) and token identity is at
  least `ver_identity` (default 0.85).

Verification is token-level, so the `.spaced` artifact carries the protected items'
token sequences, not just hashes. It is larger than the exact index and costs extra scan
time. Benchmark it on your own data before enabling it at scale.

## Usage

```bash
corpus-assay build-spaced-index --config config.yaml --out bench.spaced
corpus-assay scan --config config.yaml --index bench.native --spaced-seeds bench.spaced \
  --input data/ --out-dir scan_out
```

You can also enable it from the config:

```yaml
spaced:
  enabled: true
  path: bench.spaced
  min_loci: 3
  ver_min_span: 17
  ver_identity: 0.85
```

The `.spaced` index must be built with the same `ngram` and normalization config as the
exact index; the scan checks this against the `.meta.json` files. Documents flagged
*only* by this channel carry `"spaced_item": <item_id>` in their sample record.

`--class-aware` (mapping numbers or identifiers to classes before seeding) is reserved;
only `none` is supported by the scanner.

## Implementation and parity

- `corpus_assay/spaced_seeds.py` is the pure-Python reference implementation.
- `src/spaced.rs` is the scanner's Rust port.
- `corpus_assay/spaced_index.py` writes and reads the `.spaced` format (layout in
  [outputs.md](outputs.md)).

Parity between the two implementations is enforced by:

- `tests/fixtures/spaced_golden.json`: golden vectors for the seed streams, locus maps
  and verdicts. Regenerate them with `python tests/test_spaced_golden.py` only when
  behavior is meant to change.
- `tests/test_spaced_rust_parity.py`: the Rust and Python paths must agree on the golden
  vectors and on randomized documents.
- `tests/test_spaced_scan_integration.py`: end-to-end through the Rust scanner.
