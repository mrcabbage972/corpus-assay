# `corpus-assay`

Detect benchmark contamination in text corpora via n-gram overlap (Rust-accelerated).

**Usage**:

```console
$ corpus-assay [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-v, --verbose`: Enable verbose logging.
* `-q, --quiet`: Only log errors.
* `--version`: Show the corpus-assay version and exit.
* `--ngram-config <file>`: Path to a custom ngram normalization config JSON (stop-words, word regex, reject patterns). Defaults to the bundled config. Applied before any command runs.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `build-index`: Download HF benchmarks and build native...
* `build-stopgrams-direct`: Build stop-grams directly from a...
* `build-spaced-index`: [Experimental] Build a .spaced index for...
* `init-config`: Write a starter decontamination config...
* `list-benchmarks`: List benchmark registry keys.
* `inspect-index`: Validate an index and its sidecars.
* `inspect-run`: Validate scan outputs in a run directory.
* `reproduce`: Re-run a scan from a run manifest,...
* `describe-benchmark`: Describe a benchmark registry entry.
* `validate-config`: Validate a config file (optionally...
* `scan`: Scan dataset shards using the fast Rust...
* `run-hf`: Build (or reuse) a native index and scan a...
* `run`: Build (or reuse) a native index and scan...
* `report`: Generate reporting outputs from an...

## `corpus-assay build-index`

Download HF benchmarks and build native index (.native).

**Usage**:

```console
$ corpus-assay build-index [OPTIONS]
```

**Options**:

* `--out-index <str>`: Output path for native index (e.g., index.native).  [required]
* `--config <str>`: YAML config with decontamination hyperparameters.  [required]
* `--allow-unpinned`: Allow benchmarks without a pinned revision.
* `--allow-empty-benchmarks`: Build the index even if a benchmark loads no rows or indexes no n-grams (by default the build fails and writes nothing).
* `--item-stopgrams-share <float range>`: Drop n-grams held by more than this share of all indexed items (e.g. 0.01), such as an instruction line repeated across the items, so a wrapped copy of an item still matches in full. Overrides stopgrams.items.max_item_share and turns item stop-grams on.  [default: (config, else off); 0.0&lt;=x&lt;=1.0]
* `--item-stopgrams-min-count <int range>`: Only drop n-grams held by at least this many items. Overrides stopgrams.items.min_count and turns item stop-grams on.  [default: (config, else 10); x&gt;=2]
* `--dry-run`: Plan index build only; do not execute.
* `--help`: Show this message and exit.

## `corpus-assay build-stopgrams-direct`

Build stop-grams directly from a background corpus.

**Usage**:

```console
$ corpus-assay build-stopgrams-direct [OPTIONS]
```

**Options**:

* `--config <str>`: YAML config with background corpus settings.  [required]
* `--tau <float>`: Fractional DF cutoff (e.g., 0.001).  [required]
* `--out <str>`: Output path for stop-grams native file.  [required]
* `--workers <int>`: Number of workers to use.  [default: 8]
* `--max-docs <int>`: Optional max sampled docs to process (applies after sampling).
* `--sample-prob <float>`: Optional sampling probability (0 &lt; p &lt; 1).
* `--seed <int>`: Random seed for sampling.  [default: 0]
* `--tmp-dir <str>`: Optional directory for temp part files.
* `--sketch-size <int>`: Max distinct n-gram candidates per worker.  [default: 200000]
* `--keep-tmp`: Keep temporary part files instead of cleaning them up.
* `--help`: Show this message and exit.

## `corpus-assay build-spaced-index`

[Experimental] Build a .spaced index for the spaced-seed recall channel.

The protected eval items are loaded from the same benchmarks as ``build-index`` and
serialized with their raw tokens (for verification) and seed postings. Scan with
``--spaced-seeds &lt;out&gt;`` to OR the spaced-verified gate into the exact gate.

**Usage**:

```console
$ corpus-assay build-spaced-index [OPTIONS]
```

**Options**:

* `--config <str>`: YAML config with the protected benchmarks + ngram.  [required]
* `--out <str>`: Output path for the spaced index (e.g., index.spaced).  [required]
* `--allow-unpinned`: Allow benchmarks without a pinned revision.
* `--min-loci <int>`: Min distinct seed loci per (item, diagonal) to verify.  [default: 3]
* `--ver-min-span <int>`: Min comparable token span for verification.  [default: 17]
* `--ver-identity <float>`: Min token identity fraction for verification.  [default: 0.85]
* `--class-aware <str>`: Token class map for seeds; only &#x27;none&#x27; is currently supported.  [default: none]
* `--help`: Show this message and exit.

## `corpus-assay init-config`

Write a starter decontamination config from registry benchmarks.

**Usage**:

```console
$ corpus-assay init-config [OPTIONS]
```

**Options**:

* `--benchmarks <str>`: Benchmark registry refs to expand (repeatable).
* `--out <str>`: Output YAML config path.  [required]
* `--force`: Overwrite the output file if it exists.
* `--help`: Show this message and exit.

## `corpus-assay list-benchmarks`

List benchmark registry keys.

**Usage**:

```console
$ corpus-assay list-benchmarks [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `corpus-assay inspect-index`

Validate an index and its sidecars.

Also lists, per benchmark, the items and n-grams indexed and the items the
n-gram rule cannot flag.

**Usage**:

```console
$ corpus-assay inspect-index [OPTIONS]
```

**Options**:

* `--index <str>`: Path to the .native index.  [required]
* `--min-hits <int range>`: Count items with fewer than this many n-grams (default: the min_hits the index was built with, else 3). Needs the .items.meta.json sidecar for values other than the recorded one.  [x&gt;=1]
* `--help`: Show this message and exit.

## `corpus-assay inspect-run`

Validate scan outputs in a run directory.

**Usage**:

```console
$ corpus-assay inspect-run [OPTIONS]
```

**Options**:

* `--scan-dir <str>`: Scan output directory containing _RUN_MANIFEST.json and artifacts.  [required]
* `--validate-ndjson / --no-validate-ndjson`: Validate NDJSON samples.  [default: validate-ndjson]
* `--max-ndjson-lines <int>`: Optional cap on NDJSON lines to validate.
* `--help`: Show this message and exit.

## `corpus-assay reproduce`

Re-run a scan from a run manifest, verifying recorded inputs.

**Usage**:

```console
$ corpus-assay reproduce [OPTIONS]
```

**Options**:

* `--manifest <str>`: Path to _RUN_MANIFEST.json for a prior scan run.  [required]
* `--help`: Show this message and exit.

## `corpus-assay describe-benchmark`

Describe a benchmark registry entry.

Prints the spec, then the registry metadata (scored split, gating, short-item
share, notes) as YAML comments.

**Usage**:

```console
$ corpus-assay describe-benchmark [OPTIONS] {name}
```

**Arguments**:

* `name`: Registry benchmark key.  [required]

**Options**:

* `--help`: Show this message and exit.

## `corpus-assay validate-config`

Validate a config file (optionally resolving benchmarks on the HF Hub).

**Usage**:

```console
$ corpus-assay validate-config [OPTIONS]
```

**Options**:

* `--config <str>`: Path to YAML config.  [required]
* `--check-remote`: Attempt to resolve configs/splits via HF metadata.
* `--allow-unpinned`: Allow benchmarks without a pinned revision.
* `--help`: Show this message and exit.

## `corpus-assay scan`

Scan dataset shards using the fast Rust worker.

**Usage**:

```console
$ corpus-assay scan [OPTIONS]
```

**Options**:

* `--input, --input-glob <str>`: Input path(s): directory, glob (recursive), or file list. Repeat --input for multiple.  [required]
* `--index <str>`: Path to the .native index.  [required]
* `--out-dir <str>`: Directory for per-file outputs and summaries.  [required]
* `--text-key <str>`: Column containing document text. Repeat to scan several columns (e.g. --text-key prompt --text-key chosen --text-key rejected); each is scanned as its own document. Default: text (none when --messages-key is given).
* `--messages-key <str>`: Chat column holding a list of {role, content} messages; each turn is scanned as its own document and records carry its role and turn index.
* `--id-key <str>`: Optional stable doc id field (otherwise hashed from text).
* `--workers <int>`: Parallel worker processes.  [default: (CPU count)]
* `--index-backend <str>`: Index backend: auto, memory, or mmap.  [default: auto]
* `--no-split`: Do not split records on the packed-document separator (&lt;|endoftext|&gt; and its typo variant): each record is one document, and doc ids get no -part-&lt;k&gt; suffix. Use for chat data or decoded text that may legitimately contain the token.
* `--gate-mode <str>`: Contamination gate. &#x27;union&#x27; (default) flags a document when its matches across the whole index clear min_hits/min_coverage, with per-item attribution when an .items sidecar is present. &#x27;item&#x27; is stricter: a single protected item must clear them on its own (needs the .items sidecar). &#x27;auto&#x27; is an alias for &#x27;union&#x27;.  [default: union]
* `--min-longest-run <int>`: With --gate-mode item, also require a contiguous run of &gt;= this many normalized tokens shared with the matching item (0 disables).  [default: 0]
* `--boilerplate-threshold <int>`: Item fan-out at/above which a matched n-gram is counted as boilerplate in the per-item attribution split (unique/shared/boilerplate); must be &gt;= 2.  [default: 50]
* `--min-item-coverage <float range>`: Share of one item&#x27;s n-grams a document must hold (item coverage = hits / the item&#x27;s n-grams, with at least min_hits hits). With --gate-mode union it also flags such a document whatever its document coverage, so a long document holding a whole item is caught; with --gate-mode item it is an extra requirement on the flagging item. 0 turns it off. Needs the .items sidecar.  [default: (the config&#x27;s min_item_coverage, 0 = off); 0.0&lt;=x&lt;=1.0]
* `--exact-channel <str>`: Exact-match channel: off, short (items with fewer than min_hits n-grams, which the n-gram rule cannot flag) or all (every item). Matches each item&#x27;s question (its first field) verbatim after light normalization (Unicode NFKC, lowercase, whitespace collapsed; numbers, operators and stop words kept), on word boundaries; a match flags the document. Needs the index&#x27;s .items.text.jsonl sidecar.  [default: (the config&#x27;s exact_channel, off)]
* `--exact-min-chars <int range>`: Shortest question, in characters after light normalization, the exact channel matches (shorter stems are too generic).  [default: (the config&#x27;s exact_min_chars, 12); x&gt;=1]
* `--max-return-records <int range>`: Max flagged sub-document records written per shard to contaminated_docs/: a sample (the first N flagged, in scan order). Use 0 (or --all-records) to write every flagged record. The per-shard counts and hits.bin are always exhaustive.  [default: 500; x&gt;=0]
* `--all-records`: Write every flagged record to contaminated_docs/ (no cap; same as --max-return-records 0).
* `--stopgrams <str>`: Path to stop-grams native file (optional).
* `--allow-unverified-stopgrams`: Allow stop-grams without a compatible meta file.
* `--spaced-seeds <str>`: [Experimental] Path to a .spaced index for the spaced-seed recall channel.
* `--allow-unverified-spaced`: Allow a spaced index without a compatible meta file.
* `--allow-config-mismatch`: Scan even if the index was built with a different ngram config.
* `--dry-run`: Plan scan only; do not execute.
* `--config <str>`: YAML config with decontamination hyperparameters.  [required]
* `--help`: Show this message and exit.

## `corpus-assay run-hf`

Build (or reuse) a native index and scan a HF dataset split.

**Usage**:

```console
$ corpus-assay run-hf [OPTIONS]
```

**Options**:

* `--config <str>`: YAML config with decontamination hyperparameters.  [required]
* `--hf-dataset <str>`: HF dataset name (e.g., &#x27;allenai/c4&#x27;).  [required]
* `--split <str>`: Dataset split to scan.  [default: train]
* `--hf-revision <str>`: Pinned HF dataset revision (commit hash or tag).
* `--out-dir <str>`: Directory for per-file outputs and summaries.  [required]
* `--text-key <str>`: Column containing document text. Repeat to scan several columns (e.g. --text-key prompt --text-key chosen --text-key rejected); each is scanned as its own document. Default: text (none when --messages-key is given).
* `--messages-key <str>`: Chat column holding a list of {role, content} messages; each turn is scanned as its own document and records carry its role and turn index.
* `--id-key <str>`: Optional stable doc id field (otherwise hashed from text).
* `--workers <int>`: Parallel worker processes.  [default: (CPU count)]
* `--index-backend <str>`: Index backend: auto, memory, or mmap.  [default: auto]
* `--no-split`: Do not split records on the packed-document separator (&lt;|endoftext|&gt; and its typo variant): each record is one document, and doc ids get no -part-&lt;k&gt; suffix. Use for chat data or decoded text that may legitimately contain the token.
* `--gate-mode <str>`: Contamination gate. &#x27;union&#x27; (default) flags a document when its matches across the whole index clear min_hits/min_coverage, with per-item attribution when an .items sidecar is present. &#x27;item&#x27; is stricter: a single protected item must clear them on its own (needs the .items sidecar). &#x27;auto&#x27; is an alias for &#x27;union&#x27;.  [default: union]
* `--min-longest-run <int>`: With --gate-mode item, also require a contiguous run of &gt;= this many normalized tokens shared with the matching item (0 disables).  [default: 0]
* `--boilerplate-threshold <int>`: Item fan-out at/above which a matched n-gram is counted as boilerplate in the per-item attribution split (unique/shared/boilerplate); must be &gt;= 2.  [default: 50]
* `--min-item-coverage <float range>`: Share of one item&#x27;s n-grams a document must hold (item coverage = hits / the item&#x27;s n-grams, with at least min_hits hits). With --gate-mode union it also flags such a document whatever its document coverage, so a long document holding a whole item is caught; with --gate-mode item it is an extra requirement on the flagging item. 0 turns it off. Needs the .items sidecar.  [default: (the config&#x27;s min_item_coverage, 0 = off); 0.0&lt;=x&lt;=1.0]
* `--exact-channel <str>`: Exact-match channel: off, short (items with fewer than min_hits n-grams, which the n-gram rule cannot flag) or all (every item). Matches each item&#x27;s question (its first field) verbatim after light normalization (Unicode NFKC, lowercase, whitespace collapsed; numbers, operators and stop words kept), on word boundaries; a match flags the document. Needs the index&#x27;s .items.text.jsonl sidecar.  [default: (the config&#x27;s exact_channel, off)]
* `--exact-min-chars <int range>`: Shortest question, in characters after light normalization, the exact channel matches (shorter stems are too generic).  [default: (the config&#x27;s exact_min_chars, 12); x&gt;=1]
* `--max-return-records <int range>`: Max flagged sub-document records written per shard to contaminated_docs/: a sample (the first N flagged, in scan order). Use 0 (or --all-records) to write every flagged record. The per-shard counts and hits.bin are always exhaustive.  [default: 500; x&gt;=0]
* `--all-records`: Write every flagged record to contaminated_docs/ (no cap; same as --max-return-records 0).
* `--stopgrams <str>`: Path to stop-grams native file (optional).
* `--allow-unverified-stopgrams`: Allow stop-grams without a compatible meta file.
* `--spaced-seeds <str>`: [Experimental] Path to a .spaced index for the spaced-seed recall channel.
* `--allow-unverified-spaced`: Allow a spaced index without a compatible meta file.
* `--allow-config-mismatch`: Scan even if the index was built with a different ngram config.
* `--allow-unpinned`: Allow benchmarks without a pinned revision.
* `--allow-empty-benchmarks`: Build the index even if a benchmark loads no rows or indexes no n-grams (by default the build fails and writes nothing).
* `--index-cache-dir <str>`: Directory to cache built native indexes (honors XDG_CACHE_HOME).  [default: (~/.cache/corpus-assay/indexes)]
* `--hf-cache-dir <str>`: Optional Hugging Face datasets cache dir (defaults to HF cache).
* `--parquet-rows-per-file <int>`: Max rows per parquet shard when materializing; use 0 to force a single file.  [default: 500000]
* `--dry-run`: Plan run only; do not execute.
* `--trust-remote-code`: Allow execution of dataset loading code from the Hub.
* `--help`: Show this message and exit.

## `corpus-assay run`

Build (or reuse) a native index and scan dataset shards in one command.

**Usage**:

```console
$ corpus-assay run [OPTIONS]
```

**Options**:

* `--config <str>`: YAML config with decontamination hyperparameters.  [required]
* `--input, --input-glob <str>`: Input path(s): directory, glob (recursive), or file list. Repeat --input for multiple.  [required]
* `--out-dir <str>`: Directory for per-file outputs and summaries.  [required]
* `--text-key <str>`: Column containing document text. Repeat to scan several columns (e.g. --text-key prompt --text-key chosen --text-key rejected); each is scanned as its own document. Default: text (none when --messages-key is given).
* `--messages-key <str>`: Chat column holding a list of {role, content} messages; each turn is scanned as its own document and records carry its role and turn index.
* `--id-key <str>`: Optional stable doc id field (otherwise hashed from text).
* `--workers <int>`: Parallel worker processes.  [default: (CPU count)]
* `--index-backend <str>`: Index backend: auto, memory, or mmap.  [default: auto]
* `--no-split`: Do not split records on the packed-document separator (&lt;|endoftext|&gt; and its typo variant): each record is one document, and doc ids get no -part-&lt;k&gt; suffix. Use for chat data or decoded text that may legitimately contain the token.
* `--gate-mode <str>`: Contamination gate. &#x27;union&#x27; (default) flags a document when its matches across the whole index clear min_hits/min_coverage, with per-item attribution when an .items sidecar is present. &#x27;item&#x27; is stricter: a single protected item must clear them on its own (needs the .items sidecar). &#x27;auto&#x27; is an alias for &#x27;union&#x27;.  [default: union]
* `--min-longest-run <int>`: With --gate-mode item, also require a contiguous run of &gt;= this many normalized tokens shared with the matching item (0 disables).  [default: 0]
* `--boilerplate-threshold <int>`: Item fan-out at/above which a matched n-gram is counted as boilerplate in the per-item attribution split (unique/shared/boilerplate); must be &gt;= 2.  [default: 50]
* `--min-item-coverage <float range>`: Share of one item&#x27;s n-grams a document must hold (item coverage = hits / the item&#x27;s n-grams, with at least min_hits hits). With --gate-mode union it also flags such a document whatever its document coverage, so a long document holding a whole item is caught; with --gate-mode item it is an extra requirement on the flagging item. 0 turns it off. Needs the .items sidecar.  [default: (the config&#x27;s min_item_coverage, 0 = off); 0.0&lt;=x&lt;=1.0]
* `--exact-channel <str>`: Exact-match channel: off, short (items with fewer than min_hits n-grams, which the n-gram rule cannot flag) or all (every item). Matches each item&#x27;s question (its first field) verbatim after light normalization (Unicode NFKC, lowercase, whitespace collapsed; numbers, operators and stop words kept), on word boundaries; a match flags the document. Needs the index&#x27;s .items.text.jsonl sidecar.  [default: (the config&#x27;s exact_channel, off)]
* `--exact-min-chars <int range>`: Shortest question, in characters after light normalization, the exact channel matches (shorter stems are too generic).  [default: (the config&#x27;s exact_min_chars, 12); x&gt;=1]
* `--max-return-records <int range>`: Max flagged sub-document records written per shard to contaminated_docs/: a sample (the first N flagged, in scan order). Use 0 (or --all-records) to write every flagged record. The per-shard counts and hits.bin are always exhaustive.  [default: 500; x&gt;=0]
* `--all-records`: Write every flagged record to contaminated_docs/ (no cap; same as --max-return-records 0).
* `--stopgrams <str>`: Path to stop-grams native file (optional).
* `--allow-unverified-stopgrams`: Allow stop-grams without a compatible meta file.
* `--spaced-seeds <str>`: [Experimental] Path to a .spaced index for the spaced-seed recall channel.
* `--allow-unverified-spaced`: Allow a spaced index without a compatible meta file.
* `--allow-config-mismatch`: Scan even if the index was built with a different ngram config.
* `--allow-unpinned`: Allow benchmarks without a pinned revision.
* `--allow-empty-benchmarks`: Build the index even if a benchmark loads no rows or indexes no n-grams (by default the build fails and writes nothing).
* `--dry-run`: Plan run only; do not execute.
* `--index-cache-dir <str>`: Directory to cache built native indexes (honors XDG_CACHE_HOME).  [default: (~/.cache/corpus-assay/indexes)]
* `--help`: Show this message and exit.

## `corpus-assay report`

Generate reporting outputs from an existing scan directory.

**Usage**:

```console
$ corpus-assay report [OPTIONS]
```

**Options**:

* `--scan-dir <str>`: Scan output directory containing _RUN_MANIFEST.json and artifacts.  [default: scan_results]
* `--out-dir <str>`: Where to write report artifacts (default: &lt;scan_dir&gt;/report).
* `--topk-overall <int>`: Top-K shards to show overall.  [default: 50]
* `--topk-per-benchmark <int>`: Top-K shards to show per benchmark.  [default: 20]
* `--help`: Show this message and exit.
