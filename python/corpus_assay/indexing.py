import json
import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from datasets import disable_caching
from tqdm import tqdm

from corpus_assay._native import (
    hash_function_metadata,
    hash_ngram,
    ngram_config_hash,
    ngram_config_version,
    normalization_config_sha,
    normalize_text,
)
from corpus_assay.attribution import AttributionIndex
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import BenchmarkLoader
from corpus_assay.config import BuildIndexConfig
from corpus_assay.constants import INDEX_FORMAT_VERSION
from corpus_assay.formats import FileType, write_header
from corpus_assay.normalization import active_ngram_config_path
from corpus_assay.scanner.fingerprint import (
    library_version,
    rust_extension_version,
)
from corpus_assay.text_utils import iter_ngrams_with_filter


def _load_ngram_config() -> dict:
    # The active config (bundled default or a --ngram-config override), so stats
    # are labeled with the same reject patterns the index was built with.
    with open(active_ngram_config_path(), encoding="utf-8") as f:
        return json.load(f)


def _benchmark_spec_meta(spec: BenchmarkSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "dataset": spec.dataset,
        "revision": spec.revision,
        "data_files": spec.data_files,
        "config_name": spec.config_name,
        "splits": list(spec.splits),
        "subtraction_splits": list(spec.subtraction_splits or []),
        "fields": list(spec.fields),
        "configs": list(spec.configs),
        "drop_configs": list(spec.drop_configs or []),
        "trust_remote_code": bool(spec.trust_remote_code),
    }


@dataclass
class BuildIndexStats:
    total_candidate_ngrams: int = 0
    retained_after_filtering: int = 0
    rejected_pattern_counts: dict[int, int] = field(default_factory=dict)

    def record_candidate(self) -> None:
        self.total_candidate_ngrams += 1

    def record_retained(self) -> None:
        self.retained_after_filtering += 1

    def record_rejected(self, pattern_idx: int) -> None:
        self.rejected_pattern_counts[pattern_idx] = (
            self.rejected_pattern_counts.get(pattern_idx, 0) + 1
        )


def _iter_filtered_ngrams(
    tokens: list[str], ngram: int, stats: BuildIndexStats
) -> Iterator[str]:
    for ng, allowed, reject_idx in iter_ngrams_with_filter(tokens, n=ngram):
        stats.record_candidate()
        if not allowed:
            if reject_idx is not None:
                stats.record_rejected(reject_idx)
            continue
        stats.record_retained()
        yield ng


def _collect_unique_hashes(
    text_iter: Iterator[str],
    ngram: int,
    stats: BuildIndexStats,
    *,
    desc: str,
) -> set[int]:
    """
    Collect unique hashed n-grams from a text iterator.
    """
    hashes: set[int] = set()

    for text in tqdm(text_iter, desc=desc):
        toks = normalize_text(text)
        for ng in _iter_filtered_ngrams(toks, ngram, stats):
            hashes.add(hash_ngram(ng))

    return hashes


@dataclass(frozen=True)
class BuildIndexResult:
    ngram: int
    hashes: set[int]
    attribution: AttributionIndex
    stats: BuildIndexStats
    benchmark_specs: list[dict[str, object]]
    dataset_fingerprints: list[dict[str, object]]


def build_index(
    *,
    benchmarks: Iterable[BenchmarkSpec],
    ngram: int,
    loader_factory: Callable[[BenchmarkSpec], BenchmarkLoader] = BenchmarkLoader,
) -> BuildIndexResult:
    """
    Pure core: computes hashes + attribution in memory. No filesystem I/O.
    """
    disable_caching()

    attribution = AttributionIndex()
    stats = BuildIndexStats()
    benchmark_specs: list[dict[str, object]] = []
    dataset_fingerprints: list[dict[str, object]] = []

    for spec in benchmarks:
        benchmark_specs.append(_benchmark_spec_meta(spec))
        loader = loader_factory(spec)

        # collect negatives first
        neg: set[int] = set()
        if spec.subtraction_splits:
            neg = _collect_unique_hashes(
                loader.load_negative(), ngram, stats, desc=f"{spec.name} [Subtract]"
            )

        # stream positives and add directly (lower peak memory). Each positive
        # text is one protected evaluation item; record per-item postings so the
        # scanner can gate on single-item overlap (not benchmark-wide aggregation).
        for text in tqdm(loader.load_positive(), desc=f"{spec.name} [Index]"):
            item_idx = attribution.register_item(spec.name)
            toks = normalize_text(text)
            for ng in _iter_filtered_ngrams(toks, ngram, stats):
                h = hash_ngram(ng)
                if h in neg:
                    continue
                attribution.add_hash_for_source(h, spec.name)
                attribution.add_hash_for_item(h, item_idx)
        dataset_fingerprints.extend(getattr(loader, "dataset_fingerprints", []))

    hashes = set(attribution.hash2mask.keys())
    return BuildIndexResult(
        ngram=ngram,
        hashes=hashes,
        attribution=attribution,
        stats=stats,
        benchmark_specs=benchmark_specs,
        dataset_fingerprints=dataset_fingerprints,
    )


def build_eval_item_tokens(
    benchmarks: Iterable[BenchmarkSpec],
    *,
    loader_factory: Callable[[BenchmarkSpec], BenchmarkLoader] = BenchmarkLoader,
) -> tuple[list[list[str]], list[dict[str, object]]]:
    """Normalized token lists for every protected eval item, plus dataset fingerprints.

    These are the raw token sequences the spaced verifier aligns against. Mirrors the
    positive-item loop of :func:`build_index`, but keeps the per-item token lists (not
    hashes) and applies no negative subtraction --- the spaced artifact carries item
    tokens for token-level verification, not a hash set.
    """
    item_tokens: list[list[str]] = []
    fingerprints: list[dict[str, object]] = []
    for spec in benchmarks:
        loader = loader_factory(spec)
        for text in tqdm(loader.load_positive(), desc=f"{spec.name} [Spaced]"):
            item_tokens.append(normalize_text(text))
        fingerprints.extend(getattr(loader, "dataset_fingerprints", []))
    return item_tokens, fingerprints


def write_native_index_artifacts(result: BuildIndexResult, out_path: str) -> None:
    with open(out_path, "wb") as f:
        write_header(
            f,
            file_type=FileType.NATIVE_HASHES,
            ngram_n=result.ngram,
            record_count=len(result.hashes),
        )
        for h in sorted(result.hashes):
            f.write(struct.pack("<Q", h))

    meta_path = out_path + ".meta.json"
    from corpus_assay.schemas import INDEX_META_SCHEMA_VERSION, IndexMeta

    benchmarks_meta = [
        {
            "name": entry.get("benchmark"),
            "hf_repo": entry.get("dataset"),
            "config_name": entry.get("config"),
            "split": entry.get("split"),
            "revision": entry.get("requested_revision"),
            "resolved_revision": entry.get("resolved_revision"),
            "dataset_fingerprint": entry.get("hf_dataset_fingerprint"),
        }
        for entry in result.dataset_fingerprints
    ]

    meta = {
        "schema_version": INDEX_META_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "library_version": library_version(),
        "rust_extension_version": rust_extension_version(),
        "ngram": result.ngram,
        "ngram_config": {
            "n": result.ngram,
            "config_sha256": normalization_config_sha(),
        },
        "hash_function": hash_function_metadata(),
        "normalization_config_sha": normalization_config_sha(),
        "index_format_version": INDEX_FORMAT_VERSION,
        "ngram_config_version": ngram_config_version(),
        "ngram_config_hash": ngram_config_hash(),
        "benchmarks": benchmarks_meta,
        "benchmark_specs": result.benchmark_specs,
        "hf_datasets": result.dataset_fingerprints,
        "files": {
            "native": Path(out_path).name,
            "attr": f"{Path(out_path).name}.attr",
        },
    }

    has_items = result.attribution.has_items()
    if has_items:
        meta["files"]["items"] = f"{Path(out_path).name}.items"
        meta["files"]["items_meta"] = f"{Path(out_path).name}.items.meta.json"

    IndexMeta.model_validate(meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    result.attribution.write_sidecar(out_path, ngram=result.ngram)
    per_source_total = result.attribution.per_source_totals()
    result.attribution.write_legend(out_path, per_source_total)
    if has_items:
        result.attribution.write_items_sidecar(out_path, ngram=result.ngram)


def build_index_hf(cfg: BuildIndexConfig) -> None:
    if not cfg.allow_unpinned:
        missing = [spec.name for spec in cfg.benchmarks if not spec.revision]
        if missing:
            raise ValueError(
                "Benchmarks missing revision pin: "
                f"{', '.join(missing)}. Use --allow-unpinned to override."
            )
    # Create the output directory before the (possibly long) benchmark download.
    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
    result = build_index(benchmarks=cfg.benchmarks, ngram=cfg.ngram)
    write_native_index_artifacts(result, cfg.out_path)
    _report_build_stats(result)


def _report_build_stats(result: BuildIndexResult) -> None:
    config = _load_ngram_config()
    patterns = config.get("ngram_reject_patterns", [])
    counts = [
        (idx, count) for idx, count in result.stats.rejected_pattern_counts.items()
    ]
    counts.sort(key=lambda item: item[1], reverse=True)
    top_counts = counts[:10]

    print("[build-index] N-gram stats:")
    empty = [
        str(spec["name"])
        for spec in result.benchmark_specs
        if spec["name"] not in result.attribution.name_to_id
    ]
    if empty:
        print(
            f"  WARNING: no n-grams indexed for: {', '.join(empty)}. Their item fields "
            f"may be shorter than ngram={result.ngram} tokens (n-grams never cross "
            "fields), or every n-gram was rejected or subtracted."
        )
    print(f"  total candidate n-grams: {result.stats.total_candidate_ngrams:,}")
    print(f"  retained after filtering: {result.stats.retained_after_filtering:,}")
    if top_counts:
        print("  top rejected patterns:")
        for idx, count in top_counts:
            pattern = patterns[idx] if idx < len(patterns) else "<unknown>"
            print(f"    {count:,}  {pattern}")
    else:
        print("  top rejected patterns: (none)")
