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
from corpus_assay.benchmark_registry import REGISTRY
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import BenchmarkLoader, BenchmarkLoadStatus
from corpus_assay.config import BuildIndexConfig
from corpus_assay.constants import FIELD_SEPARATOR, INDEX_FORMAT_VERSION
from corpus_assay.formats import FileType, write_header
from corpus_assay.inspect import format_benchmark_table
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
        "label_field": spec.label_field,
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


def _positive_fields(loader: BenchmarkLoader) -> Iterator[list[str]]:
    """Each positive item as its field texts (one per spec field, in order).

    Loaders without ``load_positive_fields`` (test doubles) yield joined text;
    its ``FIELD_SEPARATOR``-separated parts are taken as the fields in order.
    """
    by_fields: Callable[[], Iterable[list[str]]] | None = getattr(
        loader, "load_positive_fields", None
    )
    if by_fields is not None:
        yield from by_fields()
        return
    for text in loader.load_positive():
        yield [part.strip() for part in text.split(FIELD_SEPARATOR)]


@dataclass
class BenchmarkBuildStats:
    """What one benchmark contributed to the index."""

    name: str
    dataset: str
    items: int = 0
    ngrams: int = 0
    items_zero_ngrams: int = 0
    items_below_min_hits: int = 0
    min_hits: int = 3
    # Why configs/splits failed to load, and hidden-label warnings (not recorded
    # in the index meta; they feed the build-time error and summary).
    load_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, object]:
        return {
            "name": self.name,
            "items": self.items,
            "ngrams": self.ngrams,
            "items_zero_ngrams": self.items_zero_ngrams,
            "items_below_min_hits": self.items_below_min_hits,
            "min_hits": self.min_hits,
        }


@dataclass(frozen=True)
class BuildIndexResult:
    ngram: int
    hashes: set[int]
    attribution: AttributionIndex
    stats: BuildIndexStats
    benchmark_specs: list[dict[str, object]]
    dataset_fingerprints: list[dict[str, object]]
    benchmark_stats: list[BenchmarkBuildStats] = field(default_factory=list)


def build_index(
    *,
    benchmarks: Iterable[BenchmarkSpec],
    ngram: int,
    min_hits: int = 3,
    loader_factory: Callable[[BenchmarkSpec], BenchmarkLoader] = BenchmarkLoader,
) -> BuildIndexResult:
    """
    Pure core: computes hashes + attribution in memory. No filesystem I/O.

    Also counts, per benchmark, the items and n-grams indexed and the items the
    n-gram rule cannot flag (fewer than ``min_hits`` n-grams); see
    :func:`empty_benchmark_problems` for the check run on those counts.
    """
    disable_caching()

    attribution = AttributionIndex()
    stats = BuildIndexStats()
    benchmark_specs: list[dict[str, object]] = []
    dataset_fingerprints: list[dict[str, object]] = []
    benchmark_stats: list[BenchmarkBuildStats] = []
    item_ranges: list[tuple[int, int]] = []

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
        # Each field is hashed on its own (n-grams never cross fields anyway), so
        # every posting records which of the spec's fields the n-gram occurs in.
        first_item = len(attribution.items_meta)
        attribution.set_benchmark_fields(spec.name, list(spec.fields))
        for fields in tqdm(_positive_fields(loader), desc=f"{spec.name} [Index]"):
            item_idx = attribution.register_item(spec.name)
            for field_idx, text in enumerate(fields):
                if not text:
                    continue
                toks = normalize_text(text)
                for ng in _iter_filtered_ngrams(toks, ngram, stats):
                    h = hash_ngram(ng)
                    if h in neg:
                        continue
                    attribution.add_hash_for_source(h, spec.name)
                    attribution.add_hash_for_item(h, item_idx, field_idx)
        dataset_fingerprints.extend(getattr(loader, "dataset_fingerprints", []))
        item_ranges.append((first_item, len(attribution.items_meta)))
        status = getattr(loader, "load_status", None)
        benchmark_stats.append(
            BenchmarkBuildStats(
                name=spec.name,
                dataset=spec.dataset,
                min_hits=min_hits,
                load_failures=(
                    list(status.failures)
                    if isinstance(status, BenchmarkLoadStatus)
                    else []
                ),
                warnings=(
                    list(status.warnings)
                    if isinstance(status, BenchmarkLoadStatus)
                    else []
                ),
            )
        )

    # Hash counts are final only after every benchmark (later ones never remove
    # hashes, but count from the finished map to match the .sources.json totals).
    per_source = attribution.per_source_totals()
    for bstats, (start, end) in zip(benchmark_stats, item_ranges):
        sid = attribution.name_to_id.get(bstats.name)
        bstats.ngrams = per_source.get(sid, 0) if sid is not None else 0
        counts = [
            int(item["n_protected_hashes"])
            for item in attribution.items_meta[start:end]
        ]
        bstats.items = len(counts)
        bstats.items_zero_ngrams = sum(1 for c in counts if c == 0)
        bstats.items_below_min_hits = sum(1 for c in counts if c < min_hits)

    hashes = set(attribution.hash2mask.keys())
    return BuildIndexResult(
        ngram=ngram,
        hashes=hashes,
        attribution=attribution,
        stats=stats,
        benchmark_specs=benchmark_specs,
        dataset_fingerprints=dataset_fingerprints,
        benchmark_stats=benchmark_stats,
    )


class EmptyBenchmarkError(ValueError):
    """A configured benchmark loaded no rows or indexed no n-grams."""


def empty_benchmark_problems(result: BuildIndexResult) -> list[str]:
    """One message per benchmark that loaded no rows or indexed no n-grams."""
    problems: list[str] = []
    for bstats in result.benchmark_stats:
        label = f"{bstats.name} ({bstats.dataset})"
        if bstats.items == 0:
            reasons = "; ".join(bstats.load_failures) or (
                "the requested splits are empty"
            )
            message = f"{label}: loaded no rows. {reasons}"
            gated = REGISTRY.is_gated_dataset(bstats.dataset) or any(
                "gated" in failure.lower() for failure in bstats.load_failures
            )
            if gated:
                message += (
                    "\n    This dataset is gated on the Hugging Face Hub: log in "
                    "with `hf auth login` and accept the dataset terms at "
                    f"https://huggingface.co/datasets/{bstats.dataset}."
                )
            problems.append(message)
        elif bstats.ngrams == 0:
            problems.append(
                f"{label}: loaded {bstats.items:,} items but indexed no n-grams. "
                f"Its item fields may be shorter than ngram={result.ngram} tokens "
                "(n-grams never cross fields), or every n-gram was rejected or "
                "subtracted."
            )
    return problems


def check_benchmarks_not_empty(
    result: BuildIndexResult, *, allow_empty_benchmarks: bool = False
) -> list[str]:
    """Raise :class:`EmptyBenchmarkError` if a benchmark would be indexed empty.

    With ``allow_empty_benchmarks`` the problems are returned instead, for the
    caller to report as warnings.
    """
    problems = empty_benchmark_problems(result)
    if problems and not allow_empty_benchmarks:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise EmptyBenchmarkError(
            f"{len(problems)} benchmark(s) would be indexed empty:\n{lines}\n"
            "No index was written. Fix the config, or pass --allow-empty-benchmarks "
            "to build the index anyway."
        )
    return problems


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
        "benchmark_stats": [b.to_meta() for b in result.benchmark_stats],
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
    result = build_index(
        benchmarks=cfg.benchmarks, ngram=cfg.ngram, min_hits=cfg.min_hits
    )
    # Checked before anything is written, so a failed build leaves no index behind.
    problems = check_benchmarks_not_empty(
        result, allow_empty_benchmarks=cfg.allow_empty_benchmarks
    )
    write_native_index_artifacts(result, cfg.out_path)
    _report_build_stats(result, problems=problems)


def _report_benchmark_table(result: BuildIndexResult) -> None:
    if not result.benchmark_stats:
        return
    min_hits = result.benchmark_stats[0].min_hits
    rows = [
        (
            b.name,
            f"{b.items:,}",
            f"{b.ngrams:,}",
            f"{b.items_zero_ngrams:,}",
            f"{b.items_below_min_hits:,}",
        )
        for b in result.benchmark_stats
    ]
    print(format_benchmark_table(rows, min_hits=min_hits, indent="  "))


def _report_build_stats(
    result: BuildIndexResult, *, problems: list[str] | None = None
) -> None:
    config = _load_ngram_config()
    patterns = config.get("ngram_reject_patterns", [])
    counts = [
        (idx, count) for idx, count in result.stats.rejected_pattern_counts.items()
    ]
    counts.sort(key=lambda item: item[1], reverse=True)
    top_counts = counts[:10]

    print("[build-index] Benchmarks:")
    _report_benchmark_table(result)
    unflaggable = sum(b.items_below_min_hits for b in result.benchmark_stats)
    if unflaggable:
        min_hits = result.benchmark_stats[0].min_hits
        print(
            f"  {unflaggable:,} item(s) have fewer than min_hits={min_hits} n-grams; "
            "the n-gram rule cannot flag them."
        )
    for message in problems or []:
        print(f"  WARNING (--allow-empty-benchmarks): {message}")
    for bstats in result.benchmark_stats:
        for warning in bstats.warnings:
            print(f"  WARNING: {warning}")

    print("[build-index] N-gram stats:")
    print(f"  total candidate n-grams: {result.stats.total_candidate_ngrams:,}")
    print(f"  retained after filtering: {result.stats.retained_after_filtering:,}")
    if top_counts:
        print("  top rejected patterns:")
        for idx, count in top_counts:
            pattern = patterns[idx] if idx < len(patterns) else "<unknown>"
            print(f"    {count:,}  {pattern}")
    else:
        print("  top rejected patterns: (none)")
