from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import typer
import yaml

from corpus_assay._native import ngram_config_hash
from corpus_assay.benchmark_registry import REGISTRY
from corpus_assay.config import (
    BuildIndexConfig,
    DecontaminationConfig,
    ScanConfig,
    SpacedSeedsConfig,
    load_decontamination_config,
)
from corpus_assay.constants import DEFAULT_INDEX_CACHE_DIR
from corpus_assay.indexing import build_index_hf
from corpus_assay.inspect import inspect_index, inspect_run
from corpus_assay.reporting import generate_report_from_scan_dir
from corpus_assay.scanner import OutputLayout, do_scan, write_run_outputs
from corpus_assay.scanner.fingerprint import _file_sha256, library_version
from corpus_assay.schemas import RunManifest
from corpus_assay.services.config_validation import ConfigValidator
from corpus_assay.services.hf_parquet import (
    dataset_slug,
    existing_parquet_files,
    resolve_hf_parquet_paths,
)
from corpus_assay.services.index_cache import (
    ensure_native_index,
    index_cache_path,
    native_index_ready,
)
from corpus_assay.services.planning import plan_scan
from corpus_assay.spaced_index import (
    load_spaced_meta,
    spaced_meta_path,
    validate_spaced_compatibility,
    write_spaced_index,
)
from corpus_assay.spaced_seeds import (
    DEFAULT_PATTERNS,
    SpacedSeedConfig,
    SpacedSeedFamily,
)
from corpus_assay.stopgrams import (
    build_stopgrams_direct_from_hf_dataset,
    load_stopgrams_meta,
    stopgrams_meta_path,
)
from corpus_assay.utils import _load_dataset_optional_revision

app = typer.Typer(
    help=(
        "Detect benchmark contamination in text corpora via n-gram overlap "
        "(Rust-accelerated)."
    )
)
logger = logging.getLogger(__name__)
_ensure_native_index = ensure_native_index

# Shared gate options for the scanning commands (scan / run / run-hf).
_GATE_MODE_HELP = (
    "Contamination gate. 'union' (default) flags a document when its matches across "
    "the whole index clear min_hits/min_coverage, with per-item attribution when an "
    ".items sidecar is present. 'item' is stricter: a single protected item must "
    "clear them on its own (needs the .items sidecar). 'auto' is an alias for 'union'."
)
_MIN_LONGEST_RUN_HELP = (
    "With --gate-mode item, also require a contiguous run of >= this many normalized "
    "tokens shared with the matching item (0 disables)."
)
_BOILERPLATE_THRESHOLD_HELP = (
    "Item fan-out at/above which a matched n-gram is counted as boilerplate in the "
    "per-item attribution split (unique/shared/boilerplate); must be >= 2."
)


# -----------------------------
# Logging
# -----------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"corpus-assay {library_version()}")
        raise typer.Exit()


@app.callback()
def configure_logging(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only log errors."),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the corpus-assay version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    ngram_config: Path | None = typer.Option(
        None,
        "--ngram-config",
        help=(
            "Path to a custom ngram normalization config JSON (stop-words, "
            "word regex, reject patterns). Defaults to the bundled config. "
            "Applied before any command runs."
        ),
        exists=True,
        dir_okay=False,
    ),
) -> None:
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet cannot be used together.")

    level = logging.INFO if verbose else logging.ERROR if quiet else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if ngram_config is not None:
        from corpus_assay.normalization import set_ngram_config_path

        set_ngram_config_path(str(ngram_config))


def _run_scan(scan_cfg: ScanConfig, out_dir: Path) -> None:
    run = do_scan(scan_cfg)
    write_run_outputs(layout=OutputLayout(out_dir), run=run, cfg=scan_cfg)


def _sample_paths(paths: list[Path], *, limit: int = 10) -> None:
    for path in paths[:limit]:
        typer.echo(f"  {path}")
    if len(paths) > limit:
        typer.echo(f"  ... {len(paths) - limit} more")


def _echo_paths(prefix: str, paths: Iterable[Path]) -> None:
    typer.echo(prefix)
    for path in paths:
        typer.echo(str(path))


def _resolve_stopgrams_path(
    cfg: DecontaminationConfig, *, stopgrams_path: str | None
) -> str | None:
    if stopgrams_path is not None:
        return stopgrams_path
    if cfg.stopgrams and cfg.stopgrams.enabled:
        return cfg.stopgrams.path
    return None


def _validate_stopgrams_compatibility(
    scan_cfg: ScanConfig, *, allow_unverified_stopgrams: bool = False
) -> None:
    if not scan_cfg.stopgrams_path:
        return
    from corpus_assay.stopgrams import validate_stopgrams_compatibility

    stopgrams_path = Path(scan_cfg.stopgrams_path)
    if not stopgrams_path.exists():
        raise typer.BadParameter(f"Stop-grams file not found: {stopgrams_path}")

    index_path = Path(scan_cfg.index_path)
    index_meta_path = index_path.with_name(f"{index_path.name}.meta.json")
    if not index_meta_path.exists():
        raise typer.BadParameter(f"Index meta not found: {index_meta_path}")
    try:
        index_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Index meta is invalid JSON: {index_meta_path}"
        ) from exc

    meta_path = stopgrams_meta_path(stopgrams_path)
    if not meta_path.exists():
        if allow_unverified_stopgrams:
            typer.echo(
                f"Warning: stop-grams meta not found: {meta_path}. Skipping compatibility checks.",
                err=True,
            )
            return
        raise typer.BadParameter(f"Stop-grams meta not found: {meta_path}")

    try:
        stopgrams_meta = load_stopgrams_meta(stopgrams_path)
        validate_stopgrams_compatibility(
            stopgrams_meta=stopgrams_meta,
            index_meta=index_meta,
            expected_ngram=scan_cfg.ngram,
        )
    except Exception as exc:
        if allow_unverified_stopgrams:
            typer.echo(
                f"Warning: stop-grams compatibility check skipped: {exc}", err=True
            )
            return
        raise typer.BadParameter(str(exc)) from exc


class _ResolvedSpaced(NamedTuple):
    path: str | None
    min_loci: int
    ver_min_span: int
    ver_identity: float


def _resolve_spaced(
    cfg: DecontaminationConfig, *, spaced_path: str | None
) -> _ResolvedSpaced:
    """Resolve the spaced index path *and* its verifier thresholds.

    Precedence mirrors ``_resolve_stopgrams_path``: an explicit ``--spaced-seeds`` flag
    wins (and uses the default verifier params, since the flag carries only a
    path); otherwise an enabled ``spaced:`` config block supplies both the path and its
    configured ``min_loci`` / ``ver_min_span`` / ``ver_identity`` -- which must reach
    ``ScanConfig`` so a non-default config actually changes what the scanner runs.
    """
    defaults = SpacedSeedsConfig()
    if spaced_path is not None:
        return _ResolvedSpaced(
            spaced_path,
            defaults.min_loci,
            defaults.ver_min_span,
            defaults.ver_identity,
        )
    if cfg.spaced and cfg.spaced.enabled:
        return _ResolvedSpaced(
            cfg.spaced.path,
            cfg.spaced.min_loci,
            cfg.spaced.ver_min_span,
            cfg.spaced.ver_identity,
        )
    return _ResolvedSpaced(
        None, defaults.min_loci, defaults.ver_min_span, defaults.ver_identity
    )


def _resolve_spaced_path(
    cfg: DecontaminationConfig, *, spaced_path: str | None
) -> str | None:
    return _resolve_spaced(cfg, spaced_path=spaced_path).path


def _validate_spaced_compatibility(
    scan_cfg: ScanConfig, *, allow_unverified_spaced: bool = False
) -> None:
    if not scan_cfg.spaced_path:
        return

    spaced_path = Path(scan_cfg.spaced_path)
    if not spaced_path.exists():
        raise typer.BadParameter(f"Spaced index file not found: {spaced_path}")

    index_path = Path(scan_cfg.index_path)
    index_meta_path = index_path.with_name(f"{index_path.name}.meta.json")
    if not index_meta_path.exists():
        raise typer.BadParameter(f"Index meta not found: {index_meta_path}")
    try:
        index_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Index meta is invalid JSON: {index_meta_path}"
        ) from exc

    meta_path = spaced_meta_path(spaced_path)
    if not meta_path.exists():
        if allow_unverified_spaced:
            typer.echo(
                f"Warning: spaced meta not found: {meta_path}. Skipping compatibility checks.",
                err=True,
            )
            return
        raise typer.BadParameter(f"Spaced index meta not found: {meta_path}")

    try:
        spaced_meta = load_spaced_meta(spaced_path)
        validate_spaced_compatibility(
            spaced_meta=spaced_meta,
            index_meta=index_meta,
            expected_ngram=scan_cfg.ngram,
        )
    except Exception as exc:
        if allow_unverified_spaced:
            typer.echo(f"Warning: spaced compatibility check skipped: {exc}", err=True)
            return
        raise typer.BadParameter(str(exc)) from exc


def _validate_index_config_compatibility(
    scan_cfg: ScanConfig, *, allow_config_mismatch: bool = False
) -> None:
    """Fail when the index was built under a different ngram normalization config.

    The scanner normalizes documents with the active config (bundled default or a
    global ``--ngram-config`` override). Matching them against an index built under
    another config silently misses contamination, so refuse unless overridden.
    """
    index_path = Path(scan_cfg.index_path)
    index_meta_path = index_path.with_name(f"{index_path.name}.meta.json")
    if not index_meta_path.exists():
        return  # the scanner reports the missing meta itself
    try:
        index_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Index meta is invalid JSON: {index_meta_path}"
        ) from exc
    index_hash = index_meta.get("ngram_config_hash")
    if not index_hash:
        return  # older index without a recorded config hash; nothing to compare
    active_hash = ngram_config_hash()
    if index_hash == active_hash:
        return
    message = (
        "Index was built with a different ngram normalization config "
        f"(index {index_hash[:12]}, active {active_hash[:12]}). Pass the same "
        "--ngram-config that was used for build-index"
    )
    if allow_config_mismatch:
        typer.echo(
            f"Warning: {message}; continuing (--allow-config-mismatch).", err=True
        )
        return
    raise typer.BadParameter(f"{message}, or --allow-config-mismatch to scan anyway.")


# -----------------------------
# Commands
# -----------------------------


@app.command("build-index")
def build_index(
    out_index: str = typer.Option(
        ..., help="Output path for native index (e.g., index.native)."
    ),
    config: str = typer.Option(
        ..., help="YAML config with decontamination hyperparameters."
    ),
    allow_unpinned: bool = typer.Option(
        False,
        "--allow-unpinned",
        help="Allow benchmarks without a pinned revision.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan index build only; do not execute."
    ),
) -> None:
    """Download HF benchmarks and build native index (.native)."""
    cfg = load_decontamination_config(config)
    if dry_run:
        typer.echo(f"Would build native index at: {out_index}")
        typer.echo(f"Benchmarks: {len(cfg.benchmarks)}")
        typer.echo(f"ngram={cfg.ngram}")
        raise typer.Exit(code=0)
    build_index_hf(
        BuildIndexConfig(
            out_path=out_index,
            ngram=cfg.ngram,
            benchmarks=cfg.benchmarks,
            allow_unpinned=allow_unpinned,
        )
    )


@app.command("build-stopgrams-direct")
def build_stopgrams_direct(
    config: str = typer.Option(
        ..., help="YAML config with background corpus settings."
    ),
    tau: float = typer.Option(..., help="Fractional DF cutoff (e.g., 0.001)."),
    out: str = typer.Option(..., help="Output path for stop-grams native file."),
    workers: int = typer.Option(8, help="Number of workers to use."),
    max_docs: int | None = typer.Option(
        None,
        "--max-docs",
        help="Optional max sampled docs to process (applies after sampling).",
    ),
    sample_prob: float | None = typer.Option(
        None, "--sample-prob", help="Optional sampling probability (0 < p < 1)."
    ),
    seed: int = typer.Option(0, help="Random seed for sampling."),
    tmp_dir: str | None = typer.Option(
        None, "--tmp-dir", help="Optional directory for temp part files."
    ),
    sketch_size: int = typer.Option(
        200_000, "--sketch-size", help="Max distinct n-gram candidates per worker."
    ),
    keep_tmp: bool = typer.Option(
        False,
        "--keep-tmp",
        help="Keep temporary part files instead of cleaning them up.",
    ),
) -> None:
    """Build stop-grams directly from a background corpus."""
    if tau <= 0 or tau >= 1:
        raise typer.BadParameter("tau must be in (0, 1)")
    if sample_prob is not None and not (0 < sample_prob < 1):
        raise typer.BadParameter("sample_prob must be in (0, 1)")
    if max_docs is not None and max_docs <= 0:
        raise typer.BadParameter("max_docs must be > 0")
    if workers <= 0:
        raise typer.BadParameter("workers must be >= 1")

    cfg = load_decontamination_config(config)
    if not cfg.background or not cfg.background.corpus:
        raise typer.BadParameter("Config missing background.corpus settings.")
    corpus = cfg.background.corpus

    build_stopgrams_direct_from_hf_dataset(
        dataset=corpus.dataset,
        local_path=corpus.local_path,
        config_name=corpus.config_name,
        split=corpus.split,
        revision=corpus.revision,
        fields=corpus.fields,
        ngram=cfg.ngram,
        packed_doc_sep=corpus.packed_doc_sep,
        packed_doc_sep_typo=corpus.packed_doc_sep_typo,
        tau=tau,
        out_path=out,
        workers=workers,
        max_docs=max_docs,
        sample_prob=sample_prob,
        seed=seed,
        tmp_dir=tmp_dir,
        sketch_size=sketch_size,
        cleanup_parts=not keep_tmp,
    )


@app.command("build-spaced-index")
def build_spaced_index_cmd(
    config: str = typer.Option(
        ..., help="YAML config with the protected benchmarks + ngram."
    ),
    out: str = typer.Option(
        ..., help="Output path for the spaced index (e.g., index.spaced)."
    ),
    allow_unpinned: bool = typer.Option(
        False,
        "--allow-unpinned",
        help="Allow benchmarks without a pinned revision.",
    ),
    min_loci: int = typer.Option(
        3, "--min-loci", help="Min distinct seed loci per (item, diagonal) to verify."
    ),
    ver_min_span: int = typer.Option(
        17, "--ver-min-span", help="Min comparable token span for verification."
    ),
    ver_identity: float = typer.Option(
        0.85, "--ver-identity", help="Min token identity fraction for verification."
    ),
    class_aware: str = typer.Option(
        "none",
        "--class-aware",
        help="Token class map for seeds; only 'none' is currently supported.",
    ),
) -> None:
    """[Experimental] Build a .spaced index for the spaced-seed recall channel.

    The protected eval items are loaded from the same benchmarks as ``build-index`` and
    serialized with their raw tokens (for verification) and seed postings. Scan with
    ``--spaced-seeds <out>`` to OR the spaced-verified gate into the exact gate.
    """
    if not (0.0 <= ver_identity <= 1.0):
        raise typer.BadParameter("ver_identity must be in [0, 1]")
    if min_loci <= 0 or ver_min_span <= 0:
        raise typer.BadParameter("min_loci and ver_min_span must be > 0")
    # The Rust scanner builds seed strings from raw tokens (no class-aware mapping;
    # see src/spaced.rs), so a class-collapsed artifact would
    # produce postings the scanner never matches. Reject non-'none' until the scan
    # path applies the same mapping.
    if class_aware != "none":
        raise typer.BadParameter(
            f"class_aware={class_aware!r} is not supported by the Rust spaced scanner; "
            "only 'none' is supported (seeds are hashed from raw tokens)."
        )

    from datetime import datetime, timezone

    from corpus_assay.indexing import build_eval_item_tokens

    cfg = load_decontamination_config(config)
    if not allow_unpinned:
        missing = [spec.name for spec in cfg.benchmarks if not spec.revision]
        if missing:
            raise typer.BadParameter(
                "Benchmarks missing revision pin: "
                f"{', '.join(missing)}. Use --allow-unpinned to override."
            )

    item_tokens, fingerprints = build_eval_item_tokens(cfg.benchmarks)
    family = SpacedSeedFamily(patterns=DEFAULT_PATTERNS)
    seed_cfg = SpacedSeedConfig(
        min_loci=min_loci, ver_min_span=ver_min_span, ver_identity=ver_identity
    )
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_spaced_index(
        out,
        item_tokens,
        family,
        seed_cfg,
        class_aware=class_aware,
        n=cfg.ngram,
        built_from={
            "library_version": library_version(),
            "benchmarks": [spec.name for spec in cfg.benchmarks],
            "hf_datasets": fingerprints,
        },
        created_at=created_at,
    )
    typer.echo(
        f"Wrote spaced index: {out} ({len(item_tokens):,} items, "
        f"min_loci={min_loci} ver_min_span={ver_min_span} ver_identity={ver_identity})"
    )


@app.command("init-config")
def init_config(
    benchmarks: list[str] = typer.Option(
        [],
        "--benchmarks",
        help="Benchmark registry refs to expand (repeatable).",
    ),
    out: str = typer.Option(..., "--out", help="Output YAML config path."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite the output file if it exists."
    ),
) -> None:
    """Write a starter decontamination config from registry benchmarks."""
    out_path = Path(out)
    if out_path.exists() and not force:
        raise typer.BadParameter(
            f"Output file already exists: {out_path}. Use --force to overwrite."
        )

    if not benchmarks:
        raise typer.BadParameter("Provide at least one --benchmarks value.")

    specs = [REGISTRY.resolve_ref(name) for name in benchmarks]
    cfg = DecontaminationConfig.model_validate({"benchmarks": specs})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8"
    )
    logger.info("Wrote config with benchmarks to %s", out_path)


@app.command("list-benchmarks")
def list_benchmarks() -> None:
    """List benchmark registry keys."""
    for key in REGISTRY.list_registry():
        typer.echo(key)


@app.command("inspect-index")
def inspect_index_cmd(
    index: str = typer.Option(..., help="Path to the .native index."),
) -> None:
    """Validate an index and its sidecars."""
    info = inspect_index(Path(index))
    typer.echo("Index inspection:")
    typer.echo(f"  ngram: {info['ngram']}")
    typer.echo(f"  index_format_version: {info['index_format_version']}")
    typer.echo(
        f"  native_records: {info['native_records']} (header={info['native_header']})"
    )
    typer.echo(f"  attr_records: {info['attr_records']} (header={info['attr_header']})")


@app.command("inspect-run")
def inspect_run_cmd(
    scan_dir: str = typer.Option(
        ..., help="Scan output directory containing _RUN_MANIFEST.json and artifacts."
    ),
    validate_ndjson: bool = typer.Option(
        True, "--validate-ndjson/--no-validate-ndjson", help="Validate NDJSON samples."
    ),
    max_ndjson_lines: int | None = typer.Option(
        None,
        "--max-ndjson-lines",
        help="Optional cap on NDJSON lines to validate.",
    ),
) -> None:
    """Validate scan outputs in a run directory."""
    info = inspect_run(
        Path(scan_dir),
        validate_ndjson=validate_ndjson,
        max_ndjson_lines=max_ndjson_lines,
    )
    typer.echo("Run inspection:")
    typer.echo(f"  summaries: {info['summaries']}")
    typer.echo(f"  hits_files: {info['hits_files']}")
    typer.echo(f"  validated_ndjson: {info['validated_ndjson']}")
    typer.echo(f"  validated_ndjson_files: {info['validated_ndjson_files']}")
    typer.echo(f"  validated_ndjson_lines: {info['validated_ndjson_lines']}")


@app.command("reproduce")
def reproduce(
    manifest: str = typer.Option(
        ..., help="Path to _RUN_MANIFEST.json for a prior scan run."
    ),
) -> None:
    """Re-run a scan from a run manifest, verifying recorded inputs."""
    manifest_path = Path(manifest)
    payload = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    base_dir = manifest_path.parent
    mismatches: list[str] = []
    for shard in payload.inputs.parquet_shards:
        shard_path = Path(shard.path)
        if not shard_path.is_absolute():
            shard_path = base_dir / shard_path
        if shard.sha256:
            computed = _file_sha256(shard_path)
            if computed != shard.sha256:
                mismatches.append(
                    f"{shard_path} expected {shard.sha256} got {computed}"
                )

    if mismatches:
        raise typer.BadParameter(
            "Parquet shard hash mismatch:\n" + "\n".join(mismatches)
        )

    for hf_entry in payload.inputs.hf_datasets:
        load_kwargs = {
            "split": hf_entry.split,
            "revision": hf_entry.revision,
            "data_files": hf_entry.data_files,
            "streaming": False,
        }
        if hf_entry.config_name:
            dataset = _load_dataset_optional_revision(
                hf_entry.hf_repo,
                name=hf_entry.config_name,
                **load_kwargs,
            )
        else:
            dataset = _load_dataset_optional_revision(
                hf_entry.hf_repo,
                **load_kwargs,
            )
        fingerprint = getattr(dataset, "_fingerprint", None)
        if hf_entry.dataset_fingerprint and fingerprint:
            if fingerprint != hf_entry.dataset_fingerprint:
                raise typer.BadParameter(
                    f"HF dataset fingerprint mismatch for {hf_entry.hf_repo} "
                    f"(expected {hf_entry.dataset_fingerprint}, got {fingerprint})."
                )

    typer.echo("Inputs verified OK.")

    if payload.config is None:
        raise typer.BadParameter("Run manifest missing full scan config.")

    scan_cfg = ScanConfig.model_validate(payload.config)
    if payload.index.meta_path is None:
        raise typer.BadParameter("Index meta missing.")
    computed = _file_sha256(Path(payload.index.meta_path))
    if payload.index.meta_sha256 and computed != payload.index.meta_sha256:
        raise typer.BadParameter("Index meta sha256 mismatch ...")

    out_dir_p = Path(scan_cfg.out_dir)
    _validate_stopgrams_compatibility(scan_cfg)
    _validate_spaced_compatibility(scan_cfg)
    _validate_index_config_compatibility(scan_cfg)
    _run_scan(scan_cfg, out_dir_p)


@app.command("describe-benchmark")
def describe_benchmark(
    name: str = typer.Argument(..., help="Registry benchmark key."),
) -> None:
    """Describe a benchmark registry entry."""
    spec = REGISTRY.resolve_ref(name)
    typer.echo(yaml.safe_dump(spec.model_dump(), sort_keys=False))


@app.command("validate-config")
def validate_config(
    config: str = typer.Option(..., help="Path to YAML config."),
    check_remote: bool = typer.Option(
        False,
        "--check-remote",
        help="Attempt to resolve configs/splits via HF metadata.",
    ),
    allow_unpinned: bool = typer.Option(
        False,
        "--allow-unpinned",
        help="Allow benchmarks without a pinned revision.",
    ),
) -> None:
    """Validate a config file (optionally resolving benchmarks on the HF Hub)."""
    validator = ConfigValidator(config_loader=load_decontamination_config)
    report = validator.validate(
        config, check_remote=check_remote, allow_unpinned=allow_unpinned
    )

    typer.echo("Config OK.")
    typer.echo(f"Benchmarks: {report.benchmark_count}")
    typer.echo(
        f"ngram={report.ngram} min_hits={report.min_hits} "
        f"min_coverage={report.min_coverage}"
    )

    if check_remote:
        for warning in report.warnings:
            typer.echo(warning)
        if report.remote_skipped:
            typer.echo(
                f"Remote checks: skipped {report.remote_skipped}, "
                f"checked {report.remote_checked}."
            )
            raise typer.Exit(code=2)
        typer.echo("Remote checks OK.")


@app.command("scan")
def scan(
    inputs: list[str] = typer.Option(
        ...,
        "--input",
        "--input-glob",
        help=(
            "Input path(s): directory, glob (recursive), or file list. Repeat --input for multiple."
        ),
    ),
    index: str = typer.Option(..., help="Path to the .native index."),
    out_dir: str = typer.Option(
        ..., help="Directory for per-file outputs and summaries."
    ),
    text_key: str = typer.Option(
        "text", help="Field name containing document text (default: text)."
    ),
    id_key: str | None = typer.Option(
        None, help="Optional stable doc id field (otherwise hashed from text)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan scan only; do not execute."
    ),
    workers: int = typer.Option(
        os.cpu_count() or 8,
        help="Parallel worker processes.",
        show_default="CPU count",
    ),
    index_backend: str = typer.Option(
        "auto", help="Index backend: auto, memory, or mmap."
    ),
    gate_mode: str = typer.Option("union", help=_GATE_MODE_HELP),
    min_longest_run: int = typer.Option(0, help=_MIN_LONGEST_RUN_HELP),
    boilerplate_threshold: int = typer.Option(50, help=_BOILERPLATE_THRESHOLD_HELP),
    max_return_records: int | None = typer.Option(
        500,
        help=(
            "Max contaminated document records to return per shard "
            "(sample only; use 0 to disable)."
        ),
    ),
    config: str = typer.Option(
        ..., help="YAML config with decontamination hyperparameters."
    ),
    stopgrams: str | None = typer.Option(
        None,
        "--stopgrams",
        help="Path to stop-grams native file (optional).",
    ),
    allow_unverified_stopgrams: bool = typer.Option(
        False,
        "--allow-unverified-stopgrams",
        help="Allow stop-grams without a compatible meta file.",
    ),
    spaced_seeds: str | None = typer.Option(
        None,
        "--spaced-seeds",
        help="[Experimental] Path to a .spaced index for the spaced-seed recall channel.",
    ),
    allow_unverified_spaced: bool = typer.Option(
        False,
        "--allow-unverified-spaced",
        help="Allow a spaced index without a compatible meta file.",
    ),
    allow_config_mismatch: bool = typer.Option(
        False,
        "--allow-config-mismatch",
        help="Scan even if the index was built with a different ngram config.",
    ),
) -> None:
    """Scan dataset shards using the fast Rust worker."""
    cfg = load_decontamination_config(Path(config))
    out_dir_p = Path(out_dir)

    if dry_run:
        plan = plan_scan(inputs, index, out_dir_p, check_index=True)
        typer.echo(f"Would scan {len(plan.files)} files")
        _sample_paths(plan.files)
        if not plan.index_ok:
            missing_list = ", ".join(str(path) for path in plan.missing_index_files)
            raise typer.BadParameter(f"Index missing required files: {missing_list}")
        typer.echo(f"Using config: {config}")
        typer.echo(
            f"ngram={cfg.ngram} min_hits={cfg.min_hits} min_coverage={cfg.min_coverage}"
        )
        typer.echo(f"Would write outputs under: {plan.out_dir}")
        raise typer.Exit(code=0)

    resolved_stopgrams = _resolve_stopgrams_path(cfg, stopgrams_path=stopgrams)
    resolved_spaced = _resolve_spaced(cfg, spaced_path=spaced_seeds)
    scan_cfg = ScanConfig(
        inputs=inputs,
        text_key=text_key,
        id_key=id_key,
        index_path=index,
        ngram=cfg.ngram,
        min_hits=cfg.min_hits,
        min_coverage=cfg.min_coverage,
        out_dir=str(out_dir_p),
        workers=workers,
        index_backend=index_backend,
        max_return_records=max_return_records,
        stopgrams_path=resolved_stopgrams,
        spaced_path=resolved_spaced.path,
        spaced_min_loci=resolved_spaced.min_loci,
        spaced_ver_min_span=resolved_spaced.ver_min_span,
        spaced_ver_identity=resolved_spaced.ver_identity,
        gate_mode=gate_mode,
        min_longest_run=min_longest_run,
        boilerplate_threshold=boilerplate_threshold,
    )
    _validate_stopgrams_compatibility(
        scan_cfg, allow_unverified_stopgrams=allow_unverified_stopgrams
    )
    _validate_spaced_compatibility(
        scan_cfg, allow_unverified_spaced=allow_unverified_spaced
    )
    _validate_index_config_compatibility(
        scan_cfg, allow_config_mismatch=allow_config_mismatch
    )
    _run_scan(scan_cfg, out_dir_p)


@app.command("run-hf")
def run_hf(
    config: str = typer.Option(
        ..., help="YAML config with decontamination hyperparameters."
    ),
    hf_dataset: str = typer.Option(..., help="HF dataset name (e.g., 'allenai/c4')."),
    split: str = typer.Option("train", help="Dataset split to scan."),
    hf_revision: str | None = typer.Option(
        None, "--hf-revision", help="Pinned HF dataset revision (commit hash or tag)."
    ),
    out_dir: str = typer.Option(
        ..., help="Directory for per-file outputs and summaries."
    ),
    text_key: str = typer.Option(
        "text", help="Field name containing document text (default: text)."
    ),
    id_key: str | None = typer.Option(
        None, help="Optional stable doc id field (otherwise hashed from text)."
    ),
    workers: int = typer.Option(
        os.cpu_count() or 8,
        help="Parallel worker processes.",
        show_default="CPU count",
    ),
    index_backend: str = typer.Option(
        "auto", help="Index backend: auto, memory, or mmap."
    ),
    gate_mode: str = typer.Option("union", help=_GATE_MODE_HELP),
    min_longest_run: int = typer.Option(0, help=_MIN_LONGEST_RUN_HELP),
    boilerplate_threshold: int = typer.Option(50, help=_BOILERPLATE_THRESHOLD_HELP),
    max_return_records: int | None = typer.Option(
        500,
        help=(
            "Max contaminated document records to return per shard "
            "(sample only; use 0 to disable)."
        ),
    ),
    allow_unpinned: bool = typer.Option(
        False,
        "--allow-unpinned",
        help="Allow benchmarks without a pinned revision.",
    ),
    index_cache_dir: str = typer.Option(
        str(DEFAULT_INDEX_CACHE_DIR),
        help="Directory to cache built native indexes (honors XDG_CACHE_HOME).",
        show_default="~/.cache/corpus-assay/indexes",
    ),
    hf_cache_dir: str | None = typer.Option(
        None,
        help="Optional Hugging Face datasets cache dir (defaults to HF cache).",
    ),
    parquet_rows_per_file: int | None = typer.Option(
        500_000,
        help=(
            "Max rows per parquet shard when materializing; "
            "use 0 to force a single file."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan run only; do not execute."
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Allow execution of dataset loading code from the Hub.",
    ),
    stopgrams: str | None = typer.Option(
        None,
        "--stopgrams",
        help="Path to stop-grams native file (optional).",
    ),
    allow_unverified_stopgrams: bool = typer.Option(
        False,
        "--allow-unverified-stopgrams",
        help="Allow stop-grams without a compatible meta file.",
    ),
    spaced_seeds: str | None = typer.Option(
        None,
        "--spaced-seeds",
        help="[Experimental] Path to a .spaced index for the spaced-seed recall channel.",
    ),
    allow_unverified_spaced: bool = typer.Option(
        False,
        "--allow-unverified-spaced",
        help="Allow a spaced index without a compatible meta file.",
    ),
    allow_config_mismatch: bool = typer.Option(
        False,
        "--allow-config-mismatch",
        help="Scan even if the index was built with a different ngram config.",
    ),
) -> None:
    """Build (or reuse) a native index and scan a HF dataset split."""
    cfg_path = Path(config)
    cfg = load_decontamination_config(cfg_path)

    if parquet_rows_per_file == 0:
        parquet_rows_per_file = None

    out_dir_p = Path(out_dir)

    if dry_run:
        cache_dir = Path(index_cache_dir)
        native_index = index_cache_path(cfg_path, cache_dir)
        if native_index_ready(native_index):
            typer.echo(f"Would use cached native index: {native_index}")
        else:
            typer.echo(f"Would build native index: {native_index}")

        parquet_dir = out_dir_p / "hf_parquet" / dataset_slug(hf_dataset) / split
        existing_parquet = existing_parquet_files(parquet_dir)
        if existing_parquet:
            typer.echo(
                f"Found {len(existing_parquet)} existing parquet shard(s) in {parquet_dir}"
            )
            _sample_paths(existing_parquet)
        else:
            typer.echo("No materialized parquet shards found.")
            if parquet_rows_per_file is None:
                typer.echo("Would materialize a single parquet file per split.")
            else:
                typer.echo(
                    "Would materialize parquet shards with up to "
                    f"{parquet_rows_per_file:,} rows each in {parquet_dir}"
                )

        typer.echo(f"Would scan split '{split}' from dataset '{hf_dataset}'.")
        typer.echo(f"Would write outputs under: {out_dir_p}")
        raise typer.Exit(code=0)

    native_index = _ensure_native_index(
        cfg_path, Path(index_cache_dir), allow_unpinned=allow_unpinned
    )

    if trust_remote_code:
        logger.warning("!" * 72)
        logger.warning("WARNING: --trust-remote-code executes code from the Hub.")
        logger.warning("Only enable this flag if you trust the dataset authors.")
        logger.warning("!" * 72)

    dataset = _load_dataset_optional_revision(
        hf_dataset,
        split=split,
        cache_dir=hf_cache_dir,
        streaming=False,
        trust_remote_code=trust_remote_code,
        revision=hf_revision,
    )
    dataset_fingerprint = getattr(dataset, "_fingerprint", None)
    parquet_paths = resolve_hf_parquet_paths(
        dataset,
        hf_dataset=hf_dataset,
        split=split,
        out_dir=out_dir_p,
        max_rows_per_file=parquet_rows_per_file,
    )
    _echo_paths("Using parquet shards:", parquet_paths)

    resolved_stopgrams = _resolve_stopgrams_path(cfg, stopgrams_path=stopgrams)
    resolved_spaced = _resolve_spaced(cfg, spaced_path=spaced_seeds)
    scan_cfg = ScanConfig(
        inputs=[str(path) for path in parquet_paths],
        text_key=text_key,
        id_key=id_key,
        index_path=str(native_index),
        ngram=cfg.ngram,
        min_hits=cfg.min_hits,
        min_coverage=cfg.min_coverage,
        out_dir=str(out_dir_p),
        workers=workers,
        index_backend=index_backend,
        max_return_records=max_return_records,
        target_hf_dataset=hf_dataset,
        target_hf_split=split,
        target_hf_revision=hf_revision,
        target_hf_fingerprint=dataset_fingerprint,
        stopgrams_path=resolved_stopgrams,
        spaced_path=resolved_spaced.path,
        spaced_min_loci=resolved_spaced.min_loci,
        spaced_ver_min_span=resolved_spaced.ver_min_span,
        spaced_ver_identity=resolved_spaced.ver_identity,
        gate_mode=gate_mode,
        min_longest_run=min_longest_run,
        boilerplate_threshold=boilerplate_threshold,
    )
    _validate_stopgrams_compatibility(
        scan_cfg, allow_unverified_stopgrams=allow_unverified_stopgrams
    )
    _validate_spaced_compatibility(
        scan_cfg, allow_unverified_spaced=allow_unverified_spaced
    )
    _validate_index_config_compatibility(
        scan_cfg, allow_config_mismatch=allow_config_mismatch
    )
    _run_scan(scan_cfg, out_dir_p)


@app.command("run")
def run(
    config: str = typer.Option(
        ..., help="YAML config with decontamination hyperparameters."
    ),
    inputs: list[str] = typer.Option(
        ...,
        "--input",
        "--input-glob",
        help=(
            "Input path(s): directory, glob (recursive), or file list. Repeat --input for multiple."
        ),
    ),
    out_dir: str = typer.Option(
        ..., help="Directory for per-file outputs and summaries."
    ),
    text_key: str = typer.Option(
        "text", help="Field name containing document text (default: text)."
    ),
    id_key: str | None = typer.Option(
        None, help="Optional stable doc id field (otherwise hashed from text)."
    ),
    workers: int = typer.Option(
        os.cpu_count() or 8,
        help="Parallel worker processes.",
        show_default="CPU count",
    ),
    index_backend: str = typer.Option(
        "auto", help="Index backend: auto, memory, or mmap."
    ),
    gate_mode: str = typer.Option("union", help=_GATE_MODE_HELP),
    min_longest_run: int = typer.Option(0, help=_MIN_LONGEST_RUN_HELP),
    boilerplate_threshold: int = typer.Option(50, help=_BOILERPLATE_THRESHOLD_HELP),
    max_return_records: int | None = typer.Option(
        500,
        help=(
            "Max contaminated document records to return per shard "
            "(sample only; use 0 to disable)."
        ),
    ),
    allow_unpinned: bool = typer.Option(
        False,
        "--allow-unpinned",
        help="Allow benchmarks without a pinned revision.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan run only; do not execute."
    ),
    index_cache_dir: str = typer.Option(
        str(DEFAULT_INDEX_CACHE_DIR),
        help="Directory to cache built native indexes (honors XDG_CACHE_HOME).",
        show_default="~/.cache/corpus-assay/indexes",
    ),
    stopgrams: str | None = typer.Option(
        None,
        "--stopgrams",
        help="Path to stop-grams native file (optional).",
    ),
    allow_unverified_stopgrams: bool = typer.Option(
        False,
        "--allow-unverified-stopgrams",
        help="Allow stop-grams without a compatible meta file.",
    ),
    spaced_seeds: str | None = typer.Option(
        None,
        "--spaced-seeds",
        help="[Experimental] Path to a .spaced index for the spaced-seed recall channel.",
    ),
    allow_unverified_spaced: bool = typer.Option(
        False,
        "--allow-unverified-spaced",
        help="Allow a spaced index without a compatible meta file.",
    ),
    allow_config_mismatch: bool = typer.Option(
        False,
        "--allow-config-mismatch",
        help="Scan even if the index was built with a different ngram config.",
    ),
) -> None:
    """Build (or reuse) a native index and scan dataset shards in one command."""
    cfg_path = Path(config)
    cfg = load_decontamination_config(cfg_path)

    out_dir_p = Path(out_dir)
    if dry_run:
        cache_dir = Path(index_cache_dir)
        native_index = index_cache_path(cfg_path, cache_dir)
        if native_index_ready(native_index):
            typer.echo(f"Would use cached native index: {native_index}")
        else:
            typer.echo(f"Would build native index: {native_index}")

        plan = plan_scan(inputs, native_index, out_dir_p, check_index=False)
        typer.echo(f"Would scan {len(plan.files)} files")
        _sample_paths(plan.files)
        typer.echo(f"Would write outputs under: {plan.out_dir}")
        raise typer.Exit(code=0)

    native_index = _ensure_native_index(
        cfg_path, Path(index_cache_dir), allow_unpinned=allow_unpinned
    )
    resolved_stopgrams = _resolve_stopgrams_path(cfg, stopgrams_path=stopgrams)
    resolved_spaced = _resolve_spaced(cfg, spaced_path=spaced_seeds)
    scan_cfg = ScanConfig(
        inputs=inputs,
        text_key=text_key,
        id_key=id_key,
        index_path=str(native_index),
        ngram=cfg.ngram,
        min_hits=cfg.min_hits,
        min_coverage=cfg.min_coverage,
        out_dir=str(out_dir_p),
        workers=workers,
        index_backend=index_backend,
        max_return_records=max_return_records,
        stopgrams_path=resolved_stopgrams,
        spaced_path=resolved_spaced.path,
        spaced_min_loci=resolved_spaced.min_loci,
        spaced_ver_min_span=resolved_spaced.ver_min_span,
        spaced_ver_identity=resolved_spaced.ver_identity,
        gate_mode=gate_mode,
        min_longest_run=min_longest_run,
        boilerplate_threshold=boilerplate_threshold,
    )
    _validate_stopgrams_compatibility(
        scan_cfg, allow_unverified_stopgrams=allow_unverified_stopgrams
    )
    _validate_spaced_compatibility(
        scan_cfg, allow_unverified_spaced=allow_unverified_spaced
    )
    _validate_index_config_compatibility(
        scan_cfg, allow_config_mismatch=allow_config_mismatch
    )
    _run_scan(scan_cfg, out_dir_p)


@app.command("report")
def report(
    scan_dir: str = typer.Option(
        "scan_results",
        help="Scan output directory containing _RUN_MANIFEST.json and artifacts.",
    ),
    out_dir: str | None = typer.Option(
        None,
        help="Where to write report artifacts (default: <scan_dir>/report).",
    ),
    topk_overall: int = typer.Option(50, help="Top-K shards to show overall."),
    topk_per_benchmark: int = typer.Option(
        20, help="Top-K shards to show per benchmark."
    ),
) -> None:
    """Generate reporting outputs from an existing scan directory."""
    report_payload = generate_report_from_scan_dir(
        scan_dir,
        out_dir=out_dir,
        topk_overall=topk_overall,
        topk_per_benchmark=topk_per_benchmark,
    )

    typer.echo("Top benchmarks by leak fraction:")
    for row in report_payload.leak_by_benchmark[:5]:
        typer.echo(
            f"  {row.benchmark}: {row.leak_fraction:.2%} "
            f"({row.leaked_hashes}/{row.total_hashes_in_index})"
        )

    typer.echo("Top leaking shards:")
    for row in report_payload.bad_shards.top_overall[:5]:
        typer.echo(
            f"  {row.shard}: {row.unique_leaked_hashes_total} "
            f"unique hashes ({row.unique_leaked_hashes_per_1k_docs:.2f}/1k docs)"
        )


def main() -> None:
    app()


__all__ = ["main", "app"]
