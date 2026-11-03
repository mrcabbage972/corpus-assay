"""Plant benchmark items into clean text and measure recall (the ``selftest`` command).

Items are sampled from the index's ``.items.text.jsonl`` sidecar. Each is planted
(its fields one per line, list fields one element per line) under each requested
variant into a host document: synthetic filler text, or a copy of one of the
first records of the input shards. The hosts are also kept unplanted, as
controls. The planted rows go to ``planted.parquet`` (with the scan's text, chat
and id columns) and the ground truth to ``truth.jsonl``.

The planted file is then scanned with the run's own settings once per gate
(``union``, and ``item`` when the index has an ``.items`` sidecar) and, when the
exact channel is on, once more with the configured gate and the exact channel.
A planted row counts as found when the scan flags any of its sub-documents.

Recall is taken over the items a run can match: an item with fewer than
``min_hits`` indexed n-grams can never be flagged by the n-gram rule, so it only
counts where the exact channel is on and would match it. Items no enabled
channel can match are listed apart.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from corpus_assay._native import light_normalize
from corpus_assay.config import ScanConfig
from corpus_assay.constants import ITEMS_TEXT_SUFFIX
from corpus_assay.scanner.fingerprint import library_version, rust_extension_version
from corpus_assay.scanner.io import (
    build_index_info,
    build_scan_config_info,
    parse_contamination_record,
)
from corpus_assay.scanner.layout import OutputLayout
from corpus_assay.scanner.runner import collect_files
from corpus_assay.scanner.schema import RunSummary
from corpus_assay.schemas import RunIndexInfo, RunScanConfig

SELFTEST_SCHEMA_VERSION = 1
VARIANTS = ("verbatim", "case", "whitespace", "wrapper")
PLANTED_NAME = "planted.parquet"
TRUTH_NAME = "truth.jsonl"
REPORT_NAME = "selftest.json"
SCANS_DIR_NAME = "scans"
_PLANTED_STEM = "planted"

HostKind = Literal["synthetic", "corpus"]

# Neutral filler vocabulary for synthetic hosts: random sequences of these words
# share no 13-gram with real benchmark text.
_FILLER_WORDS = (
    "harbour weather reports mention gentle breezes over the marina while local "
    "bakeries open early and commuters queue patiently for the morning ferry "
    "gardeners trim hedges along quiet lanes as cyclists pass the old mill and "
    "children chase kites across the meadow near a painted fence beside tall "
    "poplars where swallows gather before autumn rain drifts over distant hills"
).split()
_WRAPPER_LEADS = (
    "Question:",
    "Answer this:",
    "Solve the following problem.",
    "Here is a question from a quiz.",
)
_WRAPPER_TAILS = (
    "Answer:",
    "Please explain your reasoning.",
    "Think step by step.",
    "Give the final answer.",
)


class SelftestError(ValueError):
    """The self-test cannot run as asked."""


# -----------------------------
# Report model
# -----------------------------


class SelftestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_idx: int
    item_id: str | None = None
    benchmark: str | None = None
    # Indexed n-grams (None when the index has no .items.meta.json) and the
    # question's length after light normalization.
    ngrams: int | None = None
    question_chars: int = 0
    ngram_matchable: bool
    exact_matchable: bool


class RecallCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matchable: int = 0
    found: int = 0
    recall: float | None = None


class MissedPlant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int
    item_idx: int
    item_id: str | None = None
    benchmark: str | None = None
    variant: str


class SelftestRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    gate_mode: str
    exact_channel: str
    scan_dir: str
    # Over the planted rows whose item this run can match.
    overall: RecallCell
    by_variant: dict[str, RecallCell] = Field(default_factory=dict)
    # Per channel (only for the run with the exact channel): `ngram` over the
    # n-gram-matchable items, `exact` over the exact-matchable ones.
    by_channel: dict[str, RecallCell] = Field(default_factory=dict)
    # Found plants whose record names the planted item (item_hits, exact_hits).
    attributed: int = 0
    missed: list[MissedPlant] = Field(default_factory=list)
    controls: int = 0
    false_positives: int = 0


class SelftestReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SELFTEST_SCHEMA_VERSION
    created_at: str
    library_version: str
    rust_extension_version: str
    index: RunIndexInfo
    scan_config: RunScanConfig
    n_items: int
    seed: int
    variants: list[str]
    host: str
    planted: str = PLANTED_NAME
    truth: str = TRUTH_NAME
    items: list[SelftestItem] = Field(default_factory=list)
    # Items no enabled channel can match (excluded from every recall figure).
    unmatchable: list[SelftestItem] = Field(default_factory=list)
    runs: list[SelftestRun] = Field(default_factory=list)
    # The run with the configured gate (and exact channel): the one --min-recall
    # applies to, on its verbatim plants (all plants without `verbatim`).
    configured_run: str
    min_recall: float
    gated_recall: float | None = None
    passed: bool


# -----------------------------
# Items and plants
# -----------------------------


@dataclass(frozen=True)
class _Item:
    item_idx: int
    item_id: str | None
    benchmark: str | None
    fields: list[Any]


def load_items(index_path: Path) -> list[_Item]:
    path = index_path.with_name(f"{index_path.name}{ITEMS_TEXT_SUFFIX}")
    if not path.exists():
        raise SelftestError(
            f"selftest needs the item text sidecar {path}, which this index lacks "
            "(it was built before the sidecar existed). Rebuild the index with "
            "build-index."
        )
    items: list[_Item] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            items.append(
                _Item(
                    item_idx=int(entry["item_idx"]),
                    item_id=entry.get("item_id"),
                    benchmark=entry.get("benchmark"),
                    fields=list(entry.get("fields") or []),
                )
            )
    if not items:
        raise SelftestError(f"The item text sidecar {path} lists no items.")
    return items


def _item_ngrams(index_path: Path) -> list[int] | None:
    meta = index_path.with_name(f"{index_path.name}.items.meta.json")
    if not meta.exists():
        return None
    items = json.loads(meta.read_text(encoding="utf-8")).get("items", [])
    return [int(entry.get("n_protected_hashes", 0)) for entry in items]


def item_text(fields: Iterable[Any]) -> str:
    """An item as planted text: one field per line, list fields one element per line."""
    lines: list[str] = []
    for value in fields:
        if isinstance(value, list):
            lines.extend(str(v) for v in value if v is not None and str(v).strip())
        elif value is not None and str(value).strip():
            lines.append(str(value))
    return "\n".join(lines)


def _question(fields: list[Any]) -> str:
    """The exact channel's pattern: the light-normalized first field."""
    if not fields:
        return ""
    first = fields[0]
    if isinstance(first, list):
        first = " ".join(str(v) for v in first)
    return light_normalize(str(first or ""))


def apply_variant(text: str, variant: str, rng: random.Random) -> tuple[str, str]:
    """``(text, label)``: ``text`` perturbed by ``variant``; the label names the
    concrete perturbation (e.g. ``case:upper``)."""
    if variant == "verbatim":
        return text, "verbatim"
    if variant == "case":
        mode = rng.choice(("upper", "lower", "title"))
        changed = {"upper": text.upper, "lower": text.lower, "title": text.title}[mode]
        return changed(), f"case:{mode}"
    if variant == "whitespace":
        mode = rng.choice(("collapse", "expand"))
        if mode == "collapse":
            return " ".join(text.split()), "whitespace:collapse"
        return re.sub(r"\n", "\n\n", text.replace(" ", "  ")), "whitespace:expand"
    if variant == "wrapper":
        lead = rng.choice(_WRAPPER_LEADS)
        tail = rng.choice(_WRAPPER_TAILS)
        return f"{lead}\n{text}\n{tail}", "wrapper"
    raise SelftestError(
        f"Unknown variant {variant!r}; choose from {', '.join(VARIANTS)}"
    )


def parse_variants(value: str) -> list[str]:
    variants = [v.strip().lower() for v in value.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or not variants:
        raise SelftestError(
            f"--variants takes a comma list of {', '.join(VARIANTS)} (got {value!r})."
        )
    return list(dict.fromkeys(variants))


def _filler(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(_FILLER_WORDS) for _ in range(n_words)]
    sentences = []
    for start in range(0, len(words), 12):
        chunk = words[start : start + 12]
        sentences.append(" ".join(chunk).capitalize() + ".")
    return " ".join(sentences)


def _insert(host: str, plant: str) -> str:
    """``plant`` as its own paragraph near the middle of ``host``."""
    if not host.strip():
        return plant
    mid = len(host) // 2
    pos = host.find(" ", mid)
    if pos < 0:
        pos = host.rfind(" ", 0, mid)
    if pos < 0:
        return f"{host}\n\n{plant}"
    return f"{host[:pos]}\n\n{plant}\n\n{host[pos + 1 :]}"


# -----------------------------
# Hosts
# -----------------------------


def _host_schema(cfg: ScanConfig) -> Any:
    import pyarrow as pa

    fields = [pa.field(key, pa.string()) for key in cfg.text_keys]
    if cfg.messages_key:
        turn = pa.struct([("role", pa.string()), ("content", pa.string())])
        fields.append(pa.field(cfg.messages_key, pa.list_(turn)))
    if cfg.id_key and cfg.id_key not in cfg.text_keys:
        fields.append(pa.field(cfg.id_key, pa.string()))
    return pa.schema(fields)


def synthetic_hosts(
    cfg: ScanConfig, count: int, rng: random.Random
) -> tuple[Any, list]:
    """``(schema, rows)``: ``count`` filler documents in the scan's columns."""
    schema = _host_schema(cfg)
    rows: list[dict[str, Any]] = []
    for _ in range(count):
        row: dict[str, Any] = {key: "" for key in cfg.text_keys}
        text = _filler(rng, 120)
        if cfg.text_keys:
            row[cfg.text_keys[0]] = text
            if cfg.messages_key:
                row[cfg.messages_key] = []
        elif cfg.messages_key:
            row[cfg.messages_key] = [{"role": "user", "content": text}]
        rows.append(row)
    return schema, rows


def corpus_hosts(cfg: ScanConfig, count: int) -> tuple[Any, list]:
    """``(schema, rows)``: copies of the first ``count`` records of the input
    shards (those with the first shard's schema), cycled when there are fewer."""
    import pyarrow as pa
    import pyarrow.json as pa_json
    import pyarrow.parquet as pq

    try:
        files = collect_files(cfg.inputs)
    except FileNotFoundError as e:
        raise SelftestError(f"--host corpus needs input shards: {e}") from e
    tables: list[Any] = []
    schema = None
    have = 0
    for path in files:
        if path.name.endswith(".parquet"):
            source = pq.ParquetFile(path)
            batch = next(source.iter_batches(batch_size=count - have), None)
            table = (
                pa.Table.from_batches([batch])
                if batch is not None
                else source.schema_arrow.empty_table()
            )
        else:
            table = pa_json.read_json(str(path)).slice(0, count - have)
        if schema is None:
            schema = table.schema
        elif not table.schema.equals(schema):
            continue
        tables.append(table)
        have += table.num_rows
        if have >= count:
            break
    if schema is None or have == 0:
        raise SelftestError("--host corpus: the input shards hold no records.")
    for key in [*cfg.text_keys, cfg.messages_key]:
        if key and key not in schema.names:
            raise SelftestError(f"--host corpus: column '{key}' not in {files[0]}")
    rows = pa.concat_tables(tables).to_pylist()
    rows = [dict(rows[i % len(rows)]) for i in range(count)]
    if cfg.id_key:
        field = pa.field(cfg.id_key, pa.string())
        index = schema.get_field_index(cfg.id_key)
        schema = schema.set(index, field) if index >= 0 else schema.append(field)
    return schema, rows


def _plant_into(row: dict[str, Any], plant: str, cfg: ScanConfig) -> None:
    if cfg.text_keys:
        key = cfg.text_keys[0]
        host = row.get(key)
        row[key] = _insert(host if isinstance(host, str) else "", plant)
        return
    key = cfg.messages_key
    assert key is not None
    turns = [dict(t) for t in (row.get(key) or []) if isinstance(t, dict)]
    for turn in reversed(turns):
        if isinstance(turn.get("content"), str):
            turn["content"] = _insert(turn["content"], plant)
            break
    else:
        turns.append({"role": "user", "content": plant})
    row[key] = turns


# -----------------------------
# The self-test
# -----------------------------


@dataclass(frozen=True)
class _Truth:
    row_index: int
    kind: str  # plant | control
    item: _Item | None = None
    variant: str | None = None
    label: str | None = None


def prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise SelftestError(
                f"--out-dir {out_dir} is not empty; pass --overwrite to replace a "
                "previous selftest there."
            )
        for name in (PLANTED_NAME, TRUTH_NAME, REPORT_NAME):
            (out_dir / name).unlink(missing_ok=True)
        shutil.rmtree(out_dir / SCANS_DIR_NAME, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def build_planted(
    cfg: ScanConfig,
    out_dir: Path,
    *,
    n_items: int,
    seed: int,
    variants: list[str],
    host: HostKind,
) -> tuple[list[_Item], list[_Truth]]:
    """Write ``planted.parquet`` and ``truth.jsonl``; return the sampled items
    and every row's truth (controls first, then one row per item and variant)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    index_path = Path(cfg.index_path)
    items = load_items(index_path)
    rng = random.Random(seed)
    sample = rng.sample(items, min(n_items, len(items)))
    n_hosts = len(sample)
    if host == "corpus":
        schema, hosts = corpus_hosts(cfg, n_hosts)
    else:
        schema, hosts = synthetic_hosts(cfg, n_hosts, rng)

    rows: list[dict[str, Any]] = []
    truth: list[_Truth] = []
    for host_row in hosts:
        rows.append(dict(host_row))
        truth.append(_Truth(row_index=len(truth), kind="control"))
    for k, item in enumerate(sample):
        base = item_text(item.fields)
        for variant in variants:
            text, label = apply_variant(base, variant, rng)
            row = dict(hosts[k % len(hosts)])
            _plant_into(row, text, cfg)
            rows.append(row)
            truth.append(
                _Truth(
                    row_index=len(truth),
                    kind="plant",
                    item=item,
                    variant=variant,
                    label=label,
                )
            )
    if cfg.id_key:
        for t, row in zip(truth, rows):
            row[cfg.id_key] = f"selftest-{t.row_index}"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_dir / PLANTED_NAME)

    with (out_dir / TRUTH_NAME).open("w", encoding="utf-8") as handle:
        for t in truth:
            entry: dict[str, Any] = {"row_index": t.row_index, "kind": t.kind}
            if cfg.id_key:
                entry["doc_id"] = f"selftest-{t.row_index}"
            if t.item is not None:
                entry.update(
                    item_idx=t.item.item_idx,
                    item_id=t.item.item_id,
                    benchmark=t.item.benchmark,
                    variant=t.variant,
                    perturbation=t.label,
                )
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return sample, truth


def _classify(
    items: list[_Item], cfg: ScanConfig, ngram_counts: list[int] | None
) -> dict[int, SelftestItem]:
    out: dict[int, SelftestItem] = {}
    for item in items:
        ngrams = None
        if ngram_counts is not None:
            idx = item.item_idx
            ngrams = ngram_counts[idx] if 0 <= idx < len(ngram_counts) else 0
        qchars = len(_question(item.fields))
        # Mirrors the exact matcher: an index without item postings counts as
        # 0 n-grams per item.
        exact = cfg.exact_channel != "off" and qchars >= cfg.exact_min_chars
        if cfg.exact_channel == "short":
            exact = exact and (ngrams or 0) < cfg.min_hits
        out[item.item_idx] = SelftestItem(
            item_idx=item.item_idx,
            item_id=item.item_id,
            benchmark=item.benchmark,
            ngrams=ngrams,
            question_chars=qchars,
            ngram_matchable=ngrams is None or ngrams >= cfg.min_hits,
            exact_matchable=exact,
        )
    return out


def _cell(found: int, matchable: int) -> RecallCell:
    return RecallCell(
        matchable=matchable,
        found=found,
        recall=(found / matchable) if matchable else None,
    )


def _evaluate(
    name: str,
    run_cfg: ScanConfig,
    truth: list[_Truth],
    classes: dict[int, SelftestItem],
    variants: list[str],
    out_dir: Path,
) -> SelftestRun:
    layout = OutputLayout(Path(run_cfg.out_dir))
    path = layout.contaminated_docs_path(_PLANTED_STEM)
    channels: dict[int, set[str]] = {}
    named: dict[int, set[int]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = parse_contamination_record(line)
                if rec.row_index is None:
                    continue
                channels.setdefault(rec.row_index, set()).update(rec.channels or ["?"])
                items = named.setdefault(rec.row_index, set())
                items.update(hit.item_id for hit in rec.item_hits)
                items.update(hit.item_id for hit in rec.exact_hits)

    with_exact = run_cfg.exact_channel != "off"

    def matchable(cls: SelftestItem) -> bool:
        return cls.ngram_matchable or (with_exact and cls.exact_matchable)

    totals = {v: [0, 0] for v in variants}  # found, matchable
    per_channel: dict[str, list[int]] = {}
    if with_exact:
        per_channel = {"ngram": [0, 0], "exact": [0, 0]}
    missed: list[MissedPlant] = []
    attributed = 0
    controls = 0
    false_positives = 0
    for t in truth:
        flagged = t.row_index in channels
        if t.kind == "control":
            controls += 1
            false_positives += int(flagged)
            continue
        assert t.item is not None and t.variant is not None
        cls = classes[t.item.item_idx]
        if with_exact:
            for channel, ok in (
                ("ngram", cls.ngram_matchable),
                ("exact", cls.exact_matchable),
            ):
                if ok:
                    per_channel[channel][1] += 1
                    per_channel[channel][0] += int(
                        channel in channels.get(t.row_index, set())
                    )
        if not matchable(cls):
            continue
        totals[t.variant][1] += 1
        if flagged:
            totals[t.variant][0] += 1
            attributed += int(t.item.item_idx in named.get(t.row_index, set()))
        else:
            missed.append(
                MissedPlant(
                    row_index=t.row_index,
                    item_idx=t.item.item_idx,
                    item_id=t.item.item_id,
                    benchmark=t.item.benchmark,
                    variant=t.label or t.variant,
                )
            )
    found = sum(f for f, _ in totals.values())
    total = sum(m for _, m in totals.values())
    return SelftestRun(
        name=name,
        gate_mode=run_cfg.gate_mode,
        exact_channel=run_cfg.exact_channel,
        scan_dir=Path(run_cfg.out_dir).relative_to(out_dir).as_posix(),
        overall=_cell(found, total),
        by_variant={v: _cell(f, m) for v, (f, m) in totals.items()},
        by_channel={c: _cell(f, m) for c, (f, m) in per_channel.items()},
        attributed=attributed,
        missed=missed,
        controls=controls,
        false_positives=false_positives,
    )


def run_selftest(
    cfg: ScanConfig,
    out_dir: Path,
    *,
    scan: Callable[[ScanConfig], RunSummary],
    n_items: int = 50,
    seed: int = 0,
    variants: list[str] | None = None,
    host: HostKind = "synthetic",
    min_recall: float = 1.0,
) -> SelftestReport:
    """Plant, scan and score; writes ``selftest.json`` and returns the report.

    ``out_dir`` must exist and hold no earlier self-test (see
    ``prepare_out_dir``). ``scan`` runs one scan (``do_scan`` plus its outputs).
    """
    variants = variants or ["verbatim"]
    index_path = Path(cfg.index_path)
    has_items = index_path.with_name(f"{index_path.name}.items").exists()
    configured_gate = "item" if cfg.gate_mode == "item" else "union"
    if configured_gate == "item" and not has_items:
        raise SelftestError(
            "--gate-mode item needs the index's .items sidecar, which it lacks."
        )

    sample, truth = build_planted(
        cfg, out_dir, n_items=n_items, seed=seed, variants=variants, host=host
    )
    classes = _classify(sample, cfg, _item_ngrams(index_path))

    specs: list[tuple[str, str, str]] = [("union", "union", "off")]
    if has_items:
        specs.append(("item", "item", "off"))
    if cfg.exact_channel != "off":
        specs.append((f"{configured_gate}+exact", configured_gate, cfg.exact_channel))
        configured = f"{configured_gate}+exact"
    else:
        configured = configured_gate

    runs: list[SelftestRun] = []
    for name, gate, exact in specs:
        run_cfg = cfg.model_copy(
            update={
                "inputs": [str(out_dir / PLANTED_NAME)],
                "out_dir": str(out_dir / SCANS_DIR_NAME / name.replace("+", "_")),
                "workers": 1,
                "max_return_records": None,
                "gate_mode": gate,
                "exact_channel": exact,
            }
        )
        summary = scan(run_cfg)
        errors = [s.error for s in summary.per_file if s.error]
        if errors:
            raise SelftestError(
                f"The {name} scan of the planted file failed: {errors[0]}"
            )
        runs.append(_evaluate(name, run_cfg, truth, classes, variants, out_dir))

    configured_run = next(r for r in runs if r.name == configured)
    gated = (
        configured_run.by_variant["verbatim"]
        if "verbatim" in variants
        else configured_run.overall
    )
    passed = gated.recall is None or gated.recall >= min_recall
    report = SelftestReport(
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        library_version=library_version(),
        rust_extension_version=rust_extension_version(),
        index=build_index_info(index_path),
        scan_config=build_scan_config_info(cfg),
        n_items=len(sample),
        seed=seed,
        variants=variants,
        host=host,
        items=list(classes.values()),
        unmatchable=[
            c
            for c in classes.values()
            if not c.ngram_matchable and not c.exact_matchable
        ],
        runs=runs,
        configured_run=configured,
        min_recall=min_recall,
        gated_recall=gated.recall,
        passed=passed,
    )
    (out_dir / REPORT_NAME).write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return report


def format_report(report: SelftestReport) -> str:
    """A plain-text recall table for the terminal."""

    def fmt(cell: RecallCell) -> str:
        if cell.recall is None:
            return "-"
        return f"{cell.found}/{cell.matchable} {cell.recall:.3f}"

    columns = [*report.variants, "all"]
    header = ["run", *columns, "false pos."]
    table = [header]
    for run in report.runs:
        cells = [fmt(run.by_variant[v]) for v in report.variants]
        table.append(
            [
                run.name,
                *cells,
                fmt(run.overall),
                f"{run.false_positives}/{run.controls}",
            ]
        )
        for channel, cell in run.by_channel.items():
            table.append(
                [f"  {channel} channel", *[""] * len(report.variants), fmt(cell), ""]
            )
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in table
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    out = [
        f"Planted {report.n_items} item(s) x {len(report.variants)} variant(s) "
        f"into {report.host} hosts (seed {report.seed}).",
        "",
        *lines,
    ]
    for run in report.runs:
        if run.missed:
            ids = ", ".join(
                f"{m.item_id or m.item_idx} ({m.variant})" for m in run.missed[:20]
            )
            more = f" ... {len(run.missed) - 20} more" if len(run.missed) > 20 else ""
            out.append(f"Missed by {run.name}: {ids}{more}")
    if report.unmatchable:
        ids = ", ".join(
            f"{c.item_id or c.item_idx} ({c.ngrams} n-grams)"
            for c in report.unmatchable[:20]
        )
        more = (
            f" ... {len(report.unmatchable) - 20} more"
            if len(report.unmatchable) > 20
            else ""
        )
        out.append(
            f"No enabled channel can match {len(report.unmatchable)} sampled item(s) "
            f"(fewer than min_hits n-grams, exact channel off or not eligible), so "
            f"they are left out of recall: {ids}{more}"
        )
    recall = "n/a" if report.gated_recall is None else f"{report.gated_recall:.3f}"
    verdict = "PASS" if report.passed else "FAIL"
    out.append(
        f"{verdict}: {report.configured_run} recall {recall} "
        f"(--min-recall {report.min_recall})."
    )
    return "\n".join(out)
