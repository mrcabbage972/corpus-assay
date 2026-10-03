from __future__ import annotations

import heapq
import json
import logging
import math
import os
import random
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import corpus_assay._native as rust_text
from corpus_assay._native import (
    hash_ngram,
    ngram_config_hash,
    normalization_config_sha,
    normalize_text,
)
from corpus_assay.normalization import (
    active_ngram_config_path,
    set_ngram_config_path,
)
from corpus_assay.scanner.fingerprint import (
    library_version,
    rust_extension_version,
)
from corpus_assay.schemas import STOPGRAMS_META_SCHEMA_VERSION, StopGramsMeta
from corpus_assay.text_utils import iter_ngrams_with_filter
from corpus_assay.utils import _load_dataset_optional_revision

STOPGRAM_MAGIC = b"CASTOP1\x00"
STOPGRAM_FORMAT_VERSION = 1
_STOPGRAM_HEADER_STRUCT = struct.Struct("<8sIIQ")
STOPGRAM_HEADER_SIZE = _STOPGRAM_HEADER_STRUCT.size
_STOPGRAM_PART_STRUCT = struct.Struct("<QQQ")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StopGramsHeader:
    version: int
    ngram: int
    record_count: int


class SpaceSaving:
    __slots__ = ("max_items", "counts", "errors", "heap", "pos")

    def __init__(self, max_items: int) -> None:
        if max_items <= 0:
            raise ValueError("sketch_size must be >= 1")
        self.max_items = max_items
        self.counts: dict[int, int] = {}
        self.errors: dict[int, int] = {}
        self.heap: list[tuple[int, int]] = []
        self.pos: dict[int, int] = {}

    def _swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i][1]] = i
        self.pos[self.heap[j][1]] = j

    def _sift_down(self, i: int) -> None:
        n = len(self.heap)
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[parent][0] <= self.heap[i][0]:
                break
            self._swap(parent, i)
            i = parent

    def update(self, key: int, weight: int = 1) -> None:
        if key in self.counts:
            self.counts[key] += weight
            i = self.pos[key]
            self.heap[i] = (self.counts[key], key)
            self._sift_down(i)
            return

        if len(self.heap) < self.max_items:
            self.counts[key] = weight
            self.errors[key] = 0
            self.heap.append((weight, key))
            self.pos[key] = len(self.heap) - 1
            self._sift_up(self.pos[key])
            return

        min_count, min_key = self.heap[0]
        del self.counts[min_key]
        del self.errors[min_key]
        del self.pos[min_key]

        self.counts[key] = min_count + weight
        self.errors[key] = min_count
        self.heap[0] = (self.counts[key], key)
        self.pos[key] = 0
        self._sift_down(0)


def _hash_fn_id(payload: dict[str, Any]) -> str:
    name = str(payload.get("name", ""))
    version = str(payload.get("version", ""))
    return f"{name}:{version}"


def _hash_function_metadata() -> dict[str, Any]:
    fn = getattr(rust_text, "hash_function_metadata", None)
    if fn is None:
        return {"name": "unknown", "version": "unknown"}
    return fn()


def write_stopgrams_header(writer: Any, *, ngram: int, record_count: int) -> None:
    writer.write(
        _STOPGRAM_HEADER_STRUCT.pack(
            STOPGRAM_MAGIC, STOPGRAM_FORMAT_VERSION, int(ngram), int(record_count)
        )
    )


def read_stopgrams_header(reader: Any) -> StopGramsHeader:
    data = reader.read(STOPGRAM_HEADER_SIZE)
    if len(data) != STOPGRAM_HEADER_SIZE:
        raise ValueError(
            f"Truncated stop-grams header: expected {STOPGRAM_HEADER_SIZE} bytes, got {len(data)}."
        )
    magic, version, ngram, record_count = _STOPGRAM_HEADER_STRUCT.unpack(data)
    if magic != STOPGRAM_MAGIC:
        raise ValueError("Invalid stop-grams magic header.")
    if version != STOPGRAM_FORMAT_VERSION:
        raise ValueError(f"Unsupported stop-grams format version {version}.")
    return StopGramsHeader(version=version, ngram=ngram, record_count=record_count)


def stopgrams_meta_path(stopgrams_path: str | Path) -> Path:
    path = Path(stopgrams_path)
    if path.suffix == ".native":
        return path.with_suffix(".meta.json")
    return path.with_name(f"{path.name}.meta.json")


def load_stopgrams_meta(stopgrams_path: str | Path) -> StopGramsMeta:
    payload = json.loads(
        stopgrams_meta_path(stopgrams_path).read_text(encoding="utf-8")
    )
    return StopGramsMeta.model_validate(payload)


def _iter_part(path: Path) -> Any:
    record_size = _STOPGRAM_PART_STRUCT.size
    chunk_size = 8 << 20
    with path.open("rb") as handle:
        remainder = b""
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            buf = remainder + data
            extra = len(buf) % record_size
            if extra:
                remainder = buf[-extra:]
                buf = buf[:-extra]
            else:
                remainder = b""
            for h, est, err in _STOPGRAM_PART_STRUCT.iter_unpack(buf):
                yield h, est, err
        if remainder:
            raise ValueError(f"Truncated part record in {path}.")


def _merge_parts_to_stopgrams(
    part_paths: list[Path],
    *,
    ngram: int,
    doc_count_B: int,
    tau: float,
    out_path: Path,
) -> tuple[int, int]:
    df_threshold = int(math.ceil(tau * doc_count_B))
    if df_threshold <= 0:
        df_threshold = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop_count = 0

    with out_path.open("wb") as out:
        write_stopgrams_header(out, ngram=ngram, record_count=0)
        merged = heapq.merge(*[_iter_part(path) for path in part_paths])
        last_h = None
        est_acc = 0
        err_acc = 0
        for h, est, err in merged:
            if last_h is None:
                last_h, est_acc, err_acc = h, est, err
                continue
            if h == last_h:
                est_acc += est
                err_acc += err
            else:
                if est_acc - err_acc >= df_threshold:
                    out.write(struct.pack("<Q", int(last_h)))
                    stop_count += 1
                last_h, est_acc, err_acc = h, est, err

        if last_h is not None and est_acc - err_acc >= df_threshold:
            out.write(struct.pack("<Q", int(last_h)))
            stop_count += 1

        out.seek(0)
        write_stopgrams_header(out, ngram=ngram, record_count=stop_count)

    return df_threshold, stop_count


def _iter_documents_from_rows(
    rows: Any,
    *,
    fields: list[str],
    packed_doc_sep: str,
    packed_doc_sep_typo: str | None,
) -> Any:
    for row in rows:
        parts = []
        for field in fields:
            value = row.get(field)
            if value is None:
                continue
            if not isinstance(value, str):
                value = str(value)
            if value:
                parts.append(value)
        if not parts:
            continue
        merged = "\n".join(parts)
        if packed_doc_sep_typo:
            merged = merged.replace(packed_doc_sep_typo, packed_doc_sep)
        for segment in merged.split(packed_doc_sep):
            segment = segment.strip()
            if segment:
                yield segment


def _iter_local_jsonl_rows(
    local_path: str,
    *,
    worker_id: int,
    num_workers: int,
) -> Any:
    with open(local_path, encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            if line_idx % num_workers != worker_id:
                continue
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_stopgrams_part(
    *,
    dataset: str | None,
    local_path: str | None,
    config_name: str | None,
    split: str,
    revision: str | None,
    fields: list[str],
    ngram: int,
    packed_doc_sep: str,
    packed_doc_sep_typo: str | None,
    worker_id: int,
    num_workers: int,
    max_docs_per_worker: int | None,
    sample_prob: float | None,
    seed: int,
    sketch_size: int,
    out_part_path: Path,
    out_stats_path: Path,
    ngram_config_path: str | None = None,
) -> None:
    # Re-stage the parent's ngram config in this worker before any normalization.
    # With spawn multiprocessing, the worker re-imports the package and only sees
    # the bundled default; without this, a parent-side --ngram-config override
    # would be silently dropped here.
    if ngram_config_path is not None:
        set_ngram_config_path(ngram_config_path)

    if local_path is not None:
        rows = _iter_local_jsonl_rows(
            local_path, worker_id=worker_id, num_workers=num_workers
        )
        has_shard = True  # sharding is handled inside the iterator
    else:
        rows = _load_dataset_optional_revision(
            dataset,
            name=config_name,
            split=split,
            streaming=True,
            revision=revision,
        )
        has_shard = hasattr(rows, "shard")
        if has_shard:
            rows = rows.shard(num_shards=num_workers, index=worker_id)
        elif num_workers > 1 and worker_id == 0:
            logger.warning(
                "Dataset streaming does not support .shard(); each worker will scan the full stream. "
                "Consider --max-docs/--sample-prob to limit work."
            )

    rng = random.Random((seed * 1000003) ^ worker_id)
    sketch = SpaceSaving(sketch_size)
    doc_count = 0

    for row_idx, row in enumerate(
        _iter_documents_from_rows(
            rows,
            fields=fields,
            packed_doc_sep=packed_doc_sep,
            packed_doc_sep_typo=packed_doc_sep_typo,
        )
    ):
        # Fallback sharding happens after packed_doc_sep splitting; segments may end up
        # on different workers when the dataset lacks .shard().
        if not has_shard and (row_idx % num_workers) != worker_id:
            continue

        if sample_prob is not None and rng.random() >= sample_prob:
            continue

        doc_count += 1
        if max_docs_per_worker is not None and doc_count > max_docs_per_worker:
            break

        hashes: set[int] = set()
        tokens = normalize_text(row)
        for ng, allowed, _ in iter_ngrams_with_filter(tokens, n=ngram):
            if not allowed:
                continue
            hashes.add(hash_ngram(ng))

        for h in hashes:
            sketch.update(h)

        if max_docs_per_worker is not None and doc_count >= max_docs_per_worker:
            break

    items = sorted((h, sketch.counts[h], sketch.errors[h]) for h in sketch.counts)
    out_part_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = bytearray()
    flush_limit = 8 << 20
    with out_part_path.open("wb") as f:
        for h, est, err in items:
            buffer.extend(_STOPGRAM_PART_STRUCT.pack(int(h), int(est), int(err)))
            if len(buffer) >= flush_limit:
                f.write(buffer)
                buffer.clear()
        if buffer:
            f.write(buffer)

    out_stats_path.write_text(json.dumps({"doc_count": doc_count}), encoding="utf-8")


def _build_stopgrams_part_entry(kwargs: dict[str, Any]) -> None:
    _build_stopgrams_part(**kwargs)


def build_stopgrams_direct_from_hf_dataset(
    *,
    dataset: str | None,
    local_path: str | None = None,
    config_name: str | None,
    split: str,
    revision: str | None,
    fields: list[str],
    ngram: int,
    packed_doc_sep: str,
    packed_doc_sep_typo: str | None,
    tau: float,
    out_path: str | Path,
    workers: int,
    max_docs: int | None,
    sample_prob: float | None,
    seed: int,
    tmp_dir: str | None,
    sketch_size: int = 200_000,
    cleanup_parts: bool = True,
) -> StopGramsMeta:
    if tau <= 0 or tau >= 1:
        raise ValueError("tau must be in (0, 1)")
    if sample_prob is not None and not (0 < sample_prob < 1):
        raise ValueError("sample_prob must be in (0, 1)")
    if max_docs is not None and max_docs <= 0:
        raise ValueError("max_docs must be > 0")
    if workers <= 0:
        raise ValueError("workers must be >= 1")
    if sketch_size <= 0:
        raise ValueError("sketch_size must be >= 1")
    out_path = Path(out_path)
    temp_root = Path(tmp_dir) if tmp_dir else out_path.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}_{uuid4().hex[:8]}"
    temp_dir = temp_root / f"stopgrams_parts_{run_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    max_docs_per_worker = None
    if max_docs is not None:
        max_docs_per_worker = int(math.ceil(max_docs / max(1, workers)))

    from multiprocessing import get_context

    # Capture the ngram config path now so each spawned worker re-stages the
    # same config the parent is using, not just the bundled default it would
    # otherwise see after re-importing the package.
    ngram_config_path = active_ngram_config_path()

    part_paths: list[Path] = []
    stats_paths: list[Path] = []
    args_list = []
    try:
        for worker_id in range(workers):
            part_path = temp_dir / f"stopgrams_part_{worker_id}.bin"
            stats_path = temp_dir / f"stopgrams_part_{worker_id}.json"
            part_paths.append(part_path)
            stats_paths.append(stats_path)
            args_list.append(
                dict(
                    dataset=dataset,
                    local_path=local_path,
                    config_name=config_name,
                    split=split,
                    revision=revision,
                    fields=fields,
                    ngram=ngram,
                    packed_doc_sep=packed_doc_sep,
                    packed_doc_sep_typo=packed_doc_sep_typo,
                    worker_id=worker_id,
                    num_workers=workers,
                    max_docs_per_worker=max_docs_per_worker,
                    sample_prob=sample_prob,
                    seed=seed,
                    sketch_size=sketch_size,
                    out_part_path=part_path,
                    out_stats_path=stats_path,
                    ngram_config_path=ngram_config_path,
                )
            )

        if workers <= 1:
            for kwargs in args_list:
                _build_stopgrams_part(**kwargs)
        else:
            with get_context("spawn").Pool(processes=workers) as pool:
                pool.map(_build_stopgrams_part_entry, args_list)

        doc_count_B = 0
        for stats_path in stats_paths:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            doc_count_B += int(payload.get("doc_count", 0))

        df_threshold, stopgram_count = _merge_parts_to_stopgrams(
            part_paths,
            ngram=ngram,
            doc_count_B=doc_count_B,
            tau=tau,
            out_path=out_path,
        )
    finally:
        if cleanup_parts:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    meta = StopGramsMeta.model_validate(
        {
            "schema_version": STOPGRAMS_META_SCHEMA_VERSION,
            "format_version": STOPGRAM_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "library_version": library_version(),
            "rust_extension_version": rust_extension_version(),
            "ngram": ngram,
            "hash_fn_id": _hash_fn_id(_hash_function_metadata()),
            "doc_count_B": doc_count_B,
            "tau": tau,
            "df_threshold": df_threshold,
            "stopgram_count": stopgram_count,
            "normalization_config_sha": normalization_config_sha(),
            "ngram_config_hash": ngram_config_hash(),
            "built_from": {
                "dataset": dataset,
                "local_path": local_path,
                "config_name": config_name,
                "split": split,
                "revision": revision,
                "fields": list(fields),
                "packed_doc_sep": packed_doc_sep,
                "packed_doc_sep_typo": packed_doc_sep_typo,
                "workers": workers,
                "max_docs": max_docs,
                "sample_prob": sample_prob,
                "seed": seed,
                "tmp_dir": str(temp_root),
                "sketch_size": sketch_size,
            },
            "files": {"native": out_path.name},
        }
    )
    stopgrams_meta_path(out_path).write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return meta


def validate_stopgrams_compatibility(
    *,
    stopgrams_meta: StopGramsMeta,
    index_meta: dict[str, Any],
    expected_ngram: int,
) -> None:
    if stopgrams_meta.ngram != expected_ngram:
        raise ValueError(
            f"Stop-grams ngram {stopgrams_meta.ngram} does not match scan ngram {expected_ngram}."
        )
    if stopgrams_meta.ngram != index_meta.get("ngram", expected_ngram):
        raise ValueError(
            "Stop-grams ngram does not match index ngram "
            f"({stopgrams_meta.ngram} vs {index_meta.get('ngram')})."
        )
    index_hash = index_meta.get("hash_function") or {}
    if stopgrams_meta.hash_fn_id != _hash_fn_id(index_hash):
        raise ValueError(
            "Stop-grams hash function does not match index hash function "
            f"({stopgrams_meta.hash_fn_id} vs {_hash_fn_id(index_hash)})."
        )
    if stopgrams_meta.normalization_config_sha != index_meta.get(
        "normalization_config_sha"
    ):
        raise ValueError(
            "Stop-grams normalization config does not match index normalization config."
        )
    if stopgrams_meta.ngram_config_hash != index_meta.get("ngram_config_hash"):
        raise ValueError(
            "Stop-grams ngram config hash does not match index ngram config hash."
        )


__all__ = [
    "STOPGRAM_FORMAT_VERSION",
    "STOPGRAM_HEADER_SIZE",
    "STOPGRAM_MAGIC",
    "StopGramsHeader",
    "build_stopgrams_direct_from_hf_dataset",
    "load_stopgrams_meta",
    "read_stopgrams_header",
    "stopgrams_meta_path",
    "validate_stopgrams_compatibility",
    "_merge_parts_to_stopgrams",
    "_build_stopgrams_part",
]
