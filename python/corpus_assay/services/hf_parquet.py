from __future__ import annotations

import math
from pathlib import Path

from datasets import Dataset


def dataset_slug(name: str) -> str:
    return name.replace("/", "__")


def _parquet_cache_files(dataset: object) -> list[Path]:
    cache_files = getattr(dataset, "cache_files", None)
    if not cache_files:
        return []
    parquet_files: list[Path] = []
    for entry in cache_files:
        filename = entry.get("filename") if isinstance(entry, dict) else None
        if filename and str(filename).endswith(".parquet"):
            parquet_files.append(Path(filename))
    return sorted({path.resolve() for path in parquet_files}, key=str)


def existing_parquet_files(parquet_dir: Path) -> list[Path]:
    if not parquet_dir.exists():
        return []
    return sorted({path.resolve() for path in parquet_dir.glob("*.parquet")}, key=str)


def _materialize_parquet(
    dataset: Dataset, *, out_dir: Path, split: str, max_rows_per_file: int | None
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    num_rows = getattr(dataset, "num_rows", None)
    if num_rows is None:
        raise ValueError(
            "Dataset does not expose num_rows; cannot shard deterministically."
        )

    if max_rows_per_file is None or num_rows <= max_rows_per_file:
        out_path = out_dir / f"{split}.parquet"
        dataset.to_parquet(str(out_path))
        return [out_path.resolve()]

    num_shards = max(1, math.ceil(num_rows / max_rows_per_file))
    paths: list[Path] = []
    for shard_idx in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=shard_idx)
        out_path = out_dir / f"{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet"
        shard.to_parquet(str(out_path))
        paths.append(out_path.resolve())
    return paths


def resolve_hf_parquet_paths(
    dataset: Dataset,
    *,
    hf_dataset: str,
    split: str,
    out_dir: Path,
    max_rows_per_file: int | None,
) -> list[Path]:
    cached_parquet = _parquet_cache_files(dataset)
    if cached_parquet:
        return cached_parquet

    parquet_dir = out_dir / "hf_parquet" / dataset_slug(hf_dataset) / split
    existing_parquet = existing_parquet_files(parquet_dir)
    if existing_parquet:
        return existing_parquet

    return _materialize_parquet(
        dataset, out_dir=parquet_dir, split=split, max_rows_per_file=max_rows_per_file
    )
