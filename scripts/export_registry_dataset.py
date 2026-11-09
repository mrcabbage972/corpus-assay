"""Export the built-in benchmark registry as a Hugging Face dataset repo.

Writes ``data/registry.parquet`` (one row per registry entry) and a ``README.md``
dataset card to ``--out``. Upload the directory with:

    hf upload cabbage972/corpus-assay-benchmark-registry <out> . --repo-type dataset

Run it after any registry change that ships in a release, so the Hub copy matches the
version on PyPI.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from corpus_assay.benchmark_registry import (
    REGISTRY,
    SHORT_ITEM_MIN_HITS,
    SHORT_ITEM_NGRAM,
)

HF_REPO = "cabbage972/corpus-assay-benchmark-registry"
GITHUB = "https://github.com/mrcabbage972/corpus-assay"

SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("name", pa.string()),
        ("dataset", pa.string()),
        ("hub_url", pa.string()),
        ("revision", pa.string()),
        ("config_name", pa.string()),
        ("configs", pa.list_(pa.string())),
        ("data_files", pa.string()),
        ("splits", pa.list_(pa.string())),
        ("fields", pa.list_(pa.string())),
        ("label_field", pa.string()),
        ("scored_split", pa.string()),
        ("gated", pa.bool_()),
        ("item_count", pa.int64()),
        ("short_items", pa.int64()),
        ("short_item_share", pa.float64()),
        ("notes", pa.list_(pa.string())),
        ("spec_yaml", pa.string()),
        ("corpus_assay_version", pa.string()),
    ]
)


def registry_rows() -> list[dict[str, object]]:
    tool_version = version("corpus-assay")
    rows: list[dict[str, object]] = []
    for key in REGISTRY.list_registry():
        entry = REGISTRY.entry(key)
        spec = entry.spec
        meta = entry.metadata()
        spec_dict = spec.model_dump(exclude_defaults=True)
        rows.append(
            {
                "key": key,
                "name": spec.name,
                "dataset": spec.dataset,
                "hub_url": f"https://huggingface.co/datasets/{spec.dataset}",
                "revision": spec.revision,
                "config_name": spec.config_name,
                "configs": list(spec.configs),
                # str, list or dict depending on the dataset; JSON keeps one type.
                "data_files": (
                    json.dumps(spec.data_files) if spec.data_files is not None else None
                ),
                "splits": list(spec.splits),
                "fields": list(spec.fields),
                "label_field": spec.label_field,
                "scored_split": entry.scored_split,
                "gated": entry.gated,
                "item_count": entry.item_count,
                "short_items": entry.short_items,
                "short_item_share": meta["short_item_share"],
                "notes": list(entry.notes),
                "spec_yaml": yaml.safe_dump(
                    {"benchmarks": [spec_dict]}, sort_keys=False
                ),
                "corpus_assay_version": tool_version,
            }
        )
    return rows


def _table_row(row: dict[str, object]) -> str:
    dataset = f"[`{row['dataset']}`]({row['hub_url']})"
    if row["gated"]:
        dataset += " (gated)"
    configs = row["config_name"] or " + ".join(row["configs"])  # type: ignore[arg-type]
    if row["data_files"]:
        configs = (configs + " " if configs else "") + "`data_files`"
    splits = ", ".join(row["splits"])  # type: ignore[arg-type]
    if row["scored_split"] == "validation":
        splits = f"**{splits}**"
    fields = ", ".join(row["fields"])  # type: ignore[arg-type]
    share = row["short_item_share"]
    short = f"{share:.0%}" if isinstance(share, float) else ""
    return f"| `{row['key']}` | {dataset} | {configs} | {splits} | {fields} | {short} |"


def dataset_card(rows: list[dict[str, object]]) -> str:
    tool_version = rows[0]["corpus_assay_version"]
    front_matter = {
        "pretty_name": "corpus-assay benchmark registry",
        "license": "apache-2.0",
        "language": ["en"],
        "size_categories": ["n<1K"],
        "tags": [
            "decontamination",
            "data-contamination",
            "benchmark-contamination",
            "test-set-contamination",
            "llm",
            "pretraining",
            "pretraining-data",
            "evaluation",
            "benchmarks",
            "data-quality",
        ],
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": "train", "path": "data/registry.parquet"}],
            }
        ],
    }
    table = "\n".join(_table_row(r) for r in rows)
    return f"""---
{yaml.safe_dump(front_matter, sort_keys=False).strip()}
---

# corpus-assay benchmark registry

**{len(rows)} pinned specs of LLM evaluation benchmarks, for benchmark decontamination of
LLM pretraining data.** Each row says which Hugging Face dataset, revision, config,
split and text fields to index so that a decontamination scan protects the items an
evaluation harness actually scores.

This is the built-in registry of [corpus-assay]({GITHUB}) {tool_version}, a Rust-accelerated
tool that finds benchmark text in LLM training corpora (n-gram overlap, with
per-benchmark and per-item attribution) and writes cleaned shards. The rows hold no
benchmark text, only pointers to it.

## Why a registry

Getting the spec wrong silently weakens decontamination:

- **Wrong split.** HellaSwag, BoolQ, WinoGrande, COPA, CommonsenseQA and PIQA hide
  their test labels, so lm-eval scores validation. An index built from `test` protects
  the wrong items. These rows index the split in `scored_split` (shown in bold below).
- **Wrong text.** Indexing labels or worked solutions, or leaving out the answer
  choices, changes what counts as a leak. These rows index the text the model is shown.
- **Unpinned data.** Every row pins a Hub commit, so two indexes built from it agree.
- **Short items.** An item with fewer than {SHORT_ITEM_MIN_HITS} distinct
  {SHORT_ITEM_NGRAM}-grams can never be flagged by the standard 13-gram rule.
  `short_item_share` says how much of each benchmark that is; corpus-assay's
  exact-match channel (`--exact-channel short`) covers those items.

## Benchmarks

| Key | Dataset | Config | Split | Fields | Short items |
| --- | --- | --- | --- | --- | ---: |
{table}

Bold splits are the ones harnesses score because the test labels are hidden. "Short
items" is the share of items with fewer than {SHORT_ITEM_MIN_HITS}
{SHORT_ITEM_NGRAM}-grams; blank means not measured.

## Use

With corpus-assay, refer to rows by key:

```bash
pip install corpus-assay
corpus-assay init-config --benchmarks gsm8k --benchmarks mmlu --out decon.yaml
corpus-assay run --config decon.yaml --input ./shards --out-dir ./scan_out
```

As data, for your own pipeline:

```python
from datasets import load_dataset

registry = load_dataset("{HF_REPO}", split="train")
row = registry.filter(lambda r: r["key"] == "hellaswag")[0]
print(row["dataset"], row["revision"], row["splits"], row["fields"])
```

`spec_yaml` holds each row as a corpus-assay config block.

## Columns

| Column | Meaning |
| --- | --- |
| `key` | corpus-assay registry key |
| `name` | Benchmark name used in reports |
| `dataset`, `hub_url` | Hugging Face dataset id and its page |
| `revision` | Pinned Hub commit |
| `config_name` / `configs` | One config, or several indexed as one benchmark |
| `data_files` | JSON-encoded `data_files` override, for datasets whose card lists no splits |
| `splits` | Splits indexed |
| `fields` | Text fields indexed (each field hashed on its own) |
| `label_field` | Label column, used only to check that labels are public; never indexed |
| `scored_split` | The split evaluation harnesses score |
| `gated` | Needs a Hub login and accepted terms |
| `item_count`, `short_items`, `short_item_share` | Items, and those too short for the n-gram rule |
| `notes` | Caveats for the entry |
| `spec_yaml` | The row as a corpus-assay config block |
| `corpus_assay_version` | corpus-assay release the rows come from |

## Links

- Code: {GITHUB}
- Package: https://pypi.org/project/corpus-assay/
- Registry docs: {GITHUB}/blob/main/docs/configuration.md#the-built-in-registry
- Software DOI: https://doi.org/10.5281/zenodo.23125328

## Citation

```bibtex
@software{{may_corpus_assay,
  author    = {{May, Victor}},
  title     = {{corpus-assay: benchmark contamination detection for LLM training corpora}},
  publisher = {{Zenodo}},
  doi       = {{10.5281/zenodo.23125328}},
  url       = {{{GITHUB}}}
}}
```

Each benchmark keeps its own license; see its dataset page.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    rows = registry_rows()
    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA), data_dir / "registry.parquet"
    )
    (args.out / "README.md").write_text(dataset_card(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} entries to {args.out}")


if __name__ == "__main__":
    main()
