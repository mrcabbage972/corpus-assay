from __future__ import annotations

import re

import pytest

from corpus_assay.benchmark_registry import REGISTRY, BenchmarkRegistry, RegistryEntry
from corpus_assay.benchmark_spec import BenchmarkSpec


def test_registry_lists_keys_sorted() -> None:
    registry = BenchmarkRegistry(
        {
            "B": BenchmarkSpec(
                name="B",
                dataset="ds/b",
                splits=["test"],
                fields=["q"],
            ),
            "a": BenchmarkSpec(
                name="A",
                dataset="ds/a",
                splits=["test"],
                fields=["q"],
            ),
        }
    )

    assert registry.list_registry() == ["a", "b"]


def test_registry_resolve_returns_copy() -> None:
    spec = BenchmarkSpec(
        name="Demo",
        dataset="demo/ds",
        splits=["test"],
        fields=["question"],
    )
    registry = BenchmarkRegistry({"demo": spec})

    resolved = registry.resolve_ref("demo")
    resolved.fields.append("answer")

    assert spec.fields == ["question"]


def test_registry_unknown_ref_raises() -> None:
    registry = BenchmarkRegistry(
        {
            "demo": BenchmarkSpec(
                name="Demo",
                dataset="demo/ds",
                splits=["test"],
                fields=["question"],
            )
        }
    )

    with pytest.raises(KeyError, match="Unknown benchmark ref"):
        registry.resolve_ref("missing")


# ----------------------------
# The built-in registry
# ----------------------------

# The 22 entries of issue #14 (ARC counted per config), plus the aliases and
# extras the registry keeps.
_ISSUE_14_KEYS = {
    "aime24", "aime25", "amc23", "math500", "gsm8k", "jeebench", "gpqa_diamond",
    "humaneval", "mbpp", "livecodebench", "ifeval", "mmlu", "hellaswag",
    "commonsenseqa", "arc_challenge", "arc_easy", "piqa", "boolq", "winogrande",
    "openbookqa", "copa", "lambada",
}  # fmt: skip


def test_registry_contains_issue_14_entries_and_kept_keys() -> None:
    keys = set(REGISTRY.list_registry())
    assert _ISSUE_14_KEYS <= keys
    assert {"arc", "gsm8k_solutions", "mmlu_pro"} <= keys


@pytest.mark.parametrize("key", REGISTRY.list_registry())
def test_every_registry_entry_is_pinned_to_a_commit(key: str) -> None:
    entry = REGISTRY.entry(key)
    assert entry.spec.revision is not None
    assert re.fullmatch(r"[0-9a-f]{40}", entry.spec.revision), entry.spec.revision
    assert entry.scored_split in entry.spec.splits
    # Index what the model is shown, never the label column.
    assert entry.spec.label_field not in entry.spec.fields


def test_registry_keys_changed_in_place() -> None:
    mmlu = REGISTRY.resolve_ref("mmlu")
    assert (mmlu.config_name, mmlu.fields) == ("all", ["question", "choices"])

    hellaswag = REGISTRY.entry("hellaswag")
    assert hellaswag.spec.splits == ["validation"]
    assert hellaswag.spec.fields == ["ctx", "endings"]
    assert hellaswag.scored_split == "validation"

    arc = REGISTRY.resolve_ref("arc")
    assert arc.configs == ["ARC-Challenge", "ARC-Easy"]
    assert arc.fields == ["question", "choices"]
    assert REGISTRY.resolve_ref("arc_challenge").config_name == "ARC-Challenge"
    assert REGISTRY.resolve_ref("arc_easy").config_name == "ARC-Easy"

    assert REGISTRY.resolve_ref("gsm8k").fields == ["question"]
    assert REGISTRY.resolve_ref("gsm8k_solutions").fields == ["question", "answer"]

    boolq = REGISTRY.resolve_ref("boolq")
    assert (boolq.dataset, boolq.config_name) == ("aps/super_glue", "boolq")


def test_registry_metadata() -> None:
    gpqa = REGISTRY.entry("gpqa_diamond")
    assert gpqa.gated is True
    assert gpqa.short_item_share is None
    assert REGISTRY.is_gated_dataset("idavidrein/GPQA")
    assert not REGISTRY.is_gated_dataset("openai/gsm8k")

    copa = REGISTRY.entry("copa")
    assert copa.short_item_share == 1.0
    csqa = REGISTRY.entry("commonsenseqa")
    assert csqa.short_items == 1161 and csqa.item_count == 1221
    assert csqa.metadata()["short_item_share"] == round(1161 / 1221, 4)


def test_entry_for_spec_matches_dataset_and_config() -> None:
    def spec(dataset: str, **kw) -> BenchmarkSpec:
        return BenchmarkSpec(
            name="X", dataset=dataset, splits=["test"], fields=["q"], **kw
        )

    # One entry per dataset: matched whatever the config.
    hs = REGISTRY.entry_for_spec(spec("Rowan/hellaswag"))
    assert hs is not None and hs.scored_split == "validation"
    # Several entries share aps/super_glue: the config decides.
    copa = REGISTRY.entry_for_spec(spec("aps/super_glue", config_name="copa"))
    assert copa is not None and copa.spec.name == "COPA"
    assert REGISTRY.entry_for_spec(spec("aps/super_glue")) is None
    arc = REGISTRY.entry_for_spec(spec("allenai/ai2_arc", configs=["ARC-Easy"]))
    assert arc is not None and arc.spec.name == "ARC-Easy"
    assert REGISTRY.entry_for_spec(spec("someone/else")) is None


def test_registry_entry_returns_copies() -> None:
    entry = REGISTRY.entry("mmlu")
    entry.spec.fields.append("answer")
    assert REGISTRY.resolve_ref("mmlu").fields == ["question", "choices"]


def test_registry_accepts_registry_entries() -> None:
    spec = BenchmarkSpec(name="D", dataset="d/d", splits=["test"], fields=["q"])
    registry = BenchmarkRegistry({"d": RegistryEntry(spec, scored_split="test")})
    assert registry.entry("D").scored_split == "test"
    assert registry.resolve_ref("d") == spec
