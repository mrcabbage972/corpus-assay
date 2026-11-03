"""Registry of recommended benchmark specs.

Each entry indexes the split and the text an evaluation harness shows the model,
pinned to a Hub revision. Entries also carry metadata that is not part of the spec:
the split harnesses score, whether the dataset is gated, and how many of its items
are too short for the n-gram rule to flag. User configs keep plain
:class:`BenchmarkSpec` entries; the metadata is read by the registry commands and
by the build-time checks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Mapping

from corpus_assay.benchmark_spec import BenchmarkSpec

# Settings the short-item counts below were measured with (bundled normalization
# config, 2026-11-01; see issue #14, which also gives the counts per entry).
SHORT_ITEM_NGRAM = 13
SHORT_ITEM_MIN_HITS = 3


@dataclass(frozen=True)
class RegistryEntry:
    """A registry benchmark spec plus metadata that is not part of the spec."""

    spec: BenchmarkSpec
    # The split lm-eval-style harnesses score. Several benchmarks hide their test
    # labels, so harnesses score validation instead.
    scored_split: str | None = None
    # Gated on the Hugging Face Hub: loading needs a login and accepted terms.
    gated: bool = False
    notes: tuple[str, ...] = ()
    # Items measured with SHORT_ITEM_NGRAM, and how many of them have fewer than
    # SHORT_ITEM_MIN_HITS n-grams (the n-gram rule cannot flag those items).
    item_count: int | None = None
    short_items: int | None = None

    @property
    def short_item_share(self) -> float | None:
        if not self.item_count or self.short_items is None:
            return None
        return self.short_items / self.item_count

    def metadata(self) -> dict[str, object]:
        share = self.short_item_share
        return {
            "scored_split": self.scored_split,
            "gated": self.gated,
            "item_count": self.item_count,
            "short_items": self.short_items,
            "short_item_share": round(share, 4) if share is not None else None,
            "notes": list(self.notes),
        }


class BenchmarkRegistry:
    def __init__(self, entries: Mapping[str, BenchmarkSpec | RegistryEntry]) -> None:
        self._entries: dict[str, RegistryEntry] = {
            key.lower(): (
                value if isinstance(value, RegistryEntry) else RegistryEntry(value)
            )
            for key, value in entries.items()
        }

    def list_registry(self) -> list[str]:
        return sorted(self._entries.keys())

    def entry(self, ref: str) -> RegistryEntry:
        key = ref.strip().lower()
        if key not in self._entries:
            raise KeyError(
                "Unknown benchmark ref "
                f"'{ref}'. Available: {', '.join(self.list_registry())}"
            )
        entry = self._entries[key]
        return replace(entry, spec=deepcopy(entry.spec))

    def resolve_ref(self, ref: str) -> BenchmarkSpec:
        return self.entry(ref).spec

    def is_gated_dataset(self, dataset: str) -> bool:
        """Whether a registry entry marks ``dataset`` as gated on the Hub."""
        wanted = dataset.strip().lower()
        return any(
            e.gated and e.spec.dataset.lower() == wanted for e in self._entries.values()
        )

    def entry_for_spec(self, spec: BenchmarkSpec) -> RegistryEntry | None:
        """The registry entry for the same dataset (and config) as ``spec``.

        Matches user-written specs too, so build-time checks can use the registry
        metadata (scored split, gating) for any config that points at a registry
        dataset. Returns None when nothing, or nothing unambiguous, matches.
        """
        wanted = spec.dataset.strip().lower()
        candidates = [
            e for e in self._entries.values() if e.spec.dataset.lower() == wanted
        ]
        spec_configs = _config_set(spec)
        for match in (
            lambda e: _config_set(e.spec) == spec_configs,
            lambda e: bool(spec_configs & _config_set(e.spec)),
        ):
            hits = [e for e in candidates if match(e)]
            if hits:
                return replace(hits[0], spec=deepcopy(hits[0].spec))
        if len(candidates) == 1:
            return replace(candidates[0], spec=deepcopy(candidates[0].spec))
        return None


def _config_set(spec: BenchmarkSpec) -> set[str]:
    if spec.config_name is not None:
        return {spec.config_name}
    return set(spec.configs)


_ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
_GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
_SUPER_GLUE_REVISION = "3de24cf8022e94f4ee4b9d55a6f539891524d646"
_MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"

_HIDDEN_TEST_LABELS = (
    "The test labels are hidden, so harnesses score validation; this entry "
    "indexes validation."
)

REGISTRY = BenchmarkRegistry(
    {
        # -----------------------------
        # Reasoning and math
        # -----------------------------
        "aime24": RegistryEntry(
            BenchmarkSpec(
                name="AIME24",
                dataset="HuggingFaceH4/aime_2024",
                revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
                splits=["train"],
                fields=["problem"],
                label_field="answer",
            ),
            scored_split="train",
            item_count=30,
            short_items=0,
        ),
        "aime25": RegistryEntry(
            BenchmarkSpec(
                name="AIME25",
                dataset="opencompass/AIME2025",
                revision="a6ad95f611d72cf628a80b58bd0432ef6638f958",
                data_files={"test": ["aime2025-I.jsonl", "aime2025-II.jsonl"]},
                splits=["test"],
                fields=["question"],
                label_field="answer",
            ),
            scored_split="test",
            item_count=30,
            short_items=1,
            notes=(
                "Reads both papers (I and II) through data_files; the dataset card "
                "lists no splits.",
            ),
        ),
        "amc23": RegistryEntry(
            BenchmarkSpec(
                name="AMC23",
                dataset="math-ai/amc23",
                revision="80815d37005feb82cd7f8fbc6901d5d3eff43057",
                splits=["test"],
                fields=["question"],
                label_field="answer",
            ),
            scored_split="test",
            item_count=40,
            short_items=5,
        ),
        "math500": RegistryEntry(
            BenchmarkSpec(
                name="MATH500",
                dataset="HuggingFaceH4/MATH-500",
                revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                splits=["test"],
                fields=["problem"],
                label_field="answer",
            ),
            scored_split="test",
            item_count=500,
            short_items=160,
        ),
        "gsm8k": RegistryEntry(
            BenchmarkSpec(
                name="GSM8K",
                dataset="openai/gsm8k",
                revision=_GSM8K_REVISION,
                config_name="main",
                splits=["test"],
                fields=["question"],
                label_field="answer",
            ),
            scored_split="test",
            item_count=1319,
            short_items=22,
            notes=(
                "Indexes the question only, which is what the model is shown. Add "
                "gsm8k_solutions to also catch copies of the worked solutions.",
            ),
        ),
        "gsm8k_solutions": RegistryEntry(
            BenchmarkSpec(
                name="GSM8K-solutions",
                dataset="openai/gsm8k",
                revision=_GSM8K_REVISION,
                config_name="main",
                splits=["test"],
                fields=["question", "answer"],
            ),
            scored_split="test",
            item_count=1319,
            short_items=0,
            notes=(
                "The question plus the full worked solution, kept apart from gsm8k "
                "so solution copies stay detectable without changing the question "
                "entry's per-item n-gram counts.",
            ),
        ),
        "jeebench": RegistryEntry(
            BenchmarkSpec(
                name="JEEBench",
                dataset="daman1209arora/jeebench",
                revision="d070d1113ce365df05903763a2017ec2d0d56118",
                data_files={"test": "test.json"},
                splits=["test"],
                fields=["question"],
                label_field="gold",
            ),
            scored_split="test",
            item_count=515,
            short_items=1,
        ),
        "gpqa_diamond": RegistryEntry(
            BenchmarkSpec(
                name="GPQA-Diamond",
                dataset="Idavidrein/gpqa",
                revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
                config_name="gpqa_diamond",
                splits=["train"],
                fields=[
                    "Question",
                    "Correct Answer",
                    "Incorrect Answer 1",
                    "Incorrect Answer 2",
                    "Incorrect Answer 3",
                ],
            ),
            scored_split="train",
            gated=True,
            notes=(
                "Gated: log in with `hf auth login` and accept the dataset terms on "
                "the Hub before building. 198 items.",
            ),
        ),
        # -----------------------------
        # Code
        # -----------------------------
        "humaneval": RegistryEntry(
            BenchmarkSpec(
                name="HumanEval",
                dataset="openai/openai_humaneval",
                revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                splits=["test"],
                fields=["prompt"],
                label_field="canonical_solution",
            ),
            scored_split="test",
            item_count=164,
            short_items=1,
        ),
        "mbpp": RegistryEntry(
            BenchmarkSpec(
                name="MBPP",
                dataset="google-research-datasets/mbpp",
                revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
                config_name="full",
                splits=["test"],
                fields=["text", "test_list"],
                label_field="code",
            ),
            scored_split="test",
            item_count=500,
            short_items=19,
        ),
        "livecodebench": RegistryEntry(
            BenchmarkSpec(
                name="LiveCodeBench",
                dataset="livecodebench/code_generation_lite",
                revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
                data_files={
                    "test": [
                        "test.jsonl",
                        "test2.jsonl",
                        "test3.jsonl",
                        "test4.jsonl",
                        "test5.jsonl",
                    ]
                },
                splits=["test"],
                fields=["question_content"],
            ),
            scored_split="test",
            notes=(
                "Indexes every problem of releases v1 to v5. Harnesses usually score "
                "a contest-date window, so this is the conservative choice.",
                "Reads the JSONL files through data_files, because the dataset's "
                "loading script does not run under datasets 4.x. The files are large.",
            ),
        ),
        # -----------------------------
        # Instruction following
        # -----------------------------
        "ifeval": RegistryEntry(
            BenchmarkSpec(
                name="IFEval",
                dataset="google/IFEval",
                revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
                splits=["train"],
                fields=["prompt"],
            ),
            scored_split="train",
            item_count=541,
            short_items=67,
        ),
        # -----------------------------
        # Language suite, on the split lm-eval scores
        # -----------------------------
        "mmlu": RegistryEntry(
            BenchmarkSpec(
                name="MMLU",
                dataset="cais/mmlu",
                revision="c30699e8356da336a370243923dbaf21066bb9fe",
                config_name="all",
                splits=["test"],
                fields=["question", "choices"],
                label_field="answer",
            ),
            scored_split="test",
            item_count=14042,
            short_items=4550,
            notes=("Indexes all 57 subjects; some setups score a subset of them.",),
        ),
        "mmlu_pro": RegistryEntry(
            BenchmarkSpec(
                name="MMLU-Pro",
                dataset="TIGER-Lab/MMLU-Pro",
                revision=_MMLU_PRO_REVISION,
                splits=["test"],
                fields=["question", "options"],
                label_field="answer",
            ),
            scored_split="test",
            item_count=12032,
            short_items=950,
        ),
        "hellaswag": RegistryEntry(
            BenchmarkSpec(
                name="HellaSwag",
                dataset="Rowan/hellaswag",
                revision="218ec52e09a7e7462a5400043bb9a69a41d06b76",
                splits=["validation"],
                fields=["ctx", "endings"],
                label_field="label",
            ),
            scored_split="validation",
            item_count=10042,
            short_items=44,
            notes=(_HIDDEN_TEST_LABELS,),
        ),
        "commonsenseqa": RegistryEntry(
            BenchmarkSpec(
                name="CommonsenseQA",
                dataset="tau/commonsense_qa",
                revision="94630fe30dad47192a8546eb75f094926d47e155",
                splits=["validation"],
                fields=["question", "choices"],
                label_field="answerKey",
            ),
            scored_split="validation",
            item_count=1221,
            short_items=1161,
            notes=(_HIDDEN_TEST_LABELS,),
        ),
        "arc": RegistryEntry(
            BenchmarkSpec(
                name="ARC",
                dataset="allenai/ai2_arc",
                revision=_ARC_REVISION,
                configs=["ARC-Challenge", "ARC-Easy"],
                splits=["test"],
                fields=["question", "choices"],
                label_field="answerKey",
            ),
            scored_split="test",
            item_count=1172 + 2376,
            short_items=433 + 1303,
            notes=(
                "Both ARC configs as one benchmark. Use arc_challenge and arc_easy "
                "to report them apart; do not combine them with this entry.",
            ),
        ),
        "arc_challenge": RegistryEntry(
            BenchmarkSpec(
                name="ARC-Challenge",
                dataset="allenai/ai2_arc",
                revision=_ARC_REVISION,
                config_name="ARC-Challenge",
                splits=["test"],
                fields=["question", "choices"],
                label_field="answerKey",
            ),
            scored_split="test",
            item_count=1172,
            short_items=433,
        ),
        "arc_easy": RegistryEntry(
            BenchmarkSpec(
                name="ARC-Easy",
                dataset="allenai/ai2_arc",
                revision=_ARC_REVISION,
                config_name="ARC-Easy",
                splits=["test"],
                fields=["question", "choices"],
                label_field="answerKey",
            ),
            scored_split="test",
            item_count=2376,
            short_items=1303,
        ),
        "piqa": RegistryEntry(
            BenchmarkSpec(
                name="PIQA",
                dataset="baber/piqa",
                revision="142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
                splits=["validation"],
                fields=["goal", "sol1", "sol2"],
                label_field="label",
            ),
            scored_split="validation",
            item_count=1838,
            short_items=1253,
            notes=(_HIDDEN_TEST_LABELS,),
        ),
        "boolq": RegistryEntry(
            BenchmarkSpec(
                name="BoolQ",
                dataset="aps/super_glue",
                revision=_SUPER_GLUE_REVISION,
                config_name="boolq",
                splits=["validation"],
                fields=["passage", "question"],
                label_field="label",
            ),
            scored_split="validation",
            item_count=3270,
            short_items=45,
            notes=(_HIDDEN_TEST_LABELS,),
        ),
        "winogrande": RegistryEntry(
            BenchmarkSpec(
                name="WinoGrande",
                dataset="allenai/winogrande",
                revision="01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                config_name="winogrande_xl",
                splits=["validation"],
                fields=["sentence", "option1", "option2"],
                label_field="answer",
            ),
            scored_split="validation",
            item_count=1267,
            short_items=1077,
            notes=(_HIDDEN_TEST_LABELS,),
        ),
        "openbookqa": RegistryEntry(
            BenchmarkSpec(
                name="OpenBookQA",
                dataset="allenai/openbookqa",
                revision="388097ea7776314e93a529163e0fea805b8a6454",
                config_name="main",
                splits=["test"],
                fields=["question_stem", "choices"],
                label_field="answerKey",
            ),
            scored_split="test",
            item_count=500,
            short_items=384,
        ),
        "copa": RegistryEntry(
            BenchmarkSpec(
                name="COPA",
                dataset="aps/super_glue",
                revision=_SUPER_GLUE_REVISION,
                config_name="copa",
                splits=["validation"],
                fields=["premise", "choice1", "choice2"],
                label_field="label",
            ),
            scored_split="validation",
            item_count=100,
            short_items=100,
            notes=(
                _HIDDEN_TEST_LABELS,
                "Every field is one short sentence, so at ngram=13 no item has an "
                "n-gram and build-index fails on this benchmark. Give it its own "
                "index with a smaller ngram, or pass --allow-empty-benchmarks.",
            ),
        ),
        "lambada": RegistryEntry(
            BenchmarkSpec(
                name="LAMBADA",
                dataset="EleutherAI/lambada_openai",
                revision="900124bf3b8235c6daf21033af9948b3f07346c4",
                config_name="default",
                splits=["test"],
                fields=["text"],
            ),
            scored_split="test",
            item_count=5153,
            short_items=0,
        ),
    }
)
