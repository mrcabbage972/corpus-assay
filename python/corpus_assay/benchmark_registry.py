from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from corpus_assay.benchmark_spec import BenchmarkSpec


class BenchmarkRegistry:
    def __init__(self, entries: Mapping[str, BenchmarkSpec]) -> None:
        self._entries = {key.lower(): value for key, value in entries.items()}

    def list_registry(self) -> list[str]:
        return sorted(self._entries.keys())

    def resolve_ref(self, ref: str) -> BenchmarkSpec:
        key = ref.strip().lower()
        if key not in self._entries:
            raise KeyError(
                "Unknown benchmark ref "
                f"'{ref}'. Available: {', '.join(self.list_registry())}"
            )
        return deepcopy(self._entries[key])


REGISTRY = BenchmarkRegistry(
    {
        "mmlu": BenchmarkSpec(
            name="MMLU",
            dataset="cais/mmlu",
            configs=["*"],
            # Not a subject: auxiliary training data with no test split.
            drop_configs=["auxiliary_train"],
            splits=["test"],
            fields=["question", "choices", "answer"],
        ),
        "gsm8k": BenchmarkSpec(
            name="GSM8K",
            dataset="openai/gsm8k",
            configs=["main"],
            splits=["test"],
            fields=["question", "answer"],
        ),
        "hellaswag": BenchmarkSpec(
            name="HellaSwag",
            dataset="Rowan/hellaswag",
            splits=["test"],
            fields=["ctx", "endings", "label"],
        ),
        "arc": BenchmarkSpec(
            name="ARC",
            dataset="allenai/ai2_arc",
            configs=["ARC-Challenge", "ARC-Easy"],
            splits=["test"],
            fields=["question", "choices", "answerKey"],
        ),
        "lambada": BenchmarkSpec(
            name="LAMBADA",
            dataset="EleutherAI/lambada_openai",
            splits=["test"],
            fields=["text"],
        ),
        "boolq": BenchmarkSpec(
            name="BoolQ",
            dataset="google/boolq",
            splits=["validation"],
            fields=["question", "passage"],
        ),
        "winogrande": BenchmarkSpec(
            name="WinoGrande",
            dataset="allenai/winogrande",
            config_name="winogrande_xl",
            splits=["validation"],
            fields=["sentence", "option1", "option2"],
        ),
        "openbookqa": BenchmarkSpec(
            name="OpenBookQA",
            dataset="allenai/openbookqa",
            config_name="main",
            splits=["test"],
            fields=["question_stem", "choices"],
        ),
        "copa": BenchmarkSpec(
            name="COPA",
            dataset="aps/super_glue",
            config_name="copa",
            splits=["validation"],
            fields=["premise", "choice1", "choice2", "question"],
        ),
        "mmlu_pro": BenchmarkSpec(
            name="MMLU-Pro",
            dataset="TIGER-Lab/MMLU-Pro",
            splits=["test"],
            fields=["question", "options"],
        ),
    }
)
