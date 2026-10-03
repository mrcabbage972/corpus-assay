import textwrap

import pytest

from corpus_assay import benchmarks, utils
from corpus_assay.benchmark_spec import BenchmarkSpec
from corpus_assay.benchmarks import BenchmarkLoader
from corpus_assay.config import load_decontamination_config
from corpus_assay.constants import FIELD_SEPARATOR


def test_config_driven_benchmark_specs(tmp_path) -> None:
    config_text = textwrap.dedent(
        """
        {
          "benchmarks": [
            {
              "name": "MMLU",
              "dataset": "cais/mmlu",
              "configs": ["*"],
              "splits": ["test", "validation"],
              "subtraction_splits": ["train", "auxiliary_train"],
              "fields": ["question", "choices", "answer"]
            },
            {
              "name": "Demo",
              "dataset": "demo/ds",
              "configs": ["alpha"],
              "splits": ["test", "train"],
              "fields": ["question", "answer"],
              "trust_remote_code": true
            }
          ],
          "ngram": 13,
          "min_hits": 3,
          "min_coverage": 0.001
        }
        """
    ).strip()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    cfg = load_decontamination_config(config_path)

    assert [spec.name for spec in cfg.benchmarks] == ["MMLU", "Demo"]

    mmlu = cfg.benchmarks[0]
    assert mmlu.configs == ["*"]
    assert mmlu.splits == ["test", "validation"]
    assert mmlu.subtraction_splits == ["train", "auxiliary_train"]

    demo = cfg.benchmarks[1]
    assert demo.dataset == "demo/ds"
    assert demo.configs == ["alpha"]
    assert demo.splits == ["test", "train"]
    assert demo.fields == ["question", "answer"]
    assert demo.trust_remote_code is True


def test_config_registry_string_entry(tmp_path) -> None:
    config_text = textwrap.dedent(
        """
        {
          "benchmarks": [
            "gsm8k"
          ],
          "ngram": 13,
          "min_hits": 3,
          "min_coverage": 0.001
        }
        """
    ).strip()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    cfg = load_decontamination_config(config_path)
    spec = cfg.benchmarks[0]
    assert spec.name == "GSM8K"
    assert spec.dataset == "openai/gsm8k"
    assert spec.configs == ["main"]
    assert spec.splits == ["test"]
    assert spec.fields == ["question", "answer"]


def test_config_registry_ref_merge_overrides(tmp_path) -> None:
    config_text = textwrap.dedent(
        """
        {
          "benchmarks": [
            {
              "ref": "mmlu",
              "splits": ["test", "validation"]
            }
          ],
          "ngram": 13,
          "min_hits": 3,
          "min_coverage": 0.001
        }
        """
    ).strip()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    cfg = load_decontamination_config(config_path)
    spec = cfg.benchmarks[0]
    assert spec.name == "MMLU"
    assert spec.dataset == "cais/mmlu"
    assert spec.splits == ["test", "validation"]


def test_config_registry_unknown_ref_raises(tmp_path) -> None:
    config_text = textwrap.dedent(
        """
        {
          "benchmarks": [
            "unknown_benchmark"
          ],
          "ngram": 13,
          "min_hits": 3,
          "min_coverage": 0.001
        }
        """
    ).strip()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(KeyError, match="Unknown benchmark ref"):
        load_decontamination_config(config_path)


def test_benchmark_loader_positive_splits(monkeypatch) -> None:
    spec = BenchmarkSpec(
        name="Demo",
        dataset="demo/ds",
        configs=["alpha"],
        splits=["test"],
        fields=["a", "b"],
    )

    def fake_load_dataset(dataset, name=None, **_kwargs):
        assert dataset == "demo/ds"
        assert name == "alpha"
        return {
            "train": [{"a": "train", "b": "split"}],
            "test": [{"a": "hello", "b": "world", "c": "ignored"}],
        }

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == [f"hello {FIELD_SEPARATOR} world"]


def test_benchmark_loader_negative_splits(monkeypatch) -> None:
    spec = BenchmarkSpec(
        name="Demo",
        dataset="demo/ds",
        configs=["alpha"],
        splits=["test"],
        subtraction_splits=["train"],
        fields=["a"],
    )

    def fake_load_dataset(*_args, **_kwargs):
        return {
            "train": [{"a": "dirty_data"}],
            "test": [{"a": "clean_data"}],
        }

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["clean_data"]
    assert list(loader.load_negative()) == ["dirty_data"]


# ------------------------------
# New coverage starts here
# ------------------------------


def test_benchmark_loader_no_configs_loads_base_dataset(monkeypatch) -> None:
    """
    If spec.configs is empty, loader should still load the dataset once
    using config_name=None and yield rows.
    """
    spec = BenchmarkSpec(
        name="NoConfig",
        dataset="demo/no-config",
        configs=[],  # important
        splits=["test"],
        fields=["x"],
    )

    calls: list[tuple[str, object]] = []

    def fake_load_dataset(dataset, name=None, **_kwargs):
        calls.append((dataset, name))
        return {"test": [{"x": "ok"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["ok"]
    assert calls == [("demo/no-config", None)]


def test_benchmark_loader_wildcard_resolves_and_drops(monkeypatch) -> None:
    """
    '*' should expand via get_dataset_config_names and apply drop_configs + implicit 'all' drop.
    """
    spec = BenchmarkSpec(
        name="Wild",
        dataset="demo/wild",
        configs=["*"],
        drop_configs=["beta"],
        splits=["test"],
        fields=["x"],
    )

    monkeypatch.setattr(
        benchmarks,
        "get_dataset_config_names",
        lambda *_args, **_kwargs: ["all", "alpha", "beta"],
    )

    loaded: list[str | None] = []

    def fake_load_dataset(dataset, name=None, **_kwargs):
        assert dataset == "demo/wild"
        loaded.append(name)
        return {"test": [{"x": f"cfg={name}"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    out = list(loader.load_positive())

    # 'all' and 'beta' dropped, so only alpha should load
    assert loaded == ["alpha"]
    assert out == ["cfg=alpha"]


def test_benchmark_loader_split_not_found_warns_and_yields_nothing(monkeypatch) -> None:
    spec = BenchmarkSpec(
        name="SplitWarn",
        dataset="demo/splits",
        configs=["alpha"],
        splits=["validation"],  # doesn't exist
        fields=["x"],
    )

    def fake_load_dataset(*_args, **_kwargs):
        return {"test": [{"x": "nope"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == []


def test_benchmark_loader_multiple_splits_and_rows(monkeypatch) -> None:
    """
    Verify that loader yields all rows from all requested splits, in split order.
    """
    spec = BenchmarkSpec(
        name="MultiSplit",
        dataset="demo/multi",
        configs=["alpha"],
        splits=["test", "train"],
        fields=["a"],
    )

    def fake_load_dataset(*_args, **_kwargs):
        return {
            "test": [{"a": "t1"}, {"a": "t2"}],
            "train": [{"a": "r1"}],
        }

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["t1", "t2", "r1"]


def test_benchmark_loader_config_failure_continues(monkeypatch, capsys) -> None:
    """
    If one config fails to load, loader should warn and continue with remaining configs.
    """
    spec = BenchmarkSpec(
        name="ConfigFail",
        dataset="demo/fail",
        configs=["bad", "good"],
        splits=["test"],
        fields=["x"],
    )

    def fake_load_dataset(dataset, name=None, **_kwargs):
        if name == "bad":
            raise RuntimeError("boom")
        return {"test": [{"x": "ok"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["ok"]


def test_benchmark_loader_wildcard_resolution_failure_warns(monkeypatch) -> None:
    spec = BenchmarkSpec(
        name="WildFail",
        dataset="demo/wildfail",
        configs=["*"],
        splits=["test"],
        fields=["x"],
    )

    def fake_get_dataset_config_names(*_args, **_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(
        benchmarks, "get_dataset_config_names", fake_get_dataset_config_names
    )

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == []


def test_explicit_all_is_not_auto_dropped(monkeypatch):
    spec = BenchmarkSpec(
        name="ExplicitAll",
        dataset="demo/all",
        configs=["all"],
        splits=["test"],
        fields=["x"],
    )

    def fake_load_dataset(dataset, name=None, **_kwargs):
        assert name == "all"
        return {"test": [{"x": "ok"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["ok"]


def test_wildcard_only_all_config_kept_to_avoid_empty(monkeypatch):
    spec = BenchmarkSpec(
        name="OnlyAll",
        dataset="demo/onlyall",
        configs=["*"],
        splits=["test"],
        fields=["x"],
    )
    monkeypatch.setattr(
        benchmarks, "get_dataset_config_names", lambda *_a, **_k: ["all"]
    )

    def fake_load_dataset(dataset, name=None, **_kwargs):
        assert name == "all"
        return {"test": [{"x": "ok"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["ok"]


def test_benchmark_loader_passes_revision(monkeypatch) -> None:
    spec = BenchmarkSpec(
        name="Rev",
        dataset="demo/rev",
        revision="abc123",
        configs=["alpha"],
        splits=["test"],
        fields=["x"],
    )

    calls: list[dict[str, object]] = []

    def fake_load_dataset(dataset, name=None, **kwargs):
        calls.append({"dataset": dataset, "config": name, **kwargs})
        return {"test": [{"x": "ok"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["ok"]
    assert calls[0]["revision"] == "abc123"


def test_benchmark_loader_revision_fallback(monkeypatch) -> None:
    spec = BenchmarkSpec(
        name="RevFallback",
        dataset="demo/rev-fallback",
        revision="abc123",
        configs=["alpha"],
        splits=["test"],
        fields=["x"],
    )

    calls: list[dict[str, object]] = []

    def fake_load_dataset(dataset, name=None, **kwargs):
        calls.append({"dataset": dataset, "config": name, **kwargs})
        if "revision" in kwargs:
            raise TypeError("unexpected keyword argument 'revision'")
        return {"test": [{"x": "ok"}]}

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    loader = BenchmarkLoader(spec)
    assert list(loader.load_positive()) == ["ok"]
    assert len(calls) == 2
    assert calls[0]["revision"] == "abc123"
    assert "revision" not in calls[1]
