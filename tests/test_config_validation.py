import pytest
import typer
import yaml
from pydantic import ValidationError

from corpus_assay.config import BackgroundCorpusConfig
from corpus_assay.services.config_validation import ConfigValidator


class _FakeInfo:
    def __init__(self, splits) -> None:
        self._splits = splits

    @property
    def splits(self):
        return self._splits


class _FakeBuilderConfig:
    def __init__(self, data_files) -> None:
        self.data_files = data_files


class _FakeBuilder:
    def __init__(self, splits, *, config_data_files=None) -> None:
        self._info = _FakeInfo(splits)
        self.config = _FakeBuilderConfig(config_data_files)

    @property
    def info(self):
        return self._info


class _EmptyConfigBenchmarkLoader:
    def __init__(self, _spec) -> None:
        self.spec = _spec

    def resolve_config_names(self) -> list[str | None]:
        return []


class _SingleConfigBenchmarkLoader:
    def __init__(self, _spec) -> None:
        self.spec = _spec

    def resolve_config_names(self) -> list[str | None]:
        return [None]


def _write_config(tmp_path, payload) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(config_path)


def test_validator_reports_skipped_remote_checks(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "benchmarks": [
                {
                    "name": "Demo",
                    "dataset": "demo/ds",
                    "splits": ["train"],
                    "fields": ["text"],
                }
            ],
            "ngram": 13,
            "min_hits": 3,
            "min_coverage": 0.001,
        },
    )

    validator = ConfigValidator(
        dataset_builder_loader=lambda *_args, **_kwargs: _FakeBuilder({}),
        benchmark_loader_factory=_EmptyConfigBenchmarkLoader,
    )

    # FIX: Pass allow_unpinned=True to bypass the revision check
    report = validator.validate(config_path, check_remote=True, allow_unpinned=True)

    assert report.remote_checked == 0
    assert report.remote_skipped == 1
    assert any("could not resolve configs" in warning for warning in report.warnings)


def test_validator_raises_on_missing_splits(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "benchmarks": [
                {
                    "name": "Demo",
                    "dataset": "demo/ds",
                    "splits": ["train", "test"],
                    "fields": ["text"],
                }
            ],
            "ngram": 13,
            "min_hits": 3,
            "min_coverage": 0.001,
        },
    )

    def fake_builder_loader(*_args, **_kwargs):
        return _FakeBuilder({"train": object()})

    validator = ConfigValidator(
        dataset_builder_loader=fake_builder_loader,
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
    )

    # FIX: Pass allow_unpinned=True so we hit the 'splits not found' error instead
    with pytest.raises(typer.BadParameter, match="splits not found"):
        validator.validate(config_path, check_remote=True, allow_unpinned=True)


def test_validator_rejects_too_many_benchmarks(tmp_path) -> None:
    benchmarks = [
        {
            "name": f"Benchmark-{idx}",
            "dataset": "demo/ds",
            "splits": ["train"],
            "fields": ["text"],
        }
        for idx in range(65)
    ]
    config_path = _write_config(
        tmp_path,
        {
            "benchmarks": benchmarks,
            "ngram": 13,
            "min_hits": 3,
            "min_coverage": 0.001,
        },
    )

    validator = ConfigValidator(
        dataset_builder_loader=lambda *_args, **_kwargs: _FakeBuilder({}),
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
    )

    # Note: verify that the max benchmarks check happens before allow_unpinned check
    # or pass allow_unpinned=True to be safe.
    with pytest.raises(typer.BadParameter, match="At most 64 benchmarks"):
        validator.validate(config_path, check_remote=False, allow_unpinned=True)


def test_background_corpus_hf_requires_fields() -> None:
    """An HF dataset config must declare fields; an empty list is rejected.

    Regression test: when local_path support was added, the field-required
    invariant was inadvertently relaxed for HF as well, which would let an HF
    corpus silently iterate with no fields.
    """
    with pytest.raises(ValidationError, match="fields"):
        BackgroundCorpusConfig(dataset="some/ds")


def test_background_corpus_local_path_defaults_fields_to_text() -> None:
    """Local JSONL corpora default to a single ``text`` field when unspecified."""
    cfg = BackgroundCorpusConfig(local_path="/tmp/corpus.jsonl")
    assert cfg.fields == ["text"]


def test_background_corpus_requires_exactly_one_source() -> None:
    """Exactly one of ``dataset`` / ``local_path`` must be set."""
    with pytest.raises(ValidationError, match="Exactly one"):
        BackgroundCorpusConfig()
    with pytest.raises(ValidationError, match="Exactly one"):
        BackgroundCorpusConfig(
            dataset="some/ds", local_path="/tmp/corpus.jsonl", fields=["text"]
        )


def test_remote_check_omits_trust_remote_code_unless_requested(tmp_path) -> None:
    """`datasets` 4.x builders reject an explicit trust_remote_code key, which used to
    make every remote check fail; only pass it when the spec asks for it."""
    config_path = _write_config(
        tmp_path,
        {
            "benchmarks": [
                {
                    "name": "Plain",
                    "dataset": "demo/plain",
                    "splits": ["test"],
                    "fields": ["text"],
                },
                {
                    "name": "Scripted",
                    "dataset": "demo/scripted",
                    "splits": ["test"],
                    "fields": ["text"],
                    "trust_remote_code": True,
                },
            ]
        },
    )
    seen: dict[str, dict] = {}

    def fake_builder_loader(dataset, *_args, **kwargs):
        seen[dataset] = kwargs
        if "trust_remote_code" in kwargs and dataset == "demo/plain":
            raise ValueError("BuilderConfig doesn't have a 'trust_remote_code' key.")
        return _FakeBuilder({"test": object()})

    validator = ConfigValidator(
        dataset_builder_loader=fake_builder_loader,
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
    )
    report = validator.validate(config_path, check_remote=True, allow_unpinned=True)

    assert "trust_remote_code" not in seen["demo/plain"]
    assert seen["demo/scripted"]["trust_remote_code"] is True
    assert (report.remote_checked, report.remote_skipped) == (2, 0)


def _one_benchmark(tmp_path, **spec) -> str:
    return _write_config(
        tmp_path,
        {
            "benchmarks": [
                {"name": "Demo", "dataset": "demo/ds", "fields": ["q"], **spec}
            ]
        },
    )


@pytest.mark.parametrize(
    ("data_files", "splits"),
    [
        ({"test": ["a.jsonl", "b.jsonl"]}, ["test"]),
        ("all.jsonl", ["train"]),
        (["a.jsonl", "b.jsonl"], ["train"]),
    ],
)
def test_remote_check_falls_back_to_data_files_splits(
    tmp_path, data_files, splits
) -> None:
    """A dataset with no split metadata (e.g. opencompass/AIME2025) validates from
    its data_files keys instead of failing with 'Available: []'."""
    config_path = _one_benchmark(tmp_path, data_files=data_files, splits=splits)
    validator = ConfigValidator(
        dataset_builder_loader=lambda *_a, **_k: _FakeBuilder({}),
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
    )
    report = validator.validate(config_path, check_remote=True, allow_unpinned=True)
    assert (report.remote_checked, report.remote_skipped) == (1, 0)


def test_remote_check_data_files_fallback_still_rejects_unknown_split(
    tmp_path,
) -> None:
    config_path = _one_benchmark(
        tmp_path, data_files={"test": "t.jsonl"}, splits=["validation"]
    )
    validator = ConfigValidator(
        dataset_builder_loader=lambda *_a, **_k: _FakeBuilder({}),
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
    )
    with pytest.raises(typer.BadParameter, match=r"Available: \['test'\]"):
        validator.validate(config_path, check_remote=True, allow_unpinned=True)


def test_remote_check_uses_builder_data_files_when_spec_has_none(tmp_path) -> None:
    config_path = _one_benchmark(tmp_path, splits=["test"])
    validator = ConfigValidator(
        dataset_builder_loader=lambda *_a, **_k: _FakeBuilder(
            {}, config_data_files={"test": ["hf://x/test.jsonl"]}
        ),
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
    )
    report = validator.validate(config_path, check_remote=True, allow_unpinned=True)
    assert report.remote_checked == 1


def test_remote_check_warns_about_hidden_labels(tmp_path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "benchmarks": [
                # A registry entry pointed at its test split: labels are hidden.
                {"ref": "hellaswag", "splits": ["test"]},
                {
                    "name": "Public",
                    "dataset": "demo/public",
                    "splits": ["test"],
                    "fields": ["q"],
                    "label_field": "answer",
                },
                {
                    "name": "NoLabel",
                    "dataset": "demo/x",
                    "splits": ["test"],
                    "fields": ["q"],
                },
            ]
        },
    )
    probed: list[tuple[str, str | None, str]] = []

    def fake_probe(spec, config_name, split):
        probed.append((spec.name, config_name, split))
        return ["", ""] if spec.name == "HellaSwag" else ["A", "B"]

    validator = ConfigValidator(
        dataset_builder_loader=lambda *_a, **_k: _FakeBuilder({"test": object()}),
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
        label_probe=fake_probe,
    )
    report = validator.validate(config_path, check_remote=True, allow_unpinned=True)

    # Only specs with a label_field are probed.
    assert probed == [("HellaSwag", None, "test"), ("Public", None, "test")]
    (warning,) = report.warnings
    assert warning.startswith("[warn] HellaSwag: every 'label' value in split 'test'")
    assert "Harnesses score the 'validation' split of Rowan/hellaswag" in warning
    assert (report.remote_checked, report.remote_skipped) == (3, 0)


def test_remote_check_label_probe_failure_is_a_warning(tmp_path) -> None:
    config_path = _one_benchmark(tmp_path, splits=["test"], label_field="answer")

    def failing_probe(*_args):
        raise RuntimeError("no streaming")

    validator = ConfigValidator(
        dataset_builder_loader=lambda *_a, **_k: _FakeBuilder({"test": object()}),
        benchmark_loader_factory=_SingleConfigBenchmarkLoader,
        label_probe=failing_probe,
    )
    report = validator.validate(config_path, check_remote=True, allow_unpinned=True)
    (warning,) = report.warnings
    assert "could not read 'answer'" in warning and "no streaming" in warning
    assert report.remote_checked == 1
