import importlib
import importlib.util


def test_can_import_corpus_assay_module() -> None:
    module = importlib.import_module("corpus_assay._native")
    assert module is not None
