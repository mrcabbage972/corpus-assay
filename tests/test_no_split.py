"""``--no-split`` / ``packed_doc_sep: null`` turn packed-document splitting off (#9)."""

from __future__ import annotations

import pytest

from corpus_assay.config import BackgroundCorpusConfig, ScanConfig
from corpus_assay.stopgrams import _iter_documents_from_rows


def _scan_cfg(**kwargs: object) -> ScanConfig:
    return ScanConfig(
        inputs=["x"],
        text_key="text",
        id_key=None,
        index_path="idx.native",
        out_dir="out",
        **kwargs,  # type: ignore[arg-type]
    )


def test_scan_config_none_separator_drops_typo() -> None:
    cfg = _scan_cfg(packed_doc_sep=None)
    assert cfg.packed_doc_sep is None
    assert cfg.packed_doc_sep_typo is None
    # The default still splits on <|endoftext|> and rewrites its typo variant.
    default = _scan_cfg()
    assert (default.packed_doc_sep, default.packed_doc_sep_typo) == (
        "<|endoftext|>",
        "<|endoftext}>",
    )


def test_scan_config_rejects_empty_separator() -> None:
    with pytest.raises(ValueError, match="packed_doc_sep must be non-empty"):
        _scan_cfg(packed_doc_sep="")


def test_background_corpus_null_separator_keeps_documents_whole() -> None:
    corpus = BackgroundCorpusConfig.model_validate(
        {"local_path": "bg.jsonl", "packed_doc_sep": None}
    )
    assert corpus.packed_doc_sep is None and corpus.packed_doc_sep_typo is None

    rows = [{"text": "first part<|endoftext|>second part"}]
    whole = list(
        _iter_documents_from_rows(
            rows,
            fields=["text"],
            packed_doc_sep=corpus.packed_doc_sep,
            packed_doc_sep_typo=corpus.packed_doc_sep_typo,
        )
    )
    assert whole == ["first part<|endoftext|>second part"]

    split = list(
        _iter_documents_from_rows(
            rows,
            fields=["text"],
            packed_doc_sep="<|endoftext|>",
            packed_doc_sep_typo=None,
        )
    )
    assert split == ["first part", "second part"]
