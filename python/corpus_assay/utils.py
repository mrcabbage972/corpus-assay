from typing import Any

from datasets import load_dataset


def _load_dataset_optional_revision(
    *args: Any, revision: str | None = None, **kwargs: Any
):
    if revision is None:
        return load_dataset(*args, **kwargs)
    try:
        return load_dataset(*args, revision=revision, **kwargs)
    except TypeError:
        return load_dataset(*args, **kwargs)
