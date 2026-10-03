"""``python -m corpus_assay`` entry point (same as the ``corpus-assay`` command)."""

from corpus_assay.cli import main

# Guarded: spawned scan workers re-import the main module as ``__mp_main__``.
if __name__ == "__main__":
    main()
