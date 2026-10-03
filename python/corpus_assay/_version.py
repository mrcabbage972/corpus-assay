"""Distribution name and installed version, in one place."""

from __future__ import annotations

from importlib import metadata

DIST_NAME = "corpus-assay"


def dist_version() -> str:
    """Return the installed ``corpus-assay`` version, or ``"unknown"``."""
    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        # Running from a source tree without an installed distribution.
        return "unknown"
