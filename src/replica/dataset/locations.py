"""Dataset locations shared by discovery, fetching, and companion-data loaders."""

import os
from pathlib import Path

from replica.paths import source_checkout, user_data_root

REPO_ROOT: Path | None = source_checkout()
DATA_DIR_ENV = "PSYCHO_DATA_DIR"
UCL_ET_SUBDIR = "eye-tracking/original/ucl"
DEVARDA2023_SUBDIR = "eye-tracking/processed/devarda2023"


def data_dir() -> Path:
    """Resolve the dataset root from PSYCHO_DATA_DIR, a checkout, or user data.

    An environment override is expanded and resolved. Without it, use
    ``<checkout>/datasets`` or ``<user_data_root()>/datasets`` when installed.
    """
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT or user_data_root()) / "datasets"


def dataset_path(subdir: str, repo_root: Path | None = None) -> Path:
    """Resolve a dataset subdirectory, optionally under an explicit checkout.

    Args:
        subdir: Dataset path relative to the data root.
        repo_root: Checkout override; bypasses the environment when supplied.
    """
    if repo_root is not None:
        return Path(repo_root) / "datasets" / subdir
    return data_dir() / subdir
