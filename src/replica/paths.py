"""Writable runtime locations and explicit source-checkout discovery."""

import os
from pathlib import Path


def source_checkout() -> Path | None:
    """Find a REPLICA checkout, honoring a validated REPLICA_CHECKOUT override."""
    override = os.environ.get("REPLICA_CHECKOUT")
    candidates = (
        [Path(override).expanduser().resolve()]
        if override
        else Path(__file__).resolve().parents
    )
    for root in candidates:
        if (root / "pyproject.toml").is_file() and (root / "src/replica").is_dir():
            return root
    if override:
        raise RuntimeError("REPLICA_CHECKOUT must point at a REPLICA source checkout")
    return None


def checkout_root() -> Path:
    """Require a source checkout for a developer command."""
    root = source_checkout()
    if root is None:
        raise RuntimeError(
            "This command needs a REPLICA source checkout; set REPLICA_CHECKOUT"
        )
    return root


def _xdg_home(variable: str, default: str) -> Path:
    """An XDG base directory; an empty or relative value counts as unset."""
    value = Path(os.environ.get(variable, "")).expanduser()
    return value if value.is_absolute() else Path.home() / default


def user_data_root() -> Path:
    """Writable root for downloaded datasets and external reproduction inputs."""
    return _xdg_home("XDG_DATA_HOME", ".local/share") / "replica"


def user_cache_root() -> Path:
    """Writable root for regenerable predictor caches."""
    return _xdg_home("XDG_CACHE_HOME", ".cache") / "replica"
