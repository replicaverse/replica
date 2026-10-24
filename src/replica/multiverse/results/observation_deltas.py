"""Per-observation held-out ΔLL sidecar store for the sign-flip tests.

The joint sign-flip test over a claim's cells takes one per-observation
held-out ΔLL vector per cell (see
:func:`replica.multiverse.curve.stats.joint_signflip_inference`), so each cell's
vector must be persisted. This module owns the sidecar layout: the grid
driver writes one sidecar per cell (``grid_per_obs_path``), the master fit
one per (lm, predictor, dataset, stage, dv, variant) (``master_per_obs_path``).
Each sidecar is a 1-D float ``.npy``, one entry per held-out observation,
concatenated across CV folds.
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from replica.dataset.catalog.registry import compute_dir

from ...paths import checkout_root

logger = logging.getLogger("replica")

# The grid sidecar dir is overridable via ``PSYCHO_PER_OBS_GRID_DIR`` so an
# alternate grid run (e.g. the non-cumulative ablation grid) writes its
# per-observation arrays to a separate tree instead of clobbering the
# cumulative grid's sidecars (the cell key carries no mode tag). ``None``
# means "resolve from the env var, else the checkout default" at call time.
GRID_PER_OBS_DIR: Path | None = None


def _grid_per_obs_dir() -> Path:
    env = os.environ.get("PSYCHO_PER_OBS_GRID_DIR")
    if env:
        return Path(env)
    if GRID_PER_OBS_DIR is not None:
        return Path(GRID_PER_OBS_DIR)
    return checkout_root() / "logs" / "replication_grid" / "per_obs"


def _master_per_obs_dir() -> Path:
    env = os.environ.get("PSYCHO_PER_OBS_MASTER_DIR")
    if env:
        return Path(env)
    return compute_dir() / "per_obs" / "master"


def _slug(s) -> str:
    """Filesystem-safe token (keeps the predictor's ``=``/``.``/``+`` chars)."""
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", "" if s is None else str(s))


def grid_per_obs_path(paper, corpus, predictor, position, dv) -> Path:
    """Sidecar path for one grid cell, keyed as the grid TSV rows.

    The filename encodes the position as an ``s{position}`` token.

    Resolves the base dir at call time (env ``PSYCHO_PER_OBS_GRID_DIR``, else
    the ``GRID_PER_OBS_DIR`` module attribute, else the checkout default) so a
    caller can redirect it and have writers and readers agree in-process.

    Args:
        paper: Paper tag the cell is filed under.
        corpus: Corpus the cell was fitted on.
        predictor: Predictor being scored.
        position: Index of the cell within its recipe walk, -1 for cells
            addressed by label alone.
        dv: Dependent variable of the cell.
    """
    name = "__".join(_slug(x) for x in (paper, corpus, predictor, f"s{position}", dv))
    return _grid_per_obs_dir() / f"{name}.npy"


def master_per_obs_path(lm, predictor, dataset, stage, dv, variant="") -> Path:
    """Sidecar path for one master-replication cell (endpoint source).

    The directory is ``PSYCHO_PER_OBS_MASTER_DIR`` when set, else
    ``per_obs/master`` under :func:`~replica.dataset.catalog.registry.compute_dir`,
    resolved at call time.

    Args:
        lm: Language-model family behind the predictor.
        predictor: Predictor being scored.
        dataset: Corpus the cell was fitted on.
        stage: Regression family the cell was fitted under.
        dv: Dependent measure of the cell.
        variant: Skip treatment of an eye-tracking duration; empty for a
            measure without skipping, which leaves it out of the name.
    """
    parts = (lm, predictor, dataset, stage, dv, *((variant,) if variant else ()))
    name = "__".join(_slug(x) for x in parts)
    return _master_per_obs_dir() / f"{name}.npy"


def save(path: Path, arr: ArrayLike | None) -> None:
    """Persist *arr* as a 1-D float ``.npy`` (no-op for empty/None arrays).

    Args:
        path: Sidecar to write; parent directories are created.
        arr: Per-observation held-out ΔLL, flattened before it is written.
    """
    if arr is None:
        return
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)


def load(path: Path) -> np.ndarray | None:
    """Load a sidecar, or ``None`` if it is absent/unreadable.

    Args:
        path: Sidecar to read.
    """
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception as exc:
        # A present-but-unreadable sidecar is a real corruption, distinct from
        # an absent one; surface it instead of silently treating it as missing.
        logger.warning("per-obs sidecar %s exists but failed to load: %s", path, exc)
        return None
