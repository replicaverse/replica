"""Shared predictor-source access for the corpus-replication drivers.

A replication cell rebuilds a study's design frame for a corpus the paper
never analyzed, from the published predictor store or from a campaign's
local compute directory, then hands the frame to the study's own analysis
engine. These helpers resolve and load those sources; the per-study frame
builders live in each study package's ``replication`` module, and the sweep
driver belongs to the campaign that selects the studies and corpus universe.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from replica.dataset.catalog.registry import compute_dir

from ..features import pickle_template_for, resolve_existing_pickle


def resolve_predictor_source(
    cdir: Path,
    lm: str,
    suffix: str,
    ds_name: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> Path | str | None:
    """A local pickle if the compute dir has one, else the published store tag.

    The compute dir holds whatever the last campaign happened to build, while
    the store carries the published cell for every dataset, so a cross-dataset
    run must be able to read either. Tag and pickle stem are the same string.

    Args:
        cdir: Compute directory searched before the published store.
        lm: Canonical model identifier, or a caller-supplied alias.
        suffix: Predictor tag between the prefix and the dataset name.
        ds_name: Registry dataset name.
        aliases: Optional application aliases mapped to canonical identifiers.

    Returns:
        A local path, a published tag, or ``None`` when neither resolves.
    """
    local = resolve_existing_pickle(cdir, lm, suffix, ds_name, aliases=aliases)
    if local is not None:
        return local
    tag = Path(pickle_template_for(lm, suffix, aliases=aliases).format(ds=ds_name)).stem
    from replica.predictors import published_tags

    return tag if tag in (published_tags() or ()) else None


def load_predictor_source(src: Path | str, *, allow_reproduction: bool = False) -> dict:
    """Load a local predictor artifact or a published store tag.

    Local files require a completed, checksum-matching manifest. An explicit
    path takes precedence over the store; string tags are preserved verbatim.

    Args:
        src: A local pickle path, or a published store tag as a plain string.
        allow_reproduction: Required when the cell's declared method is kept
            for reproduction only; a local file declares it through its name
            and its manifest.

    Returns:
        Stimulus id to list of ``FeatureModelOutput``.

    Raises:
        ValueError: A missing or invalid manifest, a non-mapping payload, or
            reproduction-only data without consent.
    """
    from replica.predictors import (
        check_reproduction_consent,
        file_tag,
        load_outputs_for_tag,
    )

    if isinstance(src, Path):
        from replica.artifacts import validate_artifact

        identity = validate_artifact(src)
        if identity is None:
            if not src.exists():
                raise FileNotFoundError(src)
            raise ValueError(
                f"{src}: no provenance manifest; recompute the cell or import "
                "it with verified provenance"
            )
        check_reproduction_consent(
            file_tag(src), identity, allow_reproduction=allow_reproduction
        )
        with src.open("rb") as handle:
            outputs = pickle.load(handle)
        if not isinstance(outputs, dict):
            raise ValueError(f"{src}: predictor payload must be a mapping")
        return outputs
    return load_outputs_for_tag(src, allow_reproduction=allow_reproduction)


def load_store_cell(
    tag: str,
    *,
    allow_compute_dir: bool = False,
    cdir: Path | None = None,
    allow_reproduction: bool = False,
) -> dict | None:
    """One predictor cell by tag, from the store or the compute directory.

    Cells a campaign computed but has not published yet exist only as local
    pickles, so studies that consume them pass ``allow_compute_dir=True`` to
    read the pickle the sweep wrote.

    Args:
        tag: The cell's store tag, which doubles as the local pickle stem.
        allow_compute_dir: Also accept a local pickle with a completed,
            checksum-matching manifest.
        cdir: Compute directory holding that pickle; ``None`` takes the
            registry default.
        allow_reproduction: Required when the cell's declared method is kept
            for reproduction only.

    Returns:
        The outputs dict, or ``None`` when the cell is absent or is not a
        mapping.
    """
    local = (cdir or compute_dir()) / f"{tag}.pkl" if allow_compute_dir else None
    if local is not None and local.exists():
        return load_predictor_source(local, allow_reproduction=allow_reproduction)
    from replica.predictors import load_outputs_for_tag

    try:
        data = load_outputs_for_tag(tag, allow_reproduction=allow_reproduction)
    except FileNotFoundError:
        return None
    return data if isinstance(data, dict) else None


def lagged(values: np.ndarray, k: int) -> np.ndarray:
    """Within-stimulus lag, NaN-filled at the leading boundary.

    Args:
        values: One stimulus's per-unit values, in reading order.
        k: How many units back to shift; the first *k* entries become NaN.
    """
    out = np.full(len(values), np.nan)
    out[k:] = values[: len(values) - k]
    return out
