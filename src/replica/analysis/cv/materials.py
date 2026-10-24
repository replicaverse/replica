"""Identify the source material held out by corrected reading-time analyses.

A loader's stimulus can be a page or a paragraph. Generalization to unseen
texts requires holding out its parent material, including alternate versions.
A loader declares that material through ``DatasetInfo.material_fields``, which
``Dataset._make_stimulus`` turns into ``metadata["cv_group"]``; a caller can
set the same key directly. A stimulus without the key is its own material.
Splitters that group by stimulus ID, such as ``ItemSplitter``, and the
repetition paths do not use this module.
"""

import unicodedata
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .splitters import ItemSplitter


def material_id(stimulus) -> str:
    """Return the material identity: ``metadata["cv_group"]`` or the stimulus ID.

    Args:
        stimulus: Stimulus with a nonempty identifier and optional metadata.
    """
    sid = stimulus.stimulus_id
    if _missing(sid):
        raise ValueError("Corrected CV requires nonempty stimulus identities")
    metadata = stimulus.metadata or {}
    if "cv_group" in metadata:
        key = metadata["cv_group"]
        if _missing(key):
            raise ValueError("Empty explicit cv_group")
        return str(key)
    return str(sid)


def _missing(value) -> bool:
    """Whether an identity is absent: not a scalar, a missing value or blank."""
    return bool(
        not pd.api.types.is_scalar(value) or pd.isna(value) or not str(value).strip()
    )


def resolve_material_ids(stimuli) -> dict[str, str]:
    """Co-hold declared parent materials connected by identical stimulus text.

    Stimuli join transitively by parent ID and by exact text after Unicode and
    whitespace normalization; an empty text matches nothing.

    Args:
        stimuli: Source stimuli with unique IDs, text and parent metadata.
    """
    stimuli = list(stimuli)
    native = {s.stimulus_id: material_id(s) for s in stimuli}
    if len(native) != len(stimuli):
        raise ValueError("Duplicate stimulus identifier")
    parent = {value: value for value in native.values()}

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    content = {}
    for stimulus in stimuli:
        text = " ".join(unicodedata.normalize("NFC", stimulus.text or "").split())
        if not text:
            continue
        current = native[stimulus.stimulus_id]
        if text in content:
            a, b = sorted((root(current), root(content[text])))
            parent[b] = a
        else:
            content[text] = current
    return {sid: root(value) for sid, value in native.items()}


def material_folds(groups: Sequence[str], n_folds: int = 10) -> np.ndarray:
    """Return ItemSplitter fold IDs for a frame carrying real material IDs.

    Args:
        groups: One nonempty source-material identity per row.
        n_folds: Maximum folds; capped by the number of distinct materials.
    """
    if any(g is None or not str(g).strip() or str(g) == "nan" for g in groups):
        raise ValueError("Missing material identity")
    ids = np.asarray(groups, dtype=str)
    if ids.ndim != 1 or len(np.unique(ids)) < 2:
        raise ValueError("Corrected CV needs at least two distinct materials")
    folds = np.full(len(ids), -1, dtype=int)
    dummy = np.zeros(len(ids), dtype=int)
    for f, (train, test) in enumerate(
        ItemSplitter(n_folds).split(dummy, ids, dummy, dummy)
    ):
        if set(ids[train]) & set(ids[test]):
            raise ValueError("Material overlaps training and test")
        folds[test] = f
    if np.any(folds < 0):
        raise ValueError("Some rows have no held-out fold")
    return folds
