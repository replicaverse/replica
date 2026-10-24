"""Categorical label validation shared by preparation, models and inference."""

import numpy as np


def categorical_labels(values, *, name: str = "factor") -> np.ndarray:
    """Normalize consistently typed scalar labels without merging distinct IDs.

    Args:
        values: One-dimensional text, integer, real or boolean labels. Mixed
            scalar families require explicit normalization by the caller.
        name: Factor name used in validation errors.
    """
    if np.ma.is_masked(values):
        raise ValueError(f"Missing labels in {name!r}")
    if isinstance(values, np.ndarray) and values.ndim == 1 and values.dtype.kind == "U":
        if np.any(values == ""):
            raise ValueError(f"Missing labels in {name!r}")
        return values
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"Labels for {name!r} must be one-dimensional")
    kinds = set()
    for value in raw:
        if isinstance(value, str):
            if not value:
                raise ValueError(f"Missing labels in {name!r}")
            kinds.add("text")
        elif isinstance(value, bool | np.bool_):
            kinds.add("boolean")
        elif isinstance(value, int | np.integer):
            kinds.add("integer")
        elif isinstance(value, float | np.floating) and np.isfinite(value):
            kinds.add("real")
        else:
            raise ValueError(f"Missing, non-finite or unsupported labels in {name!r}")
    if len(kinds) > 1:
        raise ValueError(
            f"Labels for {name!r} mix scalar types; normalize them explicitly"
        )
    return raw.astype(str)
