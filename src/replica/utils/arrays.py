"""Array conversion that preserves declared missing observations."""

import numpy as np


def array_with_missing(values) -> np.ndarray:
    """Convert masked entries to NaN; unmasked inputs follow ``np.asarray``.

    Args:
        values: Array-like values, optionally carrying a NumPy mask.
    """
    array = np.asarray(values)
    if not np.ma.isMaskedArray(values):
        return array
    if array.dtype.fields is not None:
        if np.ma.flatten_mask(np.ma.getmask(values)).any():
            raise ValueError(
                "Masked structured arrays require explicit field selection"
            )
        return array
    if not np.ma.is_masked(values):
        return array
    dtype = (
        array.dtype
        if array.dtype.kind in "fc"
        else np.dtype(float)
        if array.dtype.kind in "biu"
        else np.dtype(object)
    )
    return np.ma.asarray(values, dtype=dtype).filled(np.nan)


def real_array_with_missing(values, *, name: str = "input") -> np.ndarray:
    """Normalize missing values and reject complex inputs without coercion.

    Args:
        values: Real array-like values, optionally carrying a NumPy mask.
        name: Input label used in validation errors.
    """
    array = array_with_missing(values)
    if np.iscomplexobj(array) or (
        array.dtype.kind == "O" and any(np.iscomplexobj(value) for value in array.flat)
    ):
        raise ValueError(
            f"{name} must be real-valued; complex values need explicit projection"
        )
    return array
