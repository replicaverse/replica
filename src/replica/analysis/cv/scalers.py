"""Scaler protocol and built-in implementations."""

from collections.abc import Callable
from copy import deepcopy
from numbers import Integral
from typing import Protocol, runtime_checkable

import numpy as np


def _apply_columnwise(
    train: np.ndarray,
    test: np.ndarray,
    stat_fn: Callable[[np.ndarray], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Scale both splits per column with train-fitted ``(center, scale)``.

    ``stat_fn(values)`` returns ``(center, scale)``; a non-positive scale
    falls back to centering only. Operates column-by-column for 2-D arrays.
    """
    if train.ndim == 1:
        center, scale = stat_fn(train)
        if scale > 0:
            return (train - center) / scale, (test - center) / scale
        return train - center, test - center

    train_out = np.empty_like(train, dtype=float)
    test_out = np.empty_like(test, dtype=float)
    for c in range(train.shape[1]):
        center, scale = stat_fn(train[:, c])
        if scale > 0:
            train_out[:, c] = (train[:, c] - center) / scale
            test_out[:, c] = (test[:, c] - center) / scale
        else:
            train_out[:, c] = train[:, c] - center
            test_out[:, c] = test[:, c] - center
    return train_out, test_out


def _standardize_split(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using training-set statistics, applied to both splits."""
    return _apply_columnwise(train, test, lambda v: (np.mean(v), np.std(v)))


@runtime_checkable
class Scaler(Protocol):
    """Strategy for scaling features per CV fold.

    ``fit_transform`` fits on training data and transforms both splits.
    """

    def fit_transform(
        self,
        train: np.ndarray,
        test: np.ndarray,
        ctx: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the scaling on the training rows and apply it to both splits.

        Args:
            train: Design-matrix rows of the fold's training split, one column
                per predictor; the scaling statistics are estimated from these.
            test: Held-out rows of the same fold, scaled with the statistics
                fitted on ``train`` so no held-out information leaks in.
            ctx: Per-fold context (train/test masks and the full feature
                columns). Column-uniform scalers ignore it.

        Returns:
            The scaled training and test blocks, in that order.
        """
        ...


class _FittedScaler:
    """Capture statistics used by the latest completed transformation."""

    _fit_diagnostics: dict | None = None

    def last_fit_diagnostics(self) -> dict | None:
        """Return detached training statistics, or None before a completed fit."""
        return deepcopy(self._fit_diagnostics)

    def _transform(self, train, test, stat_fn, method):
        self._fit_diagnostics = None
        centers, scales = [], []

        def capture(values):
            center, scale = stat_fn(values)
            centers.append(float(center))
            scales.append(float(scale))
            return center, scale

        result = _apply_columnwise(train, test, capture)
        self._fit_diagnostics = {
            "method": method,
            "n_columns": len(centers),
            "center": centers,
            "scale": scales,
            "effective_scale": [value if value > 0 else 1.0 for value in scales],
        }
        return result


class ZScoreScaler(_FittedScaler):
    """Per-column z-score standardisation (mean=0, std=1).

    Fits on the training split and applies the same transform to both.
    ``ddof`` selects the standard-deviation denominator ``n - ddof``;
    the default is population scaling (0), and sample scaling uses 1.
    """

    def __init__(self, ddof: int = 0):
        """Configure the standard-deviation denominator.

        Args:
            ddof: Degrees of freedom subtracted from the training row count.
        """
        if isinstance(ddof, bool) or not isinstance(ddof, Integral) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer")
        self.ddof = int(ddof)

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {"ddof": self.ddof}

    def fit_transform(
        self,
        train: np.ndarray,
        test: np.ndarray,
        ctx: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Centre and divide by the training column mean and standard deviation.

        Args:
            train: Training rows of the fold; each column's mean and standard
                deviation come from these rows.
            test: Held-out rows, shifted and divided by the same statistics.
            ctx: Per-fold context, unused: the transform is column-uniform.

        Returns:
            The standardised training and test blocks. A column with zero
            training variance is centred only.
        """
        self._fit_diagnostics = None
        if len(train) <= self.ddof:
            raise ValueError("ddof must be smaller than the number of training rows")
        result = self._transform(
            train, test, lambda v: (np.mean(v), np.std(v, ddof=self.ddof)), "zscore"
        )
        if self._fit_diagnostics is not None:
            self._fit_diagnostics["ddof"] = self.ddof
        return result


class MinMaxScaler(_FittedScaler):
    """Per-column min-max scaling to [0, 1].

    Fits min/max on the training split and applies to both.
    """

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {}

    def fit_transform(
        self,
        train: np.ndarray,
        test: np.ndarray,
        ctx: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map each column onto [0, 1] using the training minimum and range.

        Args:
            train: Training rows of the fold; each column's minimum and range
                come from these rows.
            test: Held-out rows, rescaled by the same minimum and range, so
                values outside the training range fall outside [0, 1].
            ctx: Per-fold context, unused: the transform is column-uniform.

        Returns:
            The rescaled training and test blocks. A constant training column
            is shifted to zero only.
        """
        return self._transform(
            train, test, lambda v: (np.min(v), np.max(v) - np.min(v)), "minmax"
        )


class NoScaler(_FittedScaler):
    """Pass-through: no scaling applied.

    Use when features are already pre-scaled externally.
    """

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {}

    def fit_transform(
        self,
        train: np.ndarray,
        test: np.ndarray,
        ctx: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return both splits untouched.

        Args:
            train: Training rows of the fold, passed straight through.
            test: Held-out rows of the fold, passed straight through.
            ctx: Per-fold context, unused.

        Returns:
            The two input blocks, unmodified and uncopied.
        """
        self._fit_diagnostics = None
        n_columns = 1 if train.ndim == 1 else train.shape[1]
        self._fit_diagnostics = {
            "method": "none",
            "n_columns": n_columns,
            "center": [0.0] * n_columns,
            "scale": [1.0] * n_columns,
            "effective_scale": [1.0] * n_columns,
        }
        return train, test
