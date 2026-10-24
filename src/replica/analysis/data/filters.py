"""Row-level outlier filter applied once, before cross-validation."""

from collections.abc import Callable

import numpy as np

# MAD -> consistent estimator of sigma for normal data, Leys et al. (2013).
_MAD_TO_SD = 1.4826


class OutlierFilter:
    """Drops rows by raw DV using optional floor/ceiling and groupwise outlier trims.

    - ``floor`` / ``ceiling``: keep ``y >= floor`` / ``y <= ceiling``.
    - ``method``: trims outliers within each group (``participants``), or globally if None:
        * ``"mad"``: ``|y - median| <= k * 1.4826 * MAD``
        * ``"sd"``:   ``|y - mean|   <= k * std``
      ``k`` defaults to 2.5.
    - A callable ``method(values)`` returns a boolean keep mask for each finite
      group. It can implement quantile, IQR, or application-specific rules.
      Bounds combine with the rule; ``k`` applies only to the built-in methods.
    """

    def __init__(
        self,
        *,
        floor: float | None = None,
        ceiling: float | None = None,
        method: str | Callable[[np.ndarray], np.ndarray] | None = None,
        k: float = 2.5,
    ):
        """Set the DV bounds and the dispersion trim.

        Args:
            floor: Lowest raw DV value kept; ``None`` leaves the low tail alone.
            ceiling: Highest raw DV value kept; ``None`` leaves the high tail
                alone.
            method: ``"mad"``, ``"sd"``, or a callable mapping finite group
                values to a same-shape boolean keep mask. ``None`` skips trimming.
            k: Multiplier on that estimator, so the trim keeps values within
                ``k`` scaled deviations of the group centre.

        Raises:
            ValueError: When *method* is not a supported name, callable, or None.
        """
        self.floor = floor
        self.ceiling = ceiling
        if not callable(method) and method not in (None, "mad", "sd"):
            raise ValueError(
                f"method must be None, 'mad', 'sd', or callable; got {method!r}"
            )
        self.method = method
        self.k = float(k)

    @property
    def is_active(self) -> bool:
        """True when a bound or a trim is configured, so rows can be dropped."""
        return (
            self.floor is not None
            or self.ceiling is not None
            or self.method is not None
        )

    def keep_mask(
        self, y_raw: np.ndarray, participants: np.ndarray | None = None
    ) -> np.ndarray:
        """Boolean keep-mask for raw DV values (all-True when inactive).

        Args:
            y_raw: Dependent variable in its original units, one entry per row.
            participants: Participant label per row, which groups the trim.
                ``None`` computes one trim globally over the finite values.

        Returns:
            A mask over the rows of *y_raw*, true where the row survives.
        """
        y = np.asarray(y_raw, dtype=float)
        keep = np.ones(y.shape, dtype=bool)
        if self.floor is not None:
            keep &= y >= self.floor
        if self.ceiling is not None:
            keep &= y <= self.ceiling
        if self.method is not None:
            if participants is None:
                keep &= self._trim_mask(y, np.isfinite(y))
            else:
                pids = np.asarray(participants)
                for g in np.unique(pids):
                    grp = (pids == g) & np.isfinite(y)
                    if grp.any():
                        keep &= self._trim_mask(y, grp)
        return keep

    def _trim_mask(self, y: np.ndarray, grp: np.ndarray) -> np.ndarray:
        """Keep-mask for one group: True everywhere outside ``grp`` so the
        per-group masks conjoin cleanly.
        """
        vals = y[grp]
        keep = np.ones(y.shape, dtype=bool)
        if callable(self.method):
            if vals.size:
                mask = np.asarray(self.method(vals))
                if mask.shape != vals.shape or mask.dtype != np.dtype(bool):
                    raise ValueError(
                        "Custom outlier method must return a boolean mask "
                        "with the same shape as its input"
                    )
                keep[grp] = mask
            return keep
        if self.method == "mad":
            center = float(np.median(vals))
            scale = _MAD_TO_SD * float(np.median(np.abs(vals - center)))
        else:  # "sd"
            center = float(np.mean(vals))
            scale = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        if scale > 0:
            keep[grp] = np.abs(vals - center) <= self.k * scale
        return keep

    def __repr__(self) -> str:
        return (
            f"OutlierFilter(floor={self.floor}, ceiling={self.ceiling}, "
            f"method={self.method!r}, k={self.k})"
        )
