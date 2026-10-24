"""Cross-validation with every declared grouping factor unseen at prediction."""

import math
import warnings
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from itertools import product
from numbers import Integral, Real

import numpy as np


def _values(value, label):
    """Normalize a scalar or per-axis sequence without accepting empty inputs."""
    values = (value,) if isinstance(value, Real) else tuple(value)
    if not values:
        raise ValueError(f"Provide at least one {label}")
    return values


class CrossedGroupSplitter:
    """Cross group partitions and score their test intersections once per cycle.

    Training excludes every held-out identity. Rows sharing just one held-out
    identity are unused in that fit. The default targets a nominal 90% of the
    distinct groups on each axis in training, resolving to ten folds per axis
    when enough groups exist. Whole groups and missing intersections can make
    the actual training fractions differ; :meth:`preview` reports them.

    Args:
        n_folds: Explicit maximum folds, or one maximum per axis. Mutually
            exclusive with ``train_fraction``.
        train_fraction: Nominal training fraction in [0.5, 1), shared across
            axes, supplied per axis, or keyed by factor name. Defaults to 0.9
            when fold counts are omitted. Named fractions require named
            ``groups``, which ``evaluate()`` supplies from ``Design.cv_unit``.
        seed: Shuffle group assignments reproducibly; defaults to 42. None keeps
            sorted identities in round-robin partitions. Different seeds give
            repeated complete cycles; they do not add independent groups.
    """

    uses_declared_factors = True
    uses_named_factors = True

    def configuration(self) -> dict:
        """Return declared options without fitted state."""
        return {
            "n_folds": self.n_folds,
            "train_fraction": self.train_fraction,
            "seed": self.seed,
        }

    def __init__(
        self,
        n_folds: int | tuple[int, ...] | None = None,
        *,
        train_fraction: float | tuple[float, ...] | Mapping[str, float] | None = None,
        seed: int | None = 42,
    ):
        """Validate the requested fractions or explicit fold counts."""
        if n_folds is not None and train_fraction is not None:
            raise ValueError("Choose either train_fraction or n_folds, not both")
        self.n_folds = None if n_folds is None else _values(n_folds, "fold count")
        if self.n_folds is not None and any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 2
            for n in self.n_folds
        ):
            raise ValueError("At least two integer folds are required on every axis")
        if train_fraction is None and n_folds is None:
            train_fraction = 0.9
        if isinstance(train_fraction, Mapping):
            if not train_fraction or any(
                not isinstance(key, str) or not key for key in train_fraction
            ):
                raise ValueError("Named train_fraction requires nonempty factor names")
            self.train_fraction = dict(train_fraction)
            fractions = tuple(train_fraction.values())
        else:
            self.train_fraction = (
                None
                if train_fraction is None
                else _values(train_fraction, "training fraction")
            )
            fractions = self.train_fraction or ()
        if any(
            isinstance(f, bool) or not isinstance(f, Real) or not 0.5 <= f < 1
            for f in fractions
        ):
            raise ValueError("Complete K-fold CV needs train_fraction in [0.5, 1)")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None")
        self.seed = None if seed is None else int(seed)
        self._metadata: list[dict] = []
        self._summary: dict | None = None

    @staticmethod
    def _labels(groups, participants):
        """Resolve independent identity arrays without combining their types."""
        if isinstance(groups, Mapping):
            names = tuple(groups)
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError("Grouping axes require nonempty names")
            labels = [np.asarray(groups[name]) for name in names]
        else:
            matrix = np.asarray(groups)
            if matrix.ndim == 1:
                names = ("participant", "stimulus")
                labels = [np.asarray(participants), matrix]
            elif matrix.ndim == 2:
                names = tuple(f"axis_{i}" for i in range(matrix.shape[1]))
                labels = [matrix[:, i] for i in range(matrix.shape[1])]
            else:
                raise ValueError(
                    "Crossed CV requires aligned labels for at least two axes"
                )
        if len(labels) < 2 or any(a.ndim != 1 for a in labels):
            raise ValueError("Crossed CV requires aligned labels for at least two axes")
        return names, labels

    def _requests(self, names):
        """Resolve nominal fractions to the smallest compatible integer counts."""
        fractions = self.train_fraction
        if isinstance(fractions, Mapping):
            if set(fractions) != set(names):
                raise ValueError(
                    "train_fraction keys must match the held-out factor names"
                )
            fractions = tuple(fractions[name] for name in names)
        if fractions is not None:
            fractions = tuple(float(f) for f in fractions)
            # Decimal arithmetic keeps ordinary targets such as 0.8 at five
            # folds instead of rounding a floating-point 5.000000000000001 up.
            counts = tuple(
                math.ceil(Decimal(1) / (Decimal(1) - Decimal(str(f))))
                for f in fractions
            )
        else:
            counts = self.n_folds
        if len(counts) == 1:
            counts = counts * len(names)
            if fractions is not None:
                fractions = fractions * len(names)
        if len(counts) != len(names):
            raise ValueError(
                "Provide one fold count or training fraction per grouping axis"
            )
        return tuple(int(n) for n in counts), fractions

    def _assignments(self, names, labels):
        """Balance distinct groups on each axis, recording capped requests."""
        requested, fractions = self._requests(names)
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        assignments, codes_by_axis, sizes, groups_per_axis = [], [], [], []
        for name, values, count in zip(names, labels, requested, strict=True):
            unique, codes = np.unique(values, return_inverse=True)
            size = min(count, len(unique))
            if size < 2:
                raise ValueError("Every held-out axis requires at least two groups")
            if size < count:
                warnings.warn(
                    f"Crossed CV axis {name!r} has {len(unique)} groups: using {size} "
                    f"folds instead of {count}. The requested training target is "
                    "unattainable; inspect the achieved fractions in preview().",
                    UserWarning,
                    stacklevel=3,
                )
            group_folds = np.arange(len(unique)) % size
            if rng is not None:
                rng.shuffle(group_folds)
            assignments.append(group_folds[codes])
            codes_by_axis.append(codes)
            sizes.append(size)
            groups_per_axis.append(len(unique))
        return assignments, codes_by_axis, sizes, groups_per_axis, requested, fractions

    def split(self, y, groups, participants, items):
        """Yield disjoint training and test rectangles, omitting empty tests.

        Args:
            y: Aligned responses; values never determine fold membership.
            groups: Named identity arrays, a matrix with one axis per column,
                or material labels to cross with ``participants``.
            participants: Reader labels when ``groups`` is one-dimensional.
            items: Unused; parent materials must be supplied in ``groups``.
        """
        self._metadata, self._summary = [], None
        names, labels = self._labels(groups, participants)
        if any(len(a) != len(y) for a in labels):
            raise ValueError("Crossed CV requires aligned labels")
        assignments, codes, sizes, totals, requested, fractions = self._assignments(
            names, labels
        )
        for selected in product(*(range(n) for n in sizes)):
            held = np.column_stack(
                [a == fold for a, fold in zip(assignments, selected, strict=True)]
            )
            test = held.all(axis=1)
            if not test.any():
                continue
            train = ~held.any(axis=1)
            if not train.any():
                raise ValueError("A crossed test intersection has no training rows")
            self._metadata.append(
                {
                    "axis_names": list(names),
                    "axis_folds": list(selected),
                    "folds_per_axis": sizes,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "n_unused": int((~(train | test)).sum()),
                    "train_fraction_rows": float(train.mean()),
                    "train_fraction_groups": {
                        name: len(np.unique(c[train])) / total
                        for name, c, total in zip(names, codes, totals, strict=True)
                    },
                    "seed": self.seed,
                    "held_out": "every declared axis",
                }
            )
            yield train, test
        self._summary = {
            "axis_names": list(names),
            "groups_per_axis": dict(zip(names, totals, strict=True)),
            "requested_train_fraction": (
                None if fractions is None else dict(zip(names, fractions, strict=True))
            ),
            "requested_folds": dict(zip(names, requested, strict=True)),
            "resolved_folds": dict(zip(names, sizes, strict=True)),
            "nominal_train_fraction": dict(
                zip(names, (1 - 1 / k for k in sizes), strict=True)
            ),
            "train_fraction_groups_range": {
                name: self._range(
                    [m["train_fraction_groups"][name] for m in self._metadata]
                )
                for name in names
            },
            "train_fraction_rows_range": self._range(
                [m["train_fraction_rows"] for m in self._metadata]
            ),
            "n_observations": len(y),
            "fits_per_model_per_cycle": len(self._metadata),
            "possible_test_intersections": math.prod(sizes),
            "seed": self.seed,
        }

    @staticmethod
    def _range(values):
        """Represent achieved per-fold ranges with JSON-compatible numbers."""
        return [min(values), max(values)]

    def preview(self, groups, participants=None):
        """Check all splits and report achieved fractions and fits before fitting.

        Args:
            groups: Group identities in any format accepted by :meth:`split`.
            participants: Reader labels for one-dimensional material labels.

        Returns:
            Requested and achieved group fractions, observation-fraction ranges,
            and fits per model per complete cycle. No responses are inspected.
        """
        _, labels = self._labels(groups, participants)
        for _ in self.split(np.empty(len(labels[0])), groups, participants, None):
            pass
        return self.summary()

    def summary(self):
        """Return the completed split summary, also recorded by ``evaluate()``."""
        if self._summary is None:
            raise RuntimeError(
                "Complete split() or preview() before requesting a summary"
            )
        return deepcopy(self._summary)

    def fold_metadata(self):
        """Return actual nonempty intersections from the most recent split."""
        return deepcopy(self._metadata)
