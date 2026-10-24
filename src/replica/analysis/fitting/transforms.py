"""Per-fold transforms: fit on train, apply to both splits.

Column-uniform scalers use this directly. Orthogonalization uses context to reach columns outside X.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from replica.analysis.specification.preprocessing import _orth_beta

if TYPE_CHECKING:
    from collections.abc import Mapping


class FoldContext:
    """Per-fold information available to a :class:`FoldTransform`.

    Carries the train/test boolean masks and access to the full per-column
    data (``feat_flat``), so a transform can reach a column that is not part
    of ``X``, e.g. the "against" array for orthogonalization. Column-uniform
    transforms (scaling) ignore it.
    """

    def __init__(
        self,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        columns: Mapping[str, np.ndarray],
    ):
        """Bind one fold's split masks to the full column table.

        Args:
            train_mask: Boolean mask over every row, true on the training split.
            test_mask: Boolean mask over every row, true on the held-out split.
            columns: Unsplit per-column arrays of the flattened design, keyed by
                column name, covering predictors and controls alike.
        """
        self.train_mask = train_mask
        self.test_mask = test_mask
        self._columns = columns

    def column(self, name: str) -> np.ndarray:
        """Full (unsplit) array for a named column.

        Args:
            name: Column key in the flattened design, such as a control the
                transform orthogonalizes against.
        """
        return self._columns[name]


def apply_fold_transforms(
    x_raw: np.ndarray,
    transforms: list,
    ctx: FoldContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the design matrix by the fold masks and run the transform chain.

    Args:
        x_raw: Untransformed design matrix, rows in full-dataset order.
        transforms: :class:`FoldTransform` instances applied in order, each
            fitted on the training split, e.g. ``[ZScoreScaler()]``.
        ctx: Fold masks and column access handed to every transform.

    Returns:
        The ``(train, test)`` blocks after the whole chain.
    """
    x_tr = x_raw[ctx.train_mask]
    x_te = x_raw[ctx.test_mask]
    for t in transforms:
        x_tr, x_te = t.fit_transform(x_tr, x_te, ctx)
    return x_tr, x_te


@runtime_checkable
class FoldTransform(Protocol):
    """Fit on the training split, apply to both. ``ctx`` is the per-fold
    context (masks + columns); column-uniform transforms ignore it.
    """

    def fit_transform(
        self,
        train: np.ndarray,
        test: np.ndarray,
        ctx: FoldContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit on the training block and return both blocks transformed.

        Args:
            train: Training-split design matrix, the only rows the fit reads.
            test: Held-out design matrix, transformed with the training fit.
            ctx: Per-fold masks and columns; column-uniform transforms ignore it.

        Returns:
            The transformed ``(train, test)`` pair, column layout unchanged.
        """
        ...


class _ResidualizeColumns:
    """Per-fold orthogonalization against a reference array.

    Projects planned columns out of an "against" array with
    ``β = cov(against, target) / var(against)`` fit on the training split and
    applied to both, following Opedal et al. (2024). ``plan`` is a list of
    ``(column_index, against_name)`` pairs; the against array is read per fold
    from ``ctx`` and sliced by the fold masks.
    """

    def __init__(self, plan: list[tuple[int, str]]):
        """Record which design columns to residualize and against what.

        Args:
            plan: ``(column_index, against_name)`` pairs; the column at that
                index is projected onto the named context column. An empty
                plan makes the transform a pass-through.
        """
        self.plan = plan

    def fit_transform(
        self,
        train: np.ndarray,
        test: np.ndarray,
        ctx: FoldContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subtract the training-fit projection from both splits.

        Args:
            train: Training-split design matrix; the betas come from these rows.
            test: Held-out design matrix, corrected with the training betas.
            ctx: Fold masks and the against-column lookup. Required whenever the
                plan is non-empty.

        Returns:
            The transformed ``(train, test)`` pair.

        Raises:
            ValueError: When the plan is non-empty and *ctx* is ``None``.
        """
        if not self.plan:
            return train, test
        if ctx is None:
            raise ValueError("Residualize requires a FoldContext")
        x_tr = train.astype(float, copy=True)
        x_te = test.astype(float, copy=True)
        for col_idx, against_name in self.plan:
            a_full = ctx.column(against_name)
            a_tr = a_full[ctx.train_mask]
            a_te = a_full[ctx.test_mask]
            beta = _orth_beta(a_tr, x_tr[:, col_idx])
            x_tr[:, col_idx] = x_tr[:, col_idx] - beta * a_tr
            x_te[:, col_idx] = x_te[:, col_idx] - beta * a_te
        return x_tr, x_te
