"""Sign flips over declared units of aligned held-out loss differences.

Grouping preserves within-unit dependence. It does not establish symmetry or
independence between units; overlapping CV training sets can still induce
cross-unit dependence. These are conditional tests of the stored predictions,
not a substitute for a design-appropriate refit/bootstrap analysis.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from replica.analysis.contracts.labels import categorical_labels

from .significance import (
    paired_permutation_pvalue,
    per_datapoint_permutation_pvalue,
    scale_signflip_values,
    sum_by_group,
)


def aligned_losses(analysis):
    """Read retained losses with their row identities and factor labels.

    Args:
        analysis: Evaluation result with per-fold losses and a named Design.

    Returns:
        Loss vector, row identity matrix and named factor vectors in fold order.
    """
    folds = analysis.fold_results
    if not folds or any(f.per_obs_delta_ll is None or f.row_ids is None for f in folds):
        raise ValueError(
            "Retain per-fold per-observation losses and row IDs for grouped inference"
        )
    names = set(folds[0].factor_ids)
    if any(set(f.factor_ids) != names for f in folds):
        raise ValueError("Fold factor declarations differ")
    for fold in folds:
        if any(
            np.ma.is_masked(values)
            for values in (
                fold.per_obs_delta_ll,
                fold.row_ids,
                *fold.factor_ids.values(),
            )
        ):
            raise ValueError(
                "Grouped inference requires complete losses, row IDs and factor labels"
            )
        n = len(fold.per_obs_delta_ll)
        if (
            np.shape(fold.per_obs_delta_ll) != (n,)
            or fold.row_ids.shape != (n, 3)
            or any(v.shape != (n,) for v in fold.factor_ids.values())
        ):
            raise ValueError("Retained loss, row and factor IDs do not align")
    values = np.concatenate([f.per_obs_delta_ll for f in folds])
    rows = np.concatenate([np.asarray(f.row_ids, dtype=object) for f in folds])
    # Evaluation stores source IDs and unit positions as text. Empty participant
    # IDs represent aggregate observations; stimulus and unit IDs cannot be empty.
    if any(not isinstance(v, str) for v in rows.flat) or np.any(rows[:, 1:] == ""):
        raise ValueError("Retained row identities must contain nonmissing text IDs")
    rows = rows.astype(str)
    if len(np.unique(rows, axis=0)) != len(rows):
        raise ValueError(
            "Repeated scored rows require an explicit aggregation before grouped inference"
        )
    factors = {
        n: categorical_labels(
            np.concatenate([np.asarray(f.factor_ids[n], dtype=object) for f in folds]),
            name=n,
        )
        for n in names
    }
    return values, rows, factors


def inference_scope(analysis, unit, *, allow_scope_mismatch=False):
    """Record CV and sign-flip scopes without inferring validity from either.

    Args:
        analysis: Evaluation result carrying its original Design metadata.
        unit: Selected sign-flip factor.
        allow_scope_mismatch: Acknowledge CV factors not resampled by this test.
            Callers must emit the returned warning even after acknowledgement.
    """
    if not isinstance(allow_scope_mismatch, bool):
        raise ValueError("allow_scope_mismatch must be a boolean")
    design = analysis.metadata.get("design", {})
    declared = design.get("cv_unit", ())
    axes = (declared,) if isinstance(declared, str) else tuple(declared)
    omitted = tuple(axis for axis in axes if axis != unit)
    message = None
    if omitted:
        message = (
            f"Sign flips over {unit!r} do not resample declared CV factors "
            f"{omitted!r}. This test does not by itself support the full CV "
            "generalization claim; a conditional claim still needs justified "
            "sign symmetry and dependence assumptions."
        )
        if not allow_scope_mismatch:
            raise ValueError(message + " Set allow_scope_mismatch=True to acknowledge.")
    return {
        "cv_units": axes,
        "cv_scope_recorded": "cv_unit" in design,
        "permutation_unit": unit,
        "condition_on": tuple(design.get("condition_on", ())),
        "unresampled_cv_units": omitted,
        "scope_mismatch_acknowledged": bool(omitted and allow_scope_mismatch),
        "warning": message,
        "interpretation": (
            "Conditional sign flips of saved paired predictions. Matching factor "
            "names does not establish joint sign symmetry or account for "
            "dependence from overlapping CV training sets."
        ),
    }


@dataclass(frozen=True)
class GroupedPermutationResult:
    """Recorded inference unit, effect and conditional sign-flip p-value."""

    unit: str
    n_groups: int
    delta_ll: float
    pvalue: float
    method: str
    alternative: str
    override_warning: str | None
    n_resamples: int | None
    inference_scope: dict = field(default_factory=dict)


def grouped_permutation_test(
    analysis,
    *,
    unit=None,
    allow_scope_mismatch=False,
    alternative="two-sided",
    method="monte-carlo",
    n_resamples=100_000,
    rng=None,
):
    """Test grouped loss sums using the analysis's declared permutation unit.

    Explicit overrides warn and are recorded in the returned result. Exact
    enumeration is computationally exact; inferential validity still requires
    joint sign symmetry at the selected unit.

    Args:
        analysis: Result retaining named factors, row IDs and loss differences.
        unit: Factor override; None selects Design.permutation_unit.
        allow_scope_mismatch: Explicitly acknowledge held-out factors that the
            selected sign-flip unit does not resample. Still warns and records
            both scopes; acknowledgement is not a validation of the test.
        alternative: two-sided, greater or less.
        method: exact or monte-carlo sign enumeration.
        n_resamples: Number of Monte Carlo draws.
        rng: Generator; omitted uses a reproducible seed of zero.
    """
    if method not in ("exact", "monte-carlo"):
        raise ValueError(f"unknown method: {method!r}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    declared = analysis.metadata.get("design", {}).get("permutation_unit")
    selected = declared if unit is None else unit
    if selected is None:
        raise ValueError("Declare a permutation unit or supply an explicit unit")
    scope = inference_scope(
        analysis, selected, allow_scope_mismatch=allow_scope_mismatch
    )
    if scope["warning"]:
        warnings.warn(scope["warning"], UserWarning, stacklevel=2)
    message = None
    if unit is not None and unit != declared:
        message = f"Permutation unit {unit!r} overrides declared unit {declared!r}"
        warnings.warn(message, UserWarning, stacklevel=2)
    values, _, factors = aligned_losses(analysis)
    if selected not in factors:
        raise ValueError(f"No retained labels for permutation unit {selected!r}")
    labels, sums = sum_by_group(values, factors[selected])
    try:
        total = math.fsum(sums)
    except OverflowError as exc:
        raise ValueError(
            "Grouped inference requires a finite total loss difference"
        ) from exc
    if method == "monte-carlo":
        p = per_datapoint_permutation_pvalue(
            sums,
            alternative=alternative,
            n_resamples=n_resamples,
            rng=rng,
            groups=labels,
        )
    else:
        p = paired_permutation_pvalue(
            scale_signflip_values(sums), alternative=alternative, method=method
        )
    return GroupedPermutationResult(
        selected,
        len(labels),
        total,
        p,
        method,
        alternative,
        message,
        n_resamples if method == "monte-carlo" else None,
        scope,
    )
