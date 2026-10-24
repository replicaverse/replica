"""Descriptive analysis rows with caller-selected inference and sidecar output."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from replica.analysis.inference.significance import (
    concat_fold_per_obs_delta_ll,
)

from ..spec import DatasetCell
from . import observation_deltas

if TYPE_CHECKING:
    from replica.analysis.contracts.results import AnalysisResult

TSV_COLUMNS = (
    "dataset",
    "dv",
    "response",
    "paradigm",
    "lm",
    "predictor",
    "stage",
    # How skipped words entered an eye-tracking duration: ``skips_removed``
    # (a skipped word is missing for that reader) or ``skips_as_zero`` (it
    # reads as 0 ms); empty for a measure without skipping. A table without
    # this column is read as ``skips_removed`` for eye-tracking rows.
    "variant",
    "n_obs",
    "n_participants",
    "n_items",
    "delta_ll",
    "delta_ll_se",
    "delta_ll_per_unit",
    "r2_full",
    "r2_baseline",
    "partial_r2",
    "partial_r2_se",
    "coefficient",
    "coefficient_se",
    "n_folds",
    "pvalue",
    "pvalue_per_datapoint",
    # JSON-encoded per-fold values (lists of length n_folds) so downstream
    # tools can compute their own CIs without refitting.
    "fold_delta_ll",
    "fold_delta_r2",
    "fold_n_test",
    # Scale-free effect size and optional interval on the unweighted mean of
    # per-fold ΔLL/N. That mean equals ``delta_ll_per_unit`` (total ΔLL / total N)
    # only with equal-size folds; the caller selects the interval method.
    "cohens_f2",
    "fold_dll_per_unit_ci_lo",
    "fold_dll_per_unit_ci_hi",
)


def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _std(xs: list[float]) -> float:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def row_from_analysis(
    ds: DatasetCell,
    predictor_key: str,
    stage: str,
    analysis,
    dataset,
    *,
    lm: str,
    variant: str = "",
    inference: Callable[[AnalysisResult], Mapping[str, float]] | None = None,
    per_observation_path: Path | None = None,
) -> dict:
    """Flatten an :class:`AnalysisResult` into a TSV row.

    No inference is performed and no file is written by default. An inference
    callback may supply p-values and the fold-mean interval; other columns remain
    descriptive. A supplied sidecar path receives retained per-observation ΔLL.

    Args:
        ds: The (dataset, dv) coordinate, supplying the row's identity columns.
        predictor_key: Predictor being scored, as the pickle names it.
        stage: Caller-defined identifier for the fitted analysis configuration.
        analysis: The evaluation to flatten; its ``metrics`` and
            ``fold_results`` supply every effect and per-fold series.
        dataset: The loaded corpus, counted for ``n_items`` and
            ``n_participants``; both come out ``None`` when it exposes neither.
        lm: Nonempty model identity supplied by the caller.
        variant: The skip treatment of an eye-tracking duration, see
            :data:`TSV_COLUMNS`; empty for a measure without skipping.
        inference: Optional callback over the AnalysisResult. Its mapping may contain
            only ``pvalue``, ``pvalue_per_datapoint``,
            ``fold_dll_per_unit_ci_lo`` and ``fold_dll_per_unit_ci_hi``.
            Omitted inference fields are NaN. Validate the method's assumptions
            for the evaluation design before supplying this callback.
        per_observation_path: Explicit sidecar destination, or None. Empty or
            unavailable per-observation deltas produce no file.

    Returns:
        One row keyed by :data:`TSV_COLUMNS`.
    """
    if not isinstance(lm, str) or not lm.strip():
        raise ValueError("lm must be an explicit nonempty model identity")
    inference_fields = {
        "pvalue",
        "pvalue_per_datapoint",
        "fold_dll_per_unit_ci_lo",
        "fold_dll_per_unit_ci_hi",
    }
    summary = {} if inference is None else inference(analysis)
    if not isinstance(summary, Mapping):
        raise TypeError("inference must return a mapping of inference fields")
    unexpected = set(summary) - inference_fields
    if unexpected:
        raise ValueError(
            f"inference cannot set row fields: {sorted(unexpected, key=repr)!r}"
        )
    folds = list(analysis.fold_results or [])
    delta_ll_se = float(_std([f.delta_ll for f in folds])) if folds else float("nan")
    # Across-fold SD of the per-fold ΔR², mirroring delta_ll_se.
    partial_r2_se = float(_std([f.delta_r2 for f in folds])) if folds else float("nan")
    # Per-fold values as JSON so downstream tools can compute their own CIs
    # without refitting.
    fold_delta_ll = json.dumps(
        [
            None
            if f.delta_ll is None or not _isfinite(f.delta_ll)
            else float(f.delta_ll)
            for f in folds
        ]
    )
    fold_delta_r2 = json.dumps(
        [
            None
            if f.delta_r2 is None or not _isfinite(f.delta_r2)
            else float(f.delta_r2)
            for f in folds
        ]
    )
    fold_n_test = json.dumps(
        [int(f.n_test) if f.n_test is not None else None for f in folds]
    )
    # Datasets outside the tabular base (e.g. test doubles) may lack these.
    n_items: int | None = None
    try:
        n_items = len(dataset.stimuli())
    except (AttributeError, TypeError):
        n_items = None
    try:
        n_participants = len(dataset.participants)
    except (AttributeError, TypeError):
        n_participants = None

    metrics = analysis.metrics or {}
    r2_full = metrics.get("full_r2")
    r2_base = metrics.get("baseline_r2")
    partial_r2 = (
        r2_full - r2_base if (r2_full is not None and r2_base is not None) else None
    )
    from replica.analysis.inference.effect_size import cohens_f2

    cohens_f2_val = cohens_f2(partial_r2, r2_full)
    row = {
        "dataset": ds.name,
        "dv": ds.dv,
        "response": json.dumps(getattr(analysis, "response", {}), sort_keys=True),
        "paradigm": ds.paradigm,
        "lm": lm,
        "predictor": predictor_key,
        "stage": stage,
        "variant": variant,
        "n_obs": analysis.n_observations,
        "n_participants": n_participants,
        "n_items": n_items,
        "delta_ll": metrics.get("delta_ll"),
        "delta_ll_se": delta_ll_se,
        "delta_ll_per_unit": metrics.get("delta_ll_per_unit"),
        "r2_full": r2_full,
        "r2_baseline": r2_base,
        "partial_r2": partial_r2,
        "coefficient": analysis.coefficient,
        "coefficient_se": analysis.coefficient_se,
        "partial_r2_se": partial_r2_se,
        "n_folds": len(folds),
        "pvalue": float("nan"),
        "pvalue_per_datapoint": float("nan"),
        # Per-fold raw values, JSON-encoded lists of length n_folds.
        # NaN/missing entries become JSON null.
        "fold_delta_ll": fold_delta_ll,
        "fold_delta_r2": fold_delta_r2,
        "fold_n_test": fold_n_test,
        "cohens_f2": cohens_f2_val,
        "fold_dll_per_unit_ci_lo": float("nan"),
        "fold_dll_per_unit_ci_hi": float("nan"),
    }
    row.update(summary)
    if per_observation_path is not None:
        observation_deltas.save(
            per_observation_path, concat_fold_per_obs_delta_ll(folds)
        )
    return row
