"""Owned fit snapshots and bounded, descriptive summaries of scored errors."""

import importlib.metadata
import platform
import sys
import threading
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np


def execution_environment() -> dict:
    """Record Python and versions of loaded numerical backends without importing them."""
    packages = {
        "replica": "replicaverse",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "statsmodels": "statsmodels",
        "rpy2": "rpy2",
        "torch": "torch",
        "transformers": "transformers",
    }
    versions: dict[str, str | None] = {}
    metadata_errors = {}
    for module, distribution in packages.items():
        if module in sys.modules:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = None
            except (OSError, ValueError) as error:
                versions[distribution] = None
                metadata_errors[distribution] = type(error).__name__
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
        "metadata_errors": metadata_errors,
    }


_warning_lock = threading.RLock()
_warning_captures: dict[int, list[list[dict]]] = {}
_warning_display = warnings.showwarning


def _display_and_record_warning(
    message, category, filename, lineno, file=None, line=None
):
    with _warning_lock:
        captures = _warning_captures.get(threading.get_ident())
        if captures:
            captures[-1].append(
                {"category": category.__name__, "message": str(message)}
            )
        display = _warning_display
    display(message, category, filename, lineno, file=file, line=line)


@contextmanager
def emitted_warnings():
    """Record this thread's emitted warnings without changing filters or registries."""
    # Warning display is a process-wide hook.
    global _warning_display  # noqa: PLW0603
    records: list[dict] = []
    owner = threading.get_ident()
    with _warning_lock:
        if not _warning_captures:
            _warning_display = warnings.showwarning
            warnings.showwarning = _display_and_record_warning
        _warning_captures.setdefault(owner, []).append(records)
    try:
        yield records
    finally:
        with _warning_lock:
            _warning_captures[owner].pop()
            if not _warning_captures[owner]:
                del _warning_captures[owner]
            if (
                not _warning_captures
                and warnings.showwarning is _display_and_record_warning
            ):
                warnings.showwarning = _warning_display


def group_summary(factors: Mapping[str, np.ndarray], train, test) -> dict:
    """Summarize declared group sizes and scored groups unseen during training.

    Args:
        factors: Aligned grouping labels by factor name.
        train: Boolean training-row mask.
        test: Boolean scored-row mask.
    """
    result = {}
    for name, values in factors.items():
        training, counts = np.unique(values[train], return_counts=True)
        scored, test_counts = np.unique(values[test], return_counts=True)
        result[name] = {
            "n_train_groups": len(training),
            "n_scored_groups": len(scored),
            "n_unseen_scored_groups": int((~np.isin(scored, training)).sum()),
            "n_rows_in_unseen_groups": int(
                test_counts[~np.isin(scored, training)].sum()
            ),
            "train_group_size": {"min": int(counts.min()), "max": int(counts.max())},
            "scored_group_size": {
                "min": int(test_counts.min()),
                "max": int(test_counts.max()),
            },
        }
    return result


def name_fit_columns(diagnostic: dict | None, names: list[str]) -> dict | None:
    """Attach column names to explicitly ordered coefficient records.

    Args:
        diagnostic: Owned fit snapshot to annotate, or None.
        names: Ordered design-column names, excluding the intercept.
    """
    if diagnostic is not None:
        coefficients = diagnostic.get("coefficients")
        if (
            isinstance(coefficients, dict)
            and coefficients.get("naming") == "design_columns"
        ):
            expected = ["(Intercept)", *names]
            if len(coefficients.get("estimates", ())) == len(expected):
                coefficients["names"] = expected
    return diagnostic


def retained_predictions(y, baseline, full) -> dict[str, np.ndarray]:
    """Own copies of the exact scored arrays; never recalculate predictions.

    Args:
        y: Scored responses on the model scale.
        baseline: Aligned baseline predictions.
        full: Aligned full-model predictions.
    """
    return {
        name: np.array(value, copy=True)
        for name, value in (
            ("observed", y),
            ("prediction_baseline", baseline),
            ("prediction_full", full),
        )
    }


def snapshot_fit_diagnostics(model: Any) -> dict | None:
    """Copy a backend's optional current-fit diagnostics before its next fit.

    Args:
        model: Fitted backend, optionally exposing ``last_fit_diagnostics()``.
    """
    method = getattr(model, "last_fit_diagnostics", None)
    if method is None:
        return None
    diagnostics = method()
    if diagnostics is None:
        return None
    if not isinstance(diagnostics, Mapping):
        raise TypeError("last_fit_diagnostics() must return a mapping or None")
    return deepcopy(dict(diagnostics))


def _moments(values: np.ndarray) -> tuple[float, float, float]:
    """Population moments, scaled to avoid squaring large errors directly."""
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0, 0.0, 0.0
    scaled = values / scale
    mean = float(np.mean(scaled))
    sd = float(np.sqrt(np.mean((scaled - mean) ** 2)))
    rms = float(np.sqrt(np.mean(scaled**2)))
    return mean * scale, sd * scale, rms * scale


def _ordered_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain evenly spaced order statistics, including both extremes."""
    ordered = np.sort(values)
    ranks = np.linspace(0, len(ordered) - 1, min(101, len(ordered))).astype(int)
    return (ranks + 0.5) / len(ordered), ordered[ranks]


def _summary(
    y: np.ndarray,
    prediction: np.ndarray,
    sigma: Any,
    normal_reference: bool,
) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        errors = y - prediction
    finite = np.isfinite(errors) & np.isfinite(prediction)
    errors, fitted = errors[finite], prediction[finite]
    summary: dict = {
        "n_finite_errors": int(finite.sum()),
        "n_nonfinite_errors": int((~finite).sum()),
        "error_mean": None,
        "error_sd": None,
        "error_rms": None,
        "fitted_bins": [],
        "error_quantiles": [],
        "quantile_method": "selected_order_statistics_midpoint_probability",
        "normal_qq": {
            "available": False,
            "reason": "no_declared_gaussian_reference"
            if not normal_reference
            else "no_finite_errors",
        },
    }
    if not len(errors):
        return summary
    mean, sd, rms = _moments(errors)
    summary.update(error_mean=mean, error_sd=sd, error_rms=rms)
    # Quantile boundaries partition every finite row. Ties stay together, so
    # repeated fitted values cannot create arbitrary apparent error patterns.
    ordered = np.sort(fitted)
    positions = np.arange(1, min(20, len(fitted))) * len(fitted) // min(20, len(fitted))
    boundaries = np.unique(ordered[positions])
    labels = np.searchsorted(boundaries, fitted, side="right")
    for label in np.unique(labels):
        mask = labels == label
        bin_mean, bin_sd, _ = _moments(errors[mask])
        summary["fitted_bins"].append(
            {
                "count": int(mask.sum()),
                "fitted_mean": _moments(fitted[mask])[0],
                "error_mean": bin_mean,
                "error_sd": bin_sd,
            }
        )
    probabilities, quantiles = _ordered_points(errors)
    summary["error_quantiles"] = [
        {"probability": float(p), "error": float(error)}
        for p, error in zip(probabilities, quantiles, strict=True)
    ]
    if not normal_reference:
        return summary
    qq = summary["normal_qq"]
    qq.update(
        reference="standard_normal",
        scale_source="scored_error_rms" if sigma is None else "scoring_sigma",
    )
    if sigma is None:
        qq["estimated_scale"] = rms
        scale = np.full(len(errors), rms)
    else:
        scale = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)[finite]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized = errors / scale
    valid = np.isfinite(standardized) & np.isfinite(scale) & (scale > 0)
    qq.update(n_finite=int(valid.sum()), n_nonfinite=int(len(y) - valid.sum()))
    if not valid.any():
        qq["reason"] = "no_finite_positive_standardization_scale"
        return summary
    from scipy.special import ndtri

    probabilities, quantiles = _ordered_points(standardized[valid])
    qq.update(
        available=True,
        reason=None,
        points=[
            {"theoretical": float(ndtri(p)), "standardized_error": float(error)}
            for p, error in zip(probabilities, quantiles, strict=True)
        ],
    )
    return summary


def prediction_diagnostics(
    y: Any,
    y_pred_base: Any,
    y_pred_full: Any,
    *,
    sigma_base: Any = None,
    sigma_full: Any = None,
    normal_reference: bool,
    n_train_overlap: int,
) -> dict:
    """Summarize model-scale scored errors without retaining observation arrays.

    All finite rows contribute to at most 20 fitted-value bins. Up to 101
    selected order statistics include the extremes; this is not random row
    sampling. SDs describe population spread, not uncertainty. Gaussian Q-Q
    points divide errors by the supplied scoring sigma, or by finite-error RMS
    when it is unavailable. The fallback excludes nonfinite errors, including
    subtraction overflow; it need not equal the scorer's fallback scale at
    floating-point extremes. Neither plot establishes model validity.

    Sorting costs O(n log n) time and O(n) temporary memory; retained output is
    bounded independently of row count. Inputs and random state are untouched.

    Args:
        y: Scored responses after transformation and fold preprocessing.
        y_pred_base: Aligned baseline predictions on the same response scale.
        y_pred_full: Aligned full-model predictions on the same response scale.
        sigma_base: Baseline scoring sigma, scalar or per row; None uses finite-error RMS.
        sigma_full: Full-model scoring sigma, scalar or per row; None uses finite-error RMS.
        normal_reference: Whether the scoring path declares a Gaussian reference.
        n_train_overlap: Number of scored rows also included in this fold's training set.
    """
    target, baseline, full = (
        np.asarray(values, dtype=float) for values in (y, y_pred_base, y_pred_full)
    )
    if target.ndim != 1 or baseline.shape != target.shape or full.shape != target.shape:
        raise ValueError(
            "Prediction diagnostics require aligned one-dimensional arrays"
        )
    if not 0 <= n_train_overlap <= len(target):
        raise ValueError("n_train_overlap must count overlapping scored rows")
    return {
        "population": "scored_rows",
        "response_scale": "model_after_transform_and_fold_preprocessing",
        "n_rows": len(target),
        "n_train_overlap": int(n_train_overlap),
        "baseline": _summary(target, baseline, sigma_base, normal_reference),
        "full": _summary(target, full, sigma_full, normal_reference),
    }
