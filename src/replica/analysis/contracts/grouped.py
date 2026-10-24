"""Named model frames keep categorical factors separate from numeric predictors."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from replica.analysis.contracts.design import Design
from replica.analysis.contracts.labels import categorical_labels


@dataclass(frozen=True)
class GroupedFrame:
    """One model's resolved terms and aligned numeric and categorical columns.

    Resolve ``Design.comparison`` against ``fixed`` and retain only numeric
    columns used by a fixed effect or a retained random slope.
    """

    columns: dict[str, np.ndarray]
    factors: dict[str, np.ndarray]
    fixed: tuple[str, ...]
    design: Design

    def __post_init__(self):
        object.__setattr__(self, "design", self.design.for_model(self.fixed))
        arrays = [*self.columns.values(), *self.factors.values()]
        if not arrays or any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays):
            raise ValueError(
                "Named model columns must be aligned one-dimensional arrays"
            )
        object.__setattr__(
            self,
            "factors",
            {n: categorical_labels(v, name=n) for n, v in self.factors.items()},
        )
        if len(set(self.fixed)) != len(self.fixed):
            raise ValueError("Fixed predictor names must be unique")
        if set(self.columns) & set(self.design.categorical):
            raise ValueError("Numeric and categorical fixed names must be distinct")
        required = set(self.fixed) | {
            s for r in self.design.random_effects or () for s in r.slopes
        }
        if required - self.columns.keys():
            raise ValueError(
                f"Missing numeric predictors: {sorted(required - self.columns.keys())}"
            )
        object.__setattr__(
            self, "columns", {n: v for n, v in self.columns.items() if n in required}
        )
        if {f.name for f in self.design.factors} - self.factors.keys():
            raise ValueError("Missing declared factor columns")
        if any(
            np.ma.is_masked(a) or not np.isfinite(a).all()
            for a in self.columns.values()
        ):
            raise ValueError("Numeric predictors must be finite")


@dataclass(frozen=True)
class GroupedFit:
    """Plug-in Gaussian predictions and fixed coefficients of a named fit.

    beta and se follow ``[intercept, *train.fixed]``. beta is finite on an
    accepted fit (a dropped numeric term is rejected, not reported as NaN);
    se may be NaN where a standard error is unavailable. Categorical
    coefficients belong in metadata.
    sigma is the residual scale. predictive_variance supplies per-observation
    marginal variance for entirely unseen random-effect levels when requested.
    metadata may include a compact ``fit_diagnostics`` mapping for this fit.
    """

    predictions: np.ndarray
    sigma: float
    beta: np.ndarray
    se: np.ndarray
    metadata: dict = field(default_factory=dict)
    predictive_variance: np.ndarray | None = None

    def __post_init__(self):
        if any(
            np.ma.is_masked(value)
            for value in (
                self.predictions,
                self.sigma,
                self.beta,
                self.se,
                self.predictive_variance,
            )
        ):
            raise ValueError("Grouped fits must not contain masked values")
        if np.ndim(self.sigma) != 0 or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Grouped residual sigma must be finite and positive")
        for name in ("predictions", "beta", "se"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or np.isinf(values).any():
                raise ValueError(
                    f"Grouped {name} must be a one-dimensional vector without infinities"
                )
            object.__setattr__(self, name, values)
        if not np.isfinite(self.predictions).all():
            raise ValueError("Grouped predictions must be finite")
        if self.beta.shape != self.se.shape or np.any(self.se < 0):
            raise ValueError(
                "Grouped coefficients and nonnegative standard errors must align"
            )
        if self.predictive_variance is not None:
            variance = np.asarray(self.predictive_variance, dtype=float)
            if (
                variance.shape != self.predictions.shape
                or not np.isfinite(variance).all()
                or np.any(variance <= 0)
            ):
                raise ValueError(
                    "Grouped predictive variance must be positive and aligned"
                )
            object.__setattr__(self, "predictive_variance", variance)

    def scoring_sigma(self, scoring: str) -> float | np.ndarray:
        """Return the declared forecast scale, without estimating it on test data.

        Args:
            scoring: Use residual noise alone or the marginal predictive variance.
        """
        if scoring == "residual":
            return self.sigma
        if scoring != "marginal":
            raise ValueError("Unknown Gaussian scoring convention")
        if self.predictive_variance is None:
            raise ValueError("Marginal scoring requires backend predictive variance")
        return np.sqrt(self.predictive_variance)


def validate_grouped_fit(fit: GroupedFit, n_test: int, n_fixed: int) -> dict:
    """Check fit acceptance and return compact diagnostics for either public path.

    Boundary singularity is recorded but is not itself non-convergence. Missing
    backend diagnostics remain unknown; their absence does not prove convergence.

    Args:
        fit: Fitted baseline or full model.
        n_test: Number of held-out predictions required.
        n_fixed: Number of declared numeric fixed predictors, excluding intercept.
    """
    if fit.beta.shape != (n_fixed + 1,):
        raise ValueError("Grouped coefficients must follow [intercept, *fixed]")
    if fit.predictions.shape != (n_test,) or not np.isfinite(fit.beta).all():
        raise ValueError("Invalid predictions or dropped fixed coefficients")
    codes = fit.metadata.get("optimizer_convergence", [])
    if any(v != 0 for v in codes):
        raise ValueError("Optimizer did not converge")
    messages = fit.metadata.get("convergence_messages", [])
    if any(
        "boundary" not in m.lower() or "singular" not in m.lower() for m in messages
    ):
        raise ValueError(
            "Unresolved model convergence messages: " + "; ".join(messages)
        )
    reported = fit.metadata.get("fit_diagnostics")
    if reported is not None and not isinstance(reported, Mapping):
        raise TypeError("Grouped fit_diagnostics must be a mapping or None")
    diagnostics = {
        "dropped_numeric_fixed_terms": False,
        "optimizer_convergence": list(codes)
        if "optimizer_convergence" in fit.metadata
        else None,
        "convergence_messages": list(messages)
        if "convergence_messages" in fit.metadata
        else None,
        **{
            key: fit.metadata.get(key)
            for key in ("singular", "optimizer_used", "reml", "lme4_version", "formula")
        },
    }
    if len(codes):
        diagnostics["converged"] = all(code == 0 for code in codes)
    if len(messages):
        diagnostics["warnings"] = list(messages)
    diagnostics.update(deepcopy(dict(reported or {})))
    return diagnostics


@runtime_checkable
class GroupedModel(Protocol):
    """Optional backend contract for explicit named designs and Gaussian scores."""

    def validate_design(self, design: Design) -> None:
        """Reject unsupported model terms or prediction/scoring policies.

        Args:
            design: Declaration checked before feature extraction.
        """
        ...

    def fit_grouped(
        self, y_train: np.ndarray, train: GroupedFrame, test: GroupedFrame
    ) -> GroupedFit:
        """Fit a named frame and predict the held-out rows.

        Args:
            y_train: Training response in frame row order.
            train: Training numeric and categorical columns.
            test: Held-out columns under the same declaration.
        """
        ...
