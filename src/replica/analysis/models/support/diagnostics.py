"""Compact snapshots of diagnostics from a backend's current fit."""

from copy import deepcopy
from math import isfinite, sqrt


class _FitDiagnostics:
    """Store only the current fit; callers receive independent metadata."""

    _fit_diagnostics: dict | None = None

    def last_fit_diagnostics(self) -> dict | None:
        """Return recorded diagnostics, or None before a successful fit."""
        return deepcopy(self._fit_diagnostics)


def _lme4_diagnostics(codes, messages, *, singular, **details) -> dict:
    """Describe recorded lme4 checks without changing fit acceptance policy."""
    messages = [str(message) for message in messages]
    codes = [int(code) for code in codes]
    unresolved = [
        message
        for message in messages
        if "boundary" not in message.lower() or "singular" not in message.lower()
    ]
    return {
        "convergence_applicable": True,
        "converged": not any(codes) and not unresolved if codes else None,
        "singular": bool(singular),
        "warnings": messages,
        "details": {"optimizer_convergence": codes, **details},
    }


def _r_fit_diagnostics(ro, model_var: str, n_columns: int) -> dict:
    """Read optional R telemetry without changing a completed fit's acceptance."""
    try:
        diagnostics = _read_r_fit_diagnostics(ro, model_var, n_columns)
    except Exception as error:
        # Optional diagnostic slots differ across R/backend versions. A failure
        # to inspect them must not turn a successful scientific fit into failure.
        return {"details": {"diagnostics_unavailable": str(error)}}
    try:
        diagnostics.update(_r_parameter_diagnostics(ro, model_var))
    except Exception as error:
        diagnostics.setdefault("details", {})["parameters_unavailable"] = str(error)
    return diagnostics


def _r_parameter_diagnostics(ro, model_var: str) -> dict:
    """Read fitted parameters without summary/vcov, which may fail on singular fits."""
    lm = model_var == ".lm_model"
    expression = (
        f"coef({model_var})" if lm else f"lme4::fixef({model_var}, add.dropped=TRUE)"
    )
    coefficients = _coefficient_snapshot(
        ro.r(expression), names=[str(x) for x in ro.r(f"names({expression})")]
    )
    result: dict = {
        "coefficients": coefficients,
        "residual_scale": _residual_scale(
            sigma=float(ro.r(f"sigma({model_var})")[0]),
            convention="residual_degrees_of_freedom"
            if lm
            else "REML"
            if bool(ro.r(f"lme4::isREML({model_var})")[0])
            else "ML",
        ),
    }
    if not lm:
        blocks = []
        factors = [str(x) for x in ro.r(f"names(lme4::VarCorr({model_var}))")]
        for i, factor in enumerate(factors, 1):
            expression = f"lme4::VarCorr({model_var})[[{i}]]"
            names = [str(x) for x in ro.r(f"colnames({expression})")]
            # Matrix rows are extracted explicitly to avoid conversion-order assumptions.
            covariance = [
                [float(x) for x in ro.r(f"{expression}[{j},,drop=TRUE]")]
                for j in range(1, len(names) + 1)
            ]
            blocks.append({"factor": factor, "names": names, "covariance": covariance})
        result["random_covariances"] = blocks
    return result


def _read_r_fit_diagnostics(ro, model_var: str, n_columns: int) -> dict:
    """Read diagnostics from an already fitted lm/lmer object; never refit."""
    if model_var == ".lm_model":
        return {
            "convergence_applicable": False,
            "rank": int(ro.r(f"{model_var}$rank")[0]),
            "n_columns": n_columns,
        }
    return {
        **_lme4_diagnostics(
            ro.r(f"as.integer({model_var}@optinfo$conv$opt)"),
            ro.r(f"as.character({model_var}@optinfo$conv$lme4$messages)"),
            singular=ro.r(f"lme4::isSingular({model_var})")[0],
            optimizer=str(ro.r(f"{model_var}@optinfo$optimizer")[0]),
        ),
        "rank": int(ro.r(f"ncol(lme4::getME({model_var}, 'X'))")[0]),
        "n_columns": n_columns,
    }


def _coefficient_snapshot(
    estimates, standard_errors=None, *, names=None, naming="design_columns"
) -> dict:
    """Copy existing fitted coefficients; missing/aliased entries remain unknown."""

    def values(items):
        return [float(value) if isfinite(float(value)) else None for value in items]

    estimates = values(estimates)
    result: dict = {
        "names": list(names)
        if names is not None
        else ["(Intercept)", *[f"x{i}" for i in range(len(estimates) - 1)]],
        "estimates": estimates,
        "naming": naming,
    }
    if standard_errors is not None:
        result["standard_errors"] = values(standard_errors)
    return result


def _residual_scale(*, sigma=None, variance=None, convention: str) -> dict:
    """Describe an already available fitted residual scale and its convention."""
    if sigma is None:
        sigma = sqrt(float(variance)) if float(variance) >= 0 else float("nan")
    if variance is None:
        variance = float(sigma) * float(sigma)
    return {
        "sigma": float(sigma) if isfinite(float(sigma)) else None,
        "variance": float(variance) if isfinite(float(variance)) else None,
        "convention": convention,
    }
