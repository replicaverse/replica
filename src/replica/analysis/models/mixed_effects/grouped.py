"""lme4 implementation of named-factor fitting and conditional prediction."""

import numpy as np

from replica.analysis.contracts.grouped import GroupedFit, GroupedFrame
from replica.analysis.models.support.diagnostics import (
    _coefficient_snapshot,
    _lme4_diagnostics,
    _residual_scale,
)


def validate_lmer_design(model) -> None:
    """Refuse ambiguous legacy options and unsupported likelihood semantics.

    Args:
        model: Lmer implementation whose optimizer and REML settings are retained.
    """
    if model.owns_loglik:
        raise ValueError(
            "Named Design supports plug-in Gaussian scoring, not integrated_loglik"
        )
    if (
        model.re_design is not None
        or model.random_slopes_subject
        or model.random_slopes_item
        or model.random_slope_cols_subject is not None
        or model.random_slope_cols_item is not None
        or not model.include_subject_re
        or not model.include_item_re
        or model.uncorrelated_slopes_subject
        or model.uncorrelated_slopes_item
    ):
        raise ValueError(
            "Declare random effects in Design only; legacy structural options conflict"
        )


def _check_levels(train, test):
    """Validate fixed categories and the selected unseen random-level policy."""
    for factor in train.design.categorical:
        unseen = set(test.factors[factor]) - set(train.factors[factor])
        if unseen:
            raise ValueError(
                f"Unseen fixed categorical levels for {factor!r}: {sorted(unseen)}"
            )
        if len(np.unique(train.factors[factor])) < 2:
            raise ValueError(
                f"Categorical control {factor!r} has fewer than two training levels"
            )
    if train.design.unseen == "error":
        for factor in train.design.condition_on:
            if set(test.factors[factor]) - set(train.factors[factor]):
                raise ValueError(f"Unseen conditioned random levels for {factor!r}")
    if train.design.scoring == "marginal":
        for effect in train.design.random_effects:
            if set(test.factors[effect.factor]) & set(train.factors[effect.factor]):
                raise ValueError(
                    "Marginal scoring requires entirely unseen random-effect levels; "
                    f"{effect.factor!r} overlaps training. Use scoring='residual' "
                    "for the existing conditional scoring convention."
                )


def _random_covariances(ro, numeric, groups):
    """Export lme4 covariance blocks using the original declared column names."""
    reverse_numeric = {v: k for k, v in numeric.items()}
    reverse_groups = {v: k for k, v in groups.items()}
    count = int(ro.r("length(lme4::VarCorr(.lmer_model))")[0])
    blocks = []
    for i in range(1, count + 1):
        expression = f"lme4::VarCorr(.lmer_model)[[{i}]]"
        columns = [str(x) for x in ro.r(f"colnames({expression})")]
        factor = str(ro.r(f"names(lme4::getME(.lmer_model, 'cnms'))[{i}]")[0])
        covariance = np.asarray(ro.r(f"as.matrix({expression})"), float)
        blocks.append(
            {
                "factor": reverse_groups[factor],
                "columns": [reverse_numeric.get(c, c) for c in columns],
                "covariance": covariance.tolist(),
            }
        )
    return blocks


def _marginal_variance(sigma, blocks, test):
    """Compute sigma² + z G z for each observation, including correlations."""
    n = len(next(iter(test.factors.values())))
    variance = np.full(n, sigma**2)
    for block in blocks:
        z = np.column_stack(
            [
                np.ones(n) if c == "(Intercept)" else test.columns[c]
                for c in block["columns"]
            ]
        )
        covariance = np.asarray(block["covariance"], float)
        contribution = np.einsum("ij,jk,ik->i", z, covariance, z)
        tolerance = 1e-12 * np.maximum(1, (z**2).sum(axis=1) * np.abs(covariance).max())
        if np.any(contribution < -tolerance):
            raise ValueError("Negative fitted random-effect variance")
        variance += np.maximum(0, contribution)
    return variance


def fit_lmer_grouped(
    model, ro, y, train: GroupedFrame, test: GroupedFrame
) -> GroupedFit:
    """Fit explicit named terms through lme4, with selected conditional prediction.

    Args:
        model: RLmerModel carrying optimizer and estimation options.
        ro: R objects interface inside the caller's conversion context.
        y: Training responses.
        train: Training frame, including random-only slope columns.
        test: Held-out frame in matching column order.
    """
    design = train.design
    validate_lmer_design(model)
    if (
        test.design != design
        or test.fixed != train.fixed
        or test.columns.keys() != train.columns.keys()
    ):
        raise ValueError("Training and test model declarations must match")
    if np.asarray(y).shape != (len(next(iter(train.factors.values()))),):
        raise ValueError("Training responses must align with the named frame")
    if np.ma.is_masked(y) or not np.isfinite(y).all():
        raise ValueError(
            "Training responses must be finite; filter rows before fitting"
        )
    numeric = {name: f"x{i}" for i, name in enumerate(train.columns)}
    groups = {f.name: f"g{i}" for i, f in enumerate(design.factors)}
    terms: dict[str, list[str]] = {}
    for effect in design.random_effects:
        if len(np.unique(train.factors[effect.factor])) < 2:
            raise ValueError(
                f"Random factor {effect.factor!r} has fewer than two training levels"
            )
        slopes = [numeric[s] for s in effect.slopes]
        group = groups[effect.factor]
        if effect.correlated:
            rhs = " + ".join(["1" if effect.intercept else "0", *slopes])
            terms[effect.factor] = [f"({rhs} | {group})"]
        else:
            terms[effect.factor] = ([f"(1 | {group})"] if effect.intercept else []) + [
                f"(0 + {s} | {group})" for s in slopes
            ]
    _check_levels(train, test)
    fixed = [numeric[n] for n in train.fixed] + [groups[n] for n in design.categorical]
    formula = "y ~ " + " + ".join(
        ["1", *fixed, *(t for ts in terms.values() for t in ts)]
    )
    re_formula = (
        " ~ " + " + ".join(t for f in design.condition_on for t in terms[f])
        if design.condition_on
        else None
    )

    levels = {
        name: np.unique(train.factors[name])
        if name in design.categorical
        else np.unique(np.concatenate([train.factors[name], test.factors[name]]))
        for name in groups
    }

    def frame(data, response=None):
        cols = {numeric[n]: ro.FloatVector(v) for n, v in data.columns.items()}
        for name, column in groups.items():
            cols[column] = ro.vectors.FactorVector(
                ro.StrVector(data.factors[name]), levels=ro.StrVector(levels[name])
            )
        if response is not None:
            cols["y"] = ro.FloatVector(response)
        return ro.DataFrame(cols)

    ro.globalenv[".train_data"] = frame(train, y)
    ro.globalenv[".test_data"] = frame(test)
    for name in design.categorical:
        column = groups[name]
        for data in (".train_data", ".test_data"):
            ro.r(
                f"contrasts({data}${column}) <- contr.treatment(nlevels({data}${column}))"
            )
    if terms:
        model._fit_lmer(ro, formula)
        fitted = ".lmer_model"
        ro.globalenv[".named_re"] = (
            ro.Formula(re_formula) if re_formula else ro.NA_Logical
        )
        predictions = np.asarray(
            ro.r(
                "predict(.lmer_model, newdata=.test_data, re.form=.named_re, allow.new.levels=TRUE)"
            ),
            float,
        )
        messages = list(ro.r("as.character(.lmer_model@optinfo$conv$lme4$messages)"))
        singular = bool(ro.r("lme4::isSingular(.lmer_model)")[0])
        optimizer_used = str(ro.r(".lmer_model@optinfo$optimizer")[0])
        optimizer_method = list(
            ro.r("as.character(.lmer_model@optinfo$control$method)")
        )
        optimizer_convergence = np.asarray(
            ro.r(".lmer_model@optinfo$conv$opt"), dtype=int
        ).tolist()
    else:
        ro.globalenv[".formula_str"] = formula
        ro.r(".lm_model <- lm(as.formula(.formula_str), data=.train_data)")
        fitted = ".lm_model"
        predictions = np.asarray(ro.r("predict(.lm_model, newdata=.test_data)"), float)
        messages, singular = [], False
        optimizer_used, optimizer_method, optimizer_convergence = None, [], []
    table = f"summary({fitted})$coefficients"
    names = list(ro.r(f"rownames({table})"))
    estimates = np.asarray(ro.r(table), float).reshape(len(names), -1, order="F")
    if not terms and not model.reml:
        estimates[:, 1] *= float(
            ro.r("sqrt(df.residual(.lm_model) / nobs(.lm_model))")[0]
        )
    stats = {n: row[:2] for n, row in zip(names, estimates, strict=True)}
    values = np.array(
        [
            stats.get(n, [np.nan, np.nan])
            for n in ["(Intercept)", *(numeric[n] for n in train.fixed)]
        ]
    )
    # lm's sigma uses residual degrees of freedom even when ML was requested.
    sigma_expression = (
        "sqrt(deviance(.lm_model) / nobs(.lm_model))"
        if not terms and not model.reml
        else f"sigma({fitted})"
    )
    sigma = float(ro.r(sigma_expression)[0])
    covariance = _random_covariances(ro, numeric, groups) if terms else []
    predictive_variance = (
        _marginal_variance(sigma, covariance, test)
        if design.scoring == "marginal"
        else None
    )
    version = (
        str(ro.r('as.character(utils::packageVersion("lme4"))')[0]) if terms else None
    )
    model._last_fixed = (values[:, 0].copy(), values[:, 1].copy())
    return GroupedFit(
        predictions,
        sigma,
        values[:, 0],
        values[:, 1],
        {
            "design": design.metadata(),
            "formula": formula,
            "numeric_columns": numeric,
            "factor_columns": groups,
            "conditioning_formula": re_formula,
            "scoring": "marginal_gaussian"
            if design.scoring == "marginal"
            else "residual_gaussian",
            "random_covariances": covariance,
            "predictive_variance": "residual + unseen random effects"
            if design.scoring == "marginal"
            else "residual only",
            "reml": model.reml,
            "optimizer_requested": model.optimizer,
            "optimizer_used": optimizer_used,
            "optimizer_method": optimizer_method,
            "optimizer_convergence": optimizer_convergence,
            "maxfun": model.maxfun,
            "nlminb_fallback": model.nlminb_fallback,
            "convergence_messages": messages,
            "singular": singular,
            "fit_diagnostics": {
                **(
                    _lme4_diagnostics(
                        optimizer_convergence,
                        messages,
                        singular=singular,
                        optimizer=optimizer_used,
                        reml=model.reml,
                    )
                    if terms
                    else {"convergence_applicable": False}
                ),
                "coefficients": _coefficient_snapshot(
                    [v[0] for v in stats.values()],
                    [v[1] for v in stats.values()],
                    names=[
                        {v: k for k, v in numeric.items()}.get(str(n), str(n))
                        for n in stats
                    ],
                    naming="backend_terms",
                ),
                "residual_scale": _residual_scale(
                    sigma=sigma, convention="REML" if model.reml else "ML"
                ),
                "random_covariances": [
                    {
                        "factor": block["factor"],
                        "names": list(block["columns"]),
                        "covariance": block["covariance"],
                    }
                    for block in covariance
                ],
            },
            "lme4_version": version,
            "fixed_coefficients": {str(n): v.tolist() for n, v in stats.items()},
            "categorical_contrasts": "treatment",
            "categorical_levels": {
                f: np.unique(train.factors[f]).tolist() for f in design.categorical
            },
        },
        predictive_variance=predictive_variance,
    )
