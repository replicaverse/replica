"""Gaussian mixed-model likelihood and GLS reconstruction."""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu


def verify_fit(frame, specification, fits):
    """Recover GLS coefficients and marginal log likelihood without using lme4.

    Args:
        frame: Observation frame with response, predictors and group identities.
        specification: Declared response family and random-effect specification.
        fits: Directory containing fitted coefficients, scaling and diagnostics.
    """
    if specification["family"] != "gaussian":
        raise ValueError("Gaussian marginal-likelihood verification only")
    frame = frame.copy()
    scaling = json.loads((fits / "scaling.json").read_text())
    diagnostics = json.loads((fits / "diagnostics.json").read_text())
    coefficients = pd.read_csv(fits / "coefficients.tsv", sep="\t")
    for entry in scaling:
        name = entry["column"]
        np.testing.assert_allclose(
            [frame[name].mean(), frame[name].std()],
            [entry["center"], entry["scale"]],
            rtol=1e-12,
            atol=1e-12,
        )
        frame[name] = (frame[name] - entry["center"]) / entry["scale"]
    n = len(frame)
    y = frame.response.to_numpy(float)
    results = []
    for name, terms in specification["models"].items():
        info = diagnostics[name]
        if info["nobs"] != n:
            raise ValueError("Fitted sample size differs")
        constant = info.get("constant_fixed_terms", [])
        if isinstance(constant, str):
            constant = [constant]
        terms = [term for term in terms if term not in constant]
        x = np.column_stack([np.ones(n), frame[terms].to_numpy(float)])
        blocks, residual = [], None
        for component in info["variances"]:
            variance = component["vcov"]
            if component["grp"] == "Residual":
                residual = variance
                continue
            if component.get("var2") is not None:
                raise ValueError(
                    "This verifier requires the declared diagonal random structure"
                )
            if not np.isfinite(variance) or variance < 0:
                raise ValueError("Invalid random-effect variance")
            if variance == 0:
                continue
            matches = [
                group
                for group in specification["groups"]
                if component["grp"] == group or component["grp"].startswith(group + ".")
            ]
            if len(matches) != 1:
                raise ValueError("Ambiguous R variance grouping")
            codes, levels = pd.factorize(frame[matches[0]], sort=True)
            values = (
                np.ones(n)
                if component["var1"] == "(Intercept)"
                else frame[component["var1"]].to_numpy(float)
            )
            blocks.append(
                sparse.csc_matrix(
                    (values * np.sqrt(variance), (np.arange(n), codes)),
                    shape=(n, len(levels)),
                )
            )
        if residual is None or residual <= 0:
            raise ValueError("Missing positive residual variance")
        w = sparse.hstack(blocks, format="csc") if blocks else sparse.csc_matrix((n, 0))
        if w.shape[1]:
            covariance = sparse.eye(w.shape[1], format="csc") + w.T @ w / residual
            factor = splu(covariance.tocsc())
            logdet = n * np.log(residual) + np.log(np.abs(factor.U.diagonal())).sum()

            def inverse(value, w=w, factor=factor, residual=residual):
                return (
                    value - w @ factor.solve(np.asarray(w.T @ value) / residual)
                ) / residual
        else:
            logdet = n * np.log(residual)

            def inverse(value, residual=residual):
                return value / residual

        beta = np.linalg.solve(x.T @ inverse(x), x.T @ inverse(y))
        error = y - x @ beta
        loglik = float(-0.5 * (n * np.log(2 * np.pi) + logdet + error @ inverse(error)))
        recorded = (
            coefficients.loc[coefficients.model.eq(name)]
            .set_index("term")
            .loc[["(Intercept)", *terms], "Estimate"]
            .to_numpy(float)
        )
        np.testing.assert_allclose(beta, recorded, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(loglik, info["loglik"], rtol=1e-8, atol=1e-5)
        results.append(
            dict(
                model=name,
                observations=n,
                random_coefficients=w.shape[1],
                loglik=loglik,
                loglik_absolute_error=abs(loglik - info["loglik"]),
                coefficient_max_absolute_error=float(np.max(np.abs(beta - recorded))),
            )
        )
    return results
