"""Reconstruct held-out population predictions and losses."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit

from replica.utils.digest import file_sha256


@lru_cache(maxsize=100000)
def integrated_logistic(mean, variance):
    """Integrate the logistic-normal probability using adaptive quadrature.

    Args:
        mean: Fixed-effect mean on the latent logit scale.
        variance: Nonnegative latent variance for unseen grouping levels.
    """
    if variance < 0 or not np.isfinite([mean, variance]).all():
        raise ValueError("Invalid latent predictive distribution")
    if variance == 0:
        return float(expit(mean))
    value, error = quad(
        lambda z: (
            expit(mean + np.sqrt(variance) * z)
            * np.exp(-z * z / 2)
            / np.sqrt(2 * np.pi)
        ),
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=1000,
    )
    if error > 1e-9:
        raise ValueError("Independent probability integral failed")
    return value


def latent_variance(info, family, vector, size):
    """Reconstruct the variance for new grouping levels from recorded components.

    Args:
        info: Recorded fitted-model diagnostics, including variance components.
        family: Declared response distribution.
        vector: Callable returning a design vector for a random-effect term.
        size: Number of held-out observations.
    """
    variance = np.zeros(size)
    for component in info["variances"]:
        if family == "ordinal":
            terms = component["terms"]
            terms = [terms] if isinstance(terms, str) else terms
            z = np.column_stack([vector(term) for term in terms])
            variance += np.einsum(
                "bi,ij,bj->b", z, np.array(component["covariance"]), z
            )
        elif component["grp"] != "Residual":
            variance += (
                component["vcov"] * vector(component["var1"]) ** 2
                if component.get("var2") is None
                else 2
                * component["vcov"]
                * vector(component["var1"])
                * vector(component["var2"])
            )
    return variance


def apply_training_scaling(train, test, scaling):
    """Check training-only centers and scales before transforming the test rows.

    Args:
        train: Training observations used to estimate centers and scales.
        test: Held-out frame to transform in place.
        scaling: Recorded training centers and sample standard deviations.
    """
    for entry in scaling:
        column = entry["column"]
        np.testing.assert_allclose(
            [train[column].mean(), train[column].std()],
            [entry["center"], entry["scale"]],
            rtol=1e-12,
            atol=1e-12,
        )
        test[column] = (test[column] - entry["center"]) / entry["scale"]


def verify_predictions(observations, specification, fits, predictions):
    """Reconstruct predictions from recorded coefficients, variances and training scaling.

    Args:
        observations: Path to the frozen observation table.
        specification: Path to the declared model specification.
        fits: Directory containing fitted coefficients, scaling and diagnostics.
        predictions: Path to the recorded held-out predictions and losses.
    """
    spec = json.loads(specification.read_text())
    frame = pd.read_csv(
        observations,
        sep="\t",
        dtype={g: str for g in ("participant", "material", "item")},
    )
    train, test = (
        frame.loc[frame.role.eq("train")],
        frame.loc[frame.role.eq("test")].copy(),
    )
    if train.empty or test.empty:
        raise ValueError("Missing training or independent test observations")
    for group in ("participant", "material", "item"):
        if set(train[group]) & set(test[group]):
            raise ValueError("Training and test identities overlap")
    actual = pd.read_csv(
        predictions,
        sep="\t",
        dtype={g: str for g in ("participant", "material", "item")},
    )
    keys = ["participant", "material", "item"]
    if actual.duplicated(keys).any() or test.duplicated(keys).any():
        raise ValueError("Duplicate test observations")
    test = (
        test.set_index(keys)
        .reindex(pd.MultiIndex.from_frame(actual[keys]))
        .reset_index()
    )
    if len(test) != int(frame.role.eq("test").sum()) or test.response.isna().any():
        raise ValueError("Prediction inventory differs from independent test sample")
    np.testing.assert_allclose(test.response, actual.response, rtol=1e-12, atol=1e-12)
    scaling = json.loads((fits / "scaling.json").read_text())
    apply_training_scaling(train, test, scaling)
    diagnostics = json.loads((fits / "diagnostics.json").read_text())
    coefficients = (
        None
        if spec["family"] == "ordinal"
        else pd.read_csv(fits / "coefficients.tsv", sep="\t")
    )
    results = []

    def vector(term):
        return (
            np.ones(len(test)) if term == "(Intercept)" else test[term].to_numpy(float)
        )

    for name in spec["models"]:
        info = diagnostics[name]
        if info["nobs"] != len(train) or info["optimizer_code"] != 0:
            raise ValueError("Invalid fitted model diagnostics")
        if coefficients is None:
            beta = info["coefficients"] or {}
        else:
            selected = coefficients.loc[coefficients.model.eq(name)]
            beta = selected.set_index("term").Estimate.to_dict()
        eta = sum(
            (value * vector(term) for term, value in beta.items()), np.zeros(len(test))
        )
        variance = latent_variance(info, spec["family"], vector, len(test))
        if spec["family"] == "gaussian":
            probability = eta
            loss = (test.response.to_numpy() - eta) ** 2
            columns = {name + "_prediction": probability}
        else:
            np.testing.assert_allclose(
                eta, actual[name + "_latent_mean"], rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                variance, actual[name + "_latent_variance"], rtol=1e-10, atol=1e-10
            )
            if spec["family"] == "binomial":
                probability = np.array(
                    [
                        integrated_logistic(m, v)
                        for m, v in zip(eta, variance, strict=True)
                    ]
                )
                probability = np.clip(
                    probability, np.finfo(float).eps, 1 - np.finfo(float).eps
                )
                loss = -np.where(
                    test.response.eq(1), np.log(probability), np.log1p(-probability)
                )
                columns = {name + "_prediction": probability}
            elif spec["family"] == "ordinal":
                cumulative = np.array(
                    [
                        [
                            0.0,
                            *[
                                integrated_logistic(threshold - m, v)
                                for threshold in info["thresholds"]
                            ],
                            1.0,
                        ]
                        for m, v in zip(eta, variance, strict=True)
                    ]
                )
                probability = np.diff(cumulative, axis=1)
                categories = spec["categories"]
                index = pd.Categorical(test.response, categories=categories).codes
                if (index < 0).any():
                    raise ValueError("Unknown rating category")
                loss = -np.log(
                    np.maximum(
                        np.finfo(float).eps, probability[np.arange(len(test)), index]
                    )
                )
                columns = {
                    name + "_probability_" + str(category): probability[:, i]
                    for i, category in enumerate(categories)
                }
            else:
                raise ValueError("Unknown response family")
        columns[name + "_loss"] = loss
        errors = {}
        for column, expected in columns.items():
            np.testing.assert_allclose(actual[column], expected, rtol=1e-6, atol=2e-7)
            errors[column] = float(np.max(abs(actual[column].to_numpy() - expected)))
        results.append(
            {
                "model": name,
                "observations": len(test),
                "maximum_absolute_errors": errors,
            }
        )
    inputs = [
        observations,
        specification,
        predictions,
        fits / "scaling.json",
        fits / "diagnostics.json",
    ]
    if coefficients is not None:
        inputs.append(fits / "coefficients.tsv")
    return {
        "verification": "independent SciPy reconstruction; not independent parameter optimization",
        "family": spec["family"],
        "models": results,
        "inputs": {str(p): file_sha256(p) for p in inputs},
    }
