"""Independent numerical references for unregularized least squares."""

import numpy as np
import pytest
import statsmodels.api as sm

from replica.analysis.contracts.models import AnalysisModel
from replica.analysis.models.fixed_effects.ols import OLS
from replica.analysis.models.mixed_effects.lmer_rs import RustLmerModel


@pytest.mark.parametrize("correction", ["df", "mle"])
@pytest.mark.parametrize(
    "units",
    [(1.0, 1.0), (1e-12, 1e12), (1e12, 1e-12), (1e-200, 1e200), (1e200, 1e-200)],
)
def test_predictions_and_uncertainty_match_unscaled_reference(correction, units):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 2))
    y = 3 + x @ np.array([2, -0.5]) + rng.normal(0, 0.1, len(x))
    reference = sm.OLS(y[:40], sm.add_constant(x[:40])).fit()
    scaled = x * units
    model = OLS(correction)
    assert isinstance(model, AnalysisModel)
    prediction, beta, se, sigma = model.fit_predict_with_stats(
        y[:40], scaled[:40], scaled[40:]
    )
    multiplier = 1 if correction == "df" else np.sqrt(reference.df_resid / 40)
    np.testing.assert_allclose(
        prediction, reference.predict(sm.add_constant(x[40:])), atol=1e-12
    )
    np.testing.assert_allclose(beta * np.r_[1, units], reference.params, atol=1e-12)
    np.testing.assert_allclose(
        se * np.r_[1, units], reference.bse * multiplier, atol=1e-12
    )
    assert sigma == pytest.approx(np.sqrt(reference.scale) * multiplier, rel=1e-12)
    np.testing.assert_array_equal(
        prediction, model.fit_predict(y[:40], scaled[:40], scaled[40:])
    )
    predicted, scale = model.fit_predict_with_sigma(y[:40], scaled[:40], scaled[40:])
    np.testing.assert_array_equal(prediction, predicted)
    assert sigma == scale


@pytest.mark.parametrize(
    "kind", ["intercept", "duplicate", "scaled_duplicate", "constant", "zero"]
)
@pytest.mark.parametrize("correction", ["df", "mle"])
def test_dependent_columns_use_rank_for_variance(kind, correction):
    x = np.linspace(-1, 1, 20)
    y = 4 + x + np.sin(np.arange(len(x)))
    reduced = np.ones((len(x), 1)) if kind == "intercept" else sm.add_constant(x)
    reference = sm.OLS(y, reduced).fit()
    if kind == "intercept":
        design = np.empty((len(x), 0))
        expected_beta, expected_se = reference.params, reference.bse
    elif kind in ("duplicate", "scaled_duplicate"):
        multiple = 1000 if kind == "scaled_duplicate" else 1
        design = np.column_stack([x, multiple * x])
        expected_beta = reference.params[[0, 1, 1]] / [1, 2, 2 * multiple]
        expected_se = reference.bse[[0, 1, 1]] / [1, 2, 2 * multiple]
    elif kind == "constant":
        design = np.column_stack([x, np.ones(len(x))])
        expected_beta = reference.params[[0, 1, 0]] / [2, 1, 2]
        expected_se = reference.bse[[0, 1, 0]] / [2, 1, 2]
    else:
        design = np.column_stack([x, np.zeros(len(x))])
        expected_beta, expected_se = np.r_[reference.params, 0], np.r_[reference.bse, 0]
    prediction, beta, se, sigma = OLS(correction).fit_predict_with_stats(
        y, design, design
    )
    multiplier = 1 if correction == "df" else np.sqrt(reference.df_resid / len(x))
    np.testing.assert_allclose(prediction, reference.fittedvalues, atol=1e-12)
    np.testing.assert_allclose(beta, expected_beta, atol=1e-12)
    np.testing.assert_allclose(se, expected_se * multiplier, atol=1e-12)
    assert sigma == pytest.approx(np.sqrt(reference.scale) * multiplier, rel=1e-12)


def test_underdetermined_predictions_do_not_invent_residual_degrees_of_freedom():
    design = np.eye(2)
    response = np.array([1.0, 3.0])
    np.testing.assert_allclose(OLS().fit_predict(response, design, design), response)
    for method in ("fit_predict_with_sigma", "fit_predict_with_stats"):
        with pytest.raises(ValueError, match="residual degrees of freedom"):
            getattr(OLS(), method)(response, design, design)
    prediction, sigma = OLS("mle").fit_predict_with_sigma(response, design, design)
    np.testing.assert_allclose(prediction, response)
    assert sigma < 1e-14


@pytest.mark.parametrize(("separation", "rank"), [(0.5, 2), (8.0, 3)])
def test_near_collinear_rank_cutoff_controls_residual_degrees_of_freedom(
    separation, rank
):
    x = np.tile([-1.0, 1.0], 4)
    z = np.repeat([-1.0, 1.0], 4)
    response = np.tile([-1.0, -1.0, 1.0, 1.0], 2)
    delta = separation * np.finfo(float).eps * len(x)
    design = np.column_stack([x, x + delta * z])
    prediction, sigma = OLS().fit_predict_with_sigma(response, design, design)
    rss = np.sum((response - prediction) ** 2)
    assert sigma == pytest.approx(np.sqrt(rss / (len(x) - rank)), rel=1e-12)


@pytest.mark.parametrize("bad", ["response_nan", "predictor_inf", "shape", "empty"])
def test_invalid_training_data_are_rejected(bad):
    y, x = np.arange(4.0), np.arange(4.0)[:, None]
    if bad == "response_nan":
        y[0] = np.nan
    elif bad == "predictor_inf":
        x[0] = np.inf
    elif bad == "shape":
        y = y[:, None]
    else:
        y, x = y[:0], x[:0]
    with pytest.raises(ValueError, match="OLS requires"):
        OLS().fit_predict(y, x, x)


@pytest.mark.parametrize("argument", ["y_train", "X_train", "X_test"])
@pytest.mark.parametrize("kind", ["masked", "complex", "object_complex"])
def test_array_conversion_does_not_discard_masks_or_imaginary_values(argument, kind):
    arguments = {
        "y_train": np.arange(4.0),
        "X_train": np.arange(4.0)[:, None],
        "X_test": np.arange(4.0)[:, None],
    }
    value = arguments[argument]
    arguments[argument] = (
        np.ma.array(value, mask=True) if kind == "masked" else value + 1j
    )
    if kind == "object_complex":
        arguments[argument] = value.astype(object)
        arguments[argument].flat[0] = np.complex128(value.flat[0] + 1j)
    with pytest.raises(ValueError, match="real"):
        OLS().fit_predict(**arguments)


def test_rust_single_level_fallback_preserves_small_unit_predictor(monkeypatch):
    from replica.analysis.models.mixed_effects import lmer_rs

    def forbidden_import():
        pytest.fail("The OLS fallback must not load the Rust extension")

    monkeypatch.setattr(lmer_rs, "import_lme_python", forbidden_import)
    x = np.linspace(-1, 1, 30)
    y = 2 + x + np.sin(np.arange(len(x))) * 0.1
    grouped = np.column_stack([x * 1e-16, np.zeros((len(x), 2))])
    actual = RustLmerModel().fit_predict_with_stats(y, grouped, grouped)
    reference = sm.OLS(y, sm.add_constant(x)).fit()
    np.testing.assert_allclose(actual[0], reference.fittedvalues, atol=1e-12)
    np.testing.assert_allclose(actual[1] * [1, 1e-16], reference.params, atol=1e-12)
    np.testing.assert_allclose(actual[2] * [1, 1e-16], reference.bse, atol=1e-12)
    assert actual[3] == pytest.approx(np.sqrt(reference.scale))
