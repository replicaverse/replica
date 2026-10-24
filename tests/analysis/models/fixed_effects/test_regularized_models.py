"""Shared contracts and distinct penalty limits for regularized regressions."""

import numpy as np
import pytest
from sklearn.linear_model import ElasticNetCV, RidgeCV

from replica.analysis import OLS, evaluate
from replica.analysis.models import ElasticNetCVModel, RidgeCVModel
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal, UnitPosition
from tests.analysis._fixtures import VaryingLM, linear_data, make_toy_dataset


@pytest.fixture(params=[RidgeCVModel, ElasticNetCVModel], ids=["ridge", "elasticnet"])
def model_type(request):
    return request.param


def test_sigma_is_the_training_rmse(model_type):
    y, X, X_test = linear_data()
    model = model_type()
    preds, sigma = model.fit_predict_with_sigma(y, X, X_test)
    if isinstance(model, RidgeCVModel):
        reference = RidgeCV(alphas=model.alphas).fit(X, y)
        assert model.chosen_alphas == [reference.alpha_]
    else:
        reference = ElasticNetCV(
            l1_ratio=model.l1_ratios,
            alphas=model.n_alphas,
            cv=model.cv,
            max_iter=5000,
            n_jobs=1,
        ).fit(X, y)
        assert model.chosen == [(reference.alpha_, reference.l1_ratio_)]
    np.testing.assert_array_equal(preds, reference.predict(X_test))
    diagnostics = model.last_fit_diagnostics()
    expected = {"alpha": float(reference.alpha_)}
    if isinstance(model, ElasticNetCVModel):
        expected["l1_ratio"] = float(reference.l1_ratio_)
        assert diagnostics["details"]["n_iterations"] == reference.n_iter_
        assert diagnostics["details"]["dual_gap"] == reference.dual_gap_
    assert diagnostics["selected_parameters"] == expected
    diagnostics["selected_parameters"]["alpha"] = -1
    assert model.last_fit_diagnostics()["selected_parameters"] == expected
    np.testing.assert_array_equal(
        diagnostics["coefficients"]["estimates"],
        [reference.intercept_, *reference.coef_],
    )
    assert "standard_errors" not in diagnostics["coefficients"]
    assert diagnostics["residual_scale"]["sigma"] == sigma
    residuals = y - reference.predict(X)
    assert sigma == pytest.approx(np.sqrt(np.mean(residuals**2)), rel=1e-12)


def test_evaluate_end_to_end(model_type):
    ds = make_toy_dataset(n_stimuli=6, n_words=6, n_participants=2)
    model = (
        model_type(n_alphas=10, cv=3)
        if model_type is ElasticNetCVModel
        else model_type()
    )
    results = evaluate(
        ds,
        EvaluationSpec(
            targets=[ds.feature(ds.default_measure)],
            predictors=[ModelFeature(VaryingLM(), Surprisal())],
            controls=[UnitPosition()],
        ),
        AnalysisConfig(design=None, transform_dv=np.log2, model=model, n_folds=3),
    )
    assert len(results) == 1
    assert np.isfinite(results[0].delta_ll)
    # One baseline fit and one full fit per fold.
    if isinstance(model, RidgeCVModel):
        assert len(model.chosen_alphas) == 6
        assert all(alpha in model.alphas for alpha in model.chosen_alphas)
    else:
        assert len(model.chosen) == 6


def test_small_ridge_penalty_recovers_ols():
    # Generalized cross-validation loses precision below alpha ~1e-8.
    y, X, X_test = linear_data()
    preds = RidgeCVModel(alphas=[1e-6]).fit_predict(y, X, X_test)
    np.testing.assert_allclose(preds, OLS().fit_predict(y, X, X_test), atol=1e-6)


def test_extreme_ridge_penalty_predicts_the_training_mean():
    y, X, X_test = linear_data()
    preds = RidgeCVModel(alphas=[1e12]).fit_predict(y, X, X_test)
    np.testing.assert_allclose(preds, np.full(len(X_test), y.mean()), atol=1e-6)


def test_constant_response_predicts_that_constant():
    _, X, X_test = linear_data()
    y = np.full(len(X), 5.0)
    preds = ElasticNetCVModel().fit_predict(y, X, X_test)
    np.testing.assert_allclose(preds, np.full(len(X_test), 5.0), atol=1e-12)
