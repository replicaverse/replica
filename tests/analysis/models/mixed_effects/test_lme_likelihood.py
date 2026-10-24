"""ML model comparisons must be invariant to predictor units."""

import numpy as np
import pandas as pd
import pytest

from replica.analysis.models import StatsmodelsLME
from replica.feature import ModelFeature


def test_missing_predictors_require_an_explicit_policy(monkeypatch):
    import statsmodels.regression.mixed_linear_model as mixed

    captured = []

    class Fit:
        def __init__(self, nobs):
            self.converged = True
            self.fe_params = pd.Series([0.0, 0.0], index=["Intercept", "x0"])
            self.bse_fe = pd.Series([1.0, 1.0], index=self.fe_params.index)
            self.scale = 1.0
            self.cov_re = pd.DataFrame([[1.0]], index=["Group"], columns=["Group"])
            self.nobs = nobs

        def fit(self, **kwargs):
            return self

    def record(formula, data, groups):
        captured.append(data.copy())
        return Fit(len(data))

    monkeypatch.setattr(mixed.MixedLM, "from_formula", record)
    x = np.array([[2.0], [np.nan], [4.0]])
    y, groups = np.arange(3.0), np.arange(3)
    with pytest.raises(ValueError, match="choose a NaN policy"):
        StatsmodelsLME().fit(y, x, groups)
    with pytest.raises(ValueError, match="choose a NaN policy"):
        StatsmodelsLME()._apply_test_nan_policy(x)
    assert not captured
    StatsmodelsLME(nan_policy="fill")._fit_lme(y, x, groups)
    np.testing.assert_array_equal(captured[-1]["x0"], [2, 1, 4])
    np.testing.assert_array_equal(
        StatsmodelsLME(nan_policy="fill")._apply_test_nan_policy(x)[:, 0], [2, 1, 4]
    )
    StatsmodelsLME(nan_policy="omit")._fit_lme(y, x, groups)
    assert captured[-1].index.tolist() == [0, 2]
    with pytest.raises(ValueError, match="nan_policy"):
        StatsmodelsLME(nan_policy="typo")
    with pytest.raises(ValueError, match="group_col"):
        StatsmodelsLME(group_col="typo")


@pytest.mark.parametrize("group_col,index", [("item", -1), ("participant", -2)])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1.5, float(2**63)])
def test_grouping_codes_are_validated_before_integer_conversion(
    group_col, index, invalid
):
    design = np.array([[2.0, 0.0, 0.0], [3.0, 1.0, 1.0]])
    design[0, index] = invalid
    with pytest.raises(ValueError, match="Grouping"):
        StatsmodelsLME(group_col=group_col)._split_X_and_groups(design)


@pytest.mark.parametrize("reml", [True, False])
def test_explicit_likelihood_mode_matches_statsmodels_and_respects_units(reml):
    from statsmodels.regression.mixed_linear_model import MixedLM

    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(20), 6)
    x = rng.normal(size=len(groups))
    y = 2 + 0.4 * x + rng.normal(size=20)[groups] + rng.normal(size=len(groups))
    ours = StatsmodelsLME(reml=reml).fit(y, x[:, None], groups)
    direct = MixedLM(y, np.column_stack([np.ones(len(y)), x]), groups).fit(
        method=["bfgs"], reml=reml
    )
    assert ours.converged and direct.converged
    assert ours.llf == pytest.approx(direct.llf, abs=1e-8)
    scaled = StatsmodelsLME(reml=reml).fit(y, (100 * x)[:, None], groups)
    expected_shift = -np.log(100) if reml else 0
    assert scaled.llf - ours.llf == pytest.approx(expected_shift, abs=1e-7)


def test_integrated_loglik_refuses_held_out_rows():
    """The integrated score is the training fit's own likelihood, so summing
    it over k held-out folds would count every row k - 1 times."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 6)
    x = rng.normal(size=len(groups))
    y = 2 + 0.4 * x + rng.normal(size=20)[groups] + rng.normal(size=len(groups))
    X = np.column_stack([x, np.zeros(len(x)), groups])
    model = StatsmodelsLME(integrated_loglik=True)
    with pytest.raises(ValueError, match="FullDataSplitter"):
        model.fit_score(y[:90], X[:90], X[90:], y[90:], is_full=False)
    loglik, predictions = model.fit_score(y, X, X, y, is_full=False)
    assert np.isfinite(loglik) and predictions.shape == y.shape


def test_evaluate_scores_the_integrated_likelihood_only_in_sample():
    from replica.analysis import AnalysisConfig, FullDataSplitter, ItemSplitter
    from replica.analysis.evaluation.evaluate import evaluate
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.feature import Surprisal, UnitPosition
    from tests.analysis._fixtures import VaryingLM, make_toy_dataset

    dataset = make_toy_dataset(n_stimuli=6, n_words=6, n_participants=3)
    spec = EvaluationSpec(
        targets=[dataset.feature(dataset.default_measure)],
        predictors=[ModelFeature(VaryingLM(), Surprisal())],
        controls=[UnitPosition()],
    )
    model = StatsmodelsLME(integrated_loglik=True, group_col="participant")
    with pytest.raises(ValueError, match="FullDataSplitter"):
        evaluate(
            dataset,
            spec,
            AnalysisConfig(
                design=None, transform_dv=np.log2, model=model, cv=ItemSplitter(3)
            ),
        )
    result = evaluate(
        dataset,
        spec,
        AnalysisConfig(
            design=None, transform_dv=np.log2, model=model, cv=FullDataSplitter()
        ),
    )
    assert np.isfinite(result[0].delta_ll)
