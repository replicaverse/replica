"""Evaluation dispatch, scoring helpers, and result contracts."""

import numpy as np
import pytest

from replica.analysis import (
    OLS,
    AnalysisResult,
    evaluate,
)
from replica.analysis.contracts.models import AnalysisModel, FoldScore, LikelihoodModel
from replica.analysis.cv.scalers import _standardize_split
from replica.analysis.fitting.scoring import (
    _fit_predict_sigma,
    _gaussian_log_likelihood,
    _gaussian_log_likelihood_pointwise,
)
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal, UnitLength
from tests.analysis._fixtures import ConstantLM, make_toy_dataset


class TestGaussianLogLikelihood:
    @pytest.mark.parametrize("sigma, variance", [(0.5, 0.25), (None, 0.03)])
    def test_scalar_and_pointwise_likelihood_match_gaussian_density(
        self, sigma, variance
    ):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        expected = -0.5 * np.log(2 * np.pi * variance) - np.array(
            [0.01, 0.04, 0.04]
        ) / (2 * variance)

        pointwise = _gaussian_log_likelihood_pointwise(y, y_pred, sigma)
        assert pointwise.shape == y.shape
        np.testing.assert_allclose(pointwise, expected, atol=1e-12)
        assert _gaussian_log_likelihood(y, y_pred, sigma) == pytest.approx(
            expected.sum(), abs=1e-12
        )

    @pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_scales_have_the_same_scalar_and_vector_contract(self, sigma):
        y = np.array([1.0, 2.0])
        for scale in (sigma, np.full(y.shape, sigma)):
            for scorer in (
                _gaussian_log_likelihood,
                _gaussian_log_likelihood_pointwise,
            ):
                with pytest.raises(ValueError, match="scales must be positive"):
                    scorer(y, y + 100, sigma=scale)

    def test_fitted_degenerate_scale_is_unscorable(self):
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="scales must be positive"):
            _gaussian_log_likelihood(y, y)

    @pytest.mark.parametrize("sigma", [1e-200, 1e200])
    def test_finite_scales_do_not_require_representable_variances(self, sigma):
        from scipy.stats import norm

        y = np.array([-1.0, 0.0, 2.0]) * sigma
        expected = norm.logpdf(y, scale=sigma)
        pointwise = _gaussian_log_likelihood_pointwise(y, np.zeros(3), sigma)
        np.testing.assert_allclose(pointwise, expected)
        assert _gaussian_log_likelihood(y, np.zeros(3), sigma) == pointwise.sum()

    @pytest.mark.parametrize("sigma", [None, 1e308, np.full(4, 1e308)])
    def test_representable_standardized_residuals_survive_subtraction_overflow(
        self, sigma
    ):
        y, prediction = np.array([1e308, 0, 0, 0]), np.array([-1e308, 0, 0, 0])
        expected = -0.5 * np.log(2 * np.pi) - np.log(1e308) - np.array([2, 0, 0, 0])
        pointwise = _gaussian_log_likelihood_pointwise(y, prediction, sigma)
        np.testing.assert_allclose(pointwise, expected, rtol=0, atol=1e-12)
        assert _gaussian_log_likelihood(y, prediction, sigma) == pytest.approx(
            expected.sum()
        )

    def test_zero_residual_with_tiny_scale_does_not_subtract_scaled_infinities(self):
        y = np.array([1e308])
        actual = _gaussian_log_likelihood_pointwise(y, y, 1e-308)
        expected = -0.5 * np.log(2 * np.pi) - np.log(1e-308)
        assert actual[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _standardize_split unit tests
# ---------------------------------------------------------------------------


class TestStandardizeSplit:
    def test_1d_reference(self):
        """Train mean/std applied to both splits."""
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test = np.array([6.0, 7.0])
        train_std, test_std = _standardize_split(train, test)

        m = 3.0  # np.mean([1,2,3,4,5])
        s = np.std([1.0, 2.0, 3.0, 4.0, 5.0])  # sqrt(2) ~ 1.4142

        np.testing.assert_allclose(train_std, (train - m) / s, atol=1e-12)
        np.testing.assert_allclose(test_std, (test - m) / s, atol=1e-12)

    def test_2d_column_wise(self):
        """Each column standardised independently using train stats."""
        train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        test = np.array([[4.0, 40.0]])

        train_std, test_std = _standardize_split(train, test)

        for c in range(2):
            m = np.mean(train[:, c])
            s = np.std(train[:, c])
            np.testing.assert_allclose(
                train_std[:, c], (train[:, c] - m) / s, atol=1e-12
            )
            np.testing.assert_allclose(test_std[:, c], (test[:, c] - m) / s, atol=1e-12)

    def test_constant_column_no_division_by_zero(self):
        """Constant column: std=0, should center without dividing."""
        train = np.array([5.0, 5.0, 5.0])
        test = np.array([5.0, 7.0])
        train_std, test_std = _standardize_split(train, test)
        np.testing.assert_array_equal(train_std, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(test_std, [0.0, 2.0])


# ---------------------------------------------------------------------------
# _fit_predict_sigma fallback unit test
# ---------------------------------------------------------------------------


class MinimalModel:
    """A model with fit_predict but not fit_predict_with_sigma."""

    name = "minimal"

    def fit_predict(self, y_train, X_train, X_test):
        # Predict train mean for all test points
        return np.full(X_test.shape[0], np.mean(y_train))


class TestFitPredictSigma:
    def test_fallback_uses_one_fit_for_predictions_and_scale(self):
        class ChangingFit(MinimalModel):
            calls = 0

            def fit_predict(self, y_train, X_train, X_test):
                self.calls += 1
                return super().fit_predict(y_train, X_train, X_test) + self.calls

        model = ChangingFit()
        y = np.arange(6.0)
        preds, sigma = _fit_predict_sigma(model, y, y[:, None], np.zeros((2, 1)))
        assert model.calls == 1
        np.testing.assert_array_equal(preds, [3.5, 3.5])
        assert sigma == pytest.approx(np.sqrt(np.sum((y - 3.5) ** 2) / 4))

    @pytest.mark.parametrize("needs_grouping", [False, True])
    def test_fallback_for_model_without_sigma(self, needs_grouping):
        """_fit_predict_sigma computes sigma from training residuals."""
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_train = np.ones((5, 1))  # dummy, MinimalModel ignores X
        X_test = np.ones((2, 1))

        model = MinimalModel()
        model.needs_grouping = needs_grouping
        if needs_grouping:
            X_train = np.column_stack([X_train, np.arange(5), np.arange(5)])
            X_test = np.column_stack([X_test, [0, 1], [0, 1]])
        preds, sigma = _fit_predict_sigma(model, y_train, X_train, X_test)

        # MinimalModel predicts mean(y_train) = 3.0
        np.testing.assert_array_equal(preds, [3.0, 3.0])

        # Residuals: [1-3, 2-3, 3-3, 4-3, 5-3] = [-2, -1, 0, 1, 2]
        # sum(r^2) = 10, p = 1 (X col) + 1 (assumed intercept) = 2
        # sigma = sqrt(10 / max(5-2, 1)) = sqrt(10/3)
        expected_sigma = np.sqrt(10.0 / 3.0)
        np.testing.assert_allclose(sigma, expected_sigma, atol=1e-10)

    def test_delegates_to_fit_predict_with_sigma(self):
        """When model has fit_predict_with_sigma, use it directly."""
        ols = OLS()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        y = X @ np.array([1.0, -0.5]) + 0.1 * rng.standard_normal(20)

        preds_direct, sigma_direct = ols.fit_predict_with_sigma(y[:15], X[:15], X[15:])
        preds_helper, sigma_helper = _fit_predict_sigma(ols, y[:15], X[:15], X[15:])

        np.testing.assert_array_equal(preds_direct, preds_helper)
        assert sigma_direct == sigma_helper


# ---------------------------------------------------------------------------
# evaluate() integration tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_likelihood_backend_needs_no_prediction_method(self):
        class ScoreOnly:
            name = "score_only"
            owns_loglik = True

            def __init__(self):
                self.calls = []

            def fit_score(self, y_train, X_train, X_test, y_test, *, is_full):
                self.calls.append((is_full, X_train.shape[1], X_test.shape[1]))
                pointwise = np.full(len(y_test), -0.5 if is_full else -2.0)
                predictions = np.full(len(y_test), y_train.mean())
                return FoldScore(float(pointwise.sum()), predictions, pointwise)

        model = ScoreOnly()
        assert isinstance(model, LikelihoodModel)
        assert not isinstance(model, AnalysisModel)
        dataset = make_toy_dataset()
        result = evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature("spr_rt")],
                predictors=[ModelFeature(ConstantLM(), Surprisal())],
            ),
            AnalysisConfig(
                design=None,
                model=model,
                n_folds=3,
                per_fold=True,
                retain_per_obs_delta_ll=True,
            ),
        )[0]

        assert model.calls == [(False, 0, 0), (True, 1, 1)] * 3
        assert result.delta_ll_per_unit == pytest.approx(1.5)
        for fold in result.fold_results:
            np.testing.assert_array_equal(
                fold.per_obs_delta_ll, np.full(fold.n_test, 1.5)
            )

    def test_result_metrics_and_identity(self):
        dataset = make_toy_dataset()
        (result,) = evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature("spr_rt")],
                predictors=[ModelFeature(ConstantLM(), Surprisal())],
                controls=[UnitLength()],
            ),
            AnalysisConfig(design=None, transform_dv=np.log2, model=OLS(), n_folds=3),
            language="en",
        )
        assert isinstance(result, AnalysisResult)
        assert np.isfinite(result.delta_ll)
        assert result.n_observations > 0
        for key in ("baseline_ll", "full_ll", "delta_ll", "delta_ll_per_unit"):
            assert getattr(result, key) == result.metrics[key]
        assert result.feature_name == "surprisal"
        assert result.model_name == "constant"
        assert result.dataset_name == "toy"
        assert result.language == "en"

    def test_sigma_sources_finite_and_differ(self):
        """Both ``sigma_source`` modes (the default 'training' and
        'test') produce finite results, and they disagree numerically, a
        regression that silently routed both to the same code path would
        otherwise go undetected."""
        ds = make_toy_dataset()
        lm = ConstantLM()
        spec = EvaluationSpec(
            targets=[ds.feature(ds.default_measure)],
            predictors=[ModelFeature(lm, Surprisal())],
            controls=[UnitLength()],
        )
        r_train = evaluate(
            ds,
            spec,
            AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=OLS(),
                n_folds=3,
                sigma_source="training",
            ),
        )
        r_test = evaluate(
            ds,
            spec,
            AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=OLS(),
                n_folds=3,
                sigma_source="test",
            ),
        )
        assert np.isfinite(r_train[0].delta_ll)
        assert np.isfinite(r_test[0].delta_ll)
        assert r_train[0].baseline_ll != r_test[0].baseline_ll

    def test_invalid_sigma_source_raises(self):
        ds = make_toy_dataset()
        lm = ConstantLM()
        with pytest.raises(ValueError, match="sigma_source"):
            evaluate(
                ds,
                EvaluationSpec(
                    targets=[ds.feature(ds.default_measure)],
                    predictors=[ModelFeature(lm, Surprisal())],
                ),
                AnalysisConfig(
                    design=None,
                    transform_dv=np.log2,
                    model=OLS(),
                    n_folds=3,
                    sigma_source="train",  # typo -- should be "training"
                ),
            )


# ---------------------------------------------------------------------------
# Grouping integration test
# ---------------------------------------------------------------------------


class GroupingRecorder:
    """Records X_train to verify grouping columns are appended."""

    name = "recorder"
    needs_grouping = True
    recorded_X_trains: list[np.ndarray]

    def __init__(self):
        self.recorded_X_trains = []

    def fit_predict(self, y_train, X_train, X_test):
        self.recorded_X_trains.append(X_train.copy())
        return np.full(X_test.shape[0], np.mean(y_train))


class TestGroupingIntegration:
    def test_grouping_columns_encode_participants_and_items(self):
        recorder = GroupingRecorder()
        dataset = make_toy_dataset(n_stimuli=6, n_words=4, n_participants=3)
        evaluate(
            dataset,
            EvaluationSpec(
                targets=[dataset.feature("spr_rt")],
                predictors=[ModelFeature(ConstantLM(), Surprisal())],
            ),
            AnalysisConfig(
                design=None, transform_dv=np.log2, model=recorder, n_folds=3
            ),
        )

        assert recorder.recorded_X_trains
        for design in recorder.recorded_X_trains:
            assert design.shape[1] == 3
            groups = design[:, -2:]
            np.testing.assert_array_equal(groups, groups.astype(int))
            assert np.all(groups >= 0)
        pooled = np.concatenate(recorder.recorded_X_trains)
        assert set(pooled[:, -2]) == set(range(3))
        assert set(pooled[:, -1]) == set(range(6))


# ---------------------------------------------------------------------------
# Per-fold orthogonalization
# ---------------------------------------------------------------------------


class _TableFeature:
    """Per-word values looked up by stimulus id."""

    def __init__(self, name, table):
        self.name = name
        self._table = table

    dependencies = ()

    def resolve(self, context, inputs):
        from replica.feature.core.binding import resolve_stimulus_feature

        return resolve_stimulus_feature(self, context, inputs)

    def __call__(self, stimulus, outputs, target_layer="words"):
        return self._table[stimulus.stimulus_id]


def test_orthogonalized_control_is_residualized_in_both_designs():
    """A full model must use the same residualized control as its baseline."""
    from replica.analysis import Residualize

    rng = np.random.default_rng(0)
    ds = make_toy_dataset(n_stimuli=60, n_words=10)
    target, against, noise = {}, {}, {}
    for datum in ds.data():
        sid = datum.stimulus.stimulus_id
        a = rng.normal(size=10)
        target[sid] = a
        against[sid] = a + 0.5 * rng.normal(size=10)
        noise[sid] = rng.normal(size=10)
        datum.values = a + 0.3 * rng.normal(size=10)
    control = _TableFeature("orth_target", target)
    reference = _TableFeature("against", against)
    spec = EvaluationSpec(
        targets=[ds.feature("spr_rt")],
        predictors=[_TableFeature("noise", noise)],
        controls=[control],
    )
    result = evaluate(
        ds,
        spec,
        AnalysisConfig(
            model=OLS(),
            design=None,
            preprocessing=(
                Residualize(control, against=(reference,), policy="centered_slope"),
            ),
        ),
    )[0]

    assert abs(result.delta_ll_per_unit) < 0.01
    assert result.metrics["full_r2"] == pytest.approx(
        result.metrics["baseline_r2"], abs=0.01
    )
