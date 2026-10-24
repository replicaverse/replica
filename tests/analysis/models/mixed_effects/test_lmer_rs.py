"""Tests for RustLmerModel, the lme-rs backed mixed-effects model.

The agreement tests fit the same design through ``RLmerModel`` (lme4) and
carry ``needs_r`` as well; the rest need only the ``lmer-rs`` extra.
"""

import numpy as np
import pytest

from replica.analysis import evaluate
from replica.analysis.cv import FullDataSplitter
from replica.analysis.models import RustLmerModel
from replica.analysis.models.mixed_effects.options import random_effect_terms
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal, UnitPosition
from tests.analysis._fixtures import VaryingLM, make_toy_dataset

pytestmark = pytest.mark.needs_lmer_rs


def _crossed(n_subj=30, n_item=100, seed=3, item_slope=0.0):
    """``[x0, x1, x2, participant, item]`` with subject and item intercepts."""
    rng = np.random.default_rng(seed)
    pairs = np.array([(s, i) for s in range(n_subj) for i in range(n_item)])
    n = len(pairs)
    X = rng.standard_normal((n, 3))
    y = (
        2.0
        + X @ np.array([1.0, 0.3, -1.0])
        + rng.normal(0, 1.0, n_subj)[pairs[:, 0]]
        + rng.normal(0, 0.4, n_subj)[pairs[:, 0]] * X[:, 0]
        + rng.normal(0, 0.7, n_item)[pairs[:, 1]]
        + rng.normal(0, item_slope, n_item)[pairs[:, 1]] * X[:, 0]
        + rng.normal(0, 0.5, n)
    )
    Xg = np.column_stack([X, pairs[:, 0], pairs[:, 1]])
    test = pairs[:, 1] >= n_item - 20  # held-out items unseen in training
    return y, Xg, ~test, test


def _r_model(**kw):
    from replica.analysis.models import RLmerModel

    return RLmerModel(nlminb_fallback=False, **kw)


class TestFormulaSharing:
    @pytest.mark.parametrize(
        "kw, expected",
        [
            (
                {"random_slopes": True, "uncorrelated_slopes_item": True},
                [
                    "(1 + x0 + x1 + x2|participant_id)",
                    "(1|item_id)",
                    "(0 + x0 | item_id)",
                    "(0 + x1 | item_id)",
                    "(0 + x2 | item_id)",
                ],
            ),
            # Out of range: the subject side keeps its intercept only.
            ({"random_slope_cols_subject": [5]}, ["(1|participant_id)", "(1|item_id)"]),
        ],
    )
    def test_random_terms_follow_the_options(self, kw, expected):
        fixed = ["x0", "x1", "x2"]
        terms = random_effect_terms(
            RustLmerModel(**kw), fixed, [], n_participants=4, n_items=6
        )
        assert terms == expected

    def test_single_level_factor_gets_no_term(self):
        terms = random_effect_terms(
            RustLmerModel(), ["x0"], [], n_participants=1, n_items=6
        )
        assert terms == ["(1|item_id)"]


class TestRustLmerModel:
    def test_refits_are_deterministic(self):
        y, Xg, tr, te = _crossed()
        first = RustLmerModel().fit_predict_with_sigma(y[tr], Xg[tr], Xg[te])
        second = RustLmerModel().fit_predict_with_sigma(y[tr], Xg[tr], Xg[te])
        np.testing.assert_array_equal(first[0], second[0])
        assert first[1] == second[1]

    def test_single_level_factors_fall_back_to_ols(self):
        y, Xg, _, _ = _crossed(n_subj=1, n_item=1, seed=5)
        Xg = np.repeat(Xg, 40, axis=0)
        y = np.repeat(y, 40)
        rng = np.random.default_rng(1)
        Xg[:, :3] = rng.standard_normal((len(y), 3))
        y = 1.0 + Xg[:, :3] @ np.array([0.5, -0.5, 0.25]) + rng.normal(0, 0.1, len(y))
        from replica.analysis import OLS

        preds, beta, se, sigma = RustLmerModel().fit_predict_with_stats(y, Xg, Xg[:5])
        ols = OLS(sigma_correction="df").fit_predict_with_stats(
            y, Xg[:, :3], Xg[:5, :3]
        )
        np.testing.assert_allclose(beta, ols[1], rtol=1e-10)
        np.testing.assert_allclose(sigma, ols[3], rtol=1e-10)

    def test_non_convergence_raises(self):
        y, Xg, tr, te = _crossed()
        model = RustLmerModel(
            random_slope_cols_subject=[0], max_iterations=1, max_inner_iterations=1
        )
        with pytest.raises(RuntimeError, match="did not converge"):
            model.fit_predict(y[tr], Xg[tr], Xg[te])

    def test_missing_values_are_rejected(self):
        y, Xg, tr, te = _crossed()
        y = y.copy()
        y[tr.nonzero()[0][0]] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            RustLmerModel().fit_predict(y[tr], Xg[tr], Xg[te])

    def test_rank_deficient_columns_are_dropped_later_first(self):
        y, Xg, tr, te = _crossed()
        Xd = Xg.copy()
        Xd[:, 2] = Xd[:, 1]  # x2 duplicates x1
        model = RustLmerModel()
        preds, beta, se, sigma = model.fit_predict_with_stats(y[tr], Xd[tr], Xd[te])
        # Intercept, x0, x1, and NaN where the dependent x2 was dropped.
        assert len(beta) == len(se) == 4
        assert np.isnan(beta[3]) and np.isnan(se[3])
        full = RustLmerModel().fit_predict_with_stats(
            y[tr], Xg[tr][:, [0, 1, 3, 4]], Xg[te][:, [0, 1, 3, 4]]
        )
        np.testing.assert_allclose(preds, full[0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(beta[:3], full[1], rtol=0, atol=1e-10)

    def test_dropped_fixed_column_keeps_its_random_slope(self):
        y, Xg, tr, te = _crossed()
        Xd = Xg.copy()
        Xd[:, 2] = Xd[:, 1]  # x2 duplicates x1 and is dropped from the fixed part
        model = RustLmerModel(random_slope_cols_subject=[2], include_item_re=False)
        preds, beta, _, _ = model.fit_predict_with_stats(y[tr], Xd[tr], Xd[te])
        assert len(beta) == 4 and np.isnan(beta[3])
        assert model._random_terms == ["(1 + x2|participant_id)"]
        same_slope = RustLmerModel(random_slope_cols_subject=[1], include_item_re=False)
        np.testing.assert_allclose(
            preds, same_slope.fit_predict(y[tr], Xd[tr], Xd[te]), rtol=0, atol=1e-8
        )

    def test_singular_fit_is_flagged(self):
        rng = np.random.default_rng(11)
        n_subj, n_item = 40, 60
        pairs = np.array([(s, i) for s in range(n_subj) for i in range(n_item)])
        X = rng.standard_normal((len(pairs), 2))
        base = (
            2.0
            + X @ np.array([1.0, -1.0])
            + rng.normal(0, 1.0, n_subj)[pairs[:, 0]]
            + rng.normal(0, 0.7, n_item)[pairs[:, 1]]
            + rng.normal(0, 0.5, len(pairs))
        )
        Xg = np.column_stack([X, pairs[:, 0], pairs[:, 1]])
        model = RustLmerModel(random_slope_cols_subject=[0])
        model.fit_predict(base, Xg, Xg[:5])  # no true subject slope variance
        assert model.last_singular is True
        with_slope = base + rng.normal(0, 0.6, n_subj)[pairs[:, 0]] * X[:, 0]
        model.fit_predict(with_slope, Xg, Xg[:5])
        assert model.last_singular is False

    def test_more_random_effects_than_observations_raises(self):
        # Two participants who each read every item once: a correlated by-item
        # slope has as many random effects as observations, which lme4 refuses.
        y, Xg, _, _ = _crossed(n_subj=2, n_item=40)
        model = RustLmerModel(random_slope_cols_item=[0])
        with pytest.raises(ValueError, match="number of random effects"):
            model.fit_predict(y, Xg, Xg[:5])

    def test_as_many_levels_as_observations_raises(self):
        y, Xg, _, _ = _crossed(n_subj=1, n_item=40)
        Xg = Xg.copy()
        Xg[:, -2] = np.arange(len(y))  # one participant per row
        with pytest.raises(ValueError, match="number of levels"):
            RustLmerModel(include_item_re=False).fit_predict(y, Xg, Xg[:5])

    def test_predictor_scale_warning(self):
        y, Xg, tr, te = _crossed()
        Xs = Xg.copy()
        Xs[:, 1] *= 1e4
        with pytest.warns(UserWarning, match="very different scales"):
            RustLmerModel().fit_predict(y[tr], Xs[tr], Xs[te])

    def test_re_design_needs_every_row(self):
        y, Xg, tr, te = _crossed()
        model = RustLmerModel(re_design=Xg[:, :1])
        with pytest.raises(ValueError, match="re_design has"):
            model.fit_predict(y[tr], Xg[tr], Xg[te])

    def test_evaluate_end_to_end(self):
        ds = make_toy_dataset(n_stimuli=6, n_words=6, n_participants=3)
        model = RustLmerModel()
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

    def test_evaluate_integrated_loglik(self):
        ds = make_toy_dataset(n_stimuli=6, n_words=6, n_participants=3)
        results = evaluate(
            ds,
            EvaluationSpec(
                targets=[ds.feature(ds.default_measure)],
                predictors=[ModelFeature(VaryingLM(), Surprisal())],
                controls=[UnitPosition()],
            ),
            AnalysisConfig(
                design=None,
                transform_dv=np.log2,
                model=RustLmerModel(integrated_loglik=True),
                cv=FullDataSplitter(),
            ),
        )
        assert np.isfinite(results[0].delta_ll)


@pytest.mark.needs_r
class TestAgreementWithLme4:
    @pytest.mark.parametrize(
        "kw, reml",
        [
            ({}, False),
            ({}, True),
            ({"random_slope_cols_subject": [0], "include_item_re": False}, False),
            (
                {
                    "random_slope_cols_subject": [0],
                    "uncorrelated_slopes_subject": True,
                    "include_item_re": False,
                },
                False,
            ),
            ({"random_slope_cols_subject": [0], "random_slope_cols_item": [0]}, False),
        ],
    )
    def test_matches_rlmer(self, kw, reml):
        y, Xg, tr, te = _crossed(item_slope=0.3)
        rust = RustLmerModel(reml=reml, **kw)
        r = _r_model(reml=reml, **kw)
        preds_x, beta_x, se_x, sigma_x = rust.fit_predict_with_stats(
            y[tr], Xg[tr], Xg[te]
        )
        preds_r, beta_r, se_r, sigma_r = r.fit_predict_with_stats(y[tr], Xg[tr], Xg[te])
        np.testing.assert_allclose(beta_x, beta_r, rtol=0, atol=1e-5)
        np.testing.assert_allclose(preds_x, preds_r, rtol=0, atol=1e-5)
        np.testing.assert_allclose(sigma_x, sigma_r, rtol=1e-4)
        np.testing.assert_allclose(se_x, se_r, rtol=1e-3)
        loglik_x = rust.fit_loglik(y[tr], Xg[tr], REML=reml)
        loglik_r = r.fit_loglik(y[tr], Xg[tr], REML=reml)
        assert abs(loglik_x - loglik_r) < 1e-4

    def test_fit_score_matches_rlmer(self):
        y, Xg, tr, _ = _crossed()
        loglik_x, fitted_x = RustLmerModel(integrated_loglik=True).fit_score(
            y[tr], Xg[tr], Xg[tr], y[tr], True
        )
        loglik_r, fitted_r = _r_model(integrated_loglik=True).fit_score(
            y[tr], Xg[tr], Xg[tr], y[tr], True
        )
        assert abs(loglik_x - loglik_r) < 1e-4
        np.testing.assert_allclose(fitted_x, fitted_r, rtol=0, atol=1e-4)
