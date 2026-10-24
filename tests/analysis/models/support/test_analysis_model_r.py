"""Tests for R-based analysis models (RLinearModel, RLmerModel, RGamModel).

These tests require rpy2 and R with lme4/mgcv packages installed.
They will fail if dependencies are missing.
"""

from importlib.resources import files

import numpy as np
import pytest

from replica.analysis import OLS, evaluate
from replica.analysis.models import (
    RGamModel,
    RLinearModel,
    RLmerModel,
    RobustCrossedLmer,
)
from replica.analysis.models.support.r import _r_context
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal, UnitPosition
from replica.utils.r import RScriptError, run_r_script
from tests.analysis._fixtures import VaryingLM, make_toy_dataset

pytestmark = pytest.mark.needs_r


@pytest.mark.parametrize(
    "model, expected_name",
    [
        (RLinearModel(), "lm"),
        (RLmerModel(), "lmer"),
        (RGamModel(), "gam"),
    ],
)
def test_r_model_name(model, expected_name):
    """Hardcoded `.name` attribute on R-backed analysis models."""
    assert model.name == expected_name


# ---------------------------------------------------------------------------
# RLinearModel
# ---------------------------------------------------------------------------


class TestRLinearModel:
    def test_fit_predict_matches_ols(self):
        """R lm() should closely match Python OLS on the same data."""
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 2))
        beta_true = np.array([1.0, -0.5])
        y = X @ beta_true + 0.1 * rng.standard_normal(n)

        X_train, y_train = X[:50], y[:50]
        X_test = X[50:]

        ols_pred = OLS().fit_predict(y_train, X_train, X_test)
        rlm_pred = RLinearModel().fit_predict(y_train, X_train, X_test)

        np.testing.assert_allclose(rlm_pred, ols_pred, atol=1e-4)

    def test_fit_predict_with_sigma(self):
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 2))
        beta_true = np.array([1.0, -0.5])
        y = X @ beta_true + 0.1 * rng.standard_normal(n)

        X_train, y_train = X[:50], y[:50]
        X_test = X[50:]

        preds, sigma = RLinearModel().fit_predict_with_sigma(y_train, X_train, X_test)
        assert sigma > 0
        assert sigma < 0.5  # noise is ~0.1
        assert len(preds) == 10

    def test_fit_predict_with_stats(self):
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 2))
        y = X @ np.array([1.0, -0.5]) + 0.1 * rng.standard_normal(n)

        preds, beta, se, sigma = RLinearModel().fit_predict_with_stats(
            y[:50], X[:50], X[50:]
        )
        # beta should have 3 elements (intercept + 2 predictors)
        assert len(beta) == 3
        assert len(se) == 3
        assert len(preds) == 10
        assert sigma > 0

    def test_fit_predict_with_stats_sigma_matches(self):
        """sigma from fit_predict_with_stats matches fit_predict_with_sigma."""
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 2))
        y = X @ np.array([1.0, -0.5]) + 0.1 * rng.standard_normal(n)

        _, _, _, sigma_stats = RLinearModel().fit_predict_with_stats(
            y[:50], X[:50], X[50:]
        )
        _, sigma_direct = RLinearModel().fit_predict_with_sigma(y[:50], X[:50], X[50:])
        np.testing.assert_allclose(sigma_stats, sigma_direct, atol=1e-10)


# ---------------------------------------------------------------------------
# RLmerModel
# ---------------------------------------------------------------------------


class TestRLmerModel:
    def test_needs_grouping(self):
        assert RLmerModel().needs_grouping is True

    def test_fit_predict_multi_participant(self):
        """Test lmer with multiple participants and items."""
        rng = np.random.default_rng(42)
        n_per = 30
        # 3 participants, 10 items each
        X_fixed = rng.standard_normal((n_per * 3, 2))
        participants = np.repeat([0, 1, 2], n_per)
        items = np.tile(np.arange(n_per), 3)
        y = X_fixed @ np.array([1.0, -0.5]) + 0.2 * rng.standard_normal(n_per * 3)

        # Append grouping columns
        X_full = np.column_stack([X_fixed, participants, items])

        X_train, y_train = X_full[:80], y[:80]
        X_test = X_full[80:]

        model = RLmerModel()
        preds, sigma = model.fit_predict_with_sigma(y_train, X_train, X_test)
        assert len(preds) == len(X_test)
        assert sigma > 0
        diagnostics = model.last_fit_diagnostics()
        assert diagnostics["convergence_applicable"] is True
        assert diagnostics["rank"] == diagnostics["n_columns"] == 3
        assert isinstance(diagnostics["singular"], bool)

    def test_falls_back_to_lm_single_level(self):
        """With only 1 participant and 1 item, should fall back to lm."""
        rng = np.random.default_rng(42)
        n = 50
        X_fixed = rng.standard_normal((n, 2))
        participants = np.zeros(n)
        items = np.zeros(n)
        y = X_fixed @ np.array([1.0, -0.5]) + 0.1 * rng.standard_normal(n)

        X_full = np.column_stack([X_fixed, participants, items])

        model = RLmerModel()
        preds, sigma = model.fit_predict_with_sigma(y[:40], X_full[:40], X_full[40:])
        assert len(preds) == 10
        assert sigma > 0
        assert {
            k: model.last_fit_diagnostics()[k]
            for k in ("convergence_applicable", "rank", "n_columns")
        } == {
            "convergence_applicable": False,
            "rank": 3,
            "n_columns": 3,
        }


# ---------------------------------------------------------------------------
# RGamModel
# ---------------------------------------------------------------------------


class TestRGamModel:
    def test_fit_predict_smooth(self):
        """Test GAM with default smooth on last column."""
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([rng.standard_normal(n), x])
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        model = RGamModel()
        preds, sigma = model.fit_predict_with_sigma(y[:80], X[:80], X[80:])
        assert len(preds) == 20
        assert sigma > 0

    def test_explicit_smooth_cols(self):
        """Test specifying which columns are smooth."""
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([x, rng.standard_normal(n)])
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        model = RGamModel(smooth_cols=[0])
        preds = model.fit_predict(y[:80], X[:80], X[80:])
        assert len(preds) == 20

    def test_fit_predict_with_sigma_positive(self):
        rng = np.random.default_rng(42)
        n = 80
        X = rng.standard_normal((n, 2))
        y = X @ np.array([1.0, -0.5]) + 0.3 * rng.standard_normal(n)

        _, sigma = RGamModel().fit_predict_with_sigma(y[:60], X[:60], X[60:])
        assert sigma > 0

    def test_use_bam(self):
        """Test bam() mode produces valid predictions."""
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([rng.standard_normal(n), x])
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        model = RGamModel(use_bam=True)
        preds, sigma = model.fit_predict_with_sigma(y[:80], X[:80], X[80:])
        assert len(preds) == 20
        assert sigma > 0

    def test_bam_reports_no_slope_for_a_smoothed_column(self):
        """A smooth has no single slope, so its coefficient is NaN, as on the
        gam path."""
        rng = np.random.default_rng(3)
        n = 300
        linear, curved = rng.standard_normal(n), rng.standard_normal(n)
        y = 0.5 * linear + np.sin(2 * curved) + 0.2 * rng.standard_normal(n)
        X = np.column_stack([linear, curved])
        model = RGamModel(smooth_cols=[1], use_bam=True, method="fREML")
        _, beta, se, _ = model.fit_predict_with_stats(y, X, X)
        assert beta.shape == se.shape == (3,)
        assert np.isfinite(beta[:2]).all() and np.isfinite(se[:2]).all()
        assert beta[1] == pytest.approx(0.5, abs=0.05)
        assert np.isnan(beta[2]) and np.isnan(se[2])

    def test_custom_formula(self):
        """Test passing a custom R formula string."""
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([x, rng.standard_normal(n)])
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        model = RGamModel(formula="y ~ s(x0, bs='cr') + x1")
        preds = model.fit_predict(y[:80], X[:80], X[80:])
        assert len(preds) == 20

    def test_custom_formula_with_col_names(self):
        """Test custom formula with column name remapping."""
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([x, rng.standard_normal(n)])
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        model = RGamModel(
            formula="y ~ s(surp, bs='cr') + wlen",
            col_names={"x0": "surp", "x1": "wlen"},
        )
        preds = model.fit_predict(y[:80], X[:80], X[80:])
        assert len(preds) == 20

    def test_random_smooths(self):
        """Test by-subject random effect smooths."""
        rng = np.random.default_rng(42)
        n_per = 40
        n_subj = 3
        n = n_per * n_subj
        x = np.linspace(0, 2 * np.pi, n)
        participants = np.repeat(np.arange(n_subj), n_per)
        items = np.tile(np.arange(n_per), n_subj)
        X_fixed = np.column_stack([x, rng.standard_normal(n)])
        X = np.column_stack([X_fixed, participants, items])
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = RGamModel(smooth_cols=[0], random_smooths=True)
        assert model.needs_grouping is True
        preds, sigma = model.fit_predict_with_sigma(y[:100], X[:100], X[100:])
        assert len(preds) == n - 100
        assert sigma > 0

    def test_random_smooths_drop_columns_the_ablation_removed(self):
        """A baseline fit must not carry a random slope for the dropped predictor.

        Ablation studies configure ``smooth_cols`` for the full model, then let
        ``evaluate()`` refit without the target column. The formula is rebuilt
        from the narrower array, so any random slope kept for the missing column
        names something absent from the data and R aborts the fit.
        """
        model = RGamModel(
            smooth_cols=[0, 1, 2],
            random_smooths=True,
            col_names={"x0": "wlen", "x1": "unigramsurp", "x2": "gpt"},
        )
        # 5 columns = 3 fixed + participant + item; 4 = the baseline, gpt gone.
        assert "s(gpt, participant_id, bs='re')" in model._build_formula(5)

        baseline = model._build_formula(4)
        assert "gpt" not in baseline
        assert "s(wlen, participant_id, bs='re')" in baseline
        assert "s(participant_id, bs='re')" in baseline

    def test_fit_loglik(self):
        """Test log-likelihood computation."""
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([rng.standard_normal(n), x])
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        model = RGamModel()
        ll = model.fit_loglik(y, X)
        assert np.isfinite(ll)

    @staticmethod
    def _re_data(seed: int = 7):
        """y = 1.5*x + large subject RE + noise; columns [x, subj, item]."""
        rng = np.random.default_rng(seed)
        n_subj, per = 8, 50
        re = rng.normal(0.0, 2.0, n_subj)
        xs, ys, subj = [], [], []
        for s in range(n_subj):
            x = rng.standard_normal(per)
            xs.append(x)
            ys.append(1.5 * x + re[s] + rng.normal(0, 0.3, per))
            subj.append(np.full(per, s))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        subj = np.concatenate(subj)
        items = np.zeros_like(subj)
        return y, x, subj, items

    def test_bam_heldout_predictions_exclude_random_effects(self):
        """Held-out bam predictions exclude every random-effect smooth.

        Relabelling which subject sorts into factor level 1 leaves
        fixed-effects-only predictions unchanged; a leaked BLUP shifts them.
        """
        y, x, subj, items = self._re_data()
        X_test = np.column_stack([np.linspace(-2, 2, 7), np.zeros(7), np.zeros(7)])

        def fit(labels):
            X = np.column_stack([x, labels, items])
            model = RGamModel(
                smooth_all=True,
                smooth_k=6,
                use_bam=True,
                discrete=True,
                method="fREML",
                re_subject_intercept=True,
            )
            preds, _ = model.fit_predict_with_sigma(y, X, X_test)
            return preds

        preds_a = fit(subj)  # level 1 = subject 0
        preds_b = fit(7 - subj)  # level 1 = subject 7
        # Subject 0 and subject 7 have different REs (sd=2), so the buggy
        # path differs by O(1); the fixed path is identical.
        np.testing.assert_allclose(preds_a, preds_b, atol=1e-6)

    def test_gam_heldout_predictions_invariant_to_test_ids(self):
        """rpy2 GAM path: held-out predictions must not depend on the test
        rows' participant ids (REs are excluded, per the docstring)."""
        y, x, subj, items = self._re_data()
        X_train = np.column_stack([x, subj, items])
        xg = np.linspace(-2, 2, 7)
        X_test_s0 = np.column_stack([xg, np.full(7, 0), np.zeros(7)])
        X_test_s7 = np.column_stack([xg, np.full(7, 7), np.zeros(7)])

        model = RGamModel(
            smooth_all=True,
            smooth_k=6,
            re_subject_intercept=True,
        )
        preds_s0, _ = model.fit_predict_with_sigma(y, X_train, X_test_s0)
        preds_s7, _ = model.fit_predict_with_sigma(y, X_train, X_test_s7)
        np.testing.assert_allclose(preds_s0, preds_s7, atol=1e-6)

    def test_default_formula_smooths_the_final_predictor(self):
        """The default smooth applies to the final predictor without grouping."""
        model = RGamModel()
        formula = model._build_formula(3)
        assert "s(x2" in formula
        assert "x0" in formula
        assert "x1" in formula
        assert model.needs_grouping is False

    def test_conditional_re_predict_tracks_subject_offsets(self):
        """re_predict='conditional' keeps subject random effects in held-out
        predictions; 'exclude' drops them. Covers both the rpy2 (gam) and
        subprocess (bam) paths plus sigma_mode='train_mse'."""
        rng = np.random.default_rng(0)
        n = 400
        subj = rng.integers(0, 8, n)
        x = rng.standard_normal(n)
        y = 2 * x + 0.5 * subj + rng.standard_normal(n)
        X = np.column_stack([x, subj, np.zeros(n)])
        tr, te = np.arange(300), np.arange(300, 400)
        subj_means = np.array([y[tr][subj[tr] == s].mean() for s in range(8)])
        target = subj_means[subj[te]]

        for use_bam in (False, True):
            kw = dict(
                smooth_cols=[0],
                use_bam=use_bam,
                re_subject_intercept=True,
                re_subject_slope_cols="all",
                sigma_mode="train_mse",
            )
            cond = RGamModel(re_predict="conditional", **kw)
            preds_c, sigma = cond.fit_predict_with_sigma(y[tr], X[tr], X[te])
            excl = RGamModel(re_predict="exclude", **kw)
            preds_e, _ = excl.fit_predict_with_sigma(y[tr], X[tr], X[te])
            r_cond = np.corrcoef(preds_c, target)[0, 1]
            r_excl = np.corrcoef(preds_e, target)[0, 1]
            assert sigma > 0
            assert r_cond > r_excl

    def test_re_predict_and_sigma_mode_validated(self):
        with pytest.raises(ValueError):
            RGamModel(re_predict="bogus")
        with pytest.raises(ValueError):
            RGamModel(sigma_mode="bogus")

    def test_shrink_unseen_requires_bam(self):
        """Per-row shrinkage is supported by the bam prediction path."""
        with pytest.raises(ValueError, match="use_bam"):
            RGamModel(re_predict="shrink_unseen", re_subject_intercept=True)
        model = RGamModel(
            re_predict="shrink_unseen",
            re_subject_intercept=True,
            use_bam=True,
            method="fREML",
        )
        assert model.re_predict == "shrink_unseen"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"use_bam": True},
            {"use_bam": True, "method": "ML"},
            {"method": "fREML"},
        ],
    )
    def test_discrete_requires_bam_with_a_discretizing_method(self, kwargs):
        """mgcv drops ``discrete=TRUE`` with a warning unless bam fits by
        fREML or NCV, and gam has no such option."""
        with pytest.raises(ValueError, match="discrete"):
            RGamModel(discrete=True, **kwargs)
        assert RGamModel(discrete=True, use_bam=True, method="fREML").discrete

    @pytest.mark.parametrize("use_bam", [False, True])
    def test_discrete_prediction_requires_discrete_bam(self, use_bam):
        with pytest.raises(ValueError, match="prediction_discrete"):
            RGamModel(use_bam=use_bam, prediction_discrete=True)
        assert RGamModel().configuration()["prediction_discrete"] is False

    def test_loglik_rejects_discrete_prediction(self):
        model = RGamModel(
            use_bam=True, discrete=True, method="fREML", prediction_discrete=True
        )
        with pytest.raises(ValueError, match="prediction_discrete.*fit_loglik"):
            model.fit_loglik(np.arange(5.0), np.arange(5.0)[:, None])

    @pytest.mark.parametrize("policy", ["exclude", "conditional", "shrink_unseen"])
    def test_bam_prediction_discretization_matches_mgcv(self, policy):
        """Each prediction mode matches mgcv while retaining its group policy."""
        y, x, subject, item = self._re_data()
        y = y + np.sin(4 * x)
        train = np.column_stack([x, subject, item])
        grid = np.linspace(-2.8, 2.8, 4001)
        test = np.column_stack(
            [np.tile(grid, 2), np.repeat([0, 7], len(grid)), np.zeros(2 * len(grid))]
        )
        if policy == "shrink_unseen":
            test[len(grid) :, -2] = 99
        formula = "y ~ s(x0, bs='cr', k=8) + s(participant_id, bs='re')"
        expected = {}
        with _r_context() as ro:
            ro.globalenv[".discrete_train"] = ro.r["data.frame"](
                y=ro.FloatVector(y),
                x0=ro.FloatVector(x),
                participant_id=ro.FloatVector(subject),
            )
            ro.globalenv[".discrete_test"] = ro.r["data.frame"](
                x0=ro.FloatVector(test[:, 0]),
                participant_id=ro.FloatVector(test[:, -2]),
            )
            ro.globalenv[".discrete_policy"] = policy
            ro.r(
                "suppressPackageStartupMessages(library(mgcv)); "
                ".discrete_train$participant_id <- factor(.discrete_train$participant_id); "
                ".discrete_unseen <- .discrete_test$participant_id == 99; "
                ".discrete_test$participant_id[.discrete_unseen] <- 0; "
                ".discrete_test$participant_id <- factor(.discrete_test$participant_id, "
                "levels=levels(.discrete_train$participant_id)); "
                f".discrete_model <- bam({formula}, data=.discrete_train, "
                "method='fREML', discrete=TRUE)"
            )
            for discrete in (False, True):
                ro.globalenv[".discrete_prediction"] = discrete
                expected[discrete] = np.asarray(
                    ro.r(
                        "local({"
                        "groups <- if (.discrete_policy == 'shrink_unseen') "
                        "split(seq_len(nrow(.discrete_test)), .discrete_unseen) "
                        "else list(seq_len(nrow(.discrete_test))); "
                        "out <- numeric(nrow(.discrete_test)); "
                        "for (rows in groups) {"
                        "exclude <- if (.discrete_policy == 'exclude' || "
                        "(.discrete_policy == 'shrink_unseen' && .discrete_unseen[rows[1]])) "
                        "'s(participant_id)' else NULL; "
                        "out[rows] <- predict(.discrete_model, .discrete_test[rows, ], "
                        "type='response', exclude=exclude, discrete=.discrete_prediction)}; "
                        "out})"
                    ),
                    dtype=float,
                )
        assert np.max(np.abs(expected[True] - expected[False])) > 1e-4
        for discrete in (False, True):
            model = RGamModel(
                formula=formula,
                use_bam=True,
                discrete=True,
                method="fREML",
                re_predict=policy,
                prediction_discrete=discrete,
            )
            assert model.configuration()["prediction_discrete"] is discrete
            actual = model.fit_predict(y, train, test)
            np.testing.assert_allclose(actual, expected[discrete], rtol=1e-7, atol=1e-8)


@pytest.fixture(params=["gam", "bam", "discrete_bam"])
def gam_backend(request):
    """Exercise both prediction paths and bam's discrete fitting option."""
    return {
        "use_bam": request.param != "gam",
        "discrete": request.param == "discrete_bam",
        "method": "REML" if request.param == "gam" else "fREML",
    }


@pytest.fixture
def gam_curve_data():
    """Crossed subjects/items with subject curves, holding out whole items."""
    rng = np.random.default_rng(414)
    subject = np.repeat(np.arange(6), 40)
    item = np.tile(np.arange(40), 6)
    x, z = rng.uniform(-2, 2, (2, len(subject)))
    y = (
        2
        + 0.8 * x**3
        - 0.4 * z
        + 1.2 * z**2
        + rng.normal(0, 2, 6)[subject]
        + rng.normal(0, 2, 6)[subject] * x**2
        + rng.normal(0, 0.7, 40)[item]
        + rng.normal(0, 0.2, len(subject))
    )
    X = np.column_stack([x, z, subject, item])
    train = item >= 5
    return y[train], X[train], X[~train]


def _gam_prediction_reference(y, X_train, X_test, formula, backend, *, population):
    """Independent mgcv fit, keeping the intercept and known population columns.

    This deliberately does not inspect smooth classes or grouping metadata.
    """
    with _r_context() as ro:
        columns = [
            *(f"x{i}" for i in range(X_train.shape[1] - 2)),
            "participant_id",
            "item_id",
        ]
        for name, X in (("train", X_train), ("test", X_test)):
            data = {col: ro.FloatVector(X[:, i]) for i, col in enumerate(columns)}
            if name == "train":
                data["y"] = ro.FloatVector(y)
            ro.globalenv[f".reference_{name}"] = ro.r["data.frame"](**data)
        ro.globalenv[".reference_formula"] = formula
        ro.r(
            "for (col in c('participant_id', 'item_id')) {"
            ".reference_train[[col]] <- factor(.reference_train[[col]]); "
            ".reference_test[[col]] <- factor(.reference_test[[col]], "
            "levels=levels(.reference_train[[col]]))}"
        )
        if population:
            # Supply valid placeholders only for unseen levels. Seen IDs may
            # contribute retained parametric effects and must stay unchanged.
            ro.r(
                "for (col in c('participant_id', 'item_id')) {"
                ".reference_test[[col]][is.na(.reference_test[[col]])] <- "
                "levels(.reference_train[[col]])[1]}"
            )
        fitter = "bam" if backend["use_bam"] else "gam"
        options = f", method='{backend['method']}'"
        if backend["use_bam"]:
            options += f", discrete={'TRUE' if backend['discrete'] else 'FALSE'}"
        ro.r(
            "suppressPackageStartupMessages(library(mgcv)); "
            f".reference_model <- {fitter}(as.formula(.reference_formula), "
            f"data=.reference_train{options})"
        )
        if population:
            return np.asarray(
                ro.r(
                    "local({"
                    "X <- predict(.reference_model, newdata=.reference_test, "
                    "type='lpmatrix', discrete=FALSE); "
                    "s <- .reference_model$smooth[[1]]; "
                    "keep <- c(seq_len(.reference_model$nsdf), "
                    "s$first.para:s$last.para); "
                    "drop(X[, keep, drop=FALSE] %*% coef(.reference_model)[keep])})"
                ),
                dtype=float,
            )
        return np.asarray(
            ro.r(
                "predict(.reference_model, newdata=.reference_test, "
                "type='response', discrete=FALSE)"
            ),
            dtype=float,
        )


@pytest.mark.parametrize(
    "group_terms",
    [
        pytest.param("s(participant_id, bs='re')", id="re"),
        pytest.param("s(participant_id, x0, bs='fs', m=1, k=4)", id="fs"),
        pytest.param("s(x0, by=participant_id, k=4)", id="factor_by"),
        pytest.param(
            "t2(x1, participant_id, bs=c('tp', 're'), k=c(4, 6))", id="tensor"
        ),
        pytest.param(
            "s(x0, participant_id, bs='fs', m=1, k=4) + s(item_id, bs='re')",
            id="mixed",
        ),
        pytest.param("s(x0, item_id, bs='fs', m=1, k=4)", id="item_fs"),
        pytest.param("s(participant_id, bs='re') + s(x2, bs='re')", id="other_re"),
        pytest.param("s(x2, bs='re')", id="ungrouped_re"),
    ],
)
def test_gam_grouped_smooth_exclusion_matches_reference(
    gam_backend, gam_curve_data, group_terms
):
    """Exclude grouped curves while retaining the linear and smooth controls."""
    y, train, test = gam_curve_data
    if "x2" in group_terms:
        # A penalized numeric effect with no grouping column must still be
        # excluded under the existing bs='re' rule.
        train = np.column_stack([train[:, :2], train[:, 1] ** 2, train[:, -2:]])
        test = np.column_stack([test[:, :2], test[:, 1] ** 2, test[:, -2:]])
    formula = f"y ~ s(x0, k=4) + x1 + {group_terms}"
    expected = _gam_prediction_reference(
        y, train, test, formula, gam_backend, population=True
    )
    actual = RGamModel(formula=formula, **gam_backend).fit_predict(y, train, test)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_gam_exclusion_preserves_numeric_by_smooth(gam_backend, gam_curve_data):
    """A numeric by variable is a population effect, even for a one-row fold."""
    y, train, test = gam_curve_data
    test = test[:1]
    formula = "y ~ x1 + s(x0, by=x1, k=4) + s(participant_id, bs='re')"
    expected = _gam_prediction_reference(
        y, train, test, formula, gam_backend, population=True
    )
    actual = RGamModel(formula=formula, **gam_backend).fit_predict(y, train, test)
    assert actual.shape == (1,)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("group", ["participant_id", "item_id"])
@pytest.mark.parametrize("with_smooth", [False, True])
def test_gam_exclusion_preserves_fixed_group_effects(
    gam_backend, gam_curve_data, group, with_smooth
):
    """Excluding a group smooth must preserve a retained fixed group effect."""
    from rpy2.rinterface_lib.embedded import RRuntimeError

    y, train, test = gam_curve_data
    if group == "item_id":
        train, test = train[:, [0, 1, 3, 2]], test[:, [0, 1, 3, 2]]
    formula = f"y ~ s(x0, k=4) + x1 + {group}"
    if with_smooth:
        formula += f" + s(x0, {group}, bs='fs', m=1, k=4)"
    expected = _gam_prediction_reference(
        y, train, test, formula, gam_backend, population=True
    )
    model = RGamModel(formula=formula, **gam_backend)
    actual = model.fit_predict(y, train, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    unseen = test.copy()
    unseen[:, -2 if group == "participant_id" else -1] = 99
    with pytest.raises((RRuntimeError, RScriptError), match=f"unseen {group} levels"):
        model.fit_predict(y, train, unseen)


@pytest.mark.parametrize("use_bam", [False, True], ids=["gam", "bam"])
@pytest.mark.parametrize(
    "group_terms",
    [
        "s(x0, factor(participant_id), bs='fs', m=1, k=4)",
        "s(x0, by=factor(participant_id), k=4)",
    ],
    ids=["fs", "factor_by"],
)
def test_gam_exclusion_recognizes_group_expressions(
    gam_curve_data, use_bam, group_terms
):
    """mgcv stores transformed grouping variables as expressions in metadata.

    The explicit factor() forms fit in gam and ordinary bam; mgcv's discrete
    fitter itself rejects them before prediction.
    """
    y, train, test = gam_curve_data
    formula = f"y ~ s(x0, k=4) + x1 + {group_terms}"
    backend = {"use_bam": use_bam, "discrete": False, "method": "REML"}
    expected = _gam_prediction_reference(
        y, train, test, formula, backend, population=True
    )
    actual = RGamModel(formula=formula, **backend).fit_predict(y, train, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_gam_factor_smooth_exclusion_is_invariant(gam_backend, gam_curve_data):
    """Changing test IDs or which training subject sorts first has no effect."""
    y, train, test = gam_curve_data
    test = np.tile(test[:5], (3, 1))
    test[:, -2] = np.repeat([0, 5, 99], 5)  # Two seen subjects and one unseen.
    model = RGamModel(
        formula="y ~ s(x0, k=4) + x1 + s(x0, participant_id, bs='fs', m=1, k=4)",
        **gam_backend,
    )
    original = model.fit_predict(y, train, test)
    relabeled = train.copy()
    relabeled[:, -2] = 5 - relabeled[:, -2]
    reordered = model.fit_predict(y, relabeled, test)
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original[:5], original[5:10], atol=1e-7)
    np.testing.assert_allclose(original[:5], original[10:], atol=1e-7)
    np.testing.assert_allclose(original, reordered, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("group", ["participant_id", "item_id"])
def test_gam_conditional_validates_only_used_groups(gam_backend, gam_curve_data, group):
    """Unseen unused IDs are harmless; unseen IDs in the formula still fail."""
    from rpy2.rinterface_lib.embedded import RRuntimeError

    y, train, test = gam_curve_data
    if group == "item_id":
        train, test = train[:, [0, 1, 3, 2]], test[:, [0, 1, 3, 2]]
    formula = f"y ~ s(x0, k=4) + x1 + s(x0, {group}, bs='fs', m=1, k=4)"
    expected = _gam_prediction_reference(
        y, train, test, formula, gam_backend, population=False
    )
    population = _gam_prediction_reference(
        y, train, test, formula, gam_backend, population=True
    )
    model = RGamModel(formula=formula, re_predict="conditional", **gam_backend)
    actual = model.fit_predict(y, train, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    assert np.max(np.abs(actual - population)) > 0.1

    unseen = test.copy()
    unseen[:, -2 if group == "participant_id" else -1] = 99
    with pytest.raises((RRuntimeError, RScriptError), match=f"unseen {group} levels"):
        model.fit_predict(y, train, unseen)


def test_gam_conditional_uses_selected_baseline_formula(gam_backend, gam_curve_data):
    """Groups named only by the full formula do not constrain a baseline fit."""
    y, train, test = gam_curve_data
    train, test = train[:, [0, 2, 3]], test[:, [0, 2, 3]].copy()
    test[:, -2:] = 99
    model = RGamModel(
        formula="y ~ s(x0, k=4) + x1 + s(participant_id, bs='re')",
        baseline_formula="y ~ s(x0, k=4)",
        col_names={"x0": "x0", "x1": "x1"},
        re_predict="conditional",
        **gam_backend,
    )
    actual = model.fit_predict(y, train, test)
    expected = RGamModel(formula="y ~ s(x0, k=4)", **gam_backend).fit_predict(
        y, train[:, :1], test[:, :1]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize("mode", ["exclude", "conditional"])
@pytest.mark.parametrize("group", ["participant_id", "item_id"])
def test_gam_required_groups_must_occur_in_fitted_rows(
    gam_backend, gam_curve_data, mode, group
):
    """An ID found only in rows omitted by mgcv has no usable fixed effect."""
    from rpy2.rinterface_lib.embedded import RRuntimeError

    y, train, _ = gam_curve_data
    column = -2 if group == "participant_id" else -1
    absent = train[:, column].min()
    y[train[:, column] == absent] = np.nan
    model = RGamModel(
        formula=f"y ~ s(x0, k=4) + {group}", re_predict=mode, **gam_backend
    )
    with pytest.raises((RRuntimeError, RScriptError), match=f"unseen {group} levels"):
        model.fit_predict(y, train, train[:5])


@pytest.mark.parametrize("group", ["participant_id", "item_id"])
def test_gam_exclusion_uses_observed_placeholders(gam_backend, gam_curve_data, group):
    """Omitting the first factor level must not change population predictions."""
    y, train, test = gam_curve_data
    column = -2 if group == "participant_id" else -1
    omitted = train[:, column] == train[:, column].min()
    formula = f"y ~ s(x0, k=4) + x1 + s(x0, {group}, bs='fs', m=1, k=4)"
    expected = _gam_prediction_reference(
        y[~omitted], train[~omitted], test, formula, gam_backend, population=True
    )
    y[omitted] = np.nan
    actual = RGamModel(formula=formula, **gam_backend).fit_predict(y, train, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_gam_standalone_bam_finds_prediction_helper(tmp_path, gam_curve_data):
    """The documented Rscript entry point loads its sibling helper from any cwd."""
    y, train, test = gam_curve_data
    header = "y\tx0\tx1\tparticipant_id\titem_id"
    for name, data, response in (
        ("train", train, y),
        ("test", test, np.zeros(len(test))),
    ):
        np.savetxt(
            tmp_path / f"{name}.tsv",
            np.column_stack([response, data]),
            header=header,
            comments="",
            delimiter="\t",
        )
    formula = "y ~ s(x0, k=4) + x1 + s(x0, participant_id, bs='fs', m=1, k=4)"
    run_r_script(
        files("replica.analysis.models.additive").joinpath("fit_bam.R"),
        [
            "--train",
            "train.tsv",
            "--test",
            "test.tsv",
            "--out",
            "preds.tsv",
            "--formula",
            formula,
            "--discrete",
            "FALSE",
        ],
        cwd=tmp_path,
    )
    expected = _gam_prediction_reference(
        y,
        train,
        test,
        formula,
        {"use_bam": True, "discrete": False, "method": "fREML"},
        population=True,
    )
    actual = np.loadtxt(tmp_path / "preds.tsv", skiprows=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("mode", ["exclude", "conditional", "shrink_unseen"])
@pytest.mark.parametrize("discrete", [False, True], ids=["bam", "discrete_bam"])
def test_gam_evaluate_matches_direct_predictions_and_statistics(mode, discrete):
    """Evaluation preserves group IDs, baseline selection, sigma and coefficients."""
    from scipy.stats import norm

    from replica.analysis import prepare_observations
    from replica.analysis.cv import FixedSplitter, NoScaler

    dataset = make_toy_dataset(n_stimuli=8, n_words=7, n_participants=6)
    control, focal = UnitPosition(), Surprisal()
    spec = EvaluationSpec(
        targets=[dataset.feature("spr_rt")],
        controls=[control],
        predictors=[ModelFeature(VaryingLM(), focal)],
    )
    baseline = "y ~ s(x0, k=4) + s(x0, participant_id, bs='fs', m=1, k=4)"
    model = RGamModel(
        formula=baseline + " + x1",
        baseline_formula=baseline,
        col_names={"x0": "x0", "x1": "x1"},
        use_bam=True,
        discrete=discrete,
        method="fREML",
        re_predict=mode,
    )
    config = AnalysisConfig(
        design=None,
        model=model,
        cv=FixedSplitter(mask_fn=lambda y, g, p, items: items < 5),
        scaler=NoScaler(),
        transform_dv=lambda y: y,
        per_fold=True,
        retain_per_obs_delta_ll=True,
    )
    obs = prepare_observations(dataset, spec, config)
    X = np.column_stack(
        [
            obs.columns[control.name],
            obs.columns[focal.name],
            obs.participants,
            obs.items,
        ]
    )
    train, test = obs.items < 5, obs.items >= 5
    base_X = X[:, [0, 2, 3]]
    base_pred, base_sigma = model.fit_predict_with_sigma(
        obs.y[train], base_X[train], base_X[test]
    )
    full_pred, full_sigma = model.fit_predict_with_sigma(
        obs.y[train], X[train], X[test]
    )
    stats_pred, beta, se, stats_sigma = model.fit_predict_with_stats(
        obs.y[train], X[train], X[test]
    )
    np.testing.assert_allclose(stats_pred, full_pred, atol=1e-8)
    assert stats_sigma == pytest.approx(full_sigma)
    expected = norm.logpdf(obs.y[test], full_pred, full_sigma) - norm.logpdf(
        obs.y[test], base_pred, base_sigma
    )
    result = evaluate(dataset, spec, config)[0]
    np.testing.assert_allclose(
        result.fold_results[0].per_obs_delta_ll, expected, atol=1e-8
    )
    assert result.coefficient == pytest.approx(beta[2])
    assert result.coefficient_se == pytest.approx(se[2])


# ---------------------------------------------------------------------------
# Random slopes outside the fixed-effect design


def _nested_lrt_data(seed: int = 0):
    """One dataset with two predictors, grouped by participant."""
    rng = np.random.default_rng(seed)
    n, n_subj = 400, 8
    subject = np.repeat(np.arange(n_subj), n // n_subj)
    control = rng.normal(size=n)
    focal = rng.normal(size=n)
    y = 2.0 * control + 0.8 * focal + rng.normal(scale=0.5, size=n)
    item = np.zeros(n)
    reduced = np.column_stack([control, subject, item])
    full = np.column_stack([control, focal, subject, item])
    return y, reduced, full, np.column_stack([control, focal])


def test_re_design_keeps_slopes_a_model_lacks_as_fixed_effects():
    """A nested likelihood-ratio test needs both models to carry the same
    random effects, so the reduced one holds slopes on predictors it does not
    have as fixed effects. Column indices cannot express that: they address
    the design being fit, and that design omits those columns."""
    y, reduced, full, slopes = _nested_lrt_data()

    model = RLmerModel(re_design=slopes, include_item_re=False)
    ll_reduced, _ = model.fit_score(y, reduced, reduced, y, is_full=False)
    ll_full, _ = model.fit_score(y, full, full, y, is_full=True)

    # Adding a real effect cannot lower the maximised likelihood.
    assert ll_full > ll_reduced
    # 2 * delta is the LRT statistic on one degree of freedom.
    assert 2.0 * (ll_full - ll_reduced) > 3.84

    # The slopes are really in the reduced model: dropping them costs
    # likelihood, which an intercept-only random effect would not.
    intercept_only = RLmerModel(include_item_re=False)
    ll_intercept, _ = intercept_only.fit_score(y, reduced, reduced, y, is_full=False)
    assert ll_reduced > ll_intercept


def test_re_design_rejects_a_row_count_it_cannot_align():
    """The columns are supplied whole, so a fold that subsets rows would
    silently misalign them."""
    y, reduced, _, slopes = _nested_lrt_data()

    model = RLmerModel(re_design=slopes[: len(y) // 2], include_item_re=False)
    with pytest.raises(ValueError, match="re_design has 200 rows"):
        model.fit_score(y, reduced, reduced, y, is_full=False)


def test_integrated_loglik_refuses_held_out_rows():
    """The integrated score is the training fit's own likelihood, so summing
    it over k held-out folds would count every row k - 1 times."""
    y, reduced, _, _ = _nested_lrt_data()
    model = RLmerModel(integrated_loglik=True, include_item_re=False)
    with pytest.raises(ValueError, match="FullDataSplitter"):
        model.fit_score(y[:300], reduced[:300], reduced[300:], y[300:], is_full=False)
    loglik, fitted = model.fit_score(y, reduced, reduced, y, is_full=False)
    assert np.isfinite(loglik) and fitted.shape == y.shape


# ---------------------------------------------------------------------------
# Coefficients from a model that owns its held-out likelihood


def test_last_fixed_effects_reports_the_fit_just_performed():
    """The accessor returns the fixed effects of the most recent fit.

    ``fit_score`` fits the baseline design and then the full one, so the
    values left behind belong to the full fit, which is the one whose
    coefficients a caller wants.
    """
    y, reduced, full, _ = _nested_lrt_data()
    model = RLmerModel(integrated_loglik=True, include_item_re=False)

    assert model.last_fixed_effects() is None

    model.fit_score(y, reduced, reduced, y, is_full=False)
    beta_reduced, se_reduced = model.last_fixed_effects()
    model.fit_score(y, full, full, y, is_full=True)
    beta_full, se_full = model.last_fixed_effects()

    # Intercept plus one fixed effect, then plus two.
    assert beta_reduced.shape == (2,)
    assert beta_full.shape == (3,)
    assert se_full.shape == beta_full.shape
    assert (se_full > 0).all()
    # The generator puts a real effect on the second predictor.
    assert beta_full[2] > 0


def test_evaluate_reports_a_coefficient_when_the_model_owns_its_loglik():
    """``AnalysisResult.coefficient`` has a source on the integrated-likelihood path.

    That path never calls ``fit_predict``; the coefficient comes from the
    model's ``last_fixed_effects``, and without it the field keeps its NaN
    default.
    """
    import numpy as np
    import pandas as pd

    from replica.analysis.cv import FullDataSplitter, NoScaler
    from replica.analysis.evaluation.evaluate import evaluate
    from replica.analysis.specification.config import AnalysisConfig
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.dataset.base import TabularDataset, TabularDatasetConfig
    from replica.feature.extractors import PrecomputedFeature
    from replica.studies.base import build_grouped_model

    rng = np.random.default_rng(0)
    n = 240
    frame = pd.DataFrame(
        {
            "subj": np.repeat([f"s{i}" for i in range(6)], n // 6),
            "trial": np.repeat([f"t{i}" for i in range(6)], n // 6),
            "ia": [f"w{i}" for i in range(n)],
            "ianum": list(range(n)),
            "control": rng.normal(size=n),
            "focal": rng.normal(size=n),
        }
    )
    frame["rt"] = 300 + 20 * frame["focal"] + 5 * frame["control"] + rng.normal(size=n)

    dataset = TabularDataset(
        frame,
        TabularDatasetConfig(
            name="synthetic",
            text_col="ia",
            trial_col="trial",
            unit_index_col="ianum",
            measure_col="rt",
            participant_col="subj",
        ),
    )
    model = build_grouped_model(
        frame,
        groupby_col="trial",
        sort_col="ianum",
        columns=["focal", "control"],
        model_name="synthetic",
    )
    spec = EvaluationSpec(
        predictors=[ModelFeature(model, PrecomputedFeature("focal", "focal"))],
        controls=[ModelFeature(model, PrecomputedFeature("control", "control"))],
        targets=[dataset.feature("rt")],
    )
    config = AnalysisConfig(
        design=None,
        model=RLmerModel(integrated_loglik=True, include_item_re=False),
        cv=FullDataSplitter(),
        scaler=NoScaler(),
        transform_dv=lambda x: x,
    )

    analysis = evaluate(dataset, spec, config)[0]
    assert np.isfinite(analysis.coefficient)
    # The generator's slope on the focal predictor, recovered from the fit.
    assert analysis.coefficient == pytest.approx(20.0, rel=0.05)
    assert analysis.coefficient_se > 0


# ---------------------------------------------------------------------------
# RobustCrossedLmer
# ---------------------------------------------------------------------------


def _r_error():
    from rpy2.rinterface_lib.embedded import RRuntimeError

    return RRuntimeError


def _crossed_design(n_participants, n_items, seed=0):
    """Design ``[control, focal, participant, item]``; every participant reads every item once."""
    rng = np.random.default_rng(seed)
    n = n_participants * n_items
    participants = np.repeat(np.arange(n_participants), n_items)
    items = np.tile(np.arange(n_items), n_participants)
    ctrl = rng.standard_normal(n)
    focal = rng.standard_normal(n)
    y = (
        0.5 * ctrl
        + focal
        + 0.3 * rng.standard_normal(n)
        + 0.2 * (participants - participants.mean())
    )
    X = np.column_stack([ctrl, focal, participants, items])
    train = np.arange(n) % 5 != 0
    return y, X, train


def _focal_slope_lmer(uncorrelated):
    return RLmerModel(
        random_slope_cols_subject=[1],
        random_slope_cols_item=[1],
        uncorrelated_slopes_subject=uncorrelated,
        uncorrelated_slopes_item=uncorrelated,
        nlminb_fallback=False,
    )


class TestRobustCrossedLmer:
    def test_primary_structure_is_the_focal_slope_lmer(self):
        y, X, train = _crossed_design(3, 30)
        robust = RobustCrossedLmer(n_controls=1)
        preds, sigma = robust.fit_predict_with_sigma(y[train], X[train], X[~train])
        expected_preds, expected_sigma = _focal_slope_lmer(True).fit_predict_with_sigma(
            y[train], X[train], X[~train]
        )
        np.testing.assert_array_equal(preds, expected_preds)
        assert sigma == expected_sigma
        assert robust.variant_uses == [1, 0, 0]

    def test_baseline_design_keeps_random_intercepts_only(self):
        y, X, train = _crossed_design(3, 30)
        base = [0, 2, 3]  # no focal column, so the slope index is out of range
        robust = RobustCrossedLmer(n_controls=1)
        preds, sigma = robust.fit_predict_with_sigma(
            y[train], X[train][:, base], X[~train][:, base]
        )
        expected_preds, expected_sigma = RLmerModel(
            nlminb_fallback=False
        ).fit_predict_with_sigma(y[train], X[train][:, base], X[~train][:, base])
        np.testing.assert_array_equal(preds, expected_preds)
        assert sigma == expected_sigma
        assert robust.variant_uses == [1, 0, 0]

    def test_falls_back_when_lme4_rejects_the_primary_structure(self):
        # Two participants who each read every item once: a correlated by-item
        # slope has as many random effects as observations, which lme4 refuses.
        y, X, train = _crossed_design(2, 40)
        with pytest.raises(_r_error(), match="number of observations"):
            _focal_slope_lmer(False).fit_predict_with_sigma(
                y[train], X[train], X[~train]
            )
        robust = RobustCrossedLmer(n_controls=1, uncorrelated=False)
        preds, sigma = robust.fit_predict_with_sigma(y[train], X[train], X[~train])
        expected_preds, expected_sigma = _focal_slope_lmer(True).fit_predict_with_sigma(
            y[train], X[train], X[~train]
        )
        np.testing.assert_array_equal(preds, expected_preds)
        assert sigma == expected_sigma
        assert robust.variant_uses == [0, 1, 0]

    def test_fit_predict_and_stats_share_the_fallback(self):
        y, X, train = _crossed_design(2, 40)
        robust = RobustCrossedLmer(n_controls=1, uncorrelated=False)
        preds = robust.fit_predict(y[train], X[train], X[~train])
        preds_stats, beta, se, sigma = robust.fit_predict_with_stats(
            y[train], X[train], X[~train]
        )
        np.testing.assert_array_equal(preds, preds_stats)
        assert len(beta) == len(se) == 3  # intercept, control, focal
        assert sigma > 0
        assert robust.variant_uses == [0, 2, 0]

    def test_raises_the_last_r_error_when_every_variant_fails(self, monkeypatch):
        y, X, train = _crossed_design(3, 30)
        robust = RobustCrossedLmer(n_controls=1)
        for i, variant in enumerate(robust._variants):

            def fail(*args, _i=i):
                raise _r_error()(f"variant {_i} failed")

            monkeypatch.setattr(variant, "fit_predict_with_sigma", fail)
        with pytest.raises(_r_error(), match="variant 2 failed"):
            robust.fit_predict_with_sigma(y[train], X[train], X[~train])
        assert robust.variant_uses == [0, 0, 0]

    def test_python_errors_propagate(self, monkeypatch):
        y, X, train = _crossed_design(3, 30)
        robust = RobustCrossedLmer(n_controls=1)

        def fail(*args):
            raise ValueError("bad design")

        monkeypatch.setattr(robust._variants[0], "fit_predict_with_sigma", fail)
        with pytest.raises(ValueError, match="bad design"):
            robust.fit_predict_with_sigma(y[train], X[train], X[~train])
        assert robust.variant_uses == [0, 0, 0]

    def test_evaluate_end_to_end(self):
        ds = make_toy_dataset(n_stimuli=6, n_words=6, n_participants=3)
        robust = RobustCrossedLmer(n_controls=1)
        results = evaluate(
            ds,
            EvaluationSpec(
                targets=[ds.feature("spr_rt")],
                predictors=[ModelFeature(VaryingLM(), Surprisal())],
                controls=[UnitPosition()],
            ),
            AnalysisConfig(design=None, transform_dv=np.log2, model=robust, n_folds=3),
        )
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)
        # One baseline fit and one full fit per fold, all on the primary structure.
        assert robust.variant_uses == [6, 0, 0]
