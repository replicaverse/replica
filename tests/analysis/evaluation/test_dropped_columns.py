"""Coefficients stay attached to their design columns when a backend drops one.

lm, lme4 and lme-rs remove a column that depends linearly on earlier ones. The
reported coefficients still index the design, intercept first, with NaN at the
dropped column, so ``evaluate()`` reads each feature's own slope.
"""

import numpy as np
import pandas as pd
import pytest

from replica.analysis.models import RustLmerModel


def _collinear(seed=0, n=200):
    """``[c0, 2 * c0, f0, f1]`` with slopes 1, (aliased), 3 and -2."""
    rng = np.random.default_rng(seed)
    c0, f0, f1 = rng.normal(size=(3, n))
    y = 1.0 + c0 + 3.0 * f0 - 2.0 * f1 + 0.1 * rng.normal(size=n)
    return y, np.column_stack([c0, 2.0 * c0, f0, f1])


def _assert_in_design_order(beta, se):
    assert beta.shape == se.shape == (5,)
    assert np.isnan(beta[2]) and np.isnan(se[2])
    assert np.isfinite(beta[[0, 1, 3, 4]]).all()
    np.testing.assert_allclose(beta[[3, 4]], [3.0, -2.0], atol=0.05)


def _grouped(X, n_participants=10):
    participant = np.arange(len(X)) % n_participants
    item = np.arange(len(X)) // n_participants
    return np.column_stack([X, participant, item])


@pytest.mark.needs_r
def test_lm_reports_coefficients_in_design_order():
    from replica.analysis.models import RLinearModel

    y, X = _collinear()
    _, beta, se, _ = RLinearModel().fit_predict_with_stats(y, X, X)
    _assert_in_design_order(beta, se)


@pytest.mark.needs_r
def test_lmer_reports_coefficients_in_design_order():
    from replica.analysis.models import RLmerModel

    y, X = _collinear()
    Xg = _grouped(X)
    _, beta, se, _ = RLmerModel().fit_predict_with_stats(y, Xg, Xg)
    _assert_in_design_order(beta, se)
    model = RLmerModel(integrated_loglik=True)
    model.fit_score(y, Xg, Xg, y, is_full=True)
    _assert_in_design_order(*model.last_fixed_effects())


def test_lme_rs_least_squares_path_reports_coefficients_in_design_order():
    """One participant and one item leave no random term, so the least-squares
    path runs without the lme-rs extension."""
    y, X = _collinear()
    Xg = np.column_stack([X, np.zeros(len(X)), np.zeros(len(X))])
    model = RustLmerModel()
    preds, beta, se, _ = model.fit_predict_with_stats(y, Xg, Xg)
    _assert_in_design_order(beta, se)
    _assert_in_design_order(*model.last_fixed_effects())
    independent = Xg[:, [0, 2, 3, 4, 5]]
    reduced = RustLmerModel().fit_predict_with_stats(y, independent, independent)
    np.testing.assert_allclose(preds, reduced[0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(beta[[0, 1, 3, 4]], reduced[1], rtol=0, atol=1e-10)
    # The integrated-likelihood path reports the fit it just performed.
    integrated = RustLmerModel(integrated_loglik=True)
    integrated.fit_score(y, Xg, Xg, y, is_full=True)
    _assert_in_design_order(*integrated.last_fixed_effects())


@pytest.mark.needs_r
def test_evaluate_reads_each_feature_its_own_slope():
    """A dropped control would otherwise shift every later coefficient onto
    the wrong feature."""
    from replica.analysis.cv import FullDataSplitter, NoScaler
    from replica.analysis.evaluation.evaluate import evaluate
    from replica.analysis.models import RLinearModel
    from replica.analysis.specification.config import AnalysisConfig
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.dataset.base import TabularDataset, TabularDatasetConfig
    from replica.feature import ModelFeature
    from replica.feature.extractors import PrecomputedFeature
    from replica.studies.base import build_grouped_model

    y, X = _collinear(n=240)
    columns = ["c0", "c1", "f0", "f1"]
    frame = pd.DataFrame(X, columns=columns)
    frame["subj"] = "s0"
    frame["trial"] = np.repeat([f"t{i}" for i in range(6)], 40)
    frame["ia"] = [f"w{i}" for i in range(len(frame))]
    frame["ianum"] = np.arange(len(frame))
    frame["rt"] = y
    dataset = TabularDataset(
        frame,
        TabularDatasetConfig(
            name="collinear",
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
        columns=columns,
        model_name="collinear",
    )
    spec = EvaluationSpec(
        predictors=[
            ModelFeature(model, PrecomputedFeature("f0", "f0")),
            ModelFeature(model, PrecomputedFeature("f1", "f1")),
        ],
        controls=[
            ModelFeature(model, PrecomputedFeature("c0", "c0")),
            ModelFeature(model, PrecomputedFeature("c1", "c1")),
        ],
        targets=[dataset.feature("rt")],
    )
    config = AnalysisConfig(
        design=None,
        model=RLinearModel(),
        cv=FullDataSplitter(),
        scaler=NoScaler(),
        transform_dv=lambda values: values,
    )
    analysis = evaluate(dataset, spec, config)[0]
    assert analysis.coefficients["f0"].estimate == pytest.approx(3.0, abs=0.05)
    assert analysis.coefficients["f1"].estimate == pytest.approx(-2.0, abs=0.05)
