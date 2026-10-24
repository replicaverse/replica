"""Comparison identity and boundary validation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from replica import OLS, AnalysisConfig, EvaluationSpec, RowSplitter, evaluate
from replica.analysis import AnalysisResult, NoScaler
from replica.dataset import TabularDataset, TabularDatasetConfig
from replica.feature import (
    FeatureModelOutput,
    ModelFeature,
    PrecomputedFeature,
    PrecomputedFeatureModel,
)


def _problem():
    rng = np.random.default_rng(781)
    signal, noise = rng.normal(size=(2, 200))
    frame = pd.DataFrame(
        dict(
            trialid=np.arange(200),
            ianum=0,
            ia="word",
            rt=5 * signal + rng.normal(size=200),
        )
    )
    ds = TabularDataset(frame, TabularDatasetConfig(name="toy", measure_col="rt"))
    model = PrecomputedFeatureModel(
        "toy",
        {
            f"trial_{i}": [
                FeatureModelOutput(values={"signal": signal[i], "noise": noise[i]})
            ]
            for i in range(200)
        },
    )
    spec = EvaluationSpec(
        targets=[ds.feature(ds.default_measure)],
        predictors=[
            ModelFeature(model, PrecomputedFeature("signal", "signal")),
            ModelFeature(model, PrecomputedFeature("noise", "noise")),
        ],
    )
    config = AnalysisConfig(
        model=OLS(),
        design=None,
        cv=RowSplitter(5),
        scaler=NoScaler(),
        transform_dv=lambda y: y,
        per_fold=True,
    )
    return ds, spec, config


def test_joint_result_names_the_added_set_and_each_coefficient():
    ds, spec, cfg = _problem()
    (joint,) = evaluate(ds, spec, cfg)
    assert joint.feature_name == "joint(signal, noise)"
    assert joint.comparison.added == ("signal", "noise")
    assert joint.comparison.baseline == ()
    assert joint.n_scored == 200
    assert abs(joint.coefficients["signal"].estimate - 5) < 0.3
    assert abs(joint.coefficients["noise"].estimate) < 0.3
    separate = evaluate(ds, replace(spec, comparison="individual"), cfg)
    assert separate[0].delta_ll > 300
    assert separate[1].delta_ll < 0
    for feature, result in zip(spec.predictors, separate, strict=True):
        (alone,) = evaluate(ds, replace(spec, predictors=[feature]), cfg)
        assert alone.metrics == result.metrics


@pytest.mark.parametrize("folds", [0, 1, -1])
def test_invalid_fold_counts_fail_at_configuration(folds):
    with pytest.raises(ValueError, match="at least 2"):
        RowSplitter(folds)


def test_no_scored_observations_is_an_error():
    ds, spec, cfg = _problem()

    class Empty:
        def split(self, y, *args):
            return [(np.zeros(len(y), dtype=bool), np.ones(len(y), dtype=bool))]

    with pytest.raises(ValueError, match="No observations were scored"):
        evaluate(ds, spec, replace(cfg, cv=Empty()))


def test_absent_metrics_are_unavailable():
    assert np.isnan(AnalysisResult("d", "m", "f").delta_ll)


def test_duplicate_feature_names_rejected_at_ingress():
    ds, spec, cfg = _problem()
    spec.controls.append(spec.predictors[0])
    with pytest.raises(ValueError, match="unique"):
        evaluate(ds, spec, cfg)
