"""Current-fit diagnostic snapshots preserve facts and discard stale state."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from statsmodels.tools.sm_exceptions import MissingDataError

from replica.analysis.models.fixed_effects.ols import OLS
from replica.analysis.models.fixed_effects.statsmodels import StatsmodelsOLS
from replica.analysis.models.mixed_effects import lmer_rs
from replica.analysis.models.support.diagnostics import _r_fit_diagnostics


@pytest.mark.parametrize("model_type", [OLS, StatsmodelsOLS])
def test_rank_snapshot_changes_with_design_and_failed_fit_clears_it(model_type):
    rng = np.random.default_rng(9)
    x = rng.normal(size=30)
    y = 2 + x + rng.normal(size=30)
    model = model_type()
    duplicate = np.column_stack([x, x])
    model.fit_predict(y, duplicate, duplicate[:3])
    first = model.last_fit_diagnostics()
    assert {k: first[k] for k in ("convergence_applicable", "rank", "n_columns")} == {
        "convergence_applicable": False,
        "rank": 2,
        "n_columns": 3,
    }
    assert first["coefficients"]["names"] == ["(Intercept)", "x0", "x1"]
    np.testing.assert_allclose(
        np.column_stack([np.ones(3), duplicate[:3]])
        @ first["coefficients"]["estimates"],
        model.fit_predict(y, duplicate, duplicate[:3]),
    )
    model.fit_predict(y, x[:, None], x[:3, None])
    assert model.last_fit_diagnostics()["n_columns"] == 2
    assert first["n_columns"] == 3
    first["rank"] = -1
    assert model.last_fit_diagnostics()["rank"] == 2
    invalid = np.zeros((len(y) - 1, 1))
    with pytest.raises((ValueError, MissingDataError)):
        model.fit_predict(y, invalid, invalid[:3])
    assert model.last_fit_diagnostics() is None


def test_rust_mixed_fit_diagnostics_do_not_survive_direct_fallback(monkeypatch):
    rng = np.random.default_rng(4)
    x = rng.normal(size=60)
    y = 1 + x + rng.normal(scale=0.2, size=len(x))
    mixed = np.column_stack([x, np.arange(60) % 4, np.arange(60) % 5])
    fit = SimpleNamespace(
        coefficients=[1.0, 1.0],
        std_errors=[0.2, 0.2],
        sigma2=0.04,
        theta=[1.0, 1.0],
        converged=True,
        iterations=7,
        diagnostics={"warnings": [], "native_marker": "current"},
    )
    monkeypatch.setattr(
        lmer_rs, "import_lme_python", lambda: SimpleNamespace(lmer=lambda *a, **k: fit)
    )
    model = lmer_rs.RustLmerModel()
    monkeypatch.setattr(model, "_control", lambda lme: None)
    model.fit_predict_with_stats(y, mixed, mixed[:3])
    first = model.last_fit_diagnostics()
    assert first["converged"] is True and first["details"]["n_iterations"] == 7
    first["details"]["native_marker"] = "mutated"
    assert model.last_fit_diagnostics()["details"]["native_marker"] == "current"
    direct = np.column_stack([x, np.zeros((len(x), 2))])
    predictions, *_ = model.fit_predict_with_stats(y, direct, direct[:3])
    np.testing.assert_allclose(
        predictions, OLS().fit_predict(y, x[:, None], x[:3, None])
    )
    assert {
        k: model.last_fit_diagnostics()[k]
        for k in ("convergence_applicable", "rank", "n_columns", "details")
    } == {
        "convergence_applicable": False,
        "rank": 2,
        "n_columns": 2,
        "details": {"fallback": "no_random_terms"},
    }
    assert model.last_diagnostics == {} and model.last_singular is None


def test_optional_r_telemetry_failure_does_not_change_fit_acceptance():
    def missing_slot(expression):
        raise RuntimeError("optional backend slot is unavailable")

    diagnostics = _r_fit_diagnostics(SimpleNamespace(r=missing_slot), ".lmer_model", 3)
    assert diagnostics == {
        "details": {"diagnostics_unavailable": "optional backend slot is unavailable"}
    }
    assert "converged" not in diagnostics


@pytest.mark.parametrize("scaler_name", ["ZScoreScaler", "MinMaxScaler", "NoScaler"])
def test_scaler_snapshot_uses_training_statistics_and_is_detached(scaler_name):
    from replica.analysis.cv import scalers

    scaler = getattr(scalers, scaler_name)()
    train = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    test = np.array([[101.0, 9.0]])
    transformed, held_out = scaler.fit_transform(train, test)
    snapshot = scaler.last_fit_diagnostics()
    center = np.array(snapshot["center"])
    scale = np.array(snapshot["effective_scale"])
    np.testing.assert_array_equal(transformed, (train - center) / scale)
    np.testing.assert_array_equal(held_out, (test - center) / scale)
    if scaler_name == "ZScoreScaler":
        assert center.tolist() == [3.0, 5.0]
        assert snapshot["scale"][1] == 0.0 and scale[1] == 1.0
    elif scaler_name == "MinMaxScaler":
        assert center.tolist() == [1.0, 5.0] and scale.tolist() == [4.0, 1.0]
    else:
        assert transformed is train and held_out is test
    snapshot["center"][0] = -999
    assert scaler.last_fit_diagnostics()["center"][0] != -999
    scaler.fit_transform(train[:, :1], test[:, :1])
    assert scaler.last_fit_diagnostics()["n_columns"] == 1
    if scaler_name != "NoScaler":
        for train_input, test_input in ((train + 2j, test), (train, test + 2j)):
            with pytest.raises(ValueError, match="real-valued"):
                scaler.fit_transform(train_input, test_input)
            assert scaler.last_fit_diagnostics() is None
        _, held_out = scaler.fit_transform(
            train, np.ma.array(test, mask=[[True, False]])
        )
        assert np.isnan(held_out[0, 0])
        assert held_out[0, 1] == (test[0, 1] - center[1]) / scale[1]


def test_ols_snapshot_copies_returned_statistics_and_tracks_scale():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(30, 2))
    y = 2 + x @ np.array([0.5, -0.7]) + rng.normal(size=30)
    model = OLS()
    predictions, beta, se, sigma = model.fit_predict_with_stats(y, x, x[:5])
    snapshot = model.last_fit_diagnostics()
    np.testing.assert_array_equal(snapshot["coefficients"]["estimates"], beta)
    np.testing.assert_array_equal(snapshot["coefficients"]["standard_errors"], se)
    assert snapshot["residual_scale"]["sigma"] == sigma
    np.testing.assert_allclose(predictions, np.column_stack([np.ones(5), x[:5]]) @ beta)
    beta[:] = 999
    se[:] = 999
    assert model.last_fit_diagnostics() == snapshot
    model.fit_predict(y, x[:, :1], x[:5, :1])
    assert "residual_scale" not in model.last_fit_diagnostics()
    assert len(model.last_fit_diagnostics()["coefficients"]["estimates"]) == 2


def test_optional_statsmodels_uncertainty_does_not_reject_saturated_prediction():
    model = StatsmodelsOLS()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predictions = model.fit_predict(
            np.array([1.0, 2.0]), np.array([[0.0], [1.0]]), np.array([[0.5]])
        )
    np.testing.assert_allclose(predictions, [1.5])
    diagnostics = model.last_fit_diagnostics()
    assert diagnostics["coefficients"]["standard_errors"] == [None, None]
    assert diagnostics["residual_scale"]["sigma"] is None
    assert diagnostics["residual_scale"]["variance"] is None
