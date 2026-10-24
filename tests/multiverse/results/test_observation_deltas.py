"""Per-observation ΔLL sidecar store: paths, persistence, corruption."""

import numpy as np
import pytest

from replica.analysis.contracts.results import AnalysisResult, FoldResult
from replica.multiverse.results import observation_deltas
from replica.multiverse.results.analysis_results import row_from_analysis
from replica.multiverse.spec import DatasetCell


def test_save_skips_none_and_empty(tmp_path):
    target = tmp_path / "a.npy"
    observation_deltas.save(target, None)
    observation_deltas.save(target, [])
    assert not target.exists()


def test_save_load_round_trip_flattens_to_float(tmp_path):
    target = tmp_path / "a.npy"
    observation_deltas.save(target, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(
        observation_deltas.load(target), np.array([1.0, 2.0, 3.0, 4.0])
    )


def test_load_absent_returns_none(tmp_path):
    assert observation_deltas.load(tmp_path / "missing.npy") is None


def test_load_corrupt_sidecar_warns_and_returns_none(tmp_path, caplog):
    corrupt = tmp_path / "bad.npy"
    corrupt.write_bytes(b"not a numpy file")

    with caplog.at_level("WARNING", logger="replica"):
        assert observation_deltas.load(corrupt) is None

    assert "failed to load" in caplog.text


def test_slug_keeps_predictor_chars_and_replaces_the_rest():
    assert observation_deltas._slug("mc=0.5+a.b/c d") == "mc=0.5+a.b_c_d"
    assert observation_deltas._slug(None) == ""


def test_grid_path_prefers_env_then_module_attr(tmp_path, monkeypatch):
    monkeypatch.delenv("PSYCHO_PER_OBS_GRID_DIR", raising=False)
    monkeypatch.setattr(observation_deltas, "GRID_PER_OBS_DIR", tmp_path / "attr")
    assert observation_deltas.grid_per_obs_path("p", "c", "x", 3, "dv") == (
        tmp_path / "attr" / "p__c__x__s3__dv.npy"
    )

    monkeypatch.setenv("PSYCHO_PER_OBS_GRID_DIR", str(tmp_path / "env"))
    assert observation_deltas.grid_per_obs_path("p", "c", "x", 3, "dv") == (
        tmp_path / "env" / "p__c__x__s3__dv.npy"
    )


def test_master_path_is_keyed_on_dv_and_variant(monkeypatch, tmp_path):
    monkeypatch.setenv("PSYCHO_PER_OBS_MASTER_DIR", str(tmp_path))
    path = observation_deltas.master_per_obs_path(
        "gpt2", "surprisal", "provo", "ols", "first_run"
    )
    assert path == tmp_path / "gpt2__surprisal__provo__ols__first_run.npy"
    paths = {
        observation_deltas.master_per_obs_path(
            "gpt2", "surprisal", "provo", "ols", dv, v
        )
        for dv in ("first_run", "total_time")
        for v in ("", "skips_removed", "skips_as_zero")
    }
    assert len(paths) == 6


def _analysis(per_obs):
    fold = FoldResult(
        fold_index=0,
        n_test=2,
        baseline_ll=-10.0,
        full_ll=-8.0,
        delta_ll=2.0,
        baseline_r2=0.1,
        full_r2=0.2,
        delta_r2=0.1,
        per_obs_delta_ll=None if per_obs is None else np.asarray(per_obs),
    )
    return AnalysisResult(
        dataset_name="provo",
        model_name="gpt2",
        feature_name="surprisal",
        metrics={"delta_ll": 2.0},
        n_observations=2,
        fold_results=[fold],
    )


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """No source checkout, as in an installed wheel; returns the compute dir."""
    monkeypatch.setattr("replica.paths.source_checkout", lambda: None)
    monkeypatch.setenv("PSYCHO_COMPUTE_DIR", str(tmp_path / "compute"))
    monkeypatch.delenv("PSYCHO_PER_OBS_MASTER_DIR", raising=False)
    return tmp_path / "compute"


def test_row_without_per_obs_values_needs_no_checkout(installed):
    cell = DatasetCell("provo", "first_run", "et")
    row = row_from_analysis(cell, "surprisal", "ols", _analysis(None), None, lm="gpt2")
    assert row["delta_ll"] == 2.0
    assert not installed.exists()


def test_rows_write_the_explicit_sidecar_for_each_dv_and_variant(installed):
    cells = [
        ("first_run", "skips_removed", [1.0, 1.5]),
        ("total_time", "skips_removed", [2.0, 2.5]),
        ("total_time", "skips_as_zero", [3.0, 3.5]),
    ]
    for dv, variant, per_obs in cells:
        row_from_analysis(
            DatasetCell("provo", dv, "et"),
            "surprisal",
            "ols",
            _analysis(per_obs),
            None,
            variant=variant,
            per_observation_path=observation_deltas.master_per_obs_path(
                "gpt2", "surprisal", "provo", "ols", dv, variant
            ),
            lm="gpt2",
        )
    stored = {
        path.name: observation_deltas.load(path).tolist()
        for path in (installed / "per_obs" / "master").iterdir()
    }
    assert stored == {
        f"gpt2__surprisal__provo__ols__{dv}__{variant}.npy": per_obs
        for dv, variant, per_obs in cells
    }
