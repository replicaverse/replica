"""Descriptive row formatting and explicit inference/output contracts."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from replica.dataset.measures import FIRST_PASS_RUN
from replica.multiverse.results.analysis_results import TSV_COLUMNS, row_from_analysis
from replica.multiverse.spec import DatasetCell
from tests.multiverse._analysis_fixtures import _analysis, _dataset


def test_row_has_all_columns():
    analysis = _analysis()
    analysis.response = {"measure": "first_run", "source": "released"}
    row = row_from_analysis(
        DatasetCell("dummy", FIRST_PASS_RUN, "et"),
        "surprisal",
        "ols",
        analysis,
        _dataset(),
        lm="gpt2",
    )
    assert set(TSV_COLUMNS) <= set(row)
    assert json.loads(row["response"]) == analysis.response


def test_cohens_f2_from_partial_and_full_r2():
    row = row_from_analysis(
        DatasetCell("dummy", FIRST_PASS_RUN, "et"),
        "surprisal",
        "ols",
        _analysis(),
        _dataset(),
        lm="gpt2",
    )
    # partial_r2 = 0.20 - 0.18 = 0.02; f² = 0.02 / (1 - 0.20) = 0.025.
    assert row["cohens_f2"] == pytest.approx(0.025)


def test_dataset_metadata_falls_back_to_none():
    row = row_from_analysis(
        DatasetCell("dummy", FIRST_PASS_RUN, "et"),
        "surprisal",
        "ols",
        _analysis(),
        SimpleNamespace(),  # neither stimuli() nor participants
        lm="gpt2",
    )
    assert row["n_items"] is None
    assert row["n_participants"] is None


def test_default_row_performs_no_inference_or_sidecar_write(tmp_path, monkeypatch):
    from replica.analysis.inference import significance
    from replica.multiverse.results import observation_deltas

    def unexpected(*args, **kwargs):
        raise AssertionError("descriptive formatting must not infer or write")

    monkeypatch.setattr(significance, "paired_permutation_pvalue", unexpected)
    monkeypatch.setattr(significance, "per_datapoint_permutation_pvalue", unexpected)
    monkeypatch.setattr(observation_deltas, "save", unexpected)
    analysis = _analysis()
    analysis.fold_results[0].per_obs_delta_ll = np.array([0.25, 0.75])
    row = row_from_analysis(
        DatasetCell("dummy", FIRST_PASS_RUN, "et"),
        "surprisal",
        "custom",
        analysis,
        _dataset(),
        lm="caller/model",
    )
    assert row["lm"] == "caller/model"
    assert row["delta_ll"] == 50.0
    for name in (
        "pvalue",
        "pvalue_per_datapoint",
        "fold_dll_per_unit_ci_lo",
        "fold_dll_per_unit_ci_hi",
    ):
        assert math.isnan(row[name])
    assert not list(tmp_path.iterdir())


def test_explicit_inference_and_sidecar_use_caller_choices(tmp_path):
    analysis = _analysis()
    analysis.fold_results[0].per_obs_delta_ll = np.array([0.25, 0.75])
    seen = []

    def infer(value):
        seen.append(value)
        return {"pvalue": 0.125, "fold_dll_per_unit_ci_lo": -1.0}

    sidecar = tmp_path / "chosen" / "observations.npy"
    row = row_from_analysis(
        DatasetCell("dummy", FIRST_PASS_RUN, "et"),
        "surprisal",
        "custom",
        analysis,
        _dataset(),
        lm="caller/model",
        inference=infer,
        per_observation_path=sidecar,
    )
    assert seen == [analysis]
    assert row["pvalue"] == 0.125
    assert row["fold_dll_per_unit_ci_lo"] == -1.0
    assert math.isnan(row["pvalue_per_datapoint"])
    assert row["delta_ll"] == 50.0
    np.testing.assert_array_equal(np.load(sidecar), [0.25, 0.75])


@pytest.mark.parametrize("summary", [{"lm": "guessed"}, {"delta_ll": 42.0}, None])
def test_invalid_inference_fails_before_output(tmp_path, summary):
    sidecar = tmp_path / "invalid.npy"
    analysis = _analysis()
    analysis.fold_results[0].per_obs_delta_ll = np.array([1.0])
    with pytest.raises((ValueError, TypeError), match="inference"):
        row_from_analysis(
            DatasetCell("dummy", FIRST_PASS_RUN, "et"),
            "surprisal",
            "custom",
            analysis,
            _dataset(),
            lm="caller/model",
            inference=lambda _: summary,
            per_observation_path=sidecar,
        )
    assert not sidecar.exists()


@pytest.mark.parametrize("model_id", [None, "", "   ", 123])
def test_model_identity_is_required_before_inference_or_output(tmp_path, model_id):
    def unexpected(_):
        raise AssertionError("invalid identity must fail before inference")

    sidecar = tmp_path / "invalid.npy"
    with pytest.raises(ValueError, match="nonempty model identity"):
        row_from_analysis(
            DatasetCell("dummy", FIRST_PASS_RUN, "et"),
            "surprisal",
            "custom",
            _analysis(),
            _dataset(),
            lm=model_id,
            inference=unexpected,
            per_observation_path=sidecar,
        )
    assert not sidecar.exists()


def test_model_identity_has_no_default():
    with pytest.raises(TypeError, match="lm"):
        row_from_analysis(
            DatasetCell("dummy", FIRST_PASS_RUN, "et"),
            "surprisal",
            "custom",
            _analysis(),
            _dataset(),
        )
