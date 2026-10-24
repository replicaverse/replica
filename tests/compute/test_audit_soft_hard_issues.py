"""Audit-issue classification: soft (advisory) vs hard (quarantine)."""

import math
import pickle
from pathlib import Path

import pytest

from replica.compute.validation import (
    PredictorPickleReport,
    predictor_pickle_issues,
)
from replica.feature import FeatureModelOutput


def _make_pickle(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    return p


def _outs(values: list[float], key: str = "mc_word_entropy"):
    return [FeatureModelOutput(values={key: v}) for v in values]


def test_clean_pickle_no_issues(tmp_path):
    payload = {f"s{i}": _outs([3.0, 4.0, 5.0]) for i in range(40)}
    pkl = _make_pickle(tmp_path, "clean.pkl", payload)
    r = predictor_pickle_issues(pkl)
    assert r.is_clean
    assert not r.has_hard_issues
    assert not r.soft_only


def test_mean_outlier_is_soft_only(tmp_path):
    payload = {f"s{i}": _outs([8.0, 8.0, 8.0]) for i in range(40)}
    payload["bad_short_stim"] = _outs([25.0])
    pkl = _make_pickle(tmp_path, "soft.pkl", payload)
    r = predictor_pickle_issues(pkl)
    assert "mean_outlier" in r.issues, r.issues
    assert not r.is_clean
    assert r.soft_only
    assert not r.has_hard_issues


def test_extreme_per_word_is_advisory(tmp_path):
    payload = {f"s{i}": _outs([3.0, 4.0]) for i in range(40)}
    payload["spike_stim"] = _outs([500.0, 4.0])
    pkl = _make_pickle(tmp_path, "hard.pkl", payload)
    r = predictor_pickle_issues(pkl)
    assert "extreme_per_word" in r.issues
    assert r.soft_only


def test_all_nan_stim_is_advisory_when_other_stimuli_have_values(tmp_path):
    payload = {f"s{i}": _outs([3.0, 4.0]) for i in range(40)}
    payload["dead_stim"] = _outs([math.nan, math.nan])
    pkl = _make_pickle(tmp_path, "nan.pkl", payload)
    r = predictor_pickle_issues(pkl)
    assert "all_nan_stim" in r.issues
    assert r.soft_only


def test_single_unit_nan_stim_is_soft(tmp_path):
    """A one-unit stimulus whose only value is NaN warns without quarantine."""
    payload = {f"s{i}": _outs([3.0, 4.0], key="unigram_surprisal") for i in range(40)}
    payload["lone_word"] = _outs([math.nan], key="unigram_surprisal")
    pkl = _make_pickle(tmp_path, "wordfreq_unigram_surprisal_celer.pkl", payload)
    r = predictor_pickle_issues(pkl)
    assert "single_unit_nan" in r.issues
    assert "all_nan_stim" not in r.issues, r.issues
    assert not r.has_hard_issues
    assert r.soft_only


def test_summary_line_tags_soft_vs_hard(tmp_path):
    soft = PredictorPickleReport(path=Path("x.pkl"), n_stims=10)
    soft.add("mean_outlier", "s1: ...")
    assert "[soft-only]" in soft.summary_line()

    hard = PredictorPickleReport(path=Path("y.pkl"), n_stims=10)
    hard.add("invalid_value_domain", "s1: ...")
    assert "[HARD]" in hard.summary_line()


def test_magnitude_diagnostics_do_not_depend_on_the_filename(tmp_path):
    payload = {"s": _outs([500.0, 4.0])}
    reports = [
        predictor_pickle_issues(_make_pickle(tmp_path, name, payload))
        for name in (
            "cell.pkl",
            "cell_a0.5.pkl",
            "cell_a1.pkl",
            "cell_L15.pkl",
            "cell_M30.pkl",
        )
    ]
    assert all(report.issues == reports[0].issues for report in reports)
    assert all(report.soft_only for report in reports)
    path = _make_pickle(tmp_path, "override.pkl", payload)
    assert predictor_pickle_issues(path, extreme_value_threshold=None).is_clean
    assert predictor_pickle_issues(path, extreme_value_threshold=600).is_clean


@pytest.mark.parametrize(
    "key,value,bucket",
    [
        ("log_p_observed", 0.1, "invalid_value_domain"),
        ("mc_word_entropy", -0.1, "invalid_value_domain"),
        ("surprisal", -0.1, "invalid_value_domain"),
        ("unigram_surprisal", -float("inf"), "invalid_value_domain"),
        ("mc_word_entropy", float("inf"), "nonfinite_value"),
    ],
)
def test_impossible_values_are_hard_independently_of_magnitude(
    tmp_path, key, value, bucket
):
    path = _make_pickle(tmp_path, "cell_a0.5.pkl", {"s": _outs([value], key)})
    report = predictor_pickle_issues(path, extreme_value_threshold=None)
    assert bucket in report.issues and report.has_hard_issues


@pytest.mark.parametrize(
    "key,value",
    [
        ("log_p_observed", -math.inf),
        ("unigram_surprisal", math.inf),
        ("surprisal", math.inf),
        ("similarity_adjusted_surprisal_pos", math.inf),
    ],
)
def test_zero_probability_logarithms_are_valid_extended_real_values(
    tmp_path, key, value
):
    path = _make_pickle(tmp_path, "zero_mass.pkl", {"s": _outs([value], key)})
    assert predictor_pickle_issues(path).is_clean


def test_infovalue_first_word_nan_is_definitional(tmp_path):
    """Word-infoval's first unit is NaN by definition; short stimuli with only
    that NaN (plus one documented failed word) must not hard-flag, while a
    genuinely all-NaN longer stimulus still does."""
    nan = math.nan
    payload = {
        f"s{i}": _outs([nan, 3.0, 4.0], key="infovalue_layer24") for i in range(40)
    }
    payload["one_word"] = _outs([nan], key="infovalue_layer24")
    payload["two_word_failed"] = _outs([nan, nan], key="infovalue_layer24")
    payload["three_word_one_failed"] = _outs([nan, 3.0, nan], key="infovalue_layer24")
    pkl = _make_pickle(
        tmp_path, "ai-forever_mGPT_word_infoval_N5_M3_s0_celer.pkl", payload
    )
    r = predictor_pickle_issues(pkl)
    assert "all_nan_stim" not in r.issues, r.issues
    assert "nan_burst" not in r.issues, r.issues

    payload["corrupt"] = _outs([nan, nan, nan, nan, nan], key="infovalue_layer24")
    pkl = _make_pickle(
        tmp_path, "ai-forever_mGPT_word_infoval_N5_M3_s0_provo.pkl", payload
    )
    r = predictor_pickle_issues(pkl)
    assert "all_nan_stim" in r.issues


def test_non_infovalue_all_nan_still_flags(tmp_path):
    """The definitional exemption is scoped: other keys keep the strict check."""
    payload = {f"s{i}": _outs([3.0, 4.0]) for i in range(40)}
    payload["dead"] = _outs([math.nan, math.nan])
    pkl = _make_pickle(
        tmp_path, "gpt2_mcword_entropy_N512_M30_a1_s0_provo.pkl", payload
    )
    r = predictor_pickle_issues(pkl)
    assert "all_nan_stim" in r.issues


@pytest.mark.parametrize(
    "options",
    [
        {"extreme_value_threshold": 0},
        {"extreme_value_threshold": math.nan},
        {"outlier_sigma": 0},
        {"nan_burst_fraction": -0.1},
        {"nan_burst_fraction": math.nan},
    ],
)
def test_invalid_diagnostic_thresholds_raise(tmp_path, options):
    path = _make_pickle(tmp_path, "cell.pkl", {"s": _outs([1.0])})
    with pytest.raises(ValueError):
        predictor_pickle_issues(path, **options)
