"""Malformed predictors are quarantined; distribution diagnostics remain advisory."""

import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from replica.artifacts import manifest_path
from replica.compute import runner
from replica.compute.validation import predictor_pickle_issues
from replica.feature import FeatureModelOutput

NAN = math.nan


def _pickle(path: Path, payload: dict) -> Path:
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def _rows(values: list[float], key: str = "log_p_observed") -> list:
    return [FeatureModelOutput(values={key: v}) for v in values]


def _corpus(key: str = "log_p_observed", n: int = 8) -> dict:
    return {f"s{i}": _rows([-1.0, -2.0, -3.0], key) for i in range(n)}


@pytest.mark.parametrize(
    ("payload", "bucket"),
    [
        ({f"s{i}": _rows([NAN, NAN]) for i in range(6)}, "all_nan_pickle"),
        ({f"s{i}": _rows([NAN]) for i in range(6)}, "all_nan_pickle"),
        ({**_corpus(), "inf": _rows([-1.0, math.inf])}, "invalid_value_domain"),
        ({**_corpus(), "positive": _rows([-1.0, 0.5])}, "invalid_value_domain"),
    ],
    ids=[
        "every_value_nan",
        "every_one_unit_stimulus_nan",
        "inf",
        "positive_log_probability",
    ],
)
def test_structural_damage_is_a_hard_issue(tmp_path, payload, bucket):
    report = predictor_pickle_issues(
        _pickle(tmp_path / "gpt2_surprisal_provo.pkl", payload)
    )
    assert bucket in report.issues, report.issues
    assert report.has_hard_issues


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("gpt2_surprisal_nobos_provo.pkl", {**_corpus(), "one": _rows([NAN])}),
        (
            "gpt2_surprisal_tokens_nobos_provo.pkl",
            {**_corpus(), "one": _rows([NAN]), "two": _rows([NAN, -1.0])},
        ),
        (
            "gpt2_word_infoval_N64_M10_provo.pkl",
            {f"s{i}": _rows([NAN, 1.0, 2.0], "infovalue_layer12") for i in range(6)},
        ),
        (
            "gpt2_token_renyi_entropy_a0.5_provo.pkl",
            {
                f"s{i}": _rows([1e-6, 0.0, 3e-4, 250.0], "token_renyi_entropy_alpha0.5")
                for i in range(6)
            },
        ),
    ],
    ids=["one_unit_nan", "first_token_nan", "infovalue_first_unit", "renyi_tail"],
)
def test_values_the_estimator_defines_are_published(tmp_path, filename, payload):
    report = predictor_pickle_issues(_pickle(tmp_path / filename, payload))
    assert not report.has_hard_issues, report.issues


def test_a_mean_outlier_never_quarantines(tmp_path):
    payload = {**{f"s{i}": _rows([-8.0, -8.0, -8.0]) for i in range(40)}}
    payload["far"] = _rows([-25.0])
    path, partial = tmp_path / "cell.pkl", tmp_path / "cell.pkl.partial"

    published = runner._write_and_audit_outputs(
        path, partial, payload, identity={}, expected=len(payload)
    )

    assert published == path
    report = predictor_pickle_issues(published)
    assert "mean_outlier" in report.issues and report.soft_only
    audit = json.loads(manifest_path(published).read_text())["audit"]
    assert audit["hard_issues"] == [] and audit["soft_issues"] == ["mean_outlier"]
    assert audit["examples"]["mean_outlier"] and "accepted" not in audit


def test_a_clean_cell_records_no_audit(tmp_path):
    payload = {f"s{i}": _rows([-8.0, -8.0, -8.0]) for i in range(4)}
    path, partial = tmp_path / "cell.pkl", tmp_path / "cell.pkl.partial"
    published = runner._write_and_audit_outputs(
        path, partial, payload, identity={}, expected=len(payload)
    )
    assert "audit" not in json.loads(manifest_path(published).read_text())


def test_word_similarity_outputs_are_published_when_pos_is_disabled(
    tmp_path, monkeypatch
):
    from replica.core import AnnotationLayer, Span, Stimulus
    from replica.feature.distance import similarity_surprisal

    stimulus = Stimulus(text="cat dog", stimulus_id="s")
    stimulus.add_layer(
        AnnotationLayer("words", [Span(0, 3), Span(4, 7)], ["cat", "dog"])
    )
    model = similarity_surprisal.SimilarityAdjustedSurprisalWordLM(
        "toy", pos=False, add_bos=False, device="cpu"
    )
    model._model = SimpleNamespace(config=SimpleNamespace(n_positions=64))
    monkeypatch.setattr(model, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(model, "_ensure_starter_mask", lambda: None)
    monkeypatch.setattr(
        model, "_sample_all_words", lambda *args: [["cat", "bat"], ["dog", "fog"]]
    )
    monkeypatch.setattr(
        similarity_surprisal,
        "aligned_token_ids",
        lambda *args, **kwargs: ([1, 2], None, 0, 2),
    )
    payload = {"s": model(stimulus)}
    path = tmp_path / "word_similarity.pkl"
    published = runner._write_and_audit_outputs(
        path, path.with_suffix(".partial.pkl"), payload, identity={}, expected=1
    )
    assert published == path
    report = predictor_pickle_issues(path)
    assert report.is_clean
    assert report.primary_keys == {"similarity_adjusted_surprisal_orth"}


def test_pos_only_similarity_outputs_have_a_primary_key(tmp_path):
    path = _pickle(
        tmp_path / "pos.pkl",
        {"s": _rows([1.0, 2.0], "similarity_adjusted_surprisal_pos")},
    )
    report = predictor_pickle_issues(path)
    assert report.is_clean
    assert report.primary_keys == {"similarity_adjusted_surprisal_pos"}


def test_custom_primary_key_is_explicit_and_overrides_builtin_selection(tmp_path):
    payload = {
        "s": [
            FeatureModelOutput(values={"log_p_observed": NAN, "custom_score": 0.5}),
            FeatureModelOutput(values={"log_p_observed": NAN, "custom_score": 0.8}),
        ]
    }
    path = _pickle(tmp_path / "custom.pkl", payload)
    assert predictor_pickle_issues(path).has_hard_issues
    report = predictor_pickle_issues(path, primary_key="custom_score")
    assert report.is_clean
    assert report.record(failed=0, missing=0)["primary_keys"] == ["custom_score"]


@pytest.mark.parametrize("missing_index", [0, 1])
def test_a_declared_primary_key_must_exist_in_every_record(tmp_path, missing_index):
    rows = _rows([0.5, 0.8], "custom_score")
    rows[missing_index] = FeatureModelOutput(values={"log_p_observed": -1.0})
    path = _pickle(tmp_path / "missing.pkl", {"s": rows})
    report = predictor_pickle_issues(path, primary_key="custom_score")
    assert "missing_primary_value" in report.issues
    assert report.has_hard_issues


@pytest.mark.parametrize("value", [None, "invalid", []])
def test_a_declared_primary_value_must_be_numeric(tmp_path, value):
    path = _pickle(tmp_path / "invalid.pkl", {"s": _rows([value, 0.5], "custom_score")})
    report = predictor_pickle_issues(path, primary_key="custom_score")
    assert "invalid_primary_value" in report.issues
    assert report.has_hard_issues


def test_missingness_and_magnitude_advice_do_not_quarantine_or_force_recompute(
    tmp_path,
):
    payload = {**_corpus(), "sparse": _rows([NAN, NAN, -120.0])}
    path = tmp_path / "diagnostic.pkl"
    published = runner._write_and_audit_outputs(
        path, path.with_suffix(".partial"), payload, identity={}, expected=len(payload)
    )
    assert published == path and runner._valid_cached_pickle(path)
    audit = json.loads(manifest_path(path).read_text())["audit"]
    assert audit["hard_issues"] == []
    assert set(audit["soft_issues"]) == {"nan_burst", "extreme_per_word"}


def test_missing_builtin_value_is_structural_even_in_an_otherwise_usable_table(
    tmp_path,
):
    path = _pickle(
        tmp_path / "missing.pkl",
        {"s": [*_rows([-1.0, -2.0]), FeatureModelOutput(values={})]},
    )
    report = predictor_pickle_issues(path)
    assert "missing_primary_value" in report.issues and report.has_hard_issues
