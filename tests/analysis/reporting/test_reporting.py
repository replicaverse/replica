"""Reports preserve score semantics and expose incomplete provenance safely."""

import base64
import csv
import hashlib
import json
from html.parser import HTMLParser

import numpy as np
import pytest

from replica import (
    OLS,
    AnalysisConfig,
    AnalysisResult,
    Coefficient,
    ComparisonSpec,
    EvaluationSpec,
    FoldResult,
    UnitPosition,
    evaluate,
    write_analysis_report,
)
from tests.analysis._fixtures import make_toy_dataset


class _Document(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.tags = []
        self.bars = []
        self.data = None
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if attrs.get("class") == "bar":
            self.bars.append(attrs["style"])
        if tag == "a" and attrs.get("download") == "analysis-report.json":
            self.data = json.loads(base64.b64decode(attrs["href"].split(",", 1)[1]))


def test_unequal_repeated_folds_keep_recorded_aggregate_and_joint_coefficients(
    tmp_path,
):
    result = AnalysisResult(
        "corpus",
        "language-model",
        "joint(a, b)",
        "gaze",
        metrics={"delta_ll": 2.5, "delta_ll_per_unit": 1.25, "delta_r2": float("nan")},
        n_observations=2,
        n_scored=2,
        comparison=ComparisonSpec(("length",), ("a", "b")),
        coefficients={"a": Coefficient(2, 0.1), "b": Coefficient(-3, 0.2)},
        fold_results=[
            FoldResult(0, 1, -4, -2, 2, 0.1, 0.2, 0.1),
            FoldResult(1, 3, -8, -5, 3, 0.2, 0.4, 0.2),
        ],
        metadata={"configuration": {"backend": "OLS", "cv_options": None}},
    )
    path = write_analysis_report([result, result], tmp_path / "analysis.html")
    data = _Document(path).data
    assert len(data["results"]) == 2
    first = data["results"][0]
    assert first == data["results"][1]
    assert first["feature_model"] == "language-model"
    assert first["metadata"]["configuration"]["backend"] == "OLS"
    assert first["metrics"]["delta_ll_per_unit"] == 1.25
    assert [f["delta_ll_per_observation"] for f in first["folds"]] == [2, 1]
    assert first["n_scored"] == 2
    assert first["coefficients"]["b"] == {"estimate": -3, "mean_within_fit_se": 0.2}
    assert first["metrics"]["delta_r2"] == {"nonfinite": "nan"}
    assert first["availability"]["undeclared_options"] == ["configuration.cv_options"]
    assert "fit diagnostics" in first["availability"]["not_recorded"]
    assert "not confidence intervals" in path.read_text()
    assert np.isnan(result.metrics["delta_r2"])
    result.coefficients = {}
    result.coefficient, result.coefficient_se = 0.7, 0.05
    legacy = _Document(
        write_analysis_report(result, tmp_path / "coefficient.html")
    ).data["results"][0]
    assert legacy["coefficients"] == {}
    assert legacy["coefficient"] == 0.7
    assert legacy["coefficient_mean_within_fit_se"] == 0.05


def test_report_escapes_text_and_never_traverses_observation_arrays(tmp_path):
    class UnreadableArray(np.ndarray):
        def __iter__(self):
            pytest.fail("Report traversed an observation array")

        def tolist(self):
            pytest.fail("Report materialized an observation array")

        def __deepcopy__(self, memo):
            pytest.fail("Report copied an observation array")

    raw = np.arange(1000).view(UnreadableArray)

    class UnreadableKeys(tuple):
        def __iter__(self):
            pytest.fail("Report traversed canonical observation keys")

    malicious = '</script><img src="x" onerror="alert(1)">&'
    fold = FoldResult(
        1,
        0,
        np.inf,
        -np.inf,
        np.nan,
        0,
        0,
        0,
        per_obs_delta_ll=raw,
        factor_ids={"reader": raw},
        row_ids=raw,
    )
    fold.observations = {
        "observed": raw,
        "prediction_baseline": raw,
        "prediction_full": raw,
        "in_train": raw,
    }
    fold.observation_keys = UnreadableKeys([("private_identity",)] * 1000)
    result = AnalysisResult(
        malicious,
        "LM",
        "a",
        fold_results=[fold],
        metadata={"custom_array": raw, "custom_object": object()},
    )
    path = write_analysis_report(
        result, tmp_path / "report.html", title=malicious, context={"label": malicious}
    )
    document = _Document(path)
    assert "script" not in document.tags and "img" not in document.tags
    assert document.data["title"] == malicious
    first = document.data["results"][0]
    assert first["folds"][0]["delta_ll_per_observation"] is None
    assert first["folds"][0]["baseline_ll"] == {"nonfinite": "inf"}
    assert first["metadata"]["custom_array"] == {
        "omitted_array": {"shape": [1000], "dtype": str(raw.dtype)}
    }
    assert first["availability"]["omitted_metadata_values"] == [
        "result.metadata.custom_array",
        "result.metadata.custom_object",
    ]
    assert "per_obs_delta_ll" not in first["folds"][0]
    assert first["folds"][0]["observations"]["fields"]["observed"] == {
        "shape": [1000],
        "dtype": str(raw.dtype),
    }
    assert "observation_export" not in document.data
    assert "private_identity" not in json.dumps(document.data)
    assert "configuration" in first["availability"]["not_recorded"]


def test_actual_partial_split_reports_training_count_without_inference_or_refit(
    tmp_path,
):
    class PartialSplitter:
        def split(self, y, groups, participants, items):
            index = np.arange(len(y))
            yield index < 20, index >= 24

    dataset = make_toy_dataset(n_stimuli=8, n_words=5)
    spec = EvaluationSpec(
        predictors=[UnitPosition()], targets=[dataset.feature("spr_rt")]
    )
    config = AnalysisConfig(
        model=OLS(), design=None, cv=PartialSplitter(), per_fold=True
    )
    results = evaluate(dataset, spec, config)
    metrics_before = dict(results[0].metrics)
    path = write_analysis_report(results, tmp_path / "actual.html")
    first = _Document(path).data["results"][0]
    assert first["metrics"] == metrics_before == results[0].metrics
    assert first["n_observations"] == 32
    assert first["folds"][0]["n_test"] == 8
    assert first["folds"][0]["metadata"]["n_train"] == 20
    assert first["response"]["row_exclusions"]["retained"] == 32
    assert "configuration.cv_options" in first["availability"]["undeclared_options"]
    results[0].fold_results = None
    no_folds = write_analysis_report(results, tmp_path / "no-folds.html")
    assert _Document(no_folds).data["results"][0]["folds"] is None
    assert "Fold results were not retained" in no_folds.read_text()


def test_invalid_input_does_not_overwrite_destination(tmp_path):
    path = tmp_path / "existing.html"
    path.write_text("preserve")
    for results, error in (([], ValueError), ([object()], TypeError)):
        with pytest.raises(error):
            write_analysis_report(results, path)
        assert path.read_text() == "preserve"


def test_opposite_large_finite_gains_remain_equal_visible_bars(tmp_path):
    result = AnalysisResult(
        "data",
        "model",
        "feature",
        fold_results=[
            FoldResult(i, 1, 0, value, value, 0, 0, 0)
            for i, value in enumerate([-1e308, 1e308])
        ],
    )
    document = _Document(write_analysis_report(result, tmp_path / "large.html"))
    assert len(document.bars) == 2
    assert all("width:50.000%" in bar for bar in document.bars)
    assert "left:0.000%" in document.bars[0]
    assert "left:50.000%" in document.bars[1]


def _retained_result():
    folds = []
    for index in range(2):
        fold = FoldResult(
            index,
            2,
            -2,
            -1,
            1,
            0,
            0,
            0,
            metadata={
                "observation_capture": {
                    "response_scale": "model_after_transform_and_fold_preprocessing",
                    "population": "scored_rows",
                    "row_id_columns": ["participant", "stimulus", "unit"],
                    "row_key_names": ["participant", "stimulus", "unit", "trial"],
                },
            },
        )
        fold.observations = {
            "observed": np.array([1.5, 2.5]),
            "prediction_baseline": np.array([0.75, 2.0]),
            "prediction_full": np.array([1.0, 3.0]),
            "in_train": np.array([index == 0, False]),
            "prepared_row_index": np.array([42, 43]),
        }
        fold.observation_keys = (("p1", "s1", 0, "first"), ("p1", "s1", 1, "first"))
        fold.row_ids = np.array([["p1", "s1", "0"], ["p1", "s1", "1"]])
        fold.factor_ids = {"label": np.array(['a,"quoted"', "=not_a_formula"])}
        if index == 0:
            fold.per_obs_delta_ll = np.array([0.125, -0.25])
        folds.append(fold)
    return AnalysisResult(
        "data",
        "model",
        "predictor",
        "response",
        fold_results=folds,
        response={"target_measurement": {"measurement": {"unit": "ms"}}},
        metadata={
            "configuration": {"response_transform": "log2", "score_scale": "original"}
        },
    )


def test_explicit_observation_csv_preserves_alignment_and_shares_report_record(
    tmp_path,
):
    result = _retained_result()
    csv_path = tmp_path / "rows.csv"
    path = write_analysis_report(
        [result, result],
        tmp_path / "report.html",
        observations_path=csv_path,
        context={"source": "test-revision"},
    )
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert [row["result_index"] for row in rows] == ["0"] * 4 + ["1"] * 4
    assert [row["fold_index"] for row in rows[:4]] == ["0", "0", "1", "1"]
    assert (
        json.loads(rows[0]["row_id"])
        == json.loads(rows[2]["row_id"])
        == ["p1", "s1", "0"]
    )
    assert json.loads(rows[0]["row_key"]) == ["p1", "s1", 0, "first"]
    assert rows[0]["prepared_row_index"] == rows[2]["prepared_row_index"] == "42"
    assert [float(row["error_full"]) for row in rows[:2]] == [0.5, -0.5]
    assert [float(row["error_baseline"]) for row in rows[:2]] == [0.75, 0.5]
    assert rows[0]["in_train"] == "True" and rows[2]["in_train"] == "False"
    assert json.loads(rows[0]["factor:label"]) == 'a,"quoted"'
    assert json.loads(rows[1]["factor:label"]) == "=not_a_formula"
    assert rows[0]["delta_loglik"] == "0.125" and rows[2]["delta_loglik"] == ""
    record = json.loads((tmp_path / "rows.csv.json").read_text())
    assert record == _Document(path).data
    export = record["observation_export"]
    assert export["n_rows"] == 8
    assert export["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert (
        record["results"][0]["metadata"]["configuration"]["score_scale"] == "original"
    )
    assert (
        record["results"][0]["folds"][0]["metadata"]["observation_capture"][
            "response_scale"
        ]
        == "model_after_transform_and_fold_preprocessing"
    )
    assert (
        record["results"][0]["response"]["target_measurement"]["measurement"]["unit"]
        == "ms"
    )
    assert not any(
        name.startswith("error_") for name in result.fold_results[0].observations
    )
    assert "CSV rows" in path.read_text()
    for observed, predicted in (
        (
            np.array([1, 2], dtype=np.longdouble),
            np.array(
                [np.nextafter(np.longdouble(1), np.longdouble(2)), 3],
                dtype=np.longdouble,
            ),
        ),
        (np.array([0, 1], dtype=np.uint64), np.array([2**64 - 1, 3], dtype=np.uint64)),
    ):
        result.fold_results[0].observations.update(
            observed=observed, prediction_full=predicted
        )
        write_analysis_report(result, path, observations_path=csv_path)
        with csv_path.open(newline="") as stream:
            first = next(csv.DictReader(stream))
        dtype = np.result_type(observed.dtype, predicted.dtype, np.float64)
        expected = dtype.type(observed[0]) - dtype.type(predicted[0])
        assert dtype.type(first["error_full"]) == expected < 0


@pytest.mark.parametrize("problem", ["absent", "misaligned", "identity", "collision"])
def test_invalid_observation_export_preserves_all_existing_files(tmp_path, problem):
    result = _retained_result()
    html, csv_path = tmp_path / "report.html", tmp_path / "rows.csv"
    metadata = tmp_path / "rows.csv.json"
    for path in (html, csv_path, metadata):
        path.write_text("preserve")
    if problem == "absent":
        result.fold_results[1].observations = None
    elif problem == "misaligned":
        result.fold_results[1].observations["observed"] = np.ones(1)
    elif problem == "identity":
        result.fold_results[1].factor_ids["bad"] = np.array([object(), object()])
    else:
        csv_path = html
    with pytest.raises((ValueError, TypeError)):
        write_analysis_report(result, html, observations_path=csv_path)
    for path in (html, tmp_path / "rows.csv", metadata):
        assert path.read_text() == "preserve"
    assert not list(tmp_path.glob(".*"))


def test_failed_analysis_record_does_not_fabricate_results(tmp_path):
    failure = {
        "status": "failed",
        "exception_type": "ValueError",
        "message": "<invalid>",
        "fold_attempts": [
            {
                "fold_index": 1,
                "status": "failed",
                "stage": "baseline_fit_and_predict",
                "error": {"type": "ValueError", "message": "<invalid>"},
            }
        ],
    }
    path = write_analysis_report(
        [], tmp_path / "failed.html", context={"failed_analysis": failure}
    )
    document = _Document(path)
    assert document.data["results"] == []
    assert document.data["context"]["failed_analysis"] == failure
    assert "Recorded analysis failure" in path.read_text()
    assert "baseline_fit_and_predict" in path.read_text()
    assert "invalid" not in document.tags
    with pytest.raises(ValueError, match="no observations"):
        write_analysis_report(
            [],
            path,
            context={"failed_analysis": failure},
            observations_path=tmp_path / "rows.csv",
        )
    assert not (tmp_path / "rows.csv").exists()
    with pytest.raises(ValueError, match="explicit failed_analysis"):
        write_analysis_report(
            [], path, context={"failed_analysis": {"status": "skipped"}}
        )
    assert _Document(path).data == document.data
