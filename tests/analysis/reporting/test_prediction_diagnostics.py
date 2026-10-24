"""Bounded summaries retain every finite row without asserting model validity."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtri

from replica import AnalysisResult, FoldResult, write_analysis_report
from replica.analysis.fitting.diagnostics import (
    prediction_diagnostics,
    snapshot_fit_diagnostics,
)
from replica.analysis.reporting._diagnostics import _scatter
from tests.analysis.reporting.test_reporting import _Document


def test_fit_snapshot_rejects_invalid_provider_result():
    with pytest.raises(TypeError, match="mapping or None"):
        snapshot_fit_diagnostics(SimpleNamespace(last_fit_diagnostics=lambda: []))


def test_all_row_bins_keep_ties_and_extremes_with_bounded_quantiles():
    fitted = np.repeat(np.arange(25, dtype=float), 500)
    errors = np.tile(np.linspace(-3, 4, 500), 25)
    y = fitted + errors
    diagnostic = prediction_diagnostics(
        y,
        fitted,
        fitted,
        sigma_base=2.0,
        sigma_full=np.full(len(y), 2.0),
        normal_reference=True,
        n_train_overlap=7,
    )
    assert diagnostic["n_rows"] == 12500 and diagnostic["n_train_overlap"] == 7
    baseline = diagnostic["baseline"]
    assert baseline == diagnostic["full"]
    assert len(baseline["fitted_bins"]) <= 20
    assert sum(item["count"] for item in baseline["fitted_bins"]) == len(y)
    # A tied fitted value is never divided into arbitrary residual subgroups.
    assert all(item["count"] % 500 == 0 for item in baseline["fitted_bins"])
    assert all(
        item["error_mean"] == pytest.approx(0.5) for item in baseline["fitted_bins"]
    )
    assert baseline["error_sd"] == pytest.approx(np.std(errors))
    assert baseline["error_rms"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    raw = baseline["error_quantiles"]
    qq = baseline["normal_qq"]
    assert len(raw) == len(qq["points"]) == 101
    assert raw[0]["error"] == -3 and raw[-1]["error"] == 4
    assert qq["points"][0]["standardized_error"] == -1.5
    assert qq["points"][-1]["standardized_error"] == 2
    assert qq["points"][0]["theoretical"] == pytest.approx(ndtri(0.5 / len(y)))
    assert qq["scale_source"] == "scoring_sigma"
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("normal_reference", [False, True])
def test_reference_availability_nonfinite_rows_and_empirical_scale(normal_reference):
    diagnostic = prediction_diagnostics(
        [2.0, -4.0, np.inf],
        [0.0, 0.0, 0.0],
        [2.0, -4.0, np.inf],
        normal_reference=normal_reference,
        n_train_overlap=3,
    )
    baseline, full = diagnostic["baseline"], diagnostic["full"]
    assert baseline["n_finite_errors"] == 2 and baseline["n_nonfinite_errors"] == 1
    assert baseline["error_mean"] == -1
    assert baseline["error_sd"] == 3
    assert len(baseline["fitted_bins"]) == 1
    qq = baseline["normal_qq"]
    if normal_reference:
        assert qq["n_finite"] == 2 and qq["n_nonfinite"] == 1
        assert qq["scale_source"] == "scored_error_rms"
        assert qq["estimated_scale"] == pytest.approx(np.sqrt(10))
        assert [p["standardized_error"] for p in qq["points"]] == pytest.approx(
            [-4 / np.sqrt(10), 2 / np.sqrt(10)]
        )
        assert full["normal_qq"]["reason"] == "no_finite_positive_standardization_scale"
    else:
        assert not qq["available"]
        assert qq["reason"] == "no_declared_gaussian_reference"
        assert "points" not in qq
    json.dumps(diagnostic, allow_nan=False)
    empty = prediction_diagnostics(
        [], [], [], normal_reference=normal_reference, n_train_overlap=0
    )
    assert empty["full"]["fitted_bins"] == []
    assert not empty["full"]["normal_qq"]["available"]


def test_report_keeps_checks_distinct_and_plots_separate_fold_populations(tmp_path):
    capture = prediction_diagnostics(
        [1.0, 4.0, 6.0],
        [0.0, 2.0, 4.0],
        [1.0, 3.0, 5.0],
        normal_reference=True,
        n_train_overlap=1,
    )
    nong = prediction_diagnostics(
        [9.0],
        [7.0],
        [9.0],
        normal_reference=False,
        n_train_overlap=0,
    )
    result = AnalysisResult(
        "data",
        "custom",
        "a",
        "target",
        fold_results=[
            FoldResult(
                i,
                payload["n_rows"],
                0,
                0,
                0,
                0,
                0,
                0,
                metadata={"prediction_diagnostics": payload},
            )
            for i, payload in enumerate([capture, nong])
        ],
        metadata={
            "fit_diagnostics": [
                {
                    "fold_index": 0,
                    "baseline": {
                        "convergence_applicable": False,
                        "rank": 1,
                        "n_columns": 1,
                    },
                    "full": {
                        "converged": None,
                        "diagnostic_checks_passed": True,
                        "selected_parameters": {"alpha": 2},
                        "warnings": ["<script>alert(1)</script>"],
                    },
                },
                {
                    "fold_index": 1,
                    "baseline": None,
                    "full": {"converged": False, "singular": True},
                },
            ]
        },
    )
    path = write_analysis_report(result, tmp_path / "diagnostics.html")
    document = _Document(path)
    html = path.read_text()
    assert "script" not in document.tags
    assert (
        document.tags.count("svg") == 6
    )  # Two raw and two Q-Q first fold; raw only second.
    assert "Not applicable" in html and "Unknown / not provided" in html
    assert "Sampler diagnostic checks are not convergence" in html
    assert "this backend declares no Gaussian reference" in html
    assert "finite-error RMS estimated on these scored rows" in html
    assert "Random-effect singularity" in html
    assert "1 also occur in this fold’s training data" in html
    assert "Binning can conceal local patterns" in html
    records = document.data["results"][0]["folds"]
    assert records[0]["metadata"]["prediction_diagnostics"] == capture
    assert records[1]["metadata"]["prediction_diagnostics"] == nong
    ticks = _scatter(
        [(-1.0, 0.0, None, "exact"), (1.0 - 1e-13, 0.0385406, None, "0.0385406")],
        title="Ticks",
        x_label="x",
        y_label="y",
    )
    assert ">0.0385</text>" in ticks and ">0.0385406</text>" not in ticks
    assert 'text-anchor="middle">0</text>' in ticks
    assert "<title>0.0385406</title>" in ticks
    tiny = _scatter(
        [(1e-15, 1e-15, None, "tiny"), (2e-15, 2e-15, None, "tiny")],
        title="Small scale",
        x_label="x",
        y_label="y",
    )
    assert ">1e-15</text>" in tiny
