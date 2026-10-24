"""Frozen family sizes must include explicit unavailable comparisons."""

import json

import pandas as pd
import pytest

from replica.analysis.crossed.evaluation import adjust, validate_manifest


def test_unavailable_cells_stay_in_family_and_incomplete_families_fail():
    frame = pd.DataFrame(
        [
            {
                "cell": "one",
                "name": "a",
                "family": "f",
                "kind": "prediction",
                "status": "complete",
                "p_value": 0.02,
                "effect": 1.0,
            },
            {
                "cell": "two",
                "name": "a",
                "family": "f",
                "kind": "prediction",
                "status": "unavailable",
                "p_value": 1.0,
            },
            {
                "cell": "three",
                "name": "a",
                "family": "f",
                "kind": "prediction",
                "status": "unavailable",
                "p_value": 1.0,
            },
        ]
    )
    result = adjust(frame, {"f": 3})
    assert result.p_holm.iloc[0] == pytest.approx(0.06)
    assert not result.supports.any()
    assert adjust(frame, {"f": 3}, alpha=0.1).supports.iloc[0]
    with pytest.raises(ValueError, match="Incomplete family"):
        adjust(frame.iloc[:2], {"f": 3})
    frame.loc[2, "p_value"] = 0
    with pytest.raises(ValueError, match="p=1"):
        adjust(frame, {"f": 3})


def test_manifest_rejects_missing_frozen_comparisons():
    manifest = {
        "alpha": 0.05,
        "families": {"f": 2},
        "cells": [
            {
                "id": "c",
                "family": "f",
                "kind": "association",
                "comparisons": [{"name": "one"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="Incomplete declared"):
        validate_manifest(manifest)


def test_serialized_false_direction_cannot_support_an_association():
    table = pd.DataFrame(
        [
            {
                "cell": "c",
                "name": "a",
                "family": "f",
                "kind": "association",
                "status": "complete",
                "p_value": 0.001,
                "direction_matches": "False",
            }
        ]
    )
    assert not adjust(table, {"f": 1}).supports.iloc[0]


def test_session_metadata_is_not_a_completed_model_inventory(tmp_path):
    from replica.analysis.crossed.evaluation import ready

    (tmp_path / "fits").mkdir()
    (tmp_path / "observations.tsv").write_text("response\n1\n")
    (tmp_path / "specification.json").write_text(
        json.dumps({"family": "gaussian", "models": {"control": [], "full": ["x"]}})
    )
    (tmp_path / "fits/session.txt").write_text("R session started")
    (tmp_path / "fits/diagnostics.json").write_text(json.dumps({"control": {}}))
    pd.DataFrame([{"model": "control", "term": "(Intercept)", "Estimate": 0.0}]).to_csv(
        tmp_path / "fits/coefficients.tsv", sep="\t", index=False
    )
    cell = {
        "kind": "association",
        "observations": "observations.tsv",
        "specification": "specification.json",
        "fits": "fits",
    }
    assert not ready(cell, tmp_path)


def test_manifest_alpha_controls_budget_and_is_bound_to_cell_receipts(tmp_path):
    from replica.analysis.crossed.evaluation import evaluate_cell, summarize

    manifest = {
        "alpha": 0.1,
        "families": {"f": 1},
        "cells": [
            {
                "id": "one",
                "family": "f",
                "kind": "prediction",
                "bootstrap_draws": 19,
                "bootstrap_seed": 17,
                "comparisons": [{"name": "gain"}],
            }
        ],
    }
    validate_manifest(manifest)
    with pytest.raises(ValueError, match="resolution precludes"):
        validate_manifest({**manifest, "alpha": 0.05})
    manifest["cells"][0]["unavailable"] = "fixture has no observations"
    del manifest["cells"][0]["bootstrap_draws"]
    del manifest["cells"][0]["bootstrap_seed"]
    validate_manifest(manifest)
    reports = tmp_path / "reports"
    evaluate_cell(manifest, 0, tmp_path, reports)
    assert json.loads((reports / "one/cell.json").read_text())["alpha"] == 0.1
    with pytest.raises(ValueError, match="alpha differs"):
        summarize({**manifest, "alpha": 0.2}, reports, tmp_path / "wrong")
    out = tmp_path / "summary"
    assert not summarize(manifest, reports, out).supports.any()
    assert json.loads((out / "source.json").read_text())["alpha"] == 0.1
    receipt_path = reports / "one/cell.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["alpha"]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="explicit alpha"):
        summarize(manifest, reports, tmp_path / "missing-alpha")


@pytest.mark.parametrize("alpha", [0, 1, float("nan"), True])
def test_invalid_family_alpha_fails_before_evaluation(alpha):
    with pytest.raises(ValueError, match="alpha"):
        validate_manifest({"alpha": alpha, "cells": [], "families": {}})


@pytest.mark.parametrize(
    "field,value",
    [
        ("alpha", None),
        ("bootstrap_draws", None),
        ("bootstrap_seed", None),
        ("bootstrap_seed", True),
        ("bootstrap_seed", -1),
        ("bootstrap_seed", 1.5),
        ("bootstrap_seed", "17"),
        ("bootstrap_draws", True),
        ("bootstrap_draws", 99.0),
    ],
)
def test_frozen_manifest_requires_explicit_valid_inference_settings(field, value):
    cell = {
        "id": "c",
        "family": "f",
        "kind": "prediction",
        "comparisons": [{"name": "gain"}],
        "bootstrap_draws": 99,
        "bootstrap_seed": 17,
    }
    manifest = {"alpha": 0.05, "families": {"f": 1}, "cells": [cell]}
    validate_manifest(manifest)
    target = manifest if field == "alpha" else cell
    del target[field]
    with pytest.raises(ValueError):
        validate_manifest(manifest)
    target[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)
