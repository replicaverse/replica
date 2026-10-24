"""Independent predictive integrals and rejection of altered exported losses."""

import json

import numpy as np
import pandas as pd
import pytest

from replica.analysis.crossed.verification import (
    verify_predictions as verify,
)
from replica.analysis.crossed.verification.predictions import integrated_logistic


def test_integrated_logistic_has_symmetry_and_zero_variance_limit():
    assert integrated_logistic(0.0, 100.0) == pytest.approx(0.5, abs=1e-10)
    for variance in (0.0, 0.2, 100.0):
        assert integrated_logistic(2.0, variance) + integrated_logistic(
            -2.0, variance
        ) == pytest.approx(1.0, abs=1e-10)
    assert integrated_logistic(0.0, 0.0) == 0.5
    with pytest.raises(ValueError):
        integrated_logistic(0.0, -1.0)


def test_predictive_verification_checks_every_key_and_loss(tmp_path):
    frame = pd.DataFrame(
        {
            "participant": ["a", "b", "c"],
            "material": ["x", "x", "y"],
            "item": ["i", "j", "k"],
            "role": ["train", "train", "test"],
            "response": [0.0, 1.0, 1.0],
        }
    )
    observations = tmp_path / "observations.tsv"
    frame.to_csv(observations, sep="\t", index=False)
    spec = tmp_path / "specification.json"
    spec.write_text(json.dumps({"family": "binomial", "models": {"control": []}}))
    fits = tmp_path / "fits"
    fits.mkdir()
    (fits / "scaling.json").write_text("[]")
    (fits / "diagnostics.json").write_text(
        json.dumps(
            {
                "control": {
                    "nobs": 2,
                    "optimizer_code": 0,
                    "variances": [
                        {"grp": "participant", "var1": "(Intercept)", "vcov": 2.0}
                    ],
                }
            }
        )
    )
    pd.DataFrame([{"Estimate": 0.0, "term": "(Intercept)", "model": "control"}]).to_csv(
        fits / "coefficients.tsv", sep="\t", index=False
    )
    predictions = tmp_path / "predictions.tsv"
    predicted = frame.loc[[2], ["participant", "material", "item", "response"]].assign(
        control_latent_mean=0.0,
        control_latent_variance=2.0,
        control_prediction=0.5,
        control_loss=np.log(2),
    )
    predicted.to_csv(predictions, sep="\t", index=False)
    from replica.analysis.crossed.verification.__main__ import main

    result = verify(observations, spec, fits, predictions)
    assert result["models"][0]["observations"] == 1
    receipt = tmp_path / "verification.json"
    main(
        [
            "predictions",
            "--observations",
            str(observations),
            "--specification",
            str(spec),
            "--fits",
            str(fits),
            "--predictions",
            str(predictions),
            "--out",
            str(receipt),
        ]
    )
    assert json.loads(receipt.read_text()) == result
    predicted["control_loss"] = 0.0
    predicted.to_csv(predictions, sep="\t", index=False)
    with pytest.raises(AssertionError):
        verify(observations, spec, fits, predictions)
