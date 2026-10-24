"""Verify ordinal crossed fits and integrated unseen-group category probabilities."""

import json
import shutil
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from replica.utils.r import run_r_script


@pytest.mark.needs_r
def test_ordinal_random_slopes_and_population_probabilities(tmp_path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is required")
    rng = np.random.default_rng(2)
    reader, item = np.indices((24, 30))
    reader, item = reader.ravel(), item.ravel()
    x = rng.normal(size=len(reader))
    latent = (
        x
        + rng.normal(size=24)[reader]
        + rng.normal(size=30)[item]
        + rng.normal(scale=0.4, size=24)[reader] * x
        + rng.normal(scale=0.4, size=30)[item] * x
        + rng.logistic(size=len(reader))
    )
    frame = pd.DataFrame(
        dict(
            participant=reader,
            item=item,
            material=item,
            x=x,
            response=np.digitize(latent, [-1, 0, 1]) + 1,
            role="train",
        )
    )
    test = frame.iloc[:12].copy()
    test[["participant", "item", "material"]] += 100
    test["role"] = "test"
    data = tmp_path / "observations.tsv"
    pd.concat([frame, test]).to_csv(data, sep="\t", index=False)
    spec = {
        "family": "ordinal",
        "categories": [1, 2, 3, 4],
        "groups": ["participant", "item"],
        "models": {"control": [], "full": ["x"]},
        "random_terms": [
            "(1|participant)",
            "(0+x|participant)",
            "(1|item)",
            "(0+x|item)",
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    out = tmp_path / "fit"
    run_r_script(
        files("replica.analysis.crossed.fitting").joinpath("ordinal.R"),
        [data, path, out],
        executable=rscript,
    )
    result = pd.read_csv(out / "predictions.tsv", sep="\t")
    diagnostics = json.loads((out / "diagnostics.json").read_text())["full"]
    assert diagnostics["optimizer_code"] == 0
    assert diagnostics["verified_inner_gradient"] <= 1e-6
    assert diagnostics["verified_likelihood_error"] <= 1e-5
    assert diagnostics["coefficients"]["x"] > 0
    assert len(diagnostics["variances"]) == 4
    assert sorted(v["group"] for v in diagnostics["variances"]) == [
        "item",
        "item",
        "participant",
        "participant",
    ]
    assert "0 + x | participant" in str(diagnostics["formula"])
    assert "0 + x | item" in str(diagnostics["formula"])
    columns = [f"full_probability_{i}" for i in range(1, 5)]
    np.testing.assert_allclose(result[columns].sum(axis=1), 1)
    assert np.isfinite(result.full_loss).all()
    assert np.isfinite(result.control_loss).all()
    from scipy.integrate import quad
    from scipy.special import expit
    from scipy.stats import norm

    row = result.iloc[0]
    thresholds = diagnostics["thresholds"]
    cumulative = [
        quad(
            lambda z, threshold=threshold: (
                expit(
                    threshold
                    - row.full_latent_mean
                    - np.sqrt(row.full_latent_variance) * z
                )
                * norm.pdf(z)
            ),
            -np.inf,
            np.inf,
        )[0]
        for threshold in thresholds
    ]
    np.testing.assert_allclose(
        row[columns].to_numpy(float), np.diff([0, *cumulative, 1]), atol=1e-8
    )
