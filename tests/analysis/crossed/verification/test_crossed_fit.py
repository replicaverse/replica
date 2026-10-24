"""Live mixed-model checks for common random structures and nested comparisons."""

import json
import shutil
import subprocess
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from replica.analysis.crossed.verification import verify_fit
from replica.utils.r import run_r_script

pytestmark = pytest.mark.needs_r


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
def test_reduced_model_retains_the_focal_random_slope(tmp_path, family):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is required")
    rng = np.random.default_rng(20260927)
    pairs = np.array([(reader, item) for reader in range(24) for item in range(30)])
    reader, item = pairs.T
    x = rng.normal(size=len(pairs))
    z = rng.normal(size=len(pairs))
    y = (
        3 * x
        + z
        + rng.normal(size=24)[reader]
        + rng.normal(scale=0.5, size=24)[reader] * x
        + rng.normal(size=30)[item]
        + rng.normal(scale=0.5, size=30)[item] * x
        + rng.normal(size=len(pairs))
    )
    if family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-y / 3)))
    frame = pd.DataFrame(
        {"response": y, "x": x, "z": z, "participant": reader, "item": item}
    )
    frame["constant"] = 1.0
    data = tmp_path / "observations.tsv"
    frame.to_csv(data, sep="\t", index=False)
    specification = {
        "family": family,
        "groups": ["participant", "item"],
        "models": {"reduced": ["z", "constant"], "full": ["z", "x", "constant"]},
        "random_terms": [
            "(1|participant)",
            "(0+x|participant)",
            "(1|item)",
            "(0+x|item)",
        ],
        "contrasts": [{"name": "x", "reduced": "reduced", "full": "full"}],
        "bootstrap_samples": 9,
        "seed": 20260927,
    }
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(specification))
    out = tmp_path / "fits"
    run_r_script(
        files("replica.analysis.crossed.fitting").joinpath("fit.R"),
        [data, spec, out],
        executable=rscript,
    )
    diagnostics = json.loads((out / "diagnostics.json").read_text())
    for model in diagnostics.values():
        formula = str(model["formula"])
        assert "0 + x | participant" in formula
        assert "0 + x | item" in formula
        assert model["nobs"] == len(frame)
        assert model["optimizer_code"] == 0
        assert model["constant_fixed_terms"] == "constant"
        assert "constant" not in formula
    assert diagnostics["full"]["loglik"] > diagnostics["reduced"]["loglik"]
    check_separate_control(tmp_path, rscript, specification, data, out, diagnostics)
    tests = json.loads((out / "tests.json").read_text())
    assert 0 < tests["x"]["p_value"] <= 1
    assert tests["x"]["requested_samples"] == 9
    if family == "gaussian":
        from replica.analysis.crossed.verification.replay import replay

        receipt = replay(data, spec, out, tmp_path / "fresh-replay", rscript=rscript)
        assert "coefficients.tsv" in receipt["checks"]
        check_strict_restart(tmp_path, rscript, specification, data, out, diagnostics)
    # The same frozen training fits must use training-only transforms and predict
    # completely new reader/item/material identities without their random BLUPs.
    heldout = frame.iloc[:40].copy()
    heldout["participant"] += 100
    heldout["item"] += 100
    heldout["x"] += 2
    heldout["role"] = "test"
    frame["role"] = "train"
    combined = pd.concat([frame, heldout], ignore_index=True)
    combined["material"] = combined.item
    combined.to_csv(data, sep="\t", index=False)
    destination = tmp_path / "heldout.tsv"
    run_r_script(
        files("replica.analysis.crossed.fitting").joinpath("predict.R"),
        [data, spec, out, destination],
        executable=rscript,
    )
    predictions = pd.read_csv(destination, sep="\t")
    assert len(predictions) == len(heldout)
    assert np.isfinite(predictions[["reduced_loss", "full_loss"]]).all().all()
    if family == "binomial":
        from scipy.integrate import quad
        from scipy.special import expit
        from scipy.stats import norm

        assert predictions.full_prediction.between(0, 1, inclusive="neither").all()
        for model in ("reduced", "full"):
            row = predictions.iloc[0]
            mean, variance = (
                row[model + "_latent_mean"],
                row[model + "_latent_variance"],
            )
            reference = quad(
                lambda z, mean=mean, variance=variance: (
                    expit(mean + np.sqrt(variance) * z) * norm.pdf(z)
                ),
                -np.inf,
                np.inf,
            )[0]
            assert row[model + "_prediction"] == pytest.approx(reference, abs=1e-7)
    else:
        verified = verify_fit(frame, specification, out)
        assert len(verified) == 2
        assert max(v["loglik_absolute_error"] for v in verified) < 1e-6
        check_likelihood_cli(frame, spec, out, tmp_path / "verification.json", verified)
        coefficients = pd.read_csv(out / "coefficients.tsv", sep="\t")
        beta = (
            coefficients.loc[coefficients.model.eq("full")].set_index("term").Estimate
        )
        expected = beta["(Intercept)"] + sum(
            beta[column]
            * (heldout[column] - frame[column].mean())
            / frame[column].std()
            for column in ("x", "z")
        )
        np.testing.assert_allclose(predictions.full_prediction, expected)


def check_likelihood_cli(frame, spec, fits, receipt, expected):
    """The unified fit command writes the same independently verified likelihoods."""
    from replica.analysis.crossed.verification.__main__ import main

    data = receipt.with_suffix(".tsv")
    frame.to_csv(data, sep="\t", index=False)
    main(
        [
            "fit",
            "--observations",
            str(data),
            "--specification",
            str(spec),
            "--fits",
            str(fits),
            "--out",
            str(receipt),
        ]
    )
    for actual, reference in zip(
        json.loads(receipt.read_text())["models"], expected, strict=True
    ):
        assert actual == pytest.approx(reference, rel=1e-10, abs=1e-10)


def check_partial_resume(tmp_path, rscript, specification, data, out):
    """Reuse a completed model only when its full fitting contract still matches."""
    from replica.analysis.crossed.verification.replay import compare_fits

    partial = tmp_path / "partial"
    spec = tmp_path / "resume-spec.json"
    spec.write_text(json.dumps({**specification, "bootstrap_samples": 0}))
    engine = files("replica.analysis.crossed.fitting").joinpath("fit.R")
    run_r_script(engine, [data, spec, partial, "-", "reduced"], executable=rscript)
    assert set(json.loads((partial / "diagnostics.json").read_text())) == {"reduced"}
    assert json.loads((partial / "scaling.json").read_text()) == json.loads(
        (out / "scaling.json").read_text()
    )
    resumed = tmp_path / "resumed"
    run_r_script(engine, [data, spec, resumed, partial], executable=rscript)
    compare_fits(resumed, out)
    receipt = json.loads((resumed / "resumption.json").read_text())
    assert set(receipt["models"]) == {"reduced"}

    changed = pd.read_csv(data, sep="\t")
    changed.loc[0, "response"] += 1 if specification["family"] == "gaussian" else 0
    if specification["family"] == "binomial":
        changed.loc[0, "response"] = 1 - changed.loc[0, "response"]
    other = tmp_path / "changed.tsv"
    changed.to_csv(other, sep="\t", index=False)
    process = subprocess.run(
        [
            rscript,
            str(engine),
            str(other),
            str(spec),
            str(tmp_path / "rejected"),
            str(partial),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "Resume observations differ: response" in process.stderr


def check_strict_restart(tmp_path, rscript, specification, data, out, diagnostics):
    """A tighter restart recovers the same model without removing random slopes."""
    starts = json.loads(
        subprocess.check_output(
            [
                rscript,
                "-e",
                "suppressPackageStartupMessages(library(lme4)); library(jsonlite); "
                'p <- commandArgs(TRUE)[1]; names <- c("reduced", "full"); '
                'write_json(setNames(lapply(names, function(n) list(theta=unname(getME(readRDS(file.path(p,paste0(n,".rds"))),"theta")))),names),stdout(),digits=NA)',
                str(out),
            ],
            text=True,
        )
    )
    strict_spec = dict(
        specification, optimizer_control={"rhoend": 1e-9}, starting_values=starts
    )
    strict = tmp_path / "strict-spec.json"
    strict.write_text(json.dumps(strict_spec))
    strict_out = tmp_path / "strict-fits"
    run_r_script(
        files("replica.analysis.crossed.fitting").joinpath("fit.R"),
        [data, strict, strict_out],
        executable=rscript,
    )
    strict_diagnostics = json.loads((strict_out / "diagnostics.json").read_text())
    for name in diagnostics:
        assert strict_diagnostics[name]["formula"] == diagnostics[name]["formula"]
        assert strict_diagnostics[name]["loglik"] == pytest.approx(
            diagnostics[name]["loglik"], abs=1e-5
        )


def check_separate_control(tmp_path, rscript, specification, data, out, diagnostics):
    check_partial_resume(tmp_path, rscript, specification, data, out)
    # Fitting the reduced model alone must preserve its centered random slopes.
    separate = dict(
        specification,
        models={"reduced": specification["models"]["reduced"]},
        contrasts=[],
        bootstrap_samples=0,
    )
    separate_spec = tmp_path / "separate.json"
    separate_spec.write_text(json.dumps(separate))
    separate_out = tmp_path / "separate-fits"
    run_r_script(
        files("replica.analysis.crossed.fitting").joinpath("fit.R"),
        [data, separate_spec, separate_out],
        executable=rscript,
    )
    assert json.loads((separate_out / "scaling.json").read_text()) == json.loads(
        (out / "scaling.json").read_text()
    )
    separate_diag = json.loads((separate_out / "diagnostics.json").read_text())
    assert separate_diag["reduced"]["loglik"] == pytest.approx(
        diagnostics["reduced"]["loglik"], abs=1e-8
    )
