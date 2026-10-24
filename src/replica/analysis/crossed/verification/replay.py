"""Refit a recorded crossed model in a fresh R process and compare fitted state."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from replica.utils.digest import file_sha256


def compare_value(actual, expected, path="root"):
    """Require identical structure and predeclared numerical agreement throughout.

    Args:
        actual: Freshly computed value or fitted-output path.
        expected: Authenticated reference value or fitted-output path.
        path: Diagnostic location within the compared nested values.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Changed fitted-state keys: {path}")
        for key in expected:
            compare_value(actual[key], expected[key], path + "/" + str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Changed fitted-state inventory: {path}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare_value(a, e, path + "/" + str(i))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, err_msg=path)
    elif actual != expected:
        raise ValueError(f"Changed fitted-state metadata: {path}")


def compare_fits(actual, expected):
    """Compare coefficients, covariance estimates, likelihoods and convergence evidence.

    Args:
        actual: Freshly computed value or fitted-output path.
        expected: Authenticated reference value or fitted-output path.
    """
    checks = {}
    for name in ("diagnostics.json", "scaling.json", "fixed_design.json"):
        reference = expected / name
        if not reference.is_file():
            continue
        compare_value(
            json.loads((actual / name).read_text()),
            json.loads(reference.read_text()),
            name,
        )
        checks[name] = {
            "reference_sha256": file_sha256(reference),
            "replay_sha256": file_sha256(actual / name),
        }
    if "diagnostics.json" not in checks or "scaling.json" not in checks:
        raise ValueError("Missing complete reference fit")
    reference = expected / "coefficients.tsv"
    if reference.is_file():
        a, b = (pd.read_csv(p, sep="\t") for p in (actual / reference.name, reference))
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-7)
        checks[reference.name] = {
            "reference_sha256": file_sha256(reference),
            "replay_sha256": file_sha256(actual / reference.name),
        }
    return checks


def replay(
    observations, specification, reference, out, *, rscript="Rscript", engine=None
):
    """Restart optimization with unchanged formulas, inputs and optimizer settings.

    Args:
        observations: Path to the frozen observation table.
        specification: Path to the declared model specification.
        reference: Directory containing the reference fitted state.
        out: Destination for newly computed outputs.
        rscript: Rscript executable for a fresh R process.
        engine: Pinned R fitting script; omitted, use the packaged engine.
    """
    spec = json.loads(specification.read_text())
    script = engine or (
        Path(__file__).resolve().parents[1]
        / "fitting"
        / ("ordinal.R" if spec["family"] == "ordinal" else "fit.R")
    )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "execution.log").open("w") as stream:
        subprocess.run(
            [
                rscript,
                str(script),
                str(observations),
                str(specification),
                str(out / "fits"),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    checks = compare_fits(out / "fits", reference)
    receipt = {
        "protocol": "fresh-process-crossed-refit-v1",
        "verification": "fresh optimization with the same R engine; independent numerical reconstruction is recorded separately",
        "rtol": 1e-6,
        "atol": 1e-7,
        "inputs": {
            str(p): file_sha256(p) for p in (observations, specification, script)
        },
        "checks": checks,
    }
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    """Replay one immutable full-data or training fit into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "specification", "reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--engine", type=Path, help="Pinned historical R producer source"
    )
    args = parser.parse_args()
    replay(
        args.observations,
        args.specification,
        args.reference,
        args.out,
        rscript=args.rscript,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
