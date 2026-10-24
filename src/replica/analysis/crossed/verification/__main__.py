"""Verify fitted likelihoods or held-out predictions."""

import argparse
import json
from pathlib import Path

import pandas as pd

from replica.analysis.crossed.verification.likelihood import verify_fit
from replica.analysis.crossed.verification.predictions import verify_predictions
from replica.utils.digest import file_sha256


def verify_fit_main(argv=None):
    """Write an independent marginal-likelihood verification receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("Verification receipt already exists")
    frame = pd.read_csv(
        args.observations,
        sep="\t",
        dtype={"participant": str, "material": str, "item": str},
    )
    spec = json.loads(args.specification.read_text())
    results = verify_fit(frame, spec, args.fits)
    inputs = [args.observations, args.specification, *sorted(args.fits.iterdir())]
    receipt = {
        "verification": "SciPy marginal covariance, GLS and Gaussian log likelihood",
        "models": results,
        "files": {str(p): file_sha256(p) for p in inputs if p.is_file()},
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(results))


def verify_predictions_main(argv=None):
    """Verify every held-out prediction and loss before interpreting comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "specification", "fits", "predictions", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_predictions(
        args.observations, args.specification, args.fits, args.predictions
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")


def main(argv=None):
    """Select likelihood or prediction verification from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("fit", "predictions"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.check == "fit":
        verify_fit_main(args.arguments)
    else:
        verify_predictions_main(args.arguments)


if __name__ == "__main__":
    main()
