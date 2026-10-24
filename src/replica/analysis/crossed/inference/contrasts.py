"""Nested fixed-effect likelihood-ratio contrasts for matched crossed models.

The chi-square reference is asymptotic. It is not a predictive test or a
parametric-bootstrap result; finite-group and boundary limitations remain.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from replica.utils.digest import file_sha256


def random_signature(info):
    """Require identical declared covariance designs, allowing different estimates.

    Args:
        info: Recorded fitted-model diagnostics, including variance components.
    """
    result = []
    for component in info["variances"]:
        if "grp" in component:
            if component["grp"] != "Residual":
                result.append(
                    (component["grp"], component["var1"], component.get("var2", ""))
                )
        else:
            terms = component["terms"]
            if isinstance(terms, str):
                terms = [terms]
            result.append(
                (
                    component["group"],
                    tuple(terms),
                    np.array(component["covariance"]).shape,
                )
            )
    return sorted(result)


def nested_test(reduced, full, reduced_terms, full_terms):
    """Reject different samples, random designs, nonnested terms and worse full fits.

    Args:
        reduced: Diagnostics for the reduced fitted model.
        full: Diagnostics for the full fitted model.
        reduced_terms: Declared reduced-model fixed-effect terms.
        full_terms: Declared full-model fixed-effect terms.
    """
    if reduced["nobs"] != full["nobs"] or random_signature(reduced) != random_signature(
        full
    ):
        raise ValueError("Contrast requires matched samples and random designs")
    for info in (reduced, full):
        if (
            info["optimizer_code"] != 0
            or info.get("convergence_messages")
            or not np.isfinite(info["loglik"])
        ):
            raise ValueError("Unresolved model convergence")
        if "max_gradient" in info and (
            info["max_gradient"] > 0.01 or info["verified_inner_gradient"] > 1e-6
        ):
            raise ValueError("Unresolved ordinal convergence")
    reduced_terms, full_terms = set(reduced_terms), set(full_terms)
    if not reduced_terms <= full_terms:
        raise ValueError("Fixed effects are not nested")
    degrees = len(full_terms - reduced_terms)
    delta = full["loglik"] - reduced["loglik"]
    if delta < -1e-5:
        raise ValueError("Full model likelihood is worse than nested reduced model")
    if not degrees:
        return {
            "status": "unidentifiable",
            "reason": "Focal terms add no estimable fixed-effect dimension",
            "df": 0,
            "statistic": 0.0,
            "p_value": 1.0,
        }
    statistic = max(0.0, 2 * delta)
    return {
        "status": "complete",
        "df": degrees,
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, degrees)),
        "reduced_loglik": reduced["loglik"],
        "full_loglik": full["loglik"],
        "boundary": bool(
            reduced.get("singular", reduced.get("boundary", False))
            or full.get("singular", full.get("boundary", False))
        ),
    }


def evaluate(fits, comparisons, out):
    """Evaluate a frozen nested family from independently retained fit diagnostics.

    Args:
        fits: Directory containing fitted coefficients, scaling and diagnostics.
        comparisons: Frozen named model contrasts.
        out: Destination for newly computed outputs.
    """
    diagnostics = json.loads((fits / "diagnostics.json").read_text())
    if (fits / "coefficients.tsv").exists():
        coefficients = pd.read_csv(fits / "coefficients.tsv", sep="\t")
        terms = {
            name: group.set_index("term").Estimate.to_dict()
            for name, group in coefficients.groupby("model")
        }
    else:
        terms = {
            name: value["coefficients"] or {} for name, value in diagnostics.items()
        }
    rows = []
    for comparison in comparisons:
        reduced, full = comparison["reduced"], comparison["full"]
        result = nested_test(
            diagnostics[reduced], diagnostics[full], terms[reduced], terms[full]
        )
        row = {**comparison, **result}
        if comparison.get("term") is not None:
            row["coefficient"] = terms[full].get(comparison["term"], np.nan)
            row["expected_direction"] = comparison["direction"]
            row["direction_matches"] = bool(
                row["coefficient"] * row["expected_direction"] > 0
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["p_holm"] = multipletests(frame.p_value, method="holm")[1]
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "tests.tsv", sep="\t", index=False)
    (out / "source.json").write_text(
        json.dumps(
            {
                "inference": "asymptotic nested fixed-effect likelihood-ratio test; common random design and fitted observations",
                "limitations": "finite-group and boundary approximations; not predictive superiority or independent historical model selection",
                "comparisons": comparisons,
                "inputs": {
                    p.name: file_sha256(p)
                    for p in fits.iterdir()
                    if p.name
                    in {
                        "diagnostics.json",
                        "coefficients.tsv",
                        "scaling.json",
                        "fixed_design.json",
                    }
                },
                "files": {"tests.tsv": file_sha256(out / "tests.tsv")},
            },
            indent=2,
        )
        + "\n"
    )
    return frame


def main():
    """Evaluate explicitly nested association contrasts, never nonnested rankings."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fits", "comparisons", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.fits, json.loads(args.comparisons.read_text()), args.out)


if __name__ == "__main__":
    main()
