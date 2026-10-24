"""Check grouped-CV inference on clustered synthetic nulls and alternatives.

These simulations validate the grouped-CV inference method. Null scenarios
include an irrelevant added predictor and equally predictive nonnested models,
with serially correlated noise and stimulus-level effects in both features and
responses. The predeclared gate is the 99% binomial upper count under alpha .05.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

from replica.analysis.inference.material_bootstrap import PROTOCOL, Contrast, GroupedOLS


def calibrate(n_groups: int, case: str, repetitions: int, n_resamples: int) -> dict:
    """Run one predeclared clustered simulation condition.

    Args:
        n_groups: Number of independent synthetic materials.
        case: ``nested``, ``equal`` or positive ``alternative``.
        repetitions: Independently generated datasets.
        n_resamples: Full-refit bootstrap draws per dataset.
    """
    if case not in ("nested", "equal", "alternative"):
        raise ValueError(case)
    rejected = 0
    bootstrap_rejected = 0
    failures = 0
    for replicate in range(repetitions):
        rng = np.random.default_rng(
            np.random.SeedSequence([20260909, n_groups, replicate])
        )
        sizes = rng.integers(20, 50, n_groups)
        groups = np.repeat(np.arange(n_groups).astype(str), sizes)
        n = len(groups)
        control = rng.normal(size=n)
        x = rng.normal(size=n) + np.repeat(rng.normal(size=n_groups), sizes)
        z = rng.normal(size=n) + np.repeat(rng.normal(size=n_groups), sizes)
        noise = rng.normal(size=n)
        start = 0
        for size in sizes:
            for i in range(start + 1, start + size):
                noise[i] += 0.7 * noise[i - 1]
            start += size
        y = 3 + control + noise + np.repeat(rng.normal(size=n_groups), sizes)
        baseline = ("control",)
        if case == "equal":
            y += 0.5 * x + 0.5 * z
            baseline = ("control", "z")
        if case == "alternative":
            y += x
        engine = GroupedOLS(y, {"control": control, "x": x, "z": z}, groups)
        rows, _ = engine.compare(
            [Contrast(case, baseline, ("control", "x"))],
            n_resamples=n_resamples,
            seed=replicate,
        )
        row = rows[0]
        direction = case == "equal" or row["mean_delta_ll"] > 0
        rejected += direction and row["pvalue"] < 0.05
        bootstrap_rejected += direction and row["p_bootstrap"] < 0.05
        failures += row["bootstrap_failures"]
    limit = int(binom.ppf(0.99, repetitions, 0.05))
    return {
        "protocol": PROTOCOL,
        "n_groups": n_groups,
        "case": case,
        "repetitions": repetitions,
        "n_resamples": n_resamples,
        "rejected": int(rejected),
        "rejection_rate": rejected / repetitions,
        "bootstrap_only_rejected": int(bootstrap_rejected),
        "binomial_99pct_upper_count": limit,
        "rejection_rate_ci95": [
            float(beta.ppf(0.025, rejected, repetitions - rejected + 1))
            if rejected
            else 0.0,
            float(beta.ppf(0.975, rejected + 1, repetitions - rejected))
            if rejected < repetitions
            else 1.0,
        ],
        "gate_passed": None if case == "alternative" else rejected <= limit,
        "failed_bootstrap_draws_counted_in_tail": failures,
    }


def main() -> None:
    """Run a simulation condition and write its reproducible validation result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument(
        "--case", choices=("nested", "equal", "alternative"), required=True
    )
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=399)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.groups, args.case, args.repetitions, args.resamples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["gate_passed"] is False:
        raise SystemExit("Null calibration failed")


if __name__ == "__main__":
    main()
