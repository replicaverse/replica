"""Finite-sample null simulations for conditional crossed held-out loss tests.

These are diagnostics, not a proof of validity on every response distribution.
They never use study outcomes. Retain all per-simulation p-values and coverage.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replica.analysis.crossed.inference.losses import compare_losses


def simulate(
    readers, materials, scenario, *, repetitions=1000, draws=1999, seed=20260927
):
    """Check crossed normal, skewed and incomplete heteroskedastic null samples.

    Args:
        readers: Number of independent simulated reader groups.
        materials: Number of independent simulated material groups.
        scenario: Null distribution: normal, skewed, or incomplete.
        repetitions: Number of independently generated null datasets.
        draws: Number of crossed bootstrap draws per dataset.
        seed: Seed fixing the declared random draws or identity split.
    """
    rng = np.random.default_rng(seed)
    reader, material = np.indices((readers, materials))
    rows = []
    for repetition in range(repetitions):

        def sample(shape):
            if scenario == "skewed":
                return rng.exponential(size=shape) - 1
            return rng.normal(size=shape)

        delta = (
            sample((readers, 1)) + sample((1, materials)) + sample((readers, materials))
        )
        keep = np.ones(delta.shape, dtype=bool)
        if scenario == "incomplete":
            delta *= np.exp(rng.normal(scale=0.5, size=(readers, 1)))
            keep = rng.uniform(size=delta.shape) < 0.65
        frame = pd.DataFrame({"participant": reader[keep], "material": material[keep]})
        result, _ = compare_losses(
            frame, delta[keep], np.zeros(keep.sum()), draws=draws, seed=repetition
        )
        rows.append(
            {
                "repetition": repetition,
                "p_value": result["p_value"],
                "p_bootstrap": result["p_bootstrap"],
                "p_student_guard": result["p_student_guard"],
                "covers_zero": result["ci_low"] <= 0 <= result["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def main():
    """Write null calibration draws with their exact generating design."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readers", type=int, required=True)
    parser.add_argument("--materials", type=int, required=True)
    parser.add_argument(
        "--scenario", choices=("normal", "skewed", "incomplete"), required=True
    )
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1999)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    table = simulate(
        args.readers,
        args.materials,
        args.scenario,
        repetitions=args.repetitions,
        draws=args.draws,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.out / "simulations.tsv", sep="\t", index=False)
    receipt = {
        **vars(args),
        "out": str(args.out),
        "seed": 20260927,
        "rejection_rate": float(table.p_value.lt(0.05).mean()),
        "unguarded_rejection_rate": float(table.p_bootstrap.lt(0.05).mean()),
        "coverage": float(table.covers_zero.mean()),
    }
    (args.out / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
