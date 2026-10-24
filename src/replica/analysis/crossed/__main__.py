"""Prepare a frozen crossed holdout or compare its held-out losses."""

import pandas as pd

from replica.analysis.crossed.inference.losses import evaluate
from replica.analysis.crossed.specification.holdout import prepare


def main():
    """Prepare a crossed holdout or compare its recorded test losses."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", choices=("prepare", "compare"), default="prepare"
    )
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--specification", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=9999)
    parser.add_argument(
        "--seed", type=int, default=20260927, help="Holdout or bootstrap seed"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.25,
        help="Holdout fraction per axis (prepare)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Holm significance threshold (compare)",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        if args.observations is None or args.specification is None:
            parser.error("prepare requires --observations and --specification")
        frame = pd.read_csv(
            args.observations,
            sep="\t",
            dtype={"participant": str, "material": str, "item": str},
        )
        spec = json.loads(args.specification.read_text())
        print(
            json.dumps(
                prepare(frame, spec, args.out, fraction=args.fraction, seed=args.seed)
            )
        )
    else:
        if args.predictions is None or args.comparisons is None:
            parser.error("compare requires --predictions and --comparisons")
        comparisons = json.loads(args.comparisons.read_text())
        print(
            evaluate(
                args.predictions,
                comparisons,
                args.out,
                draws=args.draws,
                seed=args.seed,
                alpha=args.alpha,
            ).to_json(orient="records")
        )


if __name__ == "__main__":
    main()
