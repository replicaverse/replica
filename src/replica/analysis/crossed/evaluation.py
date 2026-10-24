"""Evaluate frozen crossed-model cells and summarize complete multiplicity families."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from replica.analysis.crossed.inference.losses import (
    check_alpha,
    check_bootstrap_budget,
    check_seed,
)
from replica.utils.digest import file_sha256


def cell_digest(cell):
    """Bind one cell's model specification and input paths.

    Args:
        cell: Frozen model-cell specification.
    """
    return hashlib.sha256(json.dumps(cell, sort_keys=True).encode()).hexdigest()


def ready(cell, root):
    """Require every fitted model; an environment/session file is not completion.

    Args:
        cell: Frozen model-cell specification.
        root: Root directory containing the recorded analysis inputs and receipts.
    """
    if cell.get("unavailable"):
        return True
    paths = [root / cell[key] for key in ("observations", "specification")]
    fits = root / cell["fits"]
    paths.append(fits / "diagnostics.json")
    if cell["kind"] == "prediction":
        paths.append(root / cell["predictions"])
    if not all(path.is_file() for path in paths):
        return False
    specification = json.loads((root / cell["specification"]).read_text())
    diagnostics = json.loads((fits / "diagnostics.json").read_text())
    if set(diagnostics) != set(specification["models"]):
        return False
    if specification["family"] != "ordinal":
        if not (fits / "coefficients.tsv").is_file():
            return False
        coefficients = pd.read_csv(fits / "coefficients.tsv", sep="\t")
        if set(coefficients.model) != set(specification["models"]):
            return False
    return True


def validate_inventory(manifest):
    """Require safe cell identities and the declared complete comparison inventory.

    Args:
        manifest: Frozen cell inventory and complete test-family sizes.
    """
    cells = manifest["cells"]
    ids = [cell["id"] for cell in cells]
    if len(ids) != len(set(ids)) or any(
        Path(i).is_absolute() or ".." in Path(i).parts for i in ids
    ):
        raise ValueError("Duplicated or unsafe cell identity")
    counts = {family: 0 for family in manifest["families"]}
    for cell in cells:
        if cell["family"] not in counts or cell["kind"] not in (
            "prediction",
            "association",
        ):
            raise ValueError("Unknown comparison family or kind")
        comparisons = cell["comparisons"]
        if not comparisons or len({c["name"] for c in comparisons}) != len(comparisons):
            raise ValueError("Empty or duplicated within-cell comparisons")
        counts[cell["family"]] += len(comparisons)
    if counts != manifest["families"]:
        raise ValueError("Incomplete declared multiplicity family")


def validate_manifest(manifest):
    """Require explicit inference settings before executing a frozen analysis.

    Args:
        manifest: Complete inventory with alpha and each executable prediction
            cell's bootstrap_draws and bootstrap_seed.
    """
    validate_inventory(manifest)
    if "alpha" not in manifest:
        raise ValueError("Frozen manifest requires explicit alpha")
    alpha = manifest["alpha"]
    check_alpha(alpha)
    for cell in manifest["cells"]:
        if cell["kind"] == "prediction" and not cell.get("unavailable"):
            if not {"bootstrap_draws", "bootstrap_seed"} <= cell.keys():
                raise ValueError(
                    "Prediction cell requires bootstrap_draws and bootstrap_seed"
                )
            check_seed(cell["bootstrap_seed"])
            check_bootstrap_budget(
                cell["bootstrap_draws"],
                manifest["families"][cell["family"]],
                alpha=alpha,
            )


def evaluate_cell(manifest, index, root, out):
    """Verify and evaluate one complete fitted cell, or retain its explicit data limit.

    Args:
        manifest: Frozen cell inventory and complete test-family sizes.
        index: Zero-based index of the frozen model cell.
        root: Root directory containing the recorded analysis inputs and receipts.
        out: Destination for newly computed outputs.
    """
    validate_manifest(manifest)
    alpha = manifest["alpha"]
    cell = manifest["cells"][index]
    if not ready(cell, root):
        raise RuntimeError("Complete fitted-model inventory is not available yet")
    destination = out / cell["id"]
    independent = None
    if cell.get("unavailable"):
        destination.mkdir(parents=True, exist_ok=False)
        table = pd.DataFrame(
            [
                {
                    **c,
                    "status": "unavailable",
                    "reason": cell["unavailable"],
                    "p_value": 1.0,
                }
                for c in cell["comparisons"]
            ]
        )
        table.to_csv(destination / "tests.tsv", sep="\t", index=False)
    elif cell["kind"] == "prediction":
        from replica.analysis.crossed.inference.losses import evaluate
        from replica.analysis.crossed.verification.predictions import (
            verify_predictions as verify,
        )

        independent = verify(
            *[
                root / cell[key]
                for key in ("observations", "specification", "fits", "predictions")
            ]
        )
        table = evaluate(
            root / cell["predictions"],
            cell["comparisons"],
            destination,
            draws=cell["bootstrap_draws"],
            seed=cell["bootstrap_seed"],
            alpha=alpha,
        )
        table["status"] = "complete"
    else:
        from replica.analysis.crossed.inference.contrasts import evaluate
        from replica.analysis.crossed.verification.likelihood import verify_fit

        spec = json.loads((root / cell["specification"]).read_text())
        if spec["family"] == "gaussian":
            frame = pd.read_csv(
                root / cell["observations"],
                sep="\t",
                dtype={g: str for g in ("participant", "material", "item")},
            )
            independent = verify_fit(frame, spec, root / cell["fits"])
        table = evaluate(root / cell["fits"], cell["comparisons"], destination)
    if independent is not None:
        (destination / "independent.json").write_text(
            json.dumps(independent, indent=2) + "\n"
        )
    table = table.drop(columns=[c for c in ("p_holm", "holds") if c in table])
    table["cell"], table["family"], table["kind"] = (
        cell["id"],
        cell["family"],
        cell["kind"],
    )
    for key in ("study", "corpus", "response", "predictor"):
        if key in cell:
            table[key] = cell[key]
    table.to_csv(destination / "family-input.tsv", sep="\t", index=False)
    (destination / "cell.json").write_text(
        json.dumps(
            {
                "cell_sha256": cell_digest(cell),
                "alpha": alpha,
                "cell": cell,
                "input_files": {
                    cell[key]: file_sha256(root / cell[key])
                    for key in ("observations", "specification", "predictions")
                    if key in cell and not cell.get("unavailable")
                },
                "files": {p.name: file_sha256(p) for p in destination.iterdir()},
            },
            indent=2,
        )
        + "\n"
    )
    return table


def adjust(table, families, *, alpha=0.05):
    """Use complete frozen denominators; unavailable tests are never dropped.

    Args:
        table: Verified headline comparison table.
        families: Declared test families and their complete comparison counts.
        alpha: Threshold for Holm-adjusted support.
    """
    check_alpha(alpha)
    if table.duplicated(["cell", "name"]).any():
        raise ValueError("Duplicate family comparison")
    counts = table.groupby("family").size().to_dict()
    if counts != families:
        raise ValueError("Incomplete family cannot be closed out")
    if not np.isfinite(table.p_value).all() or not table.p_value.between(0, 1).all():
        raise ValueError("Invalid probability in family")
    table = table.copy()
    unavailable = table.status.ne("complete")
    if not table.loc[unavailable, "p_value"].eq(1).all():
        raise ValueError("Unavailable comparisons must retain p=1")
    table["p_holm"] = np.nan
    for _, group in table.groupby("family"):
        table.loc[group.index, "p_holm"] = multipletests(group.p_value, method="holm")[
            1
        ]
    table["supports"] = table.status.eq("complete") & table.p_holm.lt(alpha)
    if "effect" in table:
        prediction = table.kind.eq("prediction")
        table.loc[prediction, "supports"] &= table.loc[prediction, "effect"].gt(0)
    if "direction_matches" in table:
        directional = table.direction_matches.notna()
        values = table.loc[directional, "direction_matches"]
        if not values.isin([True, False, "True", "False"]).all():
            raise ValueError("Invalid serialized coefficient direction")
        table.loc[directional, "supports"] &= values.isin([True, "True"])
    return table


def summarize(manifest, reports, out):
    """Authenticate every completed cell before making a final family-level table.

    Args:
        manifest: Frozen cell inventory and complete test-family sizes.
        reports: Directory containing authenticated per-cell reports.
        out: Destination for newly computed outputs.
    """
    validate_manifest(manifest)
    alpha = manifest["alpha"]
    frames, inputs = [], {}
    for cell in manifest["cells"]:
        directory = reports / cell["id"]
        receipt = json.loads((directory / "cell.json").read_text())
        if receipt["cell_sha256"] != cell_digest(cell):
            raise ValueError("Cell contract differs from frozen manifest")
        if "alpha" not in receipt:
            raise ValueError("Evaluated cell receipt requires explicit alpha")
        check_alpha(receipt["alpha"])
        if receipt["alpha"] != alpha:
            raise ValueError("Family alpha differs from the evaluated cell")
        for name, expected in receipt["files"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError("Changed cell evidence")
        inputs[cell["id"]] = file_sha256(directory / "cell.json")
        frames.append(pd.read_csv(directory / "family-input.tsv", sep="\t"))
    table = adjust(
        pd.concat(frames, ignore_index=True), manifest["families"], alpha=alpha
    )
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out / "tests.tsv", sep="\t", index=False)
    (out / "source.json").write_text(
        json.dumps(
            {
                "protocol": "complete-crossed-families-v1",
                "alpha": alpha,
                "families": manifest["families"],
                "cell_receipts": inputs,
                "files": {"tests.tsv": file_sha256(out / "tests.tsv")},
            },
            indent=2,
        )
        + "\n"
    )
    return table


def main():
    """Compute a manifest cell or close out all of its authenticated families."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cell", "summary"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.action == "cell":
        if args.index is None:
            parser.error("cell requires --index")
        evaluate_cell(manifest, args.index, args.root, args.out)
    else:
        summarize(manifest, args.root, args.out)


if __name__ == "__main__":
    main()
