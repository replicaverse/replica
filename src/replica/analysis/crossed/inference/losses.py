"""Conditional crossed inference from paired held-out losses."""

from numbers import Integral, Real

import numpy as np
import pandas as pd


def compare_losses(frame, baseline, target, *, draws=9999, seed=20260927):
    """Paired loss reduction with a two-axis pigeonhole bootstrap on held-out data.

    Whole test participants and materials are resampled independently. Every
    observed reader-material block receives the product of its multiplicities.
    The centered one-sided bootstrap is approximate; folds are never tested as
    independent replicates. The estimand is mean observed reader-item loss.

    Args:
        frame: Independent test observations, including participant and material.
        baseline: Finite per-observation loss for the frozen baseline model.
        target: Finite per-observation loss for the frozen target model.
        draws: Number of bootstrap samples, excluding the observed sample.
        seed: NumPy RNG seed or generator; None selects fresh randomness.
    """
    from scipy.stats import t

    baseline, target = np.asarray(baseline, float), np.asarray(target, float)
    if baseline.shape != (len(frame),) or target.shape != baseline.shape:
        raise ValueError("Paired losses must match every held-out observation")
    if (
        not np.isfinite([baseline, target]).all()
        or isinstance(draws, bool)
        or not isinstance(draws, Integral)
        or draws < 2
    ):
        raise ValueError("Invalid losses or bootstrap budget")
    if frame[["participant", "material"]].isna().any().any():
        raise ValueError("Missing crossed identity")
    reader, readers = pd.factorize(frame.participant, sort=True)
    material, materials = pd.factorize(frame.material, sort=True)
    nr, nm = len(readers), len(materials)
    if min(nr, nm) < 2:
        raise ValueError("At least two independent identities per axis required")
    sums, counts = np.zeros((nr, nm)), np.zeros((nr, nm))
    np.add.at(sums, (reader, material), baseline - target)
    np.add.at(counts, (reader, material), 1)
    rng = np.random.default_rng(seed)
    values = np.full(draws, np.nan)
    for start in range(0, draws, 256):
        stop = min(draws, start + 256)
        rw = rng.multinomial(nr, np.full(nr, 1 / nr), size=stop - start)
        mw = rng.multinomial(nm, np.full(nm, 1 / nm), size=stop - start)
        denominator = np.einsum("bi,ij,bj->b", rw, counts, mw)
        numerator = np.einsum("bi,ij,bj->b", rw, sums, mw)
        values[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.full(stop - start, np.nan),
            where=denominator > 0,
        )
    effect = float(np.mean(baseline - target))
    errors = values - effect
    failed = ~np.isfinite(errors)
    extreme = (
        failed | (errors >= effect) | np.isclose(errors, effect, rtol=1e-12, atol=1e-15)
    )
    bootstrap_p = float((1 + extreme.sum()) / (1 + draws))
    # Combine bootstrap uncertainty with a finite-group Student reference.
    # This remains approximate: min(G,H)-1 is a conservative multiway reference,
    # and the resampling SE does not make a small group count asymptotic.
    degrees = min(nr, nm) - 1
    spread = float(np.std(values, ddof=1)) if not failed.any() else float("inf")
    if spread == 0:
        student_p = float(effect <= 0)
    else:
        student_p = float(t.sf(effect / spread, degrees))
    radius = float(t.ppf(0.975, degrees) * spread)
    summary = {
        "protocol": "conditional-crossed-pigeonhole-tguard-v2",
        "effect": effect,
        "direction": "positive means lower target loss",
        "p_value": max(bootstrap_p, student_p),
        "p_bootstrap": bootstrap_p,
        "p_student_guard": student_p,
        "bootstrap_standard_error": spread,
        "student_degrees_freedom": degrees,
        "ci_low": float(
            effect
            - np.quantile(
                np.where(failed, np.inf, errors), 0.975, method="inverted_cdf"
            )
        ),
        "ci_high": float(
            effect
            - np.quantile(
                np.where(failed, -np.inf, errors), 0.025, method="inverted_cdf"
            )
        ),
        "draws": draws,
        "failed_draws": int(failed.sum()),
        "seed": seed,
        "participants": nr,
        "materials": nm,
        "observations": len(frame),
        "approximation": "two-axis bootstrap; conditions on fixed training models",
    }
    summary["ci_low"] = min(summary["ci_low"], effect - radius)
    summary["ci_high"] = max(summary["ci_high"], effect + radius)
    return summary, values


def check_alpha(alpha):
    """Require a finite significance threshold strictly between zero and one.

    Args:
        alpha: Family-wise significance threshold for the assessed procedure.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")


def check_seed(seed):
    """Require a reproducible nonnegative integer bootstrap seed.

    Args:
        seed: Seed for paired participant/material multiplicities.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")


def check_bootstrap_budget(draws, family_tests, *, alpha=0.05):
    """Reject a budget whose p-value floor precludes any full-family rejection.

    Args:
        draws: Number of bootstrap replicates, excluding the observed sample.
        family_tests: Complete Holm family size, including unavailable comparisons.
        alpha: Significance threshold used for the family.
    """
    check_alpha(alpha)
    if (
        isinstance(draws, bool)
        or not isinstance(draws, Integral)
        or draws < 2
        or isinstance(family_tests, bool)
        or not isinstance(family_tests, Integral)
        or family_tests < 1
    ):
        raise ValueError("Invalid bootstrap budget or family size")
    if family_tests / (draws + 1) >= alpha:
        raise ValueError("Bootstrap resolution precludes any Holm rejection")


def evaluate(
    predictions_path, comparisons, out, *, draws=9999, seed=20260927, alpha=0.05
):
    """Evaluate a frozen family of paired held-out loss comparisons with Holm adjustment.

    Args:
        predictions_path: Authenticated output of predict.R on independent test rows.
        comparisons: Frozen list of name/baseline/target mappings, before examining losses.
        out: Fresh directory for tests, bootstrap draws and input identities.
        draws: Declared bootstrap budget; callers must check the complete family.
        seed: Seed shared by the paired bootstrap comparisons.
        alpha: Threshold for Holm-adjusted support; intervals retain 95% coverage.
    """
    import json

    from statsmodels.stats.multitest import multipletests

    from replica.utils.digest import file_sha256

    frame = pd.read_csv(
        predictions_path,
        sep="\t",
        dtype={"participant": str, "material": str, "item": str},
    )
    if frame.duplicated(["participant", "item"]).any():
        raise ValueError("Repeated test reader-item observation")
    if not comparisons or len({c["name"] for c in comparisons}) != len(comparisons):
        raise ValueError("Empty or duplicated comparison family")
    check_bootstrap_budget(draws, len(comparisons), alpha=alpha)
    check_seed(seed)
    alpha, draws = float(alpha), int(draws)
    if isinstance(seed, np.integer):
        seed = int(seed)
    tables, samples = [], {}
    for comparison in comparisons:
        baseline, target = comparison["baseline"], comparison["target"]
        result, values = compare_losses(
            frame,
            frame[baseline + "_loss"],
            frame[target + "_loss"],
            draws=draws,
            seed=seed,
        )
        tables.append({**comparison, **result})
        samples[comparison["name"]] = values
    table = pd.DataFrame(tables)
    table["p_holm"] = multipletests(table.p_value, method="holm")[1]
    table["holds"] = table.effect.gt(0) & table.p_holm.lt(alpha)
    out.mkdir(parents=True, exist_ok=False)
    table.to_csv(out / "tests.tsv", sep="\t", index=False)
    np.savez_compressed(out / "bootstrap.npz", **samples)
    receipt = {
        "predictions_sha256": file_sha256(predictions_path),
        "comparisons": comparisons,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "alpha": alpha,
        "multiplicity": "Holm within this frozen family",
        "inference": "conditional on independently fitted training models",
        "files": {p.name: file_sha256(p) for p in sorted(out.iterdir())},
    }
    (out / "source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return table
