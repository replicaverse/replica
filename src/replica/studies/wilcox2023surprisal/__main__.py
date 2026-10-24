"""End-to-end driver for the Wilcox et al. (2023) reproduction.

Runs the paper's complete design -- 11 languages x 4 language models x 3
reading measures x 3 contrasts -- through ``evaluate()`` and, for every
cell, the paper's own significance test: a two-sided sign-flip permutation
test on the per-datapoint held-out delta log-likelihood
(``analysis_surp.Rmd``/``analysis_ent.Rmd``). It also runs the Rmd's
pooled-over-languages tests, which appear as ``lang="all"`` rows.

Usage::

    python -m replica.studies.wilcox2023surprisal --out /tmp/wilcox_sig.tsv

Compare against the official engine with
``wilcox2023surprisal_r_reference.R --sig-out`` (which needs the Rmd's
``jmuOutlier``); ``--compare`` prints the cell-by-cell agreement.

``--corrected-entropy`` is the reproduction of the entropy contrasts with the
predictor the paper defines: mGPT scored live over each trial's released
text, its first-subword token entropy (and its own surprisal) replacing the
released columns, which sum the entropy over a word's subwords. It runs the
long-context mGPT slice only; the controls stay the released ones.

``--link`` runs the linear-versus-nonlinear link comparison instead. An option
that only the other run reads is a usage error.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from replica.feature import FeatureModel
from replica.studies.wilcox2023surprisal import (
    CORRECTED_ENTROPY_MODEL,
    DEFAULT_LM,
    DVS,
    LANGUAGES,
    LMS,
    Wilcox2023Surprisal,
    cached_outputs,
    corrected_entropy_model,
    signflip_summary,
)

CONTRASTS = ["surp", "ent_add", "ent_replace"]
#: Options only the significance grid reads, and options only ``--link`` reads.
GRID_OPTIONS = ("dvs", "n_resamples", "seed", "compare", "corrected_entropy")
LINK_OPTIONS = ("link_num_sim", "link_seed")
#: Values those options take when the command line leaves them unset.
DEFAULTS = {
    "dvs": DVS,
    "n_resamples": 100_000,
    "seed": 0,
    "link_num_sim": 1000,
    "link_seed": 444,
}


def link_grid(
    study: Wilcox2023Surprisal,
    languages: list[str],
    lms: list[str],
    num_sim: int,
    seed: int,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """The linear-versus-nonlinear link comparison, one row per cell.

    Same columns as ``wilcox2023surprisal_r_link_reference.tsv``, so the two engines
    can be joined cell for cell.

    Args:
        study: Study whose :meth:`~Wilcox2023Surprisal.link_cell` scores each
            cell.
        languages: MECO language codes to run.
        lms: Released language-model tags to run.
        num_sim: ``perm.test`` draws per link cell.
        seed: Seed of the link permutation test.
        data_root: Directory the study's dependency paths resolve against.
    """
    rows = []
    for lang in languages:
        for lm in lms:
            cell = study.link_cell(
                language=lang, data_root=data_root, lm=lm, num_sim=num_sim, seed=seed
            )
            rows.append(cell)
            print(
                f"[link] {lang:3s} {lm:9s} comp={cell['mean_dll_comp']:+.5f} "
                f"p_unpaired={cell['p_unpaired']:.3f} p_paired={cell['p_paired']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def significance_grid(
    study: Wilcox2023Surprisal,
    languages: list[str],
    lms: list[str],
    dvs: list[str],
    n_resamples: int,
    seed: int,
    data_root: Path | None = None,
    model: FeatureModel | None = None,
) -> pd.DataFrame:
    """Every cell's test, plus the Rmd's pooled-over-languages rows.

    Args:
        study: Study that computes each cell's per-datapoint ΔLL vectors.
        languages: MECO language codes to run.
        lms: Released language-model tags to run.
        dvs: Reading measures to run.
        n_resamples: Sign draws per test.
        seed: Sign-flip seed.
        data_root: Directory the study's dependency paths resolve against.
        model: Live model scored over each language's trials, supplying the
            surprisal and first-subword token entropy of the corrected-predictor
            reproduction; ``None`` uses the released columns.
    """
    rows: list[dict] = []
    pooled: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for lang in languages:
        cached = (
            None if model is None else cached_outputs(study, model, lang, data_root)
        )
        for lm in lms:
            for dv in dvs:
                vectors = study.per_obs_delta_ll_all(
                    language=lang,
                    data_root=data_root,
                    lm=lm,
                    dv=dv,
                    model=cached,
                    token_entropy=cached is not None,
                )
                for contrast in CONTRASTS:
                    deltas = vectors[contrast]
                    rows.append(
                        {
                            "lang": lang,
                            "lm": lm,
                            "dv": dv,
                            "contrast": contrast,
                            **signflip_summary(
                                deltas, n_resamples=n_resamples, seed=seed
                            ),
                        }
                    )
                    pooled.setdefault((lm, dv, contrast), []).append(deltas)
                print(
                    f"[cell] {lang:3s} {lm:9s} {dv:11s} "
                    + "  ".join(
                        f"{c}={rows[-len(CONTRASTS) + i]['mean_dll']:+.3e}"
                        for i, c in enumerate(CONTRASTS)
                    ),
                    flush=True,
                )
    for (lm, dv, contrast), vectors in pooled.items():
        rows.append(
            {
                "lang": "all",
                "lm": lm,
                "dv": dv,
                "contrast": contrast,
                **signflip_summary(
                    np.concatenate(vectors), n_resamples=n_resamples, seed=seed
                ),
            }
        )
    return pd.DataFrame(rows)


def compare(replica: pd.DataFrame, reference: Path, alpha: float) -> pd.DataFrame:
    """Join REPLICA's grid to the official engine's and score agreement.

    The two engines draw their own sign patterns, so the value-for-value
    comparison is on the deterministic quantities (the tested vector's
    moments and the RNG-free normal-limit p) and on the decisions.

    Args:
        replica: REPLICA's grid, as :func:`significance_grid` returns it.
        reference: Official-engine significance TSV written by
            ``wilcox2023surprisal_r_reference.R --sig-out``.
        alpha: Threshold both engines' permutation p-values are decided at.
    """
    keys = ["lang", "lm", "dv", "contrast"]
    ref = pd.read_csv(reference, sep="\t")
    both = replica.merge(ref, on=keys, suffixes=("", "_r"))
    if len(both) != len(replica):
        missing = set(map(tuple, replica[keys].values)) - set(
            map(tuple, ref[keys].values)
        )
        raise SystemExit(f"{len(missing)} REPLICA cells absent from {reference}")
    for col in ("n", "mean_dll", "sd_dll", "z_signflip", "p_normal"):
        both[f"dev_{col}"] = (both[col] - both[f"{col}_r"]).abs() / (
            both[f"{col}_r"].abs().clip(lower=1e-30)
        )
    both["sig"] = both["p_perm"] < alpha
    both["sig_r"] = both["p_perm_r"] < alpha
    both["decision_agrees"] = both["sig"] == both["sig_r"]
    both["sign_agrees"] = np.sign(both["mean_dll"]) == np.sign(both["mean_dll_r"])
    return both


def main() -> None:
    if sys.argv[1:2] == ["material-cv"]:
        from .replication import material_cv

        return material_cv(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--lms", nargs="+", default=LMS)
    ap.add_argument(
        "--dvs", nargs="+", help=f"reading measures (default: {' '.join(DVS)})"
    )
    ap.add_argument(
        "--n-resamples",
        type=int,
        help="sign draws per test (default: 100000; the Rmd uses 500)",
    )
    ap.add_argument("--seed", type=int, help="sign-flip seed (default: 0)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, help="write the grid as TSV")
    ap.add_argument(
        "--compare",
        type=Path,
        help="official-engine significance TSV from wilcox2023surprisal_r_reference.R --sig-out",
    )
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument(
        "--link",
        action="store_true",
        help="run the linear-versus-nonlinear link comparison instead",
    )
    ap.add_argument(
        "--link-num-sim",
        type=int,
        help="perm.test draws per link cell (default: 1000, as the release uses)",
    )
    ap.add_argument("--link-seed", type=int, help="link test seed (default: 444)")
    ap.add_argument(
        "--corrected-entropy",
        action="store_true",
        help="score mGPT live with first-subword token entropy in place of the "
        "released summed-subword column (long-context mGPT slice only)",
    )
    args = ap.parse_args()
    given = [
        "--" + name.replace("_", "-")
        for name in (GRID_OPTIONS if args.link else LINK_OPTIONS)
        if getattr(args, name) is not None and getattr(args, name) is not False
    ]
    if given:
        ap.error(
            f"--link does not read {', '.join(given)}"
            if args.link
            else f"only --link reads {', '.join(given)}"
        )
    for name, value in DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    study = Wilcox2023Surprisal()
    model = None
    if args.corrected_entropy:
        model = corrected_entropy_model()
        args.lms = [DEFAULT_LM]
        print(
            f"[corrected entropy] {CORRECTED_ENTROPY_MODEL} live, lm slice {DEFAULT_LM}"
        )
    if args.link:
        grid = link_grid(
            study,
            args.languages,
            args.lms,
            args.link_num_sim,
            args.link_seed,
            args.data_root,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            grid.to_csv(args.out, sep="\t", index=False)
            print(f"[wrote] {args.out} ({len(grid)} rows)")
        not_sig = grid[grid["p_unpaired"] >= args.alpha]
        print(
            f"\n{len(grid)} cells; as the release calls its test, "
            f"{len(not_sig)}/{len(grid)} are not significant at "
            f"alpha={args.alpha}; called paired, "
            f"{int((grid['p_paired'] < args.alpha).sum())} are significant"
        )
        return
    grid = significance_grid(
        study,
        args.languages,
        args.lms,
        args.dvs,
        args.n_resamples,
        args.seed,
        args.data_root,
        model=model,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.out, sep="\t", index=False)
        print(f"[wrote] {args.out} ({len(grid)} rows)")

    per_lang = grid[grid["lang"] != "all"]
    print(
        f"\n{len(per_lang)} language cells, {len(grid) - len(per_lang)} pooled rows; "
        f"significant at alpha={args.alpha}:"
    )
    for contrast in CONTRASTS:
        sub = per_lang[per_lang["contrast"] == contrast]
        pos = sub[(sub["p_perm"] < args.alpha) & (sub["mean_dll"] > 0)]
        neg = sub[(sub["p_perm"] < args.alpha) & (sub["mean_dll"] < 0)]
        print(
            f"  {contrast:12s} {len(pos):3d} positive, {len(neg):3d} negative "
            f"of {len(sub)}"
        )
        pooled = grid[(grid["lang"] == "all") & (grid["contrast"] == contrast)]
        for _, row in pooled.iterrows():
            mark = "*" if row["p_perm"] < args.alpha else " "
            print(
                f"    pooled {row['lm']:9s} {row['dv']:11s} "
                f"mean={row['mean_dll']:+.3e} p={row['p_perm']:.4f} {mark}"
            )

    if args.compare:
        both = compare(grid, args.compare, args.alpha)
        print(f"\nvs {args.compare}:")
        for col in ("n", "mean_dll", "sd_dll", "z_signflip", "p_normal"):
            print(
                f"  max relative deviation on {col:11s} = {both[f'dev_{col}'].max():.3e}"
            )
        print(
            f"  sign agreement      {int(both['sign_agrees'].sum())}/{len(both)}\n"
            f"  decision agreement  {int(both['decision_agrees'].sum())}/{len(both)}"
        )
        bad = both[~both["decision_agrees"]]
        for _, row in bad.iterrows():
            print(
                f"    disagrees: {row['lang']:3s} {row['lm']:9s} {row['dv']:11s} "
                f"{row['contrast']:12s} replica p={row['p_perm']:.4f} "
                f"official p={row['p_perm_r']:.4f} (normal-limit p={row['p_normal']:.4g})"
            )


if __name__ == "__main__":
    main()
