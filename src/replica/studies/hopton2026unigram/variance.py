"""Two-estimator bootstrap for the author branch's approximate MMC recipe.

Inputs contain per-sample probability sums, occurrence counts and boundaries.
The denominator is ``J + sum(boundaries)``. Repeated word forms are weighted
by their corpus occurrences. This recipe predates the manuscript's three-way
pathwise/expectation-equivalent comparison; its final input dumps are separate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..records.resources import resource_file
from .reference import N_BOOTSTRAP, SUBSAMPLE_SIZES


class VarianceInputs:
    """Per-sample estimator terms, laid out for the bootstrap.

    Attributes:
        tokens: The unigram surface forms, in row order of the matrices.
        rb_prob_sums: ``(n_tokens, n_samples)`` MMC numerator terms.
        mc_counts: ``(n_tokens, n_samples)`` NMC numerator terms.
        rb_boundaries: ``(n_samples,)`` boundary counts for the MMC denominator.
        mc_boundaries: ``(n_samples,)`` whitespace-token counts plus one, for
            the NMC denominator.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        rb_prob_sums: np.ndarray,
        mc_counts: np.ndarray,
        rb_boundaries: np.ndarray,
        mc_boundaries: np.ndarray,
        unit_tokens: np.ndarray | None = None,
    ):
        self.tokens = tokens
        self.rb_prob_sums = rb_prob_sums
        self.mc_counts = mc_counts
        self.rb_boundaries = rb_boundaries
        self.mc_boundaries = mc_boundaries
        self.unit_tokens = unit_tokens if unit_tokens is not None else tokens

    @property
    def n_samples(self) -> int:
        return self.rb_prob_sums.shape[1]

    @property
    def n_tokens(self) -> int:
        return self.rb_prob_sums.shape[0]

    @property
    def unit_index(self) -> np.ndarray:
        """Row of the estimate matrices backing each corpus unit."""
        position = {token: i for i, token in enumerate(self.tokens)}
        missing = {t for t in self.unit_tokens if t not in position}
        if missing:
            raise ValueError(
                f"{len(missing)} unit token(s) absent from the estimate tables, "
                f"for example {sorted(missing)[:3]}"
            )
        return np.array([position[t] for t in self.unit_tokens], dtype=int)


def load_inputs(data_dir: Path | str | None = None) -> VarianceInputs:
    """Read archived or caller-supplied per-sample terms into dense matrices.

    Args:
        data_dir: Directory of input tables; ``None`` resolves indexed resources.
    """

    def input_path(name: str) -> Path:
        if data_dir is None:
            return resource_file("hopton2026unigram", f"data/{name}")
        return Path(data_dir) / name

    long = pd.read_parquet(input_path("variance_samples.parquet"))
    bounds = pd.read_parquet(input_path("variance_boundaries.parquet")).sort_values(
        "sample_idx"
    )

    wide_rb = long.pivot(index="token", columns="sample_idx", values="rb_prob_sum")
    wide_mc = long.pivot(index="token", columns="sample_idx", values="mc_count")
    wide_mc = wide_mc.reindex(columns=wide_rb.columns, index=wide_rb.index)

    units_path = input_path("variance_units.parquet")
    unit_tokens = (
        pd.read_parquet(units_path)["token"].to_numpy() if units_path.exists() else None
    )

    return VarianceInputs(
        tokens=wide_rb.index.to_numpy(),
        rb_prob_sums=wide_rb.to_numpy(dtype=float),
        mc_counts=wide_mc.to_numpy(dtype=float),
        rb_boundaries=bounds["rb_boundary_count"].to_numpy(dtype=float),
        mc_boundaries=bounds["mc_boundary_count"].to_numpy(dtype=float),
        unit_tokens=unit_tokens,
    )


def bootstrap_variance(
    numerators: np.ndarray,
    boundaries: np.ndarray,
    subsample_size: int,
    *,
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
    shared_draws: bool = True,
) -> np.ndarray:
    """Per-token bootstrap variance of the estimate at one subsample size.

    Args:
        numerators: ``(n_tokens, n_samples)`` per-sample numerator terms.
        boundaries: ``(n_samples,)`` per-sample denominator terms.
        subsample_size: J, strings drawn with replacement per resample.
        n_boot: Resamples per token.
        rng: Generator; ``None`` seeds a fresh default. The paper's script
            leaves the draw unseeded, so the reported means carry Monte Carlo
            noise of their own, and a rerun reproduces them only up to that
            noise.
        shared_draws: Reuse one set of resamples across tokens, which turns the
            whole sweep into two matrix products. Each token's variance stays
            an unbiased estimate of its own bootstrap variance; only the errors
            across tokens become correlated, which the mean over thousands of
            tokens absorbs. Set False to draw per token as the original script
            does, at a large cost in time.

    Returns:
        ``(n_tokens,)`` variance, ``nan`` where every resample was
        non-positive.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_samples = numerators.shape[1]

    if shared_draws:
        counts = rng.multinomial(
            subsample_size, np.full(n_samples, 1.0 / n_samples), size=n_boot
        ).astype(float)
        estimates = (counts @ numerators.T) / (
            subsample_size + counts @ boundaries
        ).reshape(-1, 1)
    else:
        estimates = np.empty((n_boot, numerators.shape[0]))
        for b in range(n_boot):
            idx = rng.integers(0, n_samples, size=subsample_size)
            estimates[b] = numerators[:, idx].sum(axis=1) / (
                subsample_size + boundaries[idx].sum()
            )

    estimates = np.where(estimates > 0, estimates, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanvar(estimates, axis=0, ddof=1)


def variance_table(
    inputs: VarianceInputs | None = None,
    *,
    subsample_sizes: tuple[int, ...] = SUBSAMPLE_SIZES,
    n_boot: int = N_BOOTSTRAP,
    seed: int | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Mean variance and estimator ratio for the original two-estimator recipe.

    Columns ``J``, ``mmc``, ``nmc``, ``ratio``.
    """
    inputs = inputs if inputs is not None else load_inputs(data_dir)
    rng = np.random.default_rng(seed)
    # Expand per-form variances back to corpus units before averaging, so a
    # frequent word weighs once per occurrence as the paper reports it.
    units = inputs.unit_index

    rows = []
    for size in subsample_sizes:
        mmc = bootstrap_variance(
            inputs.rb_prob_sums, inputs.rb_boundaries, size, n_boot=n_boot, rng=rng
        )
        nmc = bootstrap_variance(
            inputs.mc_counts, inputs.mc_boundaries, size, n_boot=n_boot, rng=rng
        )
        mmc_mean = float(np.nanmean(mmc[units]))
        nmc_mean = float(np.nanmean(nmc[units]))
        rows.append(
            {
                "J": size,
                "mmc": mmc_mean,
                "nmc": nmc_mean,
                "ratio": nmc_mean / mmc_mean if mmc_mean else np.nan,
            }
        )
    return pd.DataFrame(rows)


def figure8(
    inputs: VarianceInputs | None = None,
    *,
    subsample_sizes: tuple[int, ...] = SUBSAMPLE_SIZES,
    n_boot: int = N_BOOTSTRAP,
    seed: int | None = None,
    data_dir: Path | str | None = None,
):
    """Plot the original two-estimator recipe's per-unigram variance curves.

    Returns a matplotlib figure with MMC and NMC panels.
    """
    import matplotlib.pyplot as plt

    inputs = inputs if inputs is not None else load_inputs(data_dir)
    rng = np.random.default_rng(seed)

    curves = {"mmc": [], "nmc": []}
    for size in subsample_sizes:
        curves["mmc"].append(
            bootstrap_variance(
                inputs.rb_prob_sums, inputs.rb_boundaries, size, n_boot=n_boot, rng=rng
            )
        )
        curves["nmc"].append(
            bootstrap_variance(
                inputs.mc_counts, inputs.mc_boundaries, size, n_boot=n_boot, rng=rng
            )
        )

    units = inputs.unit_index
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    styling = [
        ("mmc", "Marginalized Monte Carlo (MMC)", "#4a7fb5"),
        ("nmc", "Naive Monte Carlo (NMC)", "#e07b39"),
    ]
    for ax, (key, title, colour) in zip(axes, styling, strict=True):
        stack = np.vstack(curves[key])[:, units]
        ax.plot(subsample_sizes, stack, color=colour, alpha=0.05, linewidth=0.5)
        ax.plot(
            subsample_sizes,
            np.nanmean(stack, axis=1),
            color="black",
            linewidth=2,
            label="Mean",
        )
        ax.set_title(title)
        ax.set_xlabel("Subsample size $J$")
        ax.set_ylabel("Bootstrap variance")
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
    fig.tight_layout()
    return fig
