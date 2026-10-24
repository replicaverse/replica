"""Specification-curve inference over a multiverse grid.

* :func:`spec_curve_descriptives`: median/mean effect, share of specs in
  the predicted direction, share significant in that direction.
* :func:`joint_signflip_inference`: joint sign-flip test of an across-cell
  statistic, on the per-datapoint null of
  :mod:`replica.analysis.inference.significance`.
* :func:`swap_attribution`: type-II variance decomposition of the per-cell
  effect over the swap dimensions.

``shared_observations=True`` applies one synchronized sign vector to all
row-aligned cells, preserving their dependence; ``False`` flips
independently and is valid only for disjoint cells (on shared observations
it is anti-conservative). Swaps that change the data (outlier filter, CV
splitter, DV transform) break row alignment and would need a full refit
under the null, out of scope here.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from replica.analysis.inference.grouped_permutation import (
    aligned_losses,
    inference_scope,
)
from replica.analysis.inference.significance import (
    Alternative,
    addone_pvalue,
    scale_signflip_values,
    sign_block,
    sign_chunks,
    sum_by_group,
    tail_hits,
    validate_signflip_budget,
)

Statistic = Literal["median", "mean", "share_positive"]


@dataclass(frozen=True)
class SpecCurveSummary:
    """Descriptive statistics of one specification curve.

    Attributes:
        n_specs: Number of specifications with a finite effect.
        median_effect: Median effect across specifications.
        mean_effect: Mean effect across specifications.
        effect_min, effect_max: Range of effects.
        frac_in_direction: Fraction of specs whose effect points in the
            predicted direction.
        frac_significant: Fraction of specs in the predicted direction *and*
            significant at ``alpha`` (``NaN`` if no p-values were supplied).
        alpha: Significance threshold used for ``frac_significant``.
    """

    n_specs: int
    median_effect: float
    mean_effect: float
    effect_min: float
    effect_max: float
    frac_in_direction: float
    frac_significant: float
    alpha: float


def spec_curve_descriptives(
    effects: Sequence[float],
    pvalues: Sequence[float] | None = None,
    directions: Sequence[int] | int = 1,
    alpha: float = 0.05,
) -> SpecCurveSummary:
    """Summarize a specification curve.

    Scale-agnostic: pass only mutually comparable effects (e.g. don't mix
    ΔLL/N across the raw↔log DV transform; Cohen's f² is scale-free).

    Args:
        effects: One effect size per specification. Non-finite entries drop
            the specification.
        pvalues: Optional per-spec p-values, aligned with *effects*; when
            ``None``, ``frac_significant`` is ``NaN``.
        directions: Predicted sign per spec (``+1``/``-1``), or one int for
            all.
        alpha: Threshold for ``frac_significant``.
    """
    eff = np.asarray(list(effects), dtype=float)
    n_in = eff.size
    if isinstance(directions, int):
        dirs = np.full(n_in, directions, dtype=float)
    else:
        dirs = np.asarray(list(directions), dtype=float)
        if dirs.size != n_in:
            raise ValueError("directions and effects length mismatch")
    if pvalues is None:
        pv = np.full(n_in, np.nan)
    else:
        pv = np.asarray(list(pvalues), dtype=float)
        if pv.size != n_in:
            raise ValueError("pvalues and effects length mismatch")

    finite = np.isfinite(eff)
    eff_f = eff[finite]
    dirs_f = dirs[finite]
    pv_f = pv[finite]
    n = eff_f.size
    if n == 0:
        return SpecCurveSummary(
            n_specs=0,
            median_effect=float("nan"),
            mean_effect=float("nan"),
            effect_min=float("nan"),
            effect_max=float("nan"),
            frac_in_direction=float("nan"),
            frac_significant=float("nan"),
            alpha=alpha,
        )

    in_dir = np.sign(eff_f) == np.sign(dirs_f)
    frac_in = float(np.mean(in_dir))
    if np.all(np.isnan(pv_f)):
        frac_sig = float("nan")
    else:
        sig = in_dir & (pv_f < alpha)
        frac_sig = float(np.mean(sig))
    return SpecCurveSummary(
        n_specs=n,
        median_effect=float(np.median(eff_f)),
        mean_effect=float(np.mean(eff_f)),
        effect_min=float(np.min(eff_f)),
        effect_max=float(np.max(eff_f)),
        frac_in_direction=frac_in,
        frac_significant=frac_sig,
        alpha=alpha,
    )


@dataclass(frozen=True)
class JointInferenceResult:
    """Result of a specification-curve joint test.

    Attributes:
        statistic: Which across-cell statistic was tested.
        observed: Its observed value.
        p_value: Joint p-value (add-one Monte-Carlo estimate).
        n_specs: Number of cells that entered the test.
        n_resamples: Number of sign-flip resamples drawn.
        mode: ``"independent"`` (per-cell flips; valid for disjoint cells) or
            ``"synchronized"`` (one shared flip; valid for row-aligned cells).
    """

    statistic: str
    observed: float
    p_value: float
    n_specs: int
    n_resamples: int
    mode: str = "independent"


# -- shared sign-flip mechanics ---------------------------------------------


def _aggregate(per_cell_means: np.ndarray, statistic: Statistic, axis):
    if statistic == "median":
        return np.median(per_cell_means, axis=axis)
    if statistic == "mean":
        return np.mean(per_cell_means, axis=axis)
    if statistic == "share_positive":
        # Strict > : a flipped mean landing exactly on 0 counts as not-positive,
        # conservative tie-handling for the directional claim.
        return np.mean(per_cell_means > 0.0, axis=axis)
    raise ValueError(f"unknown statistic: {statistic!r}")


def _aligned_finite_matrix(
    per_obs_vectors: Sequence[Sequence[float]],
    requirement: str,
) -> tuple[int, np.ndarray | None]:
    """Stack row-aligned per-cell vectors into a ``(n_cells, N)`` matrix.

    Empty vectors are dropped; observations non-finite in any remaining cell
    are dropped from all cells. Returns ``(n_cells, matrix)``, with ``None``
    for the matrix when no cell or fewer than two common observations
    survive. Raises ``ValueError`` (prefixed with *requirement*) when the
    vectors differ in length.
    """
    arrs = [np.asarray(list(v), dtype=float) for v in per_obs_vectors]
    arrs = [a for a in arrs if a.size > 0]
    if not arrs:
        return 0, None
    lengths = {a.size for a in arrs}
    if len(lengths) != 1:
        raise ValueError(
            f"{requirement} requires equal-length, row-aligned vectors; "
            f"got lengths {sorted(lengths)}"
        )
    matrix = np.vstack(arrs)
    matrix = matrix[:, np.all(np.isfinite(matrix), axis=0)]
    if matrix.shape[1] < 2:
        return len(arrs), None
    return len(arrs), matrix


def _null_mean_chunks(
    matrix: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
    max_elems: int,
):
    """Yield ``(b, n_cells)`` blocks of null per-cell means, one shared sign
    vector per draw. ``max_elems`` caps the float64 cells per block.
    """
    n = matrix.shape[1]
    for signs in sign_chunks(n, n_resamples, rng, max_elems):
        yield (signs @ matrix.T) / n


def joint_signflip_inference(
    per_obs_vectors: Sequence[Sequence[float]],
    statistic: Statistic = "median",
    alternative: Alternative = "two-sided",
    shared_observations: bool = False,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
    max_elems: int = 20_000_000,
) -> JointInferenceResult:
    """Joint sign-flip test of an across-cell statistic over swap cells.

    Under the null each observation's held-out ΔLL is symmetric about zero, so
    the test compares the observed *statistic* with its values under sign
    flips. ``shared_observations`` selects the dependence assumption (see the
    module docstring); the per-cell vectors must be mutually comparable.

    Args:
        per_obs_vectors: One per-observation ΔLL vector per cell. In
            independent mode, non-finite entries and cells with < 2 finite
            entries are dropped; in synchronized mode all vectors must be
            the same length, and observations non-finite in any cell are
            dropped from all cells.
        statistic: ``"median"`` / ``"mean"`` of per-cell mean ΔLL, or
            ``"share_positive"`` (fraction of cells whose mean ΔLL > 0).
        alternative: Tail of the test; ``"greater"``/``"less"`` for a
            directional claim.
        shared_observations: Use the synchronized (dependence-preserving)
            null.
        n_resamples: Sign-flip resamples.
        rng: Random generator for the resamples; a fixed seed when omitted,
            for reproducible p-values.
        max_elems: float64 cells per chunk (memory cap).

    Raises:
        ValueError: in synchronized mode when vectors differ in length, or
            for ``share_positive`` with ``two-sided`` (its null is centred
            near 0.5, so the ``|stat| >= |obs|`` tail is malformed).
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    if statistic == "share_positive" and alternative == "two-sided":
        raise ValueError(
            "share_positive is a [0, 1] statistic with a sign-flip null "
            "centred near 0.5, a two-sided tail around 0 is malformed for "
            "it; use alternative='greater' (or 'less')."
        )
    rng = rng or np.random.default_rng(0)
    if shared_observations:
        return _joint_signflip_synchronized(
            per_obs_vectors, statistic, alternative, n_resamples, rng, max_elems
        )

    cells = []
    for v in per_obs_vectors:
        x = np.asarray(list(v), dtype=float)
        x = x[np.isfinite(x)]
        if x.size >= 2:
            cells.append(x)
    n_cells = len(cells)
    if n_cells == 0:
        return JointInferenceResult(
            statistic, float("nan"), float("nan"), 0, n_resamples, "independent"
        )

    observed_means = np.array([float(c.mean()) for c in cells], dtype=float)
    observed = float(_aggregate(observed_means, statistic, axis=None))

    max_n = max(c.size for c in cells)
    chunk = max(1, int(max_elems // max_n))
    hits = 0
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        # Per-cell mean under an independent sign flip: (signs · x) / n,
        # vectorized over the b resamples; stack cells into (b, n_cells).
        null_means = np.empty((b, n_cells), dtype=float)
        for j, c in enumerate(cells):
            null_means[:, j] = (sign_block(rng, (b, c.size)) @ c) / c.size
        null_stat = _aggregate(null_means, statistic, axis=1)
        hits += tail_hits(null_stat, observed, alternative)
        done += b
    p_value = addone_pvalue(hits, n_resamples)
    return JointInferenceResult(
        statistic, observed, p_value, n_cells, n_resamples, "independent"
    )


def _joint_signflip_synchronized(
    per_obs_vectors: Sequence[Sequence[float]],
    statistic: Statistic,
    alternative: Alternative,
    n_resamples: int,
    rng: np.random.Generator,
    max_elems: int,
) -> JointInferenceResult:
    """Dependence-preserving joint sign-flip: one shared sign vector per draw."""
    n_cells, matrix = _aligned_finite_matrix(
        per_obs_vectors, "shared_observations=True"
    )
    if matrix is None:
        return JointInferenceResult(
            statistic, float("nan"), float("nan"), n_cells, n_resamples, "synchronized"
        )
    observed = float(_aggregate(matrix.mean(axis=1), statistic, axis=None))
    hits = 0
    for null_means in _null_mean_chunks(matrix, n_resamples, rng, max_elems):
        hits += tail_hits(
            _aggregate(null_means, statistic, axis=1), observed, alternative
        )
    p_value = addone_pvalue(hits, n_resamples)
    return JointInferenceResult(
        statistic, observed, p_value, n_cells, n_resamples, "synchronized"
    )


@dataclass(frozen=True)
class FamilyWiseResult:
    """Family-wise (maxT) + share-significant sign-flip test over a claim.

    Built on the synchronized sign-flip null of
    :func:`_joint_signflip_synchronized`; each spec's statistic is the
    standardized mean ``z = mean ΔLL / SE``.

    Attributes:
        cell_z: Observed standardized z per spec (order matches the input).
        adj_pvalues: Per-spec p-value against the max-z null (maxT,
            Westfall-Young FWER adjustment).
        min_adj_p: Smallest adjusted p, the joint family-wise p-value.
        n_sig_fwer: Specs significant at ``alpha`` after FWER adjustment.
        share_significant: Observed fraction of specs whose z clears the normal
            ``alpha`` critical value in the predicted direction (unadjusted).
        share_significant_p: Joint p-value of that fraction against its null.
        n_specs: Specs that entered the test.
        n_resamples: Sign-flip resamples drawn.
        alpha: Significance threshold.
        inference_scopes: CV, conditioning and sign-flip scopes per input analysis
            when called through grouped_family_wise_signflip.
    """

    cell_z: tuple[float, ...]
    adj_pvalues: tuple[float, ...]
    min_adj_p: float
    n_sig_fwer: int
    share_significant: float
    share_significant_p: float
    n_specs: int
    n_resamples: int
    alpha: float
    permutation_unit: str | None = None
    override_warning: str | None = None
    inference_scopes: tuple[dict, ...] = ()


def _standardized_means(values: np.ndarray) -> np.ndarray:
    """Return mean / SE along the last axis, with centered sample variance."""
    means = values.mean(axis=-1)
    se = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
    standardized = np.divide(means, se, out=np.zeros_like(means), where=se > 0)
    np.copyto(standardized, np.copysign(np.inf, means), where=(se == 0) & (means != 0))
    return standardized


def family_wise_signflip(
    per_obs_vectors: Sequence[Sequence[float]],
    alternative: Alternative = "two-sided",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    max_elems: int = 20_000_000,
    *,
    groups: Sequence | None = None,
) -> FamilyWiseResult:
    """Family-wise maxT + share-significant sign-flip test over row-aligned specs.

    The cells must be row-aligned (same observations, same order), and one
    shared sign vector per draw preserves the cross-spec dependence that
    reusing observations induces. Observations non-finite in any cell are
    dropped from all cells. Nonzero constants have signed infinite statistics;
    all-zero cells have statistic zero. Null statistics at least as extreme as
    observed, including numerical ties (rtol=1e-12, atol=1e-15), count toward
    the add-one Monte Carlo p-values.

    Args:
        per_obs_vectors: One per-observation held-out ΔLL vector per spec; equal
            lengths required.
        alternative: Tail of the test; ``"two-sided"`` uses ``|z|``,
            ``"greater"``/``"less"`` test a directional claim.
        n_resamples: Sign-flip resamples.
        alpha: Threshold for ``n_sig_fwer`` / ``share_significant``.
        rng: Random generator (seeded default for reproducibility).
        max_elems: float64 cells per chunk (memory cap).
        groups: Optional common row-aligned labels. Sum each complete group
            before standardization and share its sign across all specs.
            Non-finite losses are rejected, preserving complete units.

    Returns:
        A :class:`FamilyWiseResult`; all fields ``NaN``/0 when fewer than two
        observations survive.

    Raises:
        ValueError: If vector lengths differ, the alternative is unknown,
            budgets are not positive integers, or alpha is outside (0, 1).
    """
    from scipy import stats

    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    validate_signflip_budget(n_resamples, max_elems)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between zero and one")
    rng = rng or np.random.default_rng(0)

    if groups is not None:
        per_obs_vectors = [sum_by_group(v, groups)[1] for v in per_obs_vectors]
    n_cells, matrix = _aligned_finite_matrix(per_obs_vectors, "family_wise_signflip")
    if matrix is None:
        return FamilyWiseResult(
            (),
            (),
            float("nan"),
            0,
            float("nan"),
            float("nan"),
            n_cells,
            n_resamples,
            alpha,
        )
    n = matrix.shape[1]
    matrix = np.asarray([scale_signflip_values(cell) for cell in matrix])
    z_obs = _standardized_means(matrix)
    if alternative == "two-sided":
        z_obs_dir = np.abs(z_obs)
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        in_dir = np.ones(n_cells, dtype=bool)
    else:
        sign = 1.0 if alternative == "greater" else -1.0
        z_obs_dir = sign * z_obs
        z_crit = float(stats.norm.ppf(1 - alpha))
        in_dir = (sign * z_obs) > 0
    share_obs = float(np.mean(z_obs_dir > z_crit))

    ge_max = np.zeros(n_cells, dtype=np.int64)  # #{null_max >= z_obs_j}
    ge_share = 0  # #{null_share >= share_obs}
    for signs in sign_chunks(n, n_resamples, rng, max_elems):
        z = np.column_stack([_standardized_means(signs * cell) for cell in matrix])
        z_dir = (
            np.abs(z)
            if alternative == "two-sided"
            else (z if alternative == "greater" else -z)
        )
        null_max = z_dir.max(axis=1)  # (b,)
        hits = null_max[:, None] >= z_obs_dir[None, :]
        hits |= np.isclose(
            null_max[:, None], z_obs_dir[None, :], rtol=1e-12, atol=1e-15
        )
        ge_max += hits.sum(axis=0)
        null_share = (z_dir > z_crit).mean(axis=1)  # (b,)
        ge_share += int(np.count_nonzero(null_share >= share_obs))

    adj = (ge_max + 1) / (n_resamples + 1)
    # FWER significance also requires the spec to point the predicted way.
    n_sig = int(np.count_nonzero((adj < alpha) & in_dir))
    return FamilyWiseResult(
        cell_z=tuple(float(z) for z in z_obs),
        adj_pvalues=tuple(float(p) for p in adj),
        min_adj_p=float(adj.min()),
        n_sig_fwer=n_sig,
        share_significant=share_obs,
        share_significant_p=addone_pvalue(ge_share, n_resamples),
        n_specs=n_cells,
        n_resamples=n_resamples,
        alpha=alpha,
    )


@dataclass(frozen=True)
class SwapAttribution:
    """Variance attributed to each swap dimension across the grid.

    A type-II ANOVA of the per-cell effect on the swap dimensions: which
    swap moves the effect most?

    Attributes:
        eta_squared: ``{swap_dim: SS_dim / SS_total}``, a squared semipartial
            share (the dim's type-II SS over the corrected total). Dimensions
            with fewer than two observed levels get ``0.0``. Under correlated
            swaps (e.g. the cumulative grid) these need not sum with
            ``residual_eta_squared`` to 1; only with orthogonal swaps do they.
        pvalues: ``{swap_dim: F-test p-value}`` (``NaN`` for inactive
            dimensions or a saturated fit).
        residual_eta_squared: Share of variance left unexplained (RSS / SST).
        r_squared: Model R² (= 1 - residual_eta_squared), correct regardless of
            collinearity.
        n_obs: Rows with a finite effect used in the fit.
    """

    eta_squared: dict[str, float]
    pvalues: dict[str, float]
    residual_eta_squared: float
    r_squared: float
    n_obs: int


def swap_attribution(
    df,
    effect_col: str,
    swap_cols: Sequence[str],
) -> SwapAttribution:
    """Decompose effect-size variance across grid cells by swap dimension.

    Fits ``effect ~ C(swap_0) + C(swap_1) + ...`` and reports each dimension's
    share of the total sum of squares (type-II ANOVA, each term adjusted for
    the others). Use the *ablation* grid or one whose swaps vary independently:
    in the cumulative grid the swap columns are collinear with the walk position.

    Args:
        df: A table (pandas ``DataFrame``) with one row per grid cell.
        effect_col: Numeric effect column (rows with non-finite effect drop).
        swap_cols: Categorical swap-dimension columns (e.g. the keys of
            the caller's specification annotations).

    Returns:
        A :class:`SwapAttribution`.
    """
    import statsmodels.formula.api as smf
    from scipy import stats

    work = pd.DataFrame()
    work["y"] = pd.to_numeric(df[effect_col], errors="coerce")
    safe = {f"x{i}": col for i, col in enumerate(swap_cols)}
    for safe_name, col in safe.items():
        # Cast to str so a missing swap value becomes an explicit "nan" level
        # rather than a float NaN, patsy drops NaN rows per fit, which would
        # leave the full and reduced models on different samples and break the
        # RSS_reduced - RSS_full comparison below.
        work[safe_name] = df[col].astype(str).to_numpy()
    work = work[np.isfinite(work["y"].to_numpy())].reset_index(drop=True)
    n_obs = int(work.shape[0])

    active = [s for s in safe if work[s].nunique(dropna=True) >= 2]
    eta: dict[str, float] = {safe[s]: 0.0 for s in safe}
    pvals: dict[str, float] = {safe[s]: float("nan") for s in safe}

    y = work["y"].to_numpy()
    sst = float(np.sum((y - y.mean()) ** 2)) if n_obs else float("nan")
    scale = float(np.sum(y**2)) if n_obs else 0.0
    # No usable variance / no varying swap → nothing to attribute. The
    # ``n_obs < 2`` clause overlaps the SST one (n<=1 gives sst==0); the
    # relative SST threshold additionally catches a numerically-constant
    # effect (float-noise SST) at any n.
    if (
        not active
        or n_obs < 2
        or not np.isfinite(sst)
        or sst <= 1e-12 * max(scale, 1.0)
    ):
        residual = 1.0 if n_obs else float("nan")
        r2 = 0.0 if n_obs else float("nan")
        ordered_eta = {col: eta[col] for col in swap_cols}
        ordered_pv = {col: pvals[col] for col in swap_cols}
        return SwapAttribution(ordered_eta, ordered_pv, residual, r2, n_obs)

    # Type-II sums of squares via leave-one-term-out model comparison
    # (SS_t = RSS_without_t - RSS_full). Computed directly rather than through
    # ``anova_lm`` so a saturated fit (df_resid = 0) does not crash its F-test.
    def _fit(terms):
        rhs = " + ".join(f"C({t})" for t in terms) if terms else "1"
        return smf.ols(f"y ~ {rhs}", data=work).fit()

    full = _fit(active)
    rss_full = float(np.sum(np.asarray(full.resid) ** 2))
    df_resid = float(full.df_resid)
    for s in active:
        reduced = _fit([o for o in active if o != s])
        rss_red = float(np.sum(np.asarray(reduced.resid) ** 2))
        ss_s = max(0.0, rss_red - rss_full)
        eta[safe[s]] = ss_s / sst
        df_s = float(full.df_model - reduced.df_model)
        if df_resid > 0 and df_s > 0 and rss_full > 0:
            f_stat = (ss_s / df_s) / (rss_full / df_resid)
            pvals[safe[s]] = float(stats.f.sf(f_stat, df_s, df_resid))
        else:
            pvals[safe[s]] = float("nan")
    residual = rss_full / sst
    # Preserve the caller's column order.
    ordered_eta = {col: eta[col] for col in swap_cols}
    ordered_pv = {col: pvals[col] for col in swap_cols}
    return SwapAttribution(ordered_eta, ordered_pv, residual, 1.0 - residual, n_obs)


def grouped_family_wise_signflip(
    analyses, *, unit=None, allow_scope_mismatch=False, **kwargs
):
    """Align evaluation rows by identity and flip common group sums across specs.

    Args:
        analyses: Results with identical scored observation sets and group labels.
        unit: Selected factor; None requires a common declared permutation unit.
        allow_scope_mismatch: Acknowledge held-out factors not resampled by the
            selected unit. Still warns and records each analysis's scope.
        **kwargs: Options forwarded to family_wise_signflip.
    """
    if not analyses:
        raise ValueError("Supply at least one analysis")
    declared = [a.metadata.get("design", {}).get("permutation_unit") for a in analyses]
    selected = unit if unit is not None else declared[0]
    if selected is None or (unit is None and any(d != selected for d in declared)):
        raise ValueError("Analyses must share a declared permutation unit")
    scopes = tuple(
        inference_scope(a, selected, allow_scope_mismatch=allow_scope_mismatch)
        for a in analyses
    )
    messages = dict.fromkeys(s["warning"] for s in scopes if s["warning"])
    if messages:
        warnings.warn("; ".join(messages), UserWarning, stacklevel=2)
    message = None
    if unit is not None and any(d != unit for d in declared):
        message = f"Permutation unit {unit!r} overrides an analysis declaration"
        warnings.warn(
            f"Permutation unit {unit!r} overrides an analysis declaration",
            UserWarning,
            stacklevel=2,
        )
    reference_rows = reference_groups = None
    reference_source = (
        getattr(analyses[0], "dataset_name", None),
        getattr(analyses[0], "dv_name", None),
    )
    vectors = []
    for analysis in analyses:
        if (
            getattr(analysis, "dataset_name", None),
            getattr(analysis, "dv_name", None),
        ) != reference_source:
            raise ValueError(
                "Family-wise inference requires the same dataset and response"
            )
        values, rows, factors = aligned_losses(analysis)
        order = np.lexsort(rows.T[::-1])
        rows = rows[order]
        if selected not in factors:
            raise ValueError(f"No retained labels for permutation unit {selected!r}")
        groups = factors[selected][order]
        if reference_rows is None:
            reference_rows, reference_groups = rows, groups
        elif not np.array_equal(rows, reference_rows) or not np.array_equal(
            groups, reference_groups
        ):
            raise ValueError(
                "Family-wise inference requires identical row identities and group labels"
            )
        vectors.append(values[order])
    result = family_wise_signflip(vectors, groups=reference_groups, **kwargs)
    return replace(
        result,
        permutation_unit=selected,
        override_warning=message,
        inference_scopes=scopes,
    )
