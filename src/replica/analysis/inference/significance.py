"""Sign-flip tests, grouped inference and multiple-comparison corrections."""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from replica.analysis.contracts.labels import categorical_labels

Alternative = Literal["two-sided", "greater", "less"]

#: Largest per-fold vector ``paired_permutation_pvalue`` enumerates exactly:
#: 2**24 sign patterns, one Python iteration each. Longer vectors need the
#: Monte Carlo method.
EXACT_SIGN_FLIP_MAX_N = 24


def paired_permutation_pvalue(
    per_fold_values: Sequence[float],
    alternative: Alternative = "two-sided",
    method: Literal["exact", "monte-carlo"] = "exact",
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Paired-permutation (sign-flip) p-value on per-fold differences.

    Under H₀ each ``x_i`` is symmetric around zero, so the test compares
    ``mean(sigma · x)`` over enumerated or sampled sign patterns
    ``sigma ∈ {-1,+1}^n`` to the observed ``mean(x)``.

    Args:
        per_fold_values: Paired differences ``x_i = target_i - baseline_i``,
            one per fold, or any quantity that is zero under the null;
            non-finite entries are dropped.
        alternative: ``"two-sided"`` tests ``|mean(sigma·x)| ≥ |mean(x)|``;
            ``"greater"`` tests ``mean(sigma·x) ≥ mean(x)``;
            ``"less"`` tests ``mean(sigma·x) ≤ mean(x)``.
        method: ``"exact"`` compares all ``2**n`` sign sums without roundoff and raises
            above :data:`EXACT_SIGN_FLIP_MAX_N` finite values;
            ``"monte-carlo"`` samples ``n_resamples`` patterns.
        n_resamples: Positive number of sign patterns under ``"monte-carlo"``; ignored by
            the exact branch.
        rng: Source of those draws; ``None`` seeds a fresh generator, so the
            Monte-Carlo p-value varies between calls.

    Returns:
        p-value in ``[0, 1]``; ``NaN`` if fewer than 2 finite values.
    """
    if method == "monte-carlo":
        _validate_positive_integer("n_resamples", n_resamples)
    x = np.asarray([v for v in per_fold_values if v is not None], dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    if method not in ("exact", "monte-carlo"):
        raise ValueError(f"unknown method: {method!r}")
    if method == "exact":
        return _exact_signflip_pvalue(x, alternative)

    x = _safe_signflip_values(x)
    obs = float(np.dot(np.ones(n, dtype=float), x)) / n
    if obs == 0.0 and alternative == "two-sided":
        return 1.0

    # Monte Carlo.
    rng = rng or np.random.default_rng()
    signs = rng.choice((-1.0, 1.0), size=(n_resamples, n))
    stats = signs @ x / n
    if alternative == "two-sided":
        hits = int(np.sum(np.abs(stats) >= abs(obs)))
    elif alternative == "greater":
        hits = int(np.sum(stats >= obs))
    else:
        hits = int(np.sum(stats <= obs))
    # Add-one MC estimate: observed pattern is a valid permutation, so
    # p-value is never 0 (Pimentel et al. (2023), scipy.permutation_test
    # convention).
    return (hits + 1) / (n_resamples + 1)


def _exact_signflip_pvalue(values: np.ndarray, alternative: Alternative) -> float:
    """Enumerate sign sums with exact integer arithmetic for finite float64 values."""
    ratios = [float(value).as_integer_ratio() for value in values]
    # Binary floats have power-of-two denominators, so the largest is common.
    denominator = max(den for _, den in ratios)
    weights = [num * (denominator // den) for num, den in ratios]
    observed = sum(weights)
    if observed == 0 and alternative == "two-sided":
        return 1.0
    n = len(weights)
    if n > EXACT_SIGN_FLIP_MAX_N:
        raise ValueError(
            f"exact enumeration covers at most {EXACT_SIGN_FLIP_MAX_N} values "
            f"(2**{EXACT_SIGN_FLIP_MAX_N} sign patterns), got {n}; pass "
            "method='monte-carlo' with an explicit rng"
        )
    statistic = -observed
    hits = 0
    total = 1 << n
    for index in range(total):
        if index:
            # Consecutive Gray codes change one sign; update only that summand.
            bit = index & -index
            weight = weights[bit.bit_length() - 1]
            gray = index ^ (index >> 1)
            statistic += 2 * weight if gray & bit else -2 * weight
        if alternative == "two-sided":
            hits += abs(statistic) >= abs(observed)
        elif alternative == "greater":
            hits += statistic >= observed
        else:
            hits += statistic <= observed
    return hits / total


def sign_block(rng: np.random.Generator, shape: int | tuple[int, ...]) -> np.ndarray:
    """Random ``{-1, +1}`` float64 array of *shape*.

    Args:
        rng: Source of the sign draws.
        shape: Shape of the block, either a length or a full shape tuple.
    """
    signs = rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.float64)
    signs *= 2.0
    signs -= 1.0
    return signs


def sign_chunks(
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
    max_elems: int = 20_000_000,
) -> Iterator[np.ndarray]:
    """Yield ``(b, n)`` sign blocks, about ``max_elems`` float64 cells each.

    Args:
        n: Width of a block, the number of values one resample signs.
        n_resamples: Total resamples to cover, summed over the blocks.
        rng: Source of the sign draws.
        max_elems: Memory cap in float64 cells, which sets the block height.

    Returns:
        One sign block per batch; the last one is short when *n_resamples* is
        not a multiple of the block height.
    """
    chunk = max(1, int(max_elems // n))
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        yield sign_block(rng, (b, n))
        done += b


def tail_hits(
    null_stat: np.ndarray,
    observed: float,
    alternative: Alternative,
    *,
    tolerate_roundoff: bool = False,
) -> int:
    """Count null draws at least as extreme as *observed* for the given tail.

    Args:
        null_stat: Statistic under one batch of sign draws, one entry per draw.
        observed: Statistic on the unpermuted data.
        alternative: Tail to count, ``"two-sided"`` comparing magnitudes and
            the one-sided options comparing signed values.
        tolerate_roundoff: Include numerical ties using the exact test's
            tolerance. False preserves legacy Monte Carlo comparisons.
    """
    if alternative == "two-sided":
        null_stat, observed = np.abs(null_stat), abs(observed)
    if tolerate_roundoff:
        hits = null_stat <= observed if alternative == "less" else null_stat >= observed
        return int(
            np.count_nonzero(
                hits | np.isclose(null_stat, observed, rtol=1e-12, atol=1e-15)
            )
        )
    if alternative == "two-sided":
        return int(np.count_nonzero(null_stat >= observed))
    if alternative == "greater":
        return int(np.count_nonzero(null_stat >= observed))
    return int(np.count_nonzero(null_stat <= observed))  # "less"


def addone_pvalue(hits: int, n_resamples: int) -> float:
    """Add-one Monte-Carlo p-value estimate, never 0.

    The observed pattern is itself a valid permutation, so it is counted in
    both numerator and denominator (the scipy.permutation_test convention).

    Args:
        hits: Null draws at least as extreme as the observed statistic.
        n_resamples: Draws taken.
    """
    return (hits + 1) / (n_resamples + 1)


def concat_fold_per_obs_delta_ll(fold_results) -> np.ndarray:
    """Concatenate per-observation held-out ΔLL across folds.

    Pulls ``per_obs_delta_ll`` from each fold result, skipping folds where
    it is ``None``, and concatenates them. Returns an empty array if nothing
    was retained (populate it by running ``evaluate()`` with
    ``retain_per_obs_delta_ll=True``).

    Args:
        fold_results: Per-fold results of one evaluation, each carrying a
            ``per_obs_delta_ll`` attribute. ``None`` reads as no folds.
    """
    parts = [
        np.asarray(fr.per_obs_delta_ll, dtype=float)
        for fr in (fold_results or [])
        if getattr(fr, "per_obs_delta_ll", None) is not None
    ]
    if not parts:
        return np.empty(0, dtype=float)
    return np.concatenate(parts)


def sum_by_group(values, groups):
    """Sum complete groups, rejecting missing labels or non-finite losses.

    Args:
        values: One finite loss difference per observation.
        groups: One nonmissing categorical label per observation.

    Returns:
        Sorted labels and their summed differences.
    """
    if np.ma.is_masked(values):
        raise ValueError("Grouped inference requires finite losses for complete units")
    values = np.asarray(values, float)
    groups = np.asanyarray(groups, dtype=object)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("Group labels must align with a one-dimensional loss vector")
    if not np.isfinite(values).all():
        raise ValueError("Grouped inference requires finite losses for complete units")
    groups = categorical_labels(groups, name="permutation groups")
    labels, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if not len(labels):
        return labels, np.empty(0, dtype=float)
    parts = np.split(values[np.argsort(inverse)], np.cumsum(counts)[:-1])
    try:
        sums = np.array([math.fsum(part) for part in parts])
    except OverflowError as exc:
        raise ValueError("Grouped inference requires finite group sums") from exc
    return labels, sums


def scale_signflip_values(values: np.ndarray) -> np.ndarray:
    """Rescale finite differences so sign-flip arithmetic is independent of units.

    Args:
        values: One-dimensional finite differences; zero and empty vectors work.
    """
    scale = np.max(np.abs(values), initial=0.0)
    return values / scale if scale > 0 else values


def _safe_signflip_values(values: np.ndarray) -> np.ndarray:
    """Preserve ordinary-scale arithmetic, rescaling unsafe sums and means."""
    scale = np.max(np.abs(values), initial=0.0)
    limits = np.finfo(float)
    n = values.size
    if 0 < scale < limits.tiny * n or scale > limits.max / (2 * n):
        return scale_signflip_values(values)
    return values


def validate_signflip_budget(n_resamples: int, max_elems: int) -> None:
    """Reject invalid Monte Carlo budgets before any degenerate-data shortcuts.

    Args:
        n_resamples: Positive integer number of sign draws.
        max_elems: Positive integer memory budget in float64 cells.
    """
    _validate_positive_integer("n_resamples", n_resamples)
    _validate_positive_integer("max_elems", max_elems)


def _validate_positive_integer(name: str, value: int) -> None:
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")


def per_datapoint_permutation_pvalue(
    deltas: Sequence[float],
    alternative: Alternative = "two-sided",
    n_resamples: int = 100_000,
    rng: np.random.Generator | None = None,
    max_elems: int = 20_000_000,
    *,
    groups: Sequence | None = None,
) -> float:
    """Per-datapoint sign-flip permutation on held-out ΔLL concatenated across folds.

    Under H₀ each entry is symmetric about zero, so the test compares
    ``mean(sigma·δ)`` over random sign patterns to the observed ``mean(δ)``,
    the per-observation protocol the grid uses for Wilcox et al. (2023) and
    Clark et al. (2025).

    Args:
        deltas: Per-observation ΔLL, one entry per datapoint (non-finite
            dropped).
        alternative: Tail of the sign-flip null to count against.
        n_resamples: Positive integer number of sign patterns drawn.
        rng: Source of those draws; ``None`` uses a fixed-seed generator, so
            the p-value is reproducible.
        max_elems: Memory cap (float64 cells per chunk of sign draws). One
            resample needs ``n`` cells, so route vectors longer than this to
            :func:`signflip_normal_pvalue`.
        groups: Optional aligned labels; flip complete group sums together.
            Non-finite losses are rejected in this mode. Validity requires
            joint sign symmetry at the chosen unit, including CV dependence.

    Returns:
        Monte-Carlo p-value in ``[0, 1]``; ``NaN`` if < 2 finite values.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    validate_signflip_budget(n_resamples, max_elems)
    if groups is not None:
        _, x = sum_by_group(deltas, groups)
        x = scale_signflip_values(x)
    else:
        x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    if groups is None:
        x = _safe_signflip_values(x)
    obs = float(x.mean())
    if obs == 0.0 and alternative == "two-sided":
        return 1.0

    rng = rng or np.random.default_rng(0)
    hits = 0
    for signs in sign_chunks(n, n_resamples, rng, max_elems):
        hits += tail_hits(
            (signs @ x) / n, obs, alternative, tolerate_roundoff=groups is not None
        )
    return addone_pvalue(hits, n_resamples)


def signflip_normal_pvalue(
    deltas: Sequence[float],
    alternative: Alternative = "two-sided",
) -> float:
    """Normal-approximation per-datapoint sign-flip p-value, ``z = Σx / sqrt(Σx²)``.

    The null and statistic of :func:`per_datapoint_permutation_pvalue`, with the
    tail from the central limit theorem: exact in the large-n limit and O(n),
    for pooling long held-out-ΔLL vectors across corpora.

    Args:
        deltas: Per-observation ΔLL (non-finite dropped).
        alternative: Tail of the normal null the z score is read against.

    Returns:
        Normal-approximation p-value in ``[0, 1]``; ``NaN`` if < 2 finite
        values or the vector is all-zero.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    x = scale_signflip_values(x)
    s2 = float(np.dot(x, x))
    if s2 <= 0.0:
        return float("nan")
    z = float(x.sum()) / math.sqrt(s2)
    if alternative == "two-sided":
        return float(math.erfc(abs(z) / math.sqrt(2.0)))
    if alternative == "greater":
        return float(0.5 * math.erfc(z / math.sqrt(2.0)))
    return float(0.5 * math.erfc(-z / math.sqrt(2.0)))  # "less"


def bh_correct(
    pvalues: Sequence[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Standard step-up procedure. Returns BH-corrected p-values and a boolean
    mask of "reject at FDR≤alpha" decisions, in the input order. ``NaN``
    p-values pass through as ``NaN`` and never reject.

    Args:
        pvalues: Iterable of raw p-values. Length-0 input returns
            length-0 arrays.
        alpha: Target false-discovery rate.

    Returns:
        ``(corrected, reject)``, both ndarrays of length ``len(pvalues)``.
        ``corrected[i]`` is the BH-adjusted p-value for ``pvalues[i]``
        (monotone, capped at 1.0); ``reject[i]`` is True iff
        ``corrected[i] < alpha``.
    """
    p = np.asarray(pvalues, dtype=float)
    n_total = p.size
    if n_total == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    finite_mask = np.isfinite(p)
    finite_p = p[finite_mask]
    m = finite_p.size

    corrected = np.full(n_total, np.nan, dtype=float)
    reject = np.zeros(n_total, dtype=bool)

    if m == 0:
        return corrected, reject

    # BH: sort, multiply by m/rank, then enforce monotonicity.
    order = np.argsort(finite_p)
    ranks = np.arange(1, m + 1)
    sorted_p = finite_p[order]
    bh = sorted_p * m / ranks
    # Monotonicity: q_(i) ← min(q_(i), q_(i+1), ..., q_(m))
    bh_mono = np.minimum.accumulate(bh[::-1])[::-1]
    bh_mono = np.minimum(bh_mono, 1.0)
    # Unsort.
    finite_corrected = np.empty(m, dtype=float)
    finite_corrected[order] = bh_mono
    corrected[finite_mask] = finite_corrected
    reject[finite_mask] = finite_corrected < alpha
    return corrected, reject


def pvalue_to_stars(p: float, ns: str = "") -> str:
    """APA-style significance stars from a p-value.

    ``"***"`` for ``p < .001``, ``"**"`` for ``p < .01``, ``"*"`` for
    ``p < .05``. A non-finite p-value returns ``""``.

    Args:
        p: The p-value to label.
        ns: Label for a p-value above every threshold, e.g. ``"n.s."``.
    """
    p = float(p)
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ns


def is_supported_at_alpha(effect: float, p_value: float, alpha: float = 0.05) -> bool:
    """Cell support: the effect is positive *and* significant at *alpha*.

    Equivalent to ``SupportResult(effect, p_value, claim_direction=+1,
    alpha=alpha).is_supported`` without building the dataclass, use that
    class when a non-positive ``claim_direction`` or a regime label is
    needed. Non-finite inputs are unsupported.

    Args:
        effect: Observed effect size, such as mean ΔLL per observation or ΔR².
        p_value: P-value for that effect under whichever test produced it.
        alpha: Threshold the p-value has to fall below.
    """
    effect = float(effect)
    p_value = float(p_value)
    if not (np.isfinite(effect) and np.isfinite(p_value)):
        return False
    return effect > 0 and p_value < alpha


@dataclass(frozen=True)
class SupportResult:
    """Composable "is this cell supported?" decision.

    Pairs an effect-size sign check with a p-value threshold so
    different significance regimes (two-sided uncorrected, one-sided
    BH-corrected, etc.) compose the same way at call sites.

    Attributes:
        effect: Observed effect size (e.g. mean ΔLL/N or ΔR²).
        p_value: P-value under the chosen test/correction.
        claim_direction: +1 if the hypothesis predicts ``effect > 0``,
            -1 if it predicts ``effect < 0``, 0 to skip the sign check.
        alpha: Decision threshold (BH/raw, depending on input).
        is_corrected: Tag so downstream code can label the regime.
        regime_label: Short human-readable name (e.g. ``"two-sided"``,
            ``"one-sided BH"``).
    """

    effect: float
    p_value: float
    claim_direction: int = 1
    alpha: float = 0.05
    is_corrected: bool = False
    regime_label: str = ""

    @property
    def is_supported(self) -> bool:
        """True iff the effect has the claimed sign and p < alpha."""
        if not (np.isfinite(self.effect) and np.isfinite(self.p_value)):
            return False
        if self.claim_direction > 0 and self.effect <= 0:
            return False
        if self.claim_direction < 0 and self.effect >= 0:
            return False
        return float(self.p_value) < self.alpha


def support_under_regime(
    effects: Sequence[float],
    raw_pvalues: Sequence[float],
    claim_directions: Sequence[int] | int = 1,
    alpha: float = 0.05,
    sided: Alternative = "two-sided",
    correction: Literal["none", "bh"] = "none",
) -> list[SupportResult]:
    """Apply a (sidedness x correction) regime to a batch of cells.

    ``sided="greater"``, a directional regime for claims with a predicted sign,
    halves each p-value where the effect has the claimed sign and sets it to
    ``1 - p/2`` otherwise; ``"less"`` tests the opposite sign. BH correction
    follows when asked.

    Args:
        effects: Observed effect size per cell.
        raw_pvalues: Two-sided p-value per cell, as
            :func:`paired_permutation_pvalue` returns it, aligned with
            *effects*.
        claim_directions: Sign the hypothesis predicts per cell, or one int
            broadcast to every cell: +1 for ``effect > 0``, -1 for
            ``effect < 0``, 0 to skip the sign check; a one-sided regime
            then reads 0 as its own direction (``"greater"`` tests for a
            positive effect and ``"less"`` for a negative one).
        alpha: Decision threshold carried into each result, and the target
            false-discovery rate when correcting.
        sided: Regime sidedness; ``"two-sided"`` passes the p-values through,
            the one-sided options halve them where the sign matches.
        correction: ``"bh"`` applies :func:`bh_correct` across the batch,
            ``"none"`` leaves the oriented p-values alone.

    Returns:
        One :class:`SupportResult` per cell, in input order.

    Raises:
        ValueError: On a length mismatch, or an unknown *sided* or
            *correction*.
    """
    effects = np.asarray(effects, dtype=float)
    raw = np.asarray(raw_pvalues, dtype=float)
    if isinstance(claim_directions, int):
        dirs = np.full(effects.shape, claim_directions, dtype=int)
    else:
        dirs = np.asarray(list(claim_directions), dtype=int)
    if not (effects.size == raw.size == dirs.size):
        raise ValueError("effects, raw_pvalues, claim_directions length mismatch")

    if sided == "two-sided":
        oriented = raw.copy()
    elif sided in ("greater", "less"):
        # One-sided: p_one = p_two/2 when the effect lies on the tested side,
        # else 1 - p_two/2. "greater" tests the claimed sign and "less" the
        # opposite one; a cell with no claimed sign takes the regime's own.
        claimed = np.where(dirs == 0, 1, np.sign(dirs))
        tested = claimed if sided == "greater" else -claimed
        match = np.sign(effects) == tested
        oriented = np.where(match, raw / 2.0, 1.0 - raw / 2.0)
    else:
        raise ValueError(f"unknown sided: {sided!r}")

    if correction == "none":
        pvals_final = oriented
    elif correction == "bh":
        pvals_final, _ = bh_correct(oriented, alpha=alpha)
    else:
        raise ValueError(f"unknown correction: {correction!r}")

    regime_label = sided + (" BH" if correction == "bh" else "")
    return [
        SupportResult(
            effect=float(e),
            p_value=float(p),
            claim_direction=int(d),
            alpha=alpha,
            is_corrected=(correction != "none"),
            regime_label=regime_label,
        )
        for e, p, d in zip(effects, pvals_final, dirs, strict=False)
    ]


@dataclass(frozen=True)
class PermutationResult:
    """One paired permutation comparison of two models on held-out data."""

    p_value: float
    a_stat: float
    b_stat: float
    difference: float
    n: int
    n_iter: int
    n_dropped: int = 0
    null: np.ndarray | None = None


def _joint_finite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows finite in both models, the reference's joint listwise deletion."""
    finite_a = (
        np.isfinite(a).all(axis=tuple(range(1, a.ndim)))
        if a.ndim > 1
        else np.isfinite(a)
    )
    finite_b = (
        np.isfinite(b).all(axis=tuple(range(1, b.ndim)))
        if b.ndim > 1
        else np.isfinite(b)
    )
    return finite_a & finite_b


#: Per-observation value kinds of the CDR permutation tests. Losses (``mse``
#: is the reference's name for squared error) average over rows and a larger
#: value is worse; log-likelihoods sum over rows and a smaller value is worse.
_LOSS_MODES = frozenset({"loss", "mse"})
_PERMUTATION_MODES = _LOSS_MODES | {"loglik"}


def _is_loss(mode: str) -> bool:
    """Whether *mode* names a loss rather than a log-likelihood.

    Raises:
        ValueError: On a mode outside the shared vocabulary.
    """
    if mode not in _PERMUTATION_MODES:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {sorted(_PERMUTATION_MODES)}"
        )
    return mode in _LOSS_MODES


def _directional_pass(difference: float, mode: str) -> bool:
    """Whether a directional gate lets the comparison proceed.

    The first model is the more ablated one, so the comparison is only worth
    running when it is the worse one: a higher loss, or a lower
    log-likelihood. Otherwise the reference returns a p-value of one without
    resampling.
    """
    return difference > 0 if _is_loss(mode) else difference < 0


def paired_swap_permutation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mode: str = "loss",
    n_iter: int = 10000,
    n_tails: int = 2,
    directional_gate: bool = False,
    rng: np.random.Generator | None = None,
    keep_null: bool = False,
) -> PermutationResult:
    """Paired permutation over per-observation losses or log-likelihoods.

    Each resample swaps the two models' values within an observation with
    probability one half, the procedure of the 2018 and 2021 CDR-family papers;
    ensembles need :func:`ensemble_column_permutation`. Pass (more ablated,
    fuller): the reported difference is the first minus the second.

    Args:
        a: Per-observation values of the more ablated model.
        b: Per-observation values of the fuller model, row-aligned with *a*; a
            row non-finite in either model is dropped from both.
        mode: ``"loss"`` or ``"mse"`` aggregates each model by its mean over
            rows and ``"loglik"`` by the sum, matching the reference's
            log-likelihood totals; any other value raises.
        n_iter: Sign patterns drawn.
        n_tails: 2 counts null draws exceeding the observed magnitude, 1 counts
            those at or past the observed value on its own side.
        directional_gate: Return a p-value of one without resampling unless the
            first model is the worse one; the gate reads the sign of the first
            minus the second.
        rng: Source of the sign draws; ``None`` seeds a fresh generator.
        keep_null: Carry the null distribution back on the result.

    Returns:
        The p-value, with one added to numerator and denominator so it is never
        zero, both model statistics, their difference, the retained row count,
        and the number of rows dropped as non-finite.

    Raises:
        ValueError: On a shape mismatch, an unknown *mode*, or an *n_tails*
            other than 1 or 2.
    """
    is_loss = _is_loss(mode)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    keep = _joint_finite(a, b)
    n_dropped = int((~keep).sum())
    a, b = a[keep], b[keep]
    n = len(a)

    # Losses are compared as means, log-likelihoods as sums, as in the
    # reference; the resampled statistic must use the same aggregation.
    aggregate = np.mean if is_loss else np.sum
    a_stat, b_stat = float(aggregate(a)), float(aggregate(b))
    difference = a_stat - b_stat
    if directional_gate and not _directional_pass(difference, mode):
        return PermutationResult(1.0, a_stat, b_stat, difference, n, n_iter, n_dropped)
    if difference == 0.0:
        # Two models that score identically are not distinguishable, and the
        # strict hit count would otherwise report the smallest attainable
        # p-value for them.
        return PermutationResult(1.0, a_stat, b_stat, difference, n, n_iter, n_dropped)

    rng = np.random.default_rng() if rng is None else rng
    delta = a - b
    observed = float(aggregate(delta))
    hits = 0
    null_chunks: list[np.ndarray] = []
    scale = 1.0 / n if is_loss else 1.0
    for signs in sign_chunks(n, n_iter, rng):
        null_stat = (signs @ delta) * scale
        if n_tails == 2:
            hits += int(np.count_nonzero(np.abs(null_stat) > abs(observed)))
        elif n_tails == 1:
            hits += int(
                np.count_nonzero(
                    null_stat <= observed if observed < 0 else null_stat >= observed
                )
            )
        else:
            raise ValueError("n_tails must be 1 or 2")
        if keep_null:
            null_chunks.append(null_stat)
    return PermutationResult(
        addone_pvalue(hits, n_iter),
        a_stat,
        b_stat,
        difference,
        n,
        n_iter,
        n_dropped,
        np.concatenate(null_chunks) if keep_null else None,
    )


def ensemble_column_permutation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    agg: str = "median",
    mode: str = "loglik",
    n_iter: int = 10000,
    n_tails: int = 2,
    directional_gate: bool = False,
    drop_nonfinite: bool = True,
    rng: np.random.Generator | None = None,
    keep_null: bool = False,
) -> PermutationResult:
    """Paired permutation over ensemble member columns.

    The statistic reduces the per-member totals, by default to their median.
    Each resample permutes the pooled member columns within an observation and
    reassigns the first ``k_a`` of them to the first model.

    Args:
        a: Per-observation values of the more ablated model, one column per
            ensemble member; a 1-D array reads as a single member.
        b: The fuller model's members, over the same rows.
        agg: Reduction over the per-member totals: ``"median"``, ``"mean"``,
            ``"min"``, or ``"max"``.
        mode: ``"loglik"`` totals each member by summing over rows; ``"loss"``
            and ``"mse"`` average; any other value raises.
        n_iter: Column permutations drawn.
        n_tails: 2 counts null draws exceeding the observed magnitude, 1 counts
            those at or past the observed difference on its own side.
        directional_gate: Return a p-value of one without resampling unless the
            first model is the worse one.
        drop_nonfinite: Drop rows that are non-finite in either model, jointly.
        rng: Source of the permutations; ``None`` seeds a fresh generator.
        keep_null: Carry the null distribution back on the result.

    Returns:
        The p-value with both aggregated statistics, their difference, the
        retained row count, and the number of rows dropped as non-finite.

    Raises:
        ValueError: When the inputs are not ``(n_obs, k)`` arrays of equal
            length, or *mode* is unknown.
    """
    is_loss = _is_loss(mode)
    a = np.atleast_2d(np.asarray(a, dtype=float).T).T
    b = np.atleast_2d(np.asarray(b, dtype=float).T).T
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError(
            f"expected (n_obs, k) arrays of equal length: {a.shape} {b.shape}"
        )
    n_dropped = 0
    if drop_nonfinite:
        keep = _joint_finite(a, b)
        n_dropped = int((~keep).sum())
        a, b = a[keep], b[keep]
    n, k_a = a.shape
    k_b = b.shape[1]

    reduce = {"median": np.median, "mean": np.mean, "min": np.min, "max": np.max}[agg]
    # Per member: losses average over rows, log-likelihoods sum over them.
    per_member = np.mean if is_loss else np.sum

    a_stat = float(reduce(per_member(a, axis=0)))
    b_stat = float(reduce(per_member(b, axis=0)))
    difference = a_stat - b_stat
    if directional_gate and not _directional_pass(difference, mode):
        return PermutationResult(1.0, a_stat, b_stat, difference, n, n_iter, n_dropped)
    if difference == 0.0:
        return PermutationResult(1.0, a_stat, b_stat, difference, n, n_iter, n_dropped)

    rng = np.random.default_rng() if rng is None else rng
    if k_a == 1 and k_b == 1:
        # One member each: permuting two columns per row is a sign flip of
        # their difference, at a fraction of the cost.
        return replace(
            paired_swap_permutation(
                a[:, 0],
                b[:, 0],
                mode=mode,
                n_iter=n_iter,
                n_tails=n_tails,
                rng=rng,
                keep_null=keep_null,
            ),
            n_dropped=n_dropped,
        )

    pooled = np.concatenate([a, b], axis=1)
    observed = difference
    hits = 0
    null_chunks: list[np.ndarray] = []
    chunk = max(1, int(20_000_000 // max(1, n * (k_a + k_b))))
    done = 0
    while done < n_iter:
        size = min(chunk, n_iter - done)
        # The order within each randomly assigned side matters: values in one
        # output column are summed into one ensemble member before the median
        # over members is taken. Fully sorting independent random ranks gives
        # every pooled-column permutation equal probability for every row.
        order = rng.random((size, n, k_a + k_b)).argsort(axis=2)
        drawn = np.take_along_axis(pooled[None, :, :], order, axis=2)
        left, right = drawn[:, :, :k_a], drawn[:, :, k_a:]
        null_stat = reduce(per_member(left, axis=1), axis=1) - reduce(
            per_member(right, axis=1), axis=1
        )
        if n_tails == 2:
            hits += int(np.count_nonzero(np.abs(null_stat) > abs(observed)))
        elif n_tails == 1:
            hits += int(
                np.count_nonzero(
                    null_stat <= observed if observed < 0 else null_stat >= observed
                )
            )
        else:
            raise ValueError("n_tails must be 1 or 2")
        if keep_null:
            null_chunks.append(null_stat)
        done += size
    return PermutationResult(
        addone_pvalue(hits, n_iter),
        a_stat,
        b_stat,
        difference,
        n,
        n_iter,
        n_dropped,
        np.concatenate(null_chunks) if keep_null else None,
    )
