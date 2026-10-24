"""Stateless helpers shared by the entropy estimators."""

import math

import numpy as np

from replica.feature.core.outputs.information import LN2


def _renyi_entropy_bits_from_log_p_nats(log_p_nats: np.ndarray, alpha: float) -> float:
    """Estimate Rényi-alpha entropy (in bits) from MC samples of ``log p(s)``.

    For samples ``s^(i) ~ p`` (importance-sampling at ``p`` itself):

        Σ_x p(x)^alpha  =  E_p[p(X)^(alpha-1)]  ≈  (1/N) Σ_i p(s^(i))^(alpha-1)

    so the Rényi-alpha entropy estimator is

        Ĥ_alpha  =  (1/(1-alpha)) · log_2 [ (1/N) Σ_i p(s^(i))^(alpha-1) ]

    Numerically stable via log-sum-exp on ``(alpha-1) · log_e p_i`` (kept in
    nats throughout, with a single bits-conversion at return, mixing base-2
    exponents into ``np.exp`` would compute ``p^((alpha-1)/ln 2)``). For alpha = 1,
    returns the Shannon entropy ``-(1/N) Σ log_2 p_i`` directly.

    Args:
        log_p_nats: per-sample joint log-probabilities ``ln p(s^(i))``,
            in nats. A real vector of finite nonpositive values, shape ``(N,)``.
        alpha: Finite Rényi parameter ``≥ 0``. alpha=1 gives Shannon.

    Returns:
        Estimated Rényi-alpha entropy in bits, or NaN for an empty vector.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"Rényi alpha must be finite and nonnegative, got {alpha}")
    log_p_nats = np.asarray(log_p_nats)
    invalid_samples = (
        "log_p_samples_nats must be a one-dimensional vector of "
        "finite nonpositive log probabilities representable in float64"
    )
    if (
        log_p_nats.ndim != 1
        or log_p_nats.dtype.kind not in "iuf"
        or not np.all(np.isfinite(log_p_nats) & (log_p_nats <= 0))
    ):
        raise ValueError(invalid_samples)
    if not log_p_nats.size:
        return float("nan")
    with np.errstate(over="ignore"):
        log_p_nats = log_p_nats.astype(np.float64, copy=False)
    if not np.all(np.isfinite(log_p_nats)):
        raise ValueError(invalid_samples)
    if alpha == 1.0:
        # H_Shannon = -E[log p]; ln nats → bits.
        return float(-np.mean(log_p_nats) / LN2)
    exponent = (alpha - 1.0) * log_p_nats  # ln(p_i^(alpha-1))
    m = float(np.max(exponent))
    log_sum_nats = m + math.log(float(np.sum(np.exp(exponent - m))))
    log_mean_nats = log_sum_nats - math.log(len(log_p_nats))
    # H_alpha (bits) = log_mean_nats / ((1-alpha) · ln 2)
    return float(log_mean_nats / ((1.0 - alpha) * LN2))


def _clark_half_window_starts(
    n_tokens: int,
    word_final_positions: set[int],
    half_size: int,
) -> list[int]:
    """Faithful port of Clark et al. (2025) ``mc_word_entropy.py`` half-window
    boundary computation (lines 199-214 of the upstream).

    Args:
        n_tokens: length of the (BOS-prepended) token sequence.
        word_final_positions: set of token positions where a REPLICA word
            ends (the predicting position for the *next* word's entropy).
            Equivalently: the set of ``last_token_of_prefix`` values for
            the sample positions used by the MC estimators.
        half_size: target maximum tokens per half-window. Mirrors the
            upstream ``ctx_size/2 - max_iterations`` (true division there;
            ``//`` here, plus a guard against a non-increasing start that
            the upstream lacks).

    Returns:
        Sorted ``half_window_starts`` list. Adjacent pairs define the
        boundaries of windows: window ``i`` covers tokens
        ``[starts[i] : starts[i+2]]``, with its "evaluate-from" position
        being ``starts[i]`` for the first window and ``starts[i+1]``
        otherwise. Length is at least 3 (matching the upstream padding
        when the whole story fits in one window).
    """
    if half_size < 1:
        raise ValueError(f"half_size must be ≥ 1, got {half_size}")
    half_starts = [0]
    cur_start = 0
    last_word_end = -1
    for ix in range(n_tokens):
        if ix - cur_start == half_size:
            # End this half-window. The upstream code splits at the latest
            # ``word_final`` position seen so far (so the second window
            # starts right after a word boundary).
            new_start = (
                last_word_end + 1 if last_word_end >= cur_start else cur_start + 1
            )
            cur_start = new_start
            half_starts.append(cur_start)
        elif ix in word_final_positions:
            last_word_end = ix
    half_starts.append(n_tokens)
    # Upstream padding rule: when the entire story fits in one window,
    # half_starts has only [0, n_tokens]; pad to length 3 so the
    # consumer's "pairs of consecutive half-windows" loop still produces
    # one window covering everything.
    if len(half_starts) == 2:
        half_starts.append(n_tokens)
    return half_starts


def _clark_windows(
    token_ids: list[int],
    sample_positions: list[tuple[int, int]],
    half_size: int,
) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Return Clark half-overlap windows and their local sample positions."""
    unit_at_position = {position: unit for unit, position in sample_positions}
    starts = _clark_half_window_starts(len(token_ids), set(unit_at_position), half_size)
    windows = []
    for window_index in range(len(starts) - 2):
        start, midpoint, end = starts[window_index : window_index + 3]
        evaluate_from = start if window_index == 0 else midpoint
        local_positions = [
            (unit, position - start)
            for position, unit in unit_at_position.items()
            if evaluate_from <= position < end
        ]
        if local_positions:
            windows.append((token_ids[start:end], local_positions))
    return windows
