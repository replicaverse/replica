"""Shannon and Rényi entropy in bits from caller-supplied log probabilities.

Inputs are in nats; callers choose the distribution and support. Python-float
conversion factors preserve float32 inputs under NumPy's NEP-50 rules.
"""

import math

import numpy as np

# Python floats keep float32 inputs in float32 under NEP-50.
LN2 = math.log(2.0)
LOG2_10 = 1.0 / math.log10(2.0)  # log2(10): converts log10 p to log2 p


def nats_to_bits(x):
    """Convert log-probabilities or entropies from nats to bits.

    Args:
        x: A value in nats, scalar or array. Divided elementwise by ``ln 2``,
            so the dtype and shape of the input carry through.
    """
    return x / LN2


def shannon_bits(log_p: np.ndarray):
    """Shannon entropy in bits of a distribution given as log-probs (nats).

    ``H = -Σ p·log₂p = -(Σ p·ln p) / ln2`` with ``p = exp(log_p)``. Uses
    ``nansum`` so a ``-inf`` log-prob (``p = 0``) contributes ``0`` rather than
    ``nan`` (the ``0·log 0`` convention); for a finite, renormalised
    distribution this equals ``sum``.

    Args:
        log_p: Log-probabilities in nats over the full support of one
            distribution; which distribution is the caller's choice.

    Returns:
        A numpy scalar whose dtype follows ``log_p``.
    """
    p = np.exp(log_p)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(p * log_p) / LN2


def renyi_bits(log_p: np.ndarray, alpha: float):
    """Rényi-``alpha`` entropy in bits of a distribution given as log-probs (nats).

    ``H_α = (1/(1-α))·log₂ Σ p^α`` via a max-shifted log-sum-exp on ``α·log_p``.
    ``alpha == 1`` is the Shannon limit and delegates to :func:`shannon_bits`.
    Order zero counts positive-probability outcomes. Near one, normalized
    probabilities and ``expm1``/``log1p`` avoid cancellation from rounded inputs.

    Args:
        log_p: Log-probabilities in nats over the full support of one
            distribution; which distribution is the caller's choice.
        alpha: Finite nonnegative order. ``0.5`` is the square-root order,
            ``1.0`` the Shannon limit, ``2.0`` the collision entropy.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"Rényi alpha must be finite and nonnegative, got {alpha}")
    if alpha == 0.0:
        support = np.count_nonzero(np.isfinite(log_p))
        return np.asarray(np.log2(support), dtype=log_p.dtype)[()]
    if alpha == 1.0:
        return shannon_bits(log_p)
    if abs(alpha - 1.0) < 1e-3:
        values = log_p[log_p != -np.inf].astype(np.float64)
        maximum = np.max(values)
        values -= maximum + np.log(np.sum(np.exp(values - maximum)))
        weights = np.exp(values)
        nonzero = weights != 0
        change = np.sum(weights[nonzero] * np.expm1((alpha - 1.0) * values[nonzero]))
        result = np.log1p(change) / ((1.0 - alpha) * LN2)
        return np.asarray(result, dtype=log_p.dtype)[()]
    scaled = alpha * log_p
    m = float(np.max(scaled))
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    return log_sum / ((1.0 - alpha) * LN2)


def collision_prob(log_p: np.ndarray):
    """``Σ p²`` (expected probability under the model's own distribution).

    Equals ``2^(-H₂)`` for ``H₂`` in bits (Rényi-2 / collision entropy).
    Computed via a max-shifted log-sum-exp on ``2·log_p`` for stability.

    Args:
        log_p: Log-probabilities in nats over the full support of one
            distribution; which distribution is the caller's choice.
    """
    two = 2.0 * log_p
    m = float(np.max(two))
    return np.exp(m + np.log(np.sum(np.exp(two - m))))
