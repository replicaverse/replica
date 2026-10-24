"""Closed-form Markov-chain quantities in bits from probability-array inputs."""

import numpy as np

__all__ = [
    "continuation_entropy",
    "continuation_renyi",
    "cross_entropy_bits",
    "h_bits",
    "mc_word_outcome_distribution",
    "renyi_bits",
    "stationary_distribution",
    "temperature",
    "topk_renorm",
    "truncated_sampler_xent",
]


def h_bits(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis; 0·log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=-1)


def renyi_bits(p: np.ndarray, alpha: float) -> np.ndarray:
    """Rényi-α entropy in bits; α = 1 takes the Shannon limit. At α = 0
    zero entries count as support (numpy 0^0 = 1), so the result is log2(V)
    over the full vocabulary.
    """
    if alpha == 1.0:
        return h_bits(p)
    return np.log2((p**alpha).sum(axis=-1)) / (1.0 - alpha)


def topk_renorm(p: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest entries per row, renormalize the rest to zero."""
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")
    p = np.atleast_2d(p)
    keep = np.argsort(p, axis=-1)[:, -k:]
    q = np.zeros_like(p)
    rows = np.arange(p.shape[0])[:, None]
    q[rows, keep] = p[rows, keep]
    return q / q.sum(axis=-1, keepdims=True)


def temperature(p: np.ndarray, tau: float) -> np.ndarray:
    """softmax(logits / tau) expressed on probabilities: p^(1/tau) / Z."""
    q = p ** (1.0 / tau)
    return q / q.sum(axis=-1, keepdims=True)


def cross_entropy_bits(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """-Σ q log2 p along the last axis, with 0·log convention on q."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0, q * np.log2(p), 0.0)
    return -terms.sum(axis=-1)


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Stationary distribution π with πT = π, Σπ = 1 (assumes ergodic T)."""
    v = T.shape[0]
    a = np.vstack([T.T - np.eye(v), np.ones(v)])
    b = np.concatenate([np.zeros(v), [1.0]])
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.clip(pi, 0.0, None) / np.clip(pi, 0.0, None).sum()


def continuation_entropy(T: np.ndarray, dists: np.ndarray, L: int) -> np.ndarray:
    """Shannon entropy of the length-L continuation distribution, via the
    chain rule: H(W_1..W_L | s) = H(d) + Σ_k m_{k-1} · h(T_rows).
    """
    h_rows = h_bits(T)  # (V,)
    H = h_bits(dists)
    m = dists
    for _ in range(L - 1):
        H = H + m @ h_rows
        m = m @ T
    return H


def continuation_renyi(
    T: np.ndarray, dists: np.ndarray, alpha: float, L: int
) -> np.ndarray:
    """Rényi-α of the length-L continuation distribution, via
    Σ_seq p(seq)^α = (d^α)ᵀ A^{L-1} 1 with A_ij = T_ij^α.
    """
    A = T**alpha
    v = np.ones(T.shape[0])
    for _ in range(L - 1):
        v = A @ v
    total = (dists**alpha) @ v
    return np.log2(total) / (1.0 - alpha)


def truncated_sampler_xent(
    T: np.ndarray, dists: np.ndarray, k: int, L: int
) -> np.ndarray:
    """Limit of an MC 'entropy' whose sampler is top-k truncated but whose
    scorer uses the full distribution: E_{c~Q}[-log2 P(c)].
    """
    Q = topk_renorm(T, k)
    ce_rows = cross_entropy_bits(Q, T)  # (V,)
    q0 = topk_renorm(dists, k)
    xent = cross_entropy_bits(q0, dists)
    m = q0
    for _ in range(L - 1):
        xent = xent + m @ ce_rows
        m = m @ Q
    return xent


def mc_word_outcome_distribution(
    T: np.ndarray,
    starter_ids: np.ndarray,
    subword_ids: np.ndarray,
    state: int,
    max_iter: int,
) -> np.ndarray:
    """Exact outcome distribution of the word-boundary sampler from ``state``.

    Mirrors ``MCWordEntropyLM``'s sampling scheme on a token-level Markov
    chain: the first draw is restricted to word-starter tokens
    (renormalized); each subsequent draw is over ``[subwords... | EOW]``
    with ``P(EOW) = Σ_starters P``; samples that survive ``max_iter`` draws
    are kept truncated (no EOW factor), exactly what the sampler scores.
    The estimator's Shannon/Rényi limits are ``h_bits``/``renyi_bits`` of
    the returned outcome probabilities (which sum to 1).
    """
    p_start = T[state, starter_ids]
    p_start = p_start / p_start.sum()
    outcomes: list[float] = []

    def rec(tok: int, prob: float, drawn: int) -> None:
        step = T[tok]
        if drawn == max_iter:
            outcomes.append(prob)  # truncated: kept without an EOW factor
            return
        outcomes.append(prob * step[starter_ids].sum())  # terminate here
        for sw in subword_ids:
            rec(int(sw), prob * step[sw], drawn + 1)

    for st, p in zip(starter_ids, p_start, strict=True):
        rec(int(st), float(p), 1)
    return np.asarray(outcomes)
