"""Numpy MC continuation-entropy simulator (numpy-only, no torch).

``sample_top_k`` reproduces the shape of the responsivity paper's released
``measures.py`` estimator:
continuations are *sampled* from a top-k-truncated distribution while their
log-probabilities are *scored* under the full distribution. With
``sample_top_k=None`` it is a correct MC estimator of the length-L
continuation entropy. The acceptance tests use it to contrast the released
shape with the correct estimator.
"""

import numpy as np

from replica.feature.diagnostics.references._closed_forms import topk_renorm
from replica.feature.diagnostics.references.markov import MarkovReferenceLM

from ....core import Stimulus

__all__ = ["simulate_mc_entropy"]


def simulate_mc_entropy(
    reference: MarkovReferenceLM,
    stimulus: Stimulus,
    L: int,
    n_samples: int,
    rng: np.random.Generator,
    sample_top_k: int | None = None,
) -> np.ndarray:
    """Per-position MC estimate of -E[log2 P(continuation)] in bits.

    Args:
        reference: chain supplying the exact predictive distributions and the
            transition matrix the continuations walk.
        stimulus: probe whose positions are estimated, one value per word.
        L: continuation length in words, the first word included.
        n_samples: continuations drawn per position.
        rng: generator drawing the continuations.
        sample_top_k: truncate the sampling distribution to its ``k`` most
            likely words while still scoring under the full one. ``None``
            samples from the full distribution, which is the correct estimator.
    """
    dists, _ = reference.predictive_distributions(stimulus)
    P = reference.transition
    V = P.shape[0]
    P_sample = topk_renorm(P, sample_top_k) if sample_top_k is not None else P
    cdf = np.cumsum(P_sample, axis=1)

    out = np.empty(len(dists))
    for i, d in enumerate(dists):
        d_sample = (
            topk_renorm(d[None], sample_top_k)[0] if sample_top_k is not None else d
        )
        w = rng.choice(V, size=n_samples, p=d_sample)
        log_p = np.log2(d[w])  # scored under the full distribution
        for _ in range(L - 1):
            u = rng.random(n_samples)
            # clamp: float roundoff can leave cdf[-1] marginally below u
            nxt = np.minimum((u[:, None] > cdf[w]).sum(axis=1), V - 1)
            log_p = log_p + np.log2(P[w, nxt])
            w = nxt
        out[i] = -log_p.mean()
    return out
