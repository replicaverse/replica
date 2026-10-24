"""Candidate predictor quantities evaluated on a Markov reference model.

Entropy and surprisal targets use bits; expected probability and rank are
dimensionless. Fingerprinting compares outputs with these per-unit targets.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from replica.feature.diagnostics._statistics import affine_fit
from replica.feature.diagnostics.references._closed_forms import (
    continuation_entropy,
    continuation_renyi,
    h_bits,
    renyi_bits,
    stationary_distribution,
    temperature,
    topk_renorm,
    truncated_sampler_xent,
)
from replica.feature.diagnostics.references.markov import MarkovReferenceLM

from ...core import Stimulus

__all__ = ["TargetSpec", "assert_identifiable", "default_specs"]

_SpecFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TargetSpec:
    """One candidate target quantity evaluated on a reference model.

    ``fn(transition, dists, observed) -> (n_units,)`` where ``dists[t]`` is
    the exact distribution predicting unit ``t`` and ``observed[t]`` the
    vocab index of the word at ``t``.
    """

    name: str
    description: str
    fn: _SpecFn

    def compute(
        self,
        reference: MarkovReferenceLM,
        stimulus: Stimulus,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Exact value of this quantity at every unit of one stimulus.

        Args:
            reference: chain supplying the exact predictive distributions.
            stimulus: probe whose units are scored; its words must all be in
                the reference vocabulary.
            target_layer: annotation layer defining the unit granularity.

        Returns:
            One value per unit of ``target_layer``, in reading order.
        """
        dists, observed = reference.predictive_distributions(stimulus, target_layer)
        return self.fn(reference.transition, dists, observed)


def assert_identifiable(
    specs: list[TargetSpec],
    reference: MarkovReferenceLM | None = None,
    stimuli: list[Stimulus] | None = None,
    target_layer: str = "words",
    max_r2: float = 1.0 - 1e-6,
) -> None:
    """Raise ValueError if any two specs are affinely confusable on the probe.

    Run this after extending the battery with custom specs: a spec that is
    constant, name-duplicated, or an affine near-copy of another would make
    exact/affine attributions ambiguous. The default battery is held to the
    same bound in ``tests/feature/diagnostics/test_specs.py``.

    Args:
        specs: the battery to check, pairwise and for constancy.
        reference: chain the exact values are computed on
            (default: ``MarkovReferenceLM()``).
        stimuli: probe stimuli the specs are compared over (default: the
            reference's de Bruijn probe).
        target_layer: annotation layer defining the unit granularity.
        max_r2: R² of an affine fit between two specs at or above which they
            count as confusable.
    """
    reference = reference if reference is not None else MarkovReferenceLM()
    stimuli = stimuli if stimuli is not None else [reference.probe_stimulus()]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate spec names: {dupes}")
    arrays = {
        s.name: np.concatenate(
            [s.compute(reference, st, target_layer) for st in stimuli]
        )
        for s in specs
    }
    for name, x in arrays.items():
        if np.std(x) == 0:
            raise ValueError(f"spec {name!r} is constant on the probe")
    for (na, xa), (nb, xb) in itertools.combinations(arrays.items(), 2):
        _, _, r2 = affine_fit(xa, xb)
        if r2 >= max_r2:
            raise ValueError(
                f"specs {na!r} and {nb!r} are affinely confusable on the "
                f"probe (R²={r2:.9f} ≥ {max_r2}); attribution between them "
                "would be ambiguous"
            )


def _surprisal(_T: np.ndarray, dists: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return -np.log2(dists[np.arange(len(observed)), observed])


def _unigram_surprisal(
    T: np.ndarray, _dists: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """Context-free surprisal under the chain's stationary distribution.

    The fingerprint for context-blind predictors: anything that dropped or
    mangled its left context collapses onto a per-word-type lookup, which
    matches this spec (affinely) and not the contextual ones.
    """
    pi = stationary_distribution(T)
    return -np.log2(pi[observed])


def _rank(_T: np.ndarray, dists: np.ndarray, observed: np.ndarray) -> np.ndarray:
    obs_p = dists[np.arange(len(observed)), observed]
    return 1.0 + (dists > obs_p[:, None]).sum(axis=-1)


def default_specs(
    top_k: int = 4,
    temp: float = 2.0,
    renyi_alphas: tuple[float, ...] = (0.5, 2.0),
    continuation_lengths: tuple[int, ...] = (2, 3),
) -> list[TargetSpec]:
    """The standard battery of candidate target quantities.

    Args:
        top_k: truncation size for the top-k entropy and truncated-sampler
            specs.
        temp: sampling temperature for the tempered-entropy spec.
        renyi_alphas: orders to add a next-word Rényi entropy spec for.
        continuation_lengths: continuation lengths, in words, to add a
            continuation-entropy and a truncated-sampler spec for. The first
            entry also fixes the length of the Rényi-0.5 continuation spec.
    """
    renyi_alphas = tuple(float(a) for a in renyi_alphas)  # int 2 → name alpha2.0
    if not continuation_lengths:
        raise ValueError("continuation_lengths must not be empty")
    specs = [
        TargetSpec(
            "surprisal",
            "-log2 P(w_t | context), full next-word distribution",
            _surprisal,
        ),
        TargetSpec(
            "unigram_surprisal",
            "-log2 of the context-FREE (stationary) word probability, the "
            "fingerprint of a predictor that lost its context-dependence",
            _unigram_surprisal,
        ),
        TargetSpec(
            "entropy",
            "Shannon entropy of the full next-word distribution",
            lambda _T, d, _o: h_bits(d),
        ),
        TargetSpec(
            "surprisal_reduction",
            "entropy(t) - surprisal(t): uncertainty reduction upon seeing w_t",
            lambda T, d, o: h_bits(d) - _surprisal(T, d, o),
        ),
        TargetSpec(
            "expected_probability",
            "E_{w~p}[p(w)] = Σ p(w)^2 (exp2 of negative collision entropy)",
            lambda _T, d, _o: (d**2).sum(axis=-1),
        ),
        TargetSpec(
            "rank",
            "1-indexed rank of the observed word in the next-word distribution",
            _rank,
        ),
        TargetSpec(
            f"entropy_top{top_k}",
            f"Shannon entropy of the top-{top_k}-truncated, renormalized "
            "next-word distribution",
            lambda _T, d, _o, k=top_k: h_bits(topk_renorm(d, k)),
        ),
        TargetSpec(
            f"entropy_temp{temp}",
            f"Shannon entropy at sampling temperature {temp}",
            lambda _T, d, _o, tau=temp: h_bits(temperature(d, tau)),
        ),
    ]
    for alpha in renyi_alphas:
        specs.append(
            TargetSpec(
                f"renyi_entropy_alpha{alpha}",
                f"Rényi-{alpha} entropy of the full next-word distribution",
                lambda _T, d, _o, a=alpha: renyi_bits(d, a),
            )
        )
    for L in continuation_lengths:
        specs.append(
            TargetSpec(
                f"continuation_entropy_L{L}",
                f"Shannon entropy of the length-{L} continuation distribution",
                lambda T, d, _o, n=L: continuation_entropy(T, d, n),
            )
        )
        specs.append(
            TargetSpec(
                f"truncated_sampler_xent_top{top_k}_L{L}",
                f"E over top-{top_k}-truncated samples of full-distribution "
                f"-log2 P, length {L} (truncated-sampler MC-entropy shape)",
                lambda T, d, _o, n=L, k=top_k: truncated_sampler_xent(T, d, k, n),
            )
        )
    specs.append(
        TargetSpec(
            f"continuation_renyi_alpha0.5_L{continuation_lengths[0]}",
            f"Rényi-0.5 entropy of the length-{continuation_lengths[0]} "
            "continuation distribution",
            lambda T, d, _o, n=continuation_lengths[0]: continuation_renyi(
                T, d, 0.5, n
            ),
        )
    )
    return specs
