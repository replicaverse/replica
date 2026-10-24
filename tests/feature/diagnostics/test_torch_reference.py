"""REPLICA's real MC backend against the torch Markov reference.

Drives ``ContinuationEntropyLM``'s per-prefix sampler (``_estimate``) on
:class:`MarkovTorchLM`, whose continuation entropy has a closed form.
Predictor pickles come from ``__call__``, whose one-pass driver samples with
the same ancestral step from a shared prefix cache; its draws are checked in
``tests/feature/entropy/test_continuation_entropy_one_pass.py``. No HF download: the
estimator is built on the Markov model with
``tests.feature.entropy._entropy_mocks.build_estimator``.
"""

import hashlib

import numpy as np
import pytest
import torch

from replica.feature.diagnostics import MarkovReferenceLM, fingerprint
from replica.feature.diagnostics.references._closed_forms import continuation_entropy
from replica.feature.diagnostics.references.torch_models import MarkovTorchLM
from replica.feature.entropy.estimators import ContinuationEntropyLM
from tests.feature.entropy._entropy_mocks import build_estimator

N_SAMPLES = 4096
L = 2


@pytest.fixture(scope="module")
def ref():
    return MarkovReferenceLM()


def _make_estimator(ref, n_samples=N_SAMPLES, max_tokens=L, seed=0):
    return build_estimator(
        ContinuationEntropyLM,
        MarkovTorchLM(ref.transition),
        name=f"cont_entropy_markov_N{n_samples}_L{max_tokens}",
        n_samples=n_samples,
        seed=seed,
        model_id="markov_reference",
        model_identity={"sha256": hashlib.sha256(ref.transition.tobytes()).hexdigest()},
        max_tokens=max_tokens,
    )


def test_softmax_of_logits_is_transition_row(ref):
    model = MarkovTorchLM(ref.transition)
    out = model(torch.tensor([[3, 5]]))
    probs = torch.softmax(out.logits[0], dim=-1).numpy()
    np.testing.assert_allclose(probs[0], ref.transition[3], rtol=1e-12)
    np.testing.assert_allclose(probs[1], ref.transition[5], rtol=1e-12)


@pytest.mark.parametrize("state", [0, 3, 7])
def test_estimate_matches_closed_form(ref, state):
    est = _make_estimator(ref)._estimate([state])
    exact = continuation_entropy(ref.transition, ref.transition[state][None], L)[0]
    assert abs(est - exact) < 0.3, (
        f"state {state}: MC {est:.3f} vs exact {exact:.3f} "
        f"(beyond MC noise at N={N_SAMPLES})"
    )


def test_production_sampler_fingerprints_as_continuation_entropy(ref):
    # One estimate per chain state, mapped onto the probe positions
    # (the predictive state at position t is the word at t-1; position 0
    # has no token prefix for the torch path, so it stays NaN).
    per_state = np.array(
        [_make_estimator(ref, seed=s)._estimate([s]) for s in range(len(ref.vocab))]
    )
    probe = ref.probe_stimulus()
    states = ref.state_indices(probe)
    values = np.full(len(states), np.nan)
    values[1:] = per_state[states[:-1]]

    report = fingerprint(values, ref, stimuli=[probe])
    assert report.best.name == f"continuation_entropy_L{L}"
    assert report.best.tier == "approx", report.to_text()
