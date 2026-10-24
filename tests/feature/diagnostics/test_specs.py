"""Exactness of the TargetSpec computers, and identifiability of the battery.

The closed-form sequence-level computers (chain-rule Shannon, matrix-power
Rényi, truncated-sampler cross-entropy) are validated against brute-force
enumeration over all V^L continuations, an independent implementation.

The identifiability test is the design guarantee behind fingerprinting: on
the probe stimulus, no two specs in the default battery may be affinely
related, otherwise an exact/affine match would be ambiguous.
"""

import itertools

import numpy as np
import pytest

from replica.feature.diagnostics import (
    MarkovReferenceLM,
    assert_identifiable,
    default_specs,
    make_stimulus,
)
from replica.feature.diagnostics.references._closed_forms import (
    continuation_entropy,
    continuation_renyi,
    h_bits,
    topk_renorm,
    truncated_sampler_xent,
)


@pytest.fixture(scope="module")
def ref():
    return MarkovReferenceLM()


def _spec(name):
    by_name = {s.name: s for s in default_specs()}
    return by_name[name]


# ── Brute-force enumeration (independent of the closed forms) ────────────


def _enumerate_paths(P, d, L):
    """Probabilities of all V^L length-L paths starting from dist d."""
    p, last = d.copy(), np.arange(len(d))
    for _ in range(L - 1):
        p = (p[:, None] * P[last]).ravel()
        last = np.tile(np.arange(P.shape[0]), len(last))
    return p


def _enumerate_xent(P, Q, d, dq, L):
    """E over Q-chain paths of full-chain -log2 p, by enumeration."""
    qp, log_p, last = dq.copy(), np.log2(d), np.arange(len(d))
    for _ in range(L - 1):
        qp = (qp[:, None] * Q[last]).ravel()
        log_p = (log_p[:, None] + np.log2(P[last])).ravel()
        last = np.tile(np.arange(P.shape[0]), len(last))
    return -(qp * log_p).sum()


# ── Point checks ─────────────────────────────────────────────────────────


class TestRowLevel:
    def test_near_uniform_row_entropy_hand_computed(self, ref):
        # State 0's row is the near-uniform shape (roll by 0): 4x0.13 + 4x0.12.
        stim = make_stimulus(["ba", "ba"])
        vals = _spec("entropy").compute(ref, stim)
        by_hand = -(4 * 0.13 * np.log2(0.13) + 4 * 0.12 * np.log2(0.12))
        assert np.isclose(vals[1], by_hand, rtol=1e-12)

    def test_surprisal_matches_transition_entry(self, ref):
        stim = make_stimulus(["da", "ta"])
        vals = _spec("surprisal").compute(ref, stim)
        expected = -np.log2(ref.transition[1, ref.vocab.index("ta")])
        assert np.isclose(vals[1], expected)

    def test_rank_of_most_probable_word_is_1(self, ref):
        best = ref.vocab[int(np.argmax(ref.initial))]
        vals = _spec("rank").compute(ref, make_stimulus([best]))
        assert vals[0] == 1.0

    def test_topk_renorm_keeps_k_and_normalizes(self):
        p = np.array([[0.4, 0.3, 0.2, 0.1]])
        q = topk_renorm(p, 2)
        assert np.isclose(q.sum(), 1.0)
        np.testing.assert_allclose(q[0], [4 / 7, 3 / 7, 0.0, 0.0])


class TestSequenceLevel:
    def test_continuation_L1_equals_entropy(self, ref):
        dists, _ = ref.predictive_distributions(ref.probe_stimulus())
        np.testing.assert_allclose(
            continuation_entropy(ref.transition, dists, 1), h_bits(dists)
        )

    @pytest.mark.parametrize("L", [2, 3])
    def test_continuation_entropy_matches_enumeration(self, ref, L):
        for d in [ref.initial, *ref.transition]:
            p = _enumerate_paths(ref.transition, d, L)
            brute = -(p * np.log2(p)).sum()
            closed = continuation_entropy(ref.transition, d[None], L)[0]
            assert np.isclose(closed, brute, rtol=1e-12)

    @pytest.mark.parametrize("alpha", [0.5, 2.0])
    def test_continuation_renyi_matches_enumeration(self, ref, alpha):
        for d in [ref.initial, *ref.transition]:
            p = _enumerate_paths(ref.transition, d, 3)
            brute = np.log2((p**alpha).sum()) / (1.0 - alpha)
            closed = continuation_renyi(ref.transition, d[None], alpha, 3)[0]
            assert np.isclose(closed, brute, rtol=1e-12)

    @pytest.mark.parametrize("L", [2, 3])
    def test_truncated_sampler_xent_matches_enumeration(self, ref, L):
        k = 4
        Q = topk_renorm(ref.transition, k)
        for d in [ref.initial, *ref.transition]:
            dq = topk_renorm(d[None], k)[0]
            brute = _enumerate_xent(ref.transition, Q, d, dq, L)
            closed = truncated_sampler_xent(ref.transition, d[None], k, L)[0]
            assert np.isclose(closed, brute, rtol=1e-12)


# ── Identifiability of the default battery ───────────────────────────────


class TestIdentifiability:
    def test_no_spec_is_constant_on_probe(self, ref):
        stim = ref.probe_stimulus()
        for spec in default_specs():
            vals = spec.compute(ref, stim)
            assert np.std(vals) > 1e-3, f"{spec.name} is ~constant on the probe"

    def test_default_battery_is_identifiable(self, ref):
        assert_identifiable(default_specs(), ref)  # raises on confusion

    def test_assert_identifiable_catches_affine_copy(self, ref):
        from replica.feature.diagnostics import TargetSpec
        from replica.feature.diagnostics.references._closed_forms import h_bits

        copy = TargetSpec(
            "entropy_rescaled", "affine copy", lambda T, d, o: 2.0 * h_bits(d) + 1.0
        )
        with pytest.raises(ValueError, match="affinely confusable"):
            assert_identifiable([*default_specs(), copy], ref)

    def test_assert_identifiable_catches_duplicate_names(self, ref):
        specs = default_specs()
        with pytest.raises(ValueError, match="duplicate spec names"):
            assert_identifiable([*specs, specs[0]], ref)

    def test_assert_identifiable_catches_constant_spec(self, ref):
        from replica.feature.diagnostics import TargetSpec

        const = TargetSpec(
            "constant", "no spread", lambda T, d, o: np.full(len(o), 3.0)
        )
        with pytest.raises(ValueError, match="constant on the probe"):
            assert_identifiable([*default_specs(), const], ref)

    def test_approx_tier_identity_rms_floor(self, ref):
        """The identity-RMS criterion is the only thing separating a
        bug-shape spec from its target in the approx tier (e.g. truncated
        xent vs continuation entropy). Require every in-slope-band spec
        pair to sit ≥ 1.4x the RMS limit apart, so a noiseless predictor
        of A can never earn 'approx' on B after a matrix edit."""
        from replica.feature.diagnostics.fingerprint import (
            _APPROX_IDENTITY_RMS,
            _APPROX_SLOPE,
        )

        stim = ref.probe_stimulus()
        arrays = {s.name: s.compute(ref, stim) for s in default_specs()}
        for (na, a), (nb, b) in itertools.permutations(arrays.items(), 2):
            slope, _ = np.polyfit(b, a, 1)
            if not (_APPROX_SLOPE[0] <= slope <= _APPROX_SLOPE[1]):
                continue
            rms = float(np.sqrt(np.mean((a - b) ** 2)))
            assert rms > 1.4 * _APPROX_IDENTITY_RMS * np.std(b), (
                f"predictor of {na!r} could pass as approx-{nb!r} "
                f"(rms/std = {rms / np.std(b):.3f})"
            )

    def test_no_topk_boundary_ties_in_reference(self, ref):
        # truncated-sampler specs propagate the kept indices through the
        # chain; a value tie at the top-k boundary would make their exact
        # value depend on argsort order. k = default_specs() top_k.
        k = 4
        for row in [ref.initial, *ref.transition]:
            s = np.sort(row)[::-1]
            assert s[k - 1] > s[k] + 1e-12, f"top-{k} boundary tie in {row}"
