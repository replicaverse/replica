"""Production MC samplers on the exactly-solvable token-level chain.

Unlike ``test_torch_reference.py`` (which calls ``_estimate`` on single
prefixes), these tests run the full production ``__call__`` paths, the
code that builds predictor pickles, on ``TokenChainReference``: real
``align_from_tokenizer`` alignment over multi-token words, KV-cache
forking, and ``MCWordEntropyLM``'s starter/subword masks and EOW
stopping rule, all checked against enumerated closed forms.
"""

import hashlib

import numpy as np
import pytest

from replica.feature.diagnostics.references._closed_forms import (
    h_bits,
    mc_word_outcome_distribution,
    renyi_bits,
)
from replica.feature.diagnostics.references.torch_models import TokenChainReference
from replica.feature.entropy.estimators import (
    ContinuationEntropyLM,
    ForwardMarginalEntropyLM,
    ForwardWordLookaheadEntropyLM,
    MCWordEntropyLM,
)
from replica.feature.extractors import ContinuationEntropy, MCWordEntropy
from tests.feature.entropy._entropy_mocks import build_estimator

N_SAMPLES = 4096
MAX_ITER = 4
WORDS = ["ba", "dax", "gaxy", "ka", "bax", "da"]
ATOL_BITS = 0.3  # ≫ MC se at N=4096 (≈ sd/64), deterministic given the seed


@pytest.fixture(scope="module")
def token_ref():
    return TokenChainReference()


def _chain(token_ref, n_samples, seed) -> dict:
    """Settings every estimator on the token chain shares."""
    return {
        "model": token_ref.model,
        "tokenizer": token_ref.tokenizer,
        "model_id": "markov_token_chain",
        "model_identity": {
            "sha256": hashlib.sha256(token_ref.transition.tobytes()).hexdigest()
        },
        "n_samples": n_samples,
        "seed": seed,
    }


def _make_mcword(token_ref, *, alpha=1.0, n_samples=N_SAMPLES, seed=0):
    return build_estimator(
        MCWordEntropyLM,
        name=f"mcword_token_chain_N{n_samples}_a{alpha}",
        max_iter=MAX_ITER,
        alpha=float(alpha),
        **_chain(token_ref, n_samples, seed),
    )


def _make_continuation(token_ref, *, max_tokens=2, n_samples=N_SAMPLES, seed=0):
    return build_estimator(
        ContinuationEntropyLM,
        name=f"cont_token_chain_N{n_samples}_L{max_tokens}",
        max_tokens=max_tokens,
        **_chain(token_ref, n_samples, seed),
    )


def _make_forward_word_lookahead(
    token_ref,
    *,
    max_words=3,
    max_iter=4,
    n_samples=N_SAMPLES,
    seed=0,
):
    return build_estimator(
        ForwardWordLookaheadEntropyLM,
        name=f"fwl_token_chain_N{n_samples}_M{max_iter}_K{max_words}",
        max_iter=max_iter,
        max_words=max_words,
        alpha=1.0,
        **_chain(token_ref, n_samples, seed),
    )


def _make_forward_marginal(token_ref, *, max_tokens=3, n_samples=N_SAMPLES, seed=0):
    return build_estimator(
        ForwardMarginalEntropyLM,
        name=f"fwd_marg_token_chain_N{n_samples}_L{max_tokens}",
        max_tokens=max_tokens,
        **_chain(token_ref, n_samples, seed),
    )


class TestTokenizer:
    def test_multi_token_words_tokenize_greedily(self, token_ref):
        enc = token_ref.tokenizer("ba dax gaxy")
        assert enc["input_ids"] == [1, 2, 5, 3, 5, 6]
        assert enc["offset_mapping"][:3] == [(0, 2), (3, 5), (5, 6)]

    def test_masks_match_production_partition(self, token_ref):
        # BOS has no Ġ prefix, so it lands in the SUBWORD set, same rule
        # as MCWordEntropyLM._compute_token_masks on real GPT-2 vocabs.
        lm = _make_mcword(token_ref)
        np.testing.assert_array_equal(lm._space_ids.numpy(), token_ref.starter_ids)
        np.testing.assert_array_equal(lm._subword_ids.numpy(), token_ref.subword_ids)

    def test_outcome_distribution_is_normalized(self, token_ref):
        for state in range(token_ref.transition.shape[0]):
            q = mc_word_outcome_distribution(
                token_ref.transition,
                token_ref.starter_ids,
                token_ref.subword_ids,
                state,
                MAX_ITER,
            )
            assert np.isclose(q.sum(), 1.0, atol=1e-12)


class TestMCWordEntropyProduction:
    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_call_matches_enumerated_word_entropy(self, token_ref, alpha):
        stim = token_ref.make_stimulus(WORDS)
        # Rényi-0.5 is nonlinear in the sample mean, so its standard error is
        # larger than Shannon's; 4x the samples restores the margin at the same
        # atol.
        n = 16384 if alpha == 0.5 else N_SAMPLES
        lm = _make_mcword(token_ref, alpha=alpha, n_samples=n)
        est = MCWordEntropy()(stim, lm(stim))
        exact = token_ref.exact_mc_word_entropy(stim, MAX_ITER, alpha=alpha)

        assert np.isnan(est[0]) and np.isnan(exact[0])  # no left context
        np.testing.assert_allclose(est[1:], exact[1:], atol=ATOL_BITS)

    def test_estimate_path_agrees_with_one_pass_target(self, token_ref):
        # Per-prefix path (_estimate) and one-pass path (__call__) are two
        # implementations of the same sampler; both must sit on the same
        # enumerated target.
        state = 5  # subword 'x' as last prefix token
        lm = _make_mcword(token_ref)
        est = lm._estimate([1, state])
        q = mc_word_outcome_distribution(
            token_ref.transition,
            token_ref.starter_ids,
            token_ref.subword_ids,
            state,
            MAX_ITER,
        )
        assert abs(est - h_bits(q)) < ATOL_BITS

    def test_renyi_target_differs_from_shannon(self, token_ref):
        # Guard against the Rényi/Shannon base-mixing bug class: the two
        # exact targets must differ enough that the estimators could not
        # pass each other's test.
        stim = token_ref.make_stimulus(WORDS)
        h1 = token_ref.exact_mc_word_entropy(stim, MAX_ITER, alpha=1.0)
        h05 = token_ref.exact_mc_word_entropy(stim, MAX_ITER, alpha=0.5)
        assert np.nanmax(np.abs(h1 - h05)) > 2 * ATOL_BITS


class TestContinuationEntropyProduction:
    def test_call_matches_closed_form(self, token_ref):
        stim = token_ref.make_stimulus(WORDS)
        lm = _make_continuation(token_ref, max_tokens=2)
        est = ContinuationEntropy()(stim, lm(stim))
        exact = token_ref.exact_continuation_entropy(stim, max_tokens=2)

        assert np.isnan(est[0]) and np.isnan(exact[0])
        np.testing.assert_allclose(est[1:], exact[1:], atol=ATOL_BITS)


class TestForwardMarginalEntropyProduction:
    def test_call_matches_closed_form(self, token_ref):
        # All three profiles (marginal mixture, free per-step conditional,
        # joint cumulative) are exact matrix algebra of the chain (see
        # TokenChainReference.exact_forward_marginal). Full-vocab ancestral
        # sampling → unbiased for free/joint; the marginal is a plug-in
        # entropy of an unbiased mixture (bias vanishes with N), so a
        # generous atol well below the marg–free gap keeps it non-hollow.
        stim = token_ref.make_stimulus(WORDS)
        L = 3
        lm = _make_forward_marginal(token_ref, max_tokens=L, n_samples=16384)
        outs = lm(stim)
        marg = np.array([o.values["forward_marginal_entropy"] for o in outs])
        free = np.array([o.values["free_step_entropy"] for o in outs])
        joint = np.array([o.values["joint_entropy"] for o in outs])
        ex_marg, ex_free, ex_joint = token_ref.exact_forward_marginal(stim, L)

        assert np.all(np.isnan(marg[0])) and np.all(np.isnan(ex_marg[0]))
        np.testing.assert_allclose(marg[1:], ex_marg[1:], atol=0.1)
        np.testing.assert_allclose(free[1:], ex_free[1:], atol=0.1)
        np.testing.assert_allclose(joint[1:], ex_joint[1:], atol=0.1)

    def test_identities(self, token_ref):
        # Built-in algebraic invariants of the estimator (single draw):
        # marg[1]==free[1], marg>=free elementwise, joint non-decreasing.
        stim = token_ref.make_stimulus(WORDS)
        lm = _make_forward_marginal(token_ref, max_tokens=4, n_samples=2048)
        outs = lm(stim)
        for o in outs[1:]:
            marg = o.values["forward_marginal_entropy"]
            free = o.values["free_step_entropy"]
            joint = o.values["joint_entropy"]
            # depth-1 has nothing to mix; equal up to the float32 mixture-
            # accumulation path (marg sums softmaxes in fp32, free sums fp64).
            assert abs(marg[0] - free[0]) < 1e-6
            assert np.all(marg - free > -1e-9)  # mixing raises entropy
            assert np.all(np.diff(joint) > -1e-9)  # cumulative joint monotone

    def test_marginal_differs_from_free(self, token_ref):
        # Anti-hollow: the marginal must be meaningfully above the free
        # conditional at some depth (else the test could pass on the wrong key).
        stim = token_ref.make_stimulus(WORDS)
        lm = _make_forward_marginal(token_ref, max_tokens=3, n_samples=16384)
        outs = lm(stim)
        marg = np.array([o.values["forward_marginal_entropy"] for o in outs[1:]])
        free = np.array([o.values["free_step_entropy"] for o in outs[1:]])
        assert np.nanmax(marg - free) > 0.1


class TestForwardWordLookaheadProduction:
    """ForwardWordLookaheadEntropyLM has no closed-form referee here: its
    K-word boundary tree is not enumerated. These tests pin the estimator's
    exact by-construction invariants, seeded determinism, output contract,
    and the truncation rule that distinguishes it from ``mc_word_entropy``."""

    def test_saved_identities_reproduce_marginal_entropy(self):
        lm = object.__new__(ForwardWordLookaheadEntropyLM)
        identities = np.array([[1], [1 + 2**32]], dtype=np.uint64)
        values = lm._aggregate(np.full((2, 1), -1.0), identities)
        saved = values["fwd_word_id_hash"]
        np.testing.assert_array_equal(saved, identities)
        assert saved.dtype == np.uint64
        _, counts = np.unique(saved[:, 0], return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -(probabilities * np.log2(probabilities)).sum()
        assert entropy == values["marginal_word_entropy_plugin"][0] == 1.0

    def test_chain_rule_identities(self, token_ref):
        stim = token_ref.make_stimulus(WORDS)
        lm = _make_forward_word_lookahead(
            token_ref, max_words=3, max_iter=4, n_samples=2048
        )
        for o in lm(stim)[1:]:
            joint = o.values["joint_word_entropy"]
            cond = o.values["conditional_word_entropy"]
            # joint is the running sum of the per-step conditionals (exact).
            np.testing.assert_allclose(joint, np.cumsum(cond), atol=1e-9)
            assert np.all(np.diff(joint) > -1e-9)  # joint monotone (cond >= 0)
            # Miller-Madow corrected marginal >= raw plug-in.
            assert np.all(
                o.values["marginal_word_entropy"]
                - o.values["marginal_word_entropy_plugin"]
                > -1e-12
            )

    def test_seeded_determinism(self, token_ref):
        stim = token_ref.make_stimulus(WORDS)
        a = _make_forward_word_lookahead(
            token_ref, max_words=2, max_iter=4, n_samples=1024, seed=7
        )(stim)
        b = _make_forward_word_lookahead(
            token_ref, max_words=2, max_iter=4, n_samples=1024, seed=7
        )(stim)
        for oa, ob in zip(a[1:], b[1:], strict=False):
            np.testing.assert_array_equal(
                oa.values["fwd_word_log_p_step_nats"],
                ob.values["fwd_word_log_p_step_nats"],
            )

    def test_output_contract(self, token_ref):
        stim = token_ref.make_stimulus(WORDS)
        K = 3
        outs = _make_forward_word_lookahead(
            token_ref, max_words=K, max_iter=4, n_samples=64
        )(stim)
        assert np.all(np.isnan(outs[0].values["joint_word_entropy"]))  # no left context
        for o in outs[1:]:
            assert set(o.values) == {
                "joint_word_entropy",
                "conditional_word_entropy",
                "marginal_word_entropy",
                "marginal_word_entropy_plugin",
                "fwd_word_log_p_step_nats",
                "fwd_word_id_hash",
            }
            assert o.values["joint_word_entropy"].shape == (K,)
            assert o.values["fwd_word_log_p_step_nats"].shape[1] == K
            assert o.values["fwd_word_log_p_step_nats"].dtype == np.float32
            assert o.values["fwd_word_id_hash"].dtype == np.uint64

    def test_truncation_rule_differs_from_mcword(self, token_ref):
        # Anti-trap: FWL's per-word conditional uses a deeper global token
        # budget (K*max_iter) than ``mc_word_entropy``'s per-word max_iter,
        # so its first-word conditional must not collapse onto the MCWord exact
        # word entropy. Guards against a regression that reuses the wrong rule.
        stim = token_ref.make_stimulus(WORDS)
        mi = 2
        outs = _make_forward_word_lookahead(
            token_ref,
            max_words=2,
            max_iter=mi,
            n_samples=16384,
            seed=0,
        )(stim)
        mcword = token_ref.exact_mc_word_entropy(stim, max_iter=mi, alpha=1.0)
        diffs = [
            abs(o.values["conditional_word_entropy"][0] - mcword[t])
            for t, o in enumerate(outs)
            if t > 0 and not np.isnan(mcword[t])
        ]
        assert max(diffs) > 0.1


def test_renyi_exact_helper_consistency(token_ref):
    # renyi_bits(q, alpha→1) ≈ h_bits(q) on an outcome distribution.
    q = mc_word_outcome_distribution(
        token_ref.transition,
        token_ref.starter_ids,
        token_ref.subword_ids,
        1,
        MAX_ITER,
    )
    assert np.isclose(renyi_bits(q, 1.0 + 1e-9), h_bits(q), atol=1e-5)
