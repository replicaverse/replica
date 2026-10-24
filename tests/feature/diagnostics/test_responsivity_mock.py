"""ResponsivityLM against exact embedding-space limits on the token chain.

With a known next-token distribution p (a chain row) and known input
embeddings, both responsivity quantities have closed forms:

- decontextualised distance      →  Σ_v p(v) · d(e_obs, e_v)
- expected decontextualised dist →  Σ_{v,v'} p(v) p(v') · d(e_v, e_v')

(d = cosine distance; self-pair exclusion leaves i.i.d. pairs, so it does
not change the limit.)

The importance-sampling path (``importance_temp ≠ 1``) must leave both
estimands invariant to the proposal. The *observed* correction reweights
its single draw; the *expected* correction reweights both draws by
w = p/q, so both recover the same pure limits as temp=1: Σ p(v) d(obs, v)
and Σ p(v) p(v') d(v, v'). Recovering the pure limit, which differs from the
mixed Σ p(v) q(v') d, shows that the second draw was reweighted too.
"""

import numpy as np
import pytest
import torch

from replica.feature import ResponsivityLM
from replica.feature.diagnostics.references._closed_forms import temperature
from replica.feature.diagnostics.references.torch_models import TokenChainReference
from replica.feature.extractors import ResponsivityDistance

N_SAMPLES = 4000
WORDS = ["ba", "dax", "gaxy", "ka", "bax", "da"]
# Importance test routes through "ga": prefix state Ġga has the largest
# tempered-proposal shift, so the mixed-vs-pure limits separate decisively.
WORDS_IS = ["ba", "ga", "dax", "ga", "kay", "ga"]
ATOL = 0.04  # ≥ 4σ of the per-position MC se at N=4000

# Deterministic, pairwise non-parallel embedding directions (7 tokens, dim 4).
# Token 0 (<bos>) is rare under every chain row and placed antipodal to the
# rest, so tempering the proposal moves the expected distance decisively,
# the importance test's mixed-vs-pure limits separate by ≫ ATOL.
EMBEDS = np.array(
    [
        [-1.0, -0.8, -0.9, -0.6],
        [0.0, 1.0, 0.0, 0.1],
        [0.1, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0, 0.7],
        [0.0, 0.6, 0.8, 0.1],
        [0.3, 0.0, 0.3, 0.9],
        [0.7, 0.7, 0.1, 0.0],
    ],
    dtype=np.float64,
)

pytestmark = pytest.mark.resource_free(
    "ResponsivityLM runs on the in-memory token chain; no checkpoint loads"
)


@pytest.fixture(scope="module")
def token_ref():
    return TokenChainReference()


@pytest.fixture(scope="module")
def make_responsivity(token_ref):
    """Build ResponsivityLM instances over the token chain's model and embeddings."""

    def make(
        *,
        importance_temp=1.0,
        n_samples=N_SAMPLES,
        expected_is_correction="unbiased",
    ):
        # The constructor is lazy (no HF load); pre-setting the internals makes
        # _ensure_loaded a no-op, same pattern as the MC-backend mocks.
        lm = ResponsivityLM(
            "markov_token_chain",
            n_samples=n_samples,
            importance_temp=importance_temp,
            device="cpu",
            seed=0,
            add_bos=False,
            expected_is_correction=expected_is_correction,
        )
        lm._model = token_ref.model
        lm._tokenizer = token_ref.tokenizer
        lm._token_embeds = torch.from_numpy(EMBEDS)
        lm._wpe0 = torch.zeros(1, EMBEDS.shape[1], dtype=torch.float64)
        return lm

    return make


def _cosine_distance_matrix() -> np.ndarray:
    e = EMBEDS / np.linalg.norm(EMBEDS, axis=1, keepdims=True)
    return 1.0 - e @ e.T


def _exact_limits(token_ref, stimulus, temp=1.0):
    """(observed, expected) per word position; expected uses the mixed
    p×q limit that the paper's importance correction converges to."""
    d_mat = _cosine_distance_matrix()
    states = token_ref._prefix_states(stimulus)
    # Observed token = each word's first token, via the production aligner.
    first_tok = [toks[0] for toks in token_ref.word_token_ids(stimulus)]
    n = len(first_tok)
    observed = np.full(n, np.nan)
    expected = np.full(n, np.nan)
    for t, state in enumerate(states):
        if state < 0:
            continue
        p = token_ref.transition[int(state)]
        q = temperature(p[None], temp)[0] if temp != 1.0 else p
        observed[t] = float(q @ d_mat[first_tok[t]])
        expected[t] = float(p @ d_mat @ q)
    return observed, expected


class TestPlainSampling:
    @pytest.fixture(scope="class")
    @classmethod
    def outputs(cls, token_ref, make_responsivity):
        stim = token_ref.make_stimulus(WORDS, "resp_probe")
        return stim, make_responsivity()(stim)

    def test_observed_distance_matches_exact_limit(self, token_ref, outputs):
        stim, outs = outputs
        vals = ResponsivityDistance("observed")(stim, outs)
        exact, _ = _exact_limits(token_ref, stim)
        assert np.isnan(vals[0]) and np.isnan(exact[0])
        np.testing.assert_allclose(vals[1:], exact[1:], atol=ATOL)

    def test_expected_distance_matches_exact_limit(self, token_ref, outputs):
        stim, outs = outputs
        vals = ResponsivityDistance("expected")(stim, outs)
        _, exact = _exact_limits(token_ref, stim)
        np.testing.assert_allclose(vals[1:], exact[1:], atol=ATOL)


class TestImportanceSampling:
    TEMP = 4.0
    # Importance weights at TEMP=4 inflate the observed-path variance, so
    # these tests draw twice N_SAMPLES.
    N_IS = 8000

    @pytest.fixture(scope="class")
    @classmethod
    def outputs(cls, token_ref, make_responsivity):
        stim = token_ref.make_stimulus(WORDS_IS, "resp_is_probe")
        lm = make_responsivity(importance_temp=cls.TEMP, n_samples=cls.N_IS)
        return stim, lm(stim)

    def test_observed_correction_is_unbiased(self, token_ref, outputs):
        # Full importance weight on the single draw → same p-limit as temp=1.
        stim, outs = outputs
        vals = ResponsivityDistance("observed")(stim, outs)
        exact, _ = _exact_limits(token_ref, stim, temp=1.0)
        np.testing.assert_allclose(vals[1:], exact[1:], atol=ATOL)

    def test_expected_correction_recovers_pure_limit(self, token_ref, outputs):
        # Reweighting both draws by w = p/q makes the expected correction
        # proposal-invariant: from a tempered (temp=4) proposal it recovers the
        # same pure limit Σ p(v) p(v') d(v, v') = p·D·p as temp=1. The mixed
        # limit p·D·q would mean the second draw was left unweighted.
        stim, outs = outputs
        vals = ResponsivityDistance("expected")(stim, outs)
        _, pure = _exact_limits(token_ref, stim, temp=1.0)
        _, mixed = _exact_limits(token_ref, stim, temp=self.TEMP)
        np.testing.assert_allclose(vals[1:], pure[1:], atol=ATOL)
        # Pure and mixed must be distinguishable at this tolerance, or matching
        # pure would not prove the second draw was reweighted.
        assert np.nanmax(np.abs(pure - mixed)) > 3 * ATOL

    @pytest.fixture(scope="class")
    @classmethod
    def reference_outputs(cls, token_ref, make_responsivity):
        stim = token_ref.make_stimulus(WORDS_IS, "resp_is_ref_probe")
        with pytest.warns(UserWarning, match="biased toward the mixed limit"):
            lm = make_responsivity(
                importance_temp=cls.TEMP,
                n_samples=cls.N_IS,
                expected_is_correction="reference",
            )
        return stim, lm(stim)

    def test_reference_estimator_matches_authors_mixed_limit(
        self, token_ref, reference_outputs
    ):
        # 'reference' reproduces the released estimator of Giulianelli et al. (2024a):
        # weighting only one draw makes the expected correction converge to the
        # mixed limit p·D·q (biased vs the paper's estimand E_{a,a'~p}[d]).
        stim, outs = reference_outputs
        vals = ResponsivityDistance("expected")(stim, outs)
        _, mixed = _exact_limits(token_ref, stim, temp=self.TEMP)
        _, pure = _exact_limits(token_ref, stim, temp=1.0)
        np.testing.assert_allclose(vals[1:], mixed[1:], atol=ATOL)
        assert np.nanmax(np.abs(mixed - pure)) > 3 * ATOL

    def test_reference_mode_warns_of_bias(self, make_responsivity):
        # Selecting the biased reference estimator with importance sampling on
        # must warn, so the deviation from the paper's estimand is never silent.
        with pytest.warns(UserWarning, match="biased toward the mixed limit"):
            make_responsivity(
                importance_temp=self.TEMP,
                expected_is_correction="reference",
            )

    def test_reference_mode_silent_without_importance_sampling(self, make_responsivity):
        # At temp=1 the two estimators coincide, so 'reference' is harmless and
        # must not warn.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            make_responsivity(expected_is_correction="reference")
