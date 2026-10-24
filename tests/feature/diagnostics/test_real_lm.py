"""Real-LM diagnostics on gpt2: backend plumbing + MC bias with known truth.

Token-level quantities are exactly computable from one forward pass over a
real model's logits; ``diagnostics.references.causal_lm`` recomputes them with raw
``transformers`` calls. These tests check that

- ``HuggingFaceLM``'s surprisal / token-entropy / token-Rényi pipeline
  (tokenization, BOS, subword aggregation, base conversion) reproduces the
  referee values on real GPT-2, and
- the production ``ContinuationEntropyLM`` at ``max_tokens=1``, whose
  target is the exact token entropy, is unbiased on a real model.
"""

import numpy as np
import pytest

from replica.feature import (
    ContinuationEntropy,
    Surprisal,
    TokenEntropy,
    TokenRenyiEntropy,
)
from replica.feature.diagnostics import make_stimulus
from replica.feature.diagnostics.references.causal_lm import exact_token_quantities

pytestmark = [pytest.mark.needs_model, pytest.mark.slow]

WORDS = "the cat sat on the windowsill because it was unquestionably tired".split()


@pytest.fixture(scope="module")
def stim():
    return make_stimulus(WORDS, stimulus_id="real_lm_probe")


@pytest.fixture(scope="module")
def exact():
    # The referee gets its own stimulus: it must never share the mutable
    # Stimulus (token layers are added by every backend) with the systems
    # under test.
    return exact_token_quantities(
        "gpt2", make_stimulus(WORDS, "real_lm_referee"), renyi_alphas=(0.5,)
    )


class TestBackendPlumbing:
    """HuggingFaceLM vs the independent raw-transformers recompute.

    float32 backend vs float64 referee → atol absorbs accumulation noise.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def outputs(cls, lm_cache, stim):
        lm = lm_cache(
            "gpt2",
            add_bos=True,
            token_level_entropy=True,
            # tuple: lm_cache hashes kwargs; HuggingFaceLM only iterates it
            token_renyi_alphas=(0.5,),
        )
        return lm(stim)

    def test_surprisal_matches_referee(self, stim, outputs, exact):
        vals = Surprisal()(stim, outputs)
        np.testing.assert_allclose(vals, exact["surprisal"], rtol=1e-4, atol=5e-3)

    def test_token_entropy_matches_referee(self, stim, outputs, exact):
        vals = TokenEntropy()(stim, outputs)
        np.testing.assert_allclose(vals, exact["token_entropy"], rtol=1e-4, atol=5e-3)

    def test_token_renyi_matches_referee(self, stim, outputs, exact):
        vals = TokenRenyiEntropy(alpha=0.5)(stim, outputs)
        np.testing.assert_allclose(
            vals, exact["token_renyi_entropy_alpha0.5"], rtol=1e-4, atol=5e-3
        )


def _leading_space_stimulus(words: list[str], stimulus_id: str):
    """Stimulus whose text starts with a space, so GPT-2 tokenizes every
    word with a Ġ token, the same token sequence LogitLensLM builds
    internally (``bos + " " + " ".join(words)``). Required for the referee
    to see identical contexts."""
    from replica.core import AnnotationLayer, Span, Stimulus

    text = " " + " ".join(words)
    spans, off = [], 1
    for w in words:
        spans.append(Span(off, off + len(w)))
        off += len(w) + 1
    stim = Stimulus(text=text, stimulus_id=stimulus_id)
    stim.add_layer(AnnotationLayer(name="words", spans=spans, values=list(words)))
    return stim


class TestLogitLens:
    """LogitLensLM's final layer is, by construction, the model's own
    output distribution, so its per-word surprisal must equal the
    independent referee's. Intermediate layers must differ (otherwise the
    lens is silently projecting the wrong states)."""

    @pytest.fixture(scope="class")
    @classmethod
    def lens_outputs(cls, stim):
        from replica.feature import LogitLensLM

        lm = LogitLensLM(
            "gpt2", layers=[6, -1], device="cpu", assert_final_matches=True
        )
        return lm(stim)

    @pytest.fixture(scope="class")
    @classmethod
    def exact_ls(cls, stim):
        return exact_token_quantities(
            "gpt2",
            _leading_space_stimulus(WORDS, "real_lm_probe_ls"),
            renyi_alphas=(),
        )

    def test_final_layer_matches_referee_surprisal(self, stim, lens_outputs, exact_ls):
        from replica.feature import LayerSurprisal

        vals = LayerSurprisal(12)(stim, lens_outputs)
        np.testing.assert_allclose(vals, exact_ls["surprisal"], rtol=1e-4, atol=5e-3)

    def test_mirrored_log_p_observed_feeds_surprisal_extractor(
        self, stim, lens_outputs, exact_ls
    ):
        vals = Surprisal()(stim, lens_outputs)
        np.testing.assert_allclose(vals, exact_ls["surprisal"], rtol=1e-4, atol=5e-3)

    def test_intermediate_layer_differs_from_output(self, stim, lens_outputs):
        from replica.feature import LayerSurprisal

        mid = LayerSurprisal(6)(stim, lens_outputs)
        final = LayerSurprisal(12)(stim, lens_outputs)
        assert np.nanmax(np.abs(mid - final)) > 0.5


class TestMCOnRealModel:
    def test_continuation_L1_is_unbiased_for_token_entropy(self, stim, exact):
        from replica.feature import ContinuationEntropyLM

        lm = ContinuationEntropyLM(
            "gpt2", n_samples=1024, max_tokens=1, seed=0, add_bos=True
        )
        est = ContinuationEntropy()(stim, lm(stim))
        truth = exact["token_entropy"]
        mask = np.isfinite(est) & np.isfinite(truth)
        assert mask.sum() >= len(WORDS) - 1

        diff = est[mask] - truth[mask]
        # MC se at N=1024 is ≈ sd(-log p)/32.
        assert np.abs(diff).mean() < 0.3, (est[mask], truth[mask])
        assert np.abs(diff).max() < 1.0
        r = np.corrcoef(est[mask], truth[mask])[0, 1]
        assert r > 0.95


class TestContinuationPool:
    """continuation_log2p_pool + bootstrap subsampling reproduces the
    unbiased-Shannon / biased-Renyi story on real gpt2."""

    @pytest.fixture(scope="class")
    @classmethod
    def pool(cls):
        from replica.feature.diagnostics.references.causal_lm import (
            continuation_log2p_pool,
        )

        return continuation_log2p_pool(
            "gpt2",
            "the cat sat on the".split(),
            n_samples=1500,
            max_tokens=2,
            seed=0,
            chunk_size=256,
        )

    def test_pool_shape_and_finite(self, pool):
        assert pool.shape == (1500,)
        assert np.isfinite(pool).all()

    def test_shannon_subsample_is_unbiased(self, pool):
        from replica.feature.diagnostics import diagnose_estimator

        theta = float(-pool.mean())

        def est(n, seed):
            sub = np.random.default_rng(seed).choice(pool, n, replace=True)
            return float(-sub.mean()), float(sub.std(ddof=1) / np.sqrt(n))

        rep = diagnose_estimator(est, theta, n_grid=(16, 64, 256, 1024), n_reps=400)
        assert rep.classification == "no_bias_detected", rep.to_text()

    def test_renyi_subsample_is_biased_but_consistent(self, pool):
        from replica.feature.diagnostics import diagnose_estimator

        a = 0.5
        theta = float((1 / (1 - a)) * np.log2(np.mean(2.0 ** ((a - 1) * pool))))

        def est(n, seed):
            sub = np.random.default_rng(seed).choice(pool, n, replace=True)
            return float((1 / (1 - a)) * np.log2(np.mean(2.0 ** ((a - 1) * sub))))

        rep = diagnose_estimator(est, theta, n_grid=(8, 32, 128, 512), n_reps=800)
        assert rep.classification == "decreasing_bias", rep.to_text()
