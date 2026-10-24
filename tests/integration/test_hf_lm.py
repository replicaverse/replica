"""Integration tests for HuggingFaceLM with real GPT-2.

Verifies surprisal extraction and full evaluate() pipeline using a real
HuggingFace model on hand-crafted and dataset stimuli.

Run with: pytest tests/integration/test_hf_lm.py -v --data-root datasets
"""

import os

import numpy as np
import pytest

from replica import (
    OLS,
    AnalysisConfig,
    EvaluationSpec,
    ItemSplitter,
    ModelFeature,
    evaluate,
)
from replica.analysis import AnalysisResult
from replica.dataset import PROVO
from replica.feature import Surprisal, UnitLength
from tests._stimuli import word_stimulus

pytestmark = [pytest.mark.needs_model, pytest.mark.needs_data]


@pytest.fixture(scope="module")
def hf_lm(lm_cache):
    """Shared GPT-2 instance (session-scoped ``lm_cache``)."""
    return lm_cache("gpt2", device="cpu")


# Run GPT-2 inference once at module scope; every downstream assertion shares
# the same outputs. Without this, each test re-runs the forward pass.
@pytest.fixture(scope="module")
def single_sentence_outputs(hf_lm):
    stim = word_stimulus("The cat sat on the mat.", "test_1")
    return stim, hf_lm(stim), hf_lm.vocab


class TestHuggingFaceLMSurprisal:
    """GPT-2 surprisal values on a hand-crafted stimulus.

    All assertions share one forward pass via the ``single_sentence_outputs``
    fixture; each test checks a different invariant on the same outputs.
    """

    def test_output_count(self, single_sentence_outputs):
        """One output per word."""
        stim, outputs, _ = single_sentence_outputs
        assert len(outputs) == len(stim.layers["words"])

    def test_first_word_scored_from_bos(self, single_sentence_outputs):
        """With ``add_bos=True`` (default), the first word is scored from the
        BOS position, so its log_p_observed is finite and negative."""
        _, outputs, _ = single_sentence_outputs
        lp = outputs[0].values["log_p_observed"]
        assert np.isfinite(lp)
        assert lp < 0

    def test_remaining_words_finite_negative(self, single_sentence_outputs):
        """Non-first words have finite negative log_p_observed."""
        _, outputs, _ = single_sentence_outputs
        for t in range(1, len(outputs)):
            lp = outputs[t].values["log_p_observed"]
            assert np.isfinite(lp), f"Word {t}: log_p_observed is not finite"
            assert lp < 0, f"Word {t}: log_p_observed={lp} is not negative"

    def test_next_log_probs_shape(self, single_sentence_outputs):
        """next_log_probs arrays have correct vocab size."""
        _, outputs, vocab = single_sentence_outputs
        vocab_size = len(vocab)
        for t in range(1, len(outputs)):
            nlp = outputs[t].values.get("next_log_probs")
            assert nlp is not None, f"Word {t}: next_log_probs is missing"
            assert len(nlp) == vocab_size, (
                f"Word {t}: next_log_probs length {len(nlp)} != vocab size {vocab_size}"
            )

    def test_two_sentence_stimulus(self, hf_lm):
        """Multi-sentence input still yields one finite output per word."""
        stim = word_stimulus(
            "The dog ran quickly. It was a beautiful day.",
            stimulus_id="test_2sent",
        )
        outputs = hf_lm(stim)
        assert len(outputs) == len(stim.layers["words"])
        # add_bos=True (default): first word scored from BOS → finite.
        for t in range(len(outputs)):
            assert np.isfinite(outputs[t].values["log_p_observed"])


class TestHuggingFaceLMEvaluate:
    """Test GPT-2 through the full evaluate() pipeline."""

    def test_evaluate_pipeline(self, hf_lm, data_root):
        """End-to-end: GPT-2 + OLS on PROVO produces finite delta_ll."""
        path = os.path.join(data_root, "eye-tracking", "original", "provo")
        if not os.path.isdir(path):
            pytest.skip(f"PROVO data not found at {path}")

        ds = PROVO()
        ds.load(path)

        spec = EvaluationSpec(
            predictors=[ModelFeature(hf_lm, Surprisal())],
            controls=[UnitLength()],
            targets=[ds.feature(ds.default_measure)],
        )
        cfg = AnalysisConfig(
            design=None,
            transform_dv=np.log2,
            model=OLS(),
            cv=ItemSplitter(5),
            n_folds=5,
        )
        results = evaluate(ds, spec, cfg)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, AnalysisResult)
            assert np.isfinite(r.delta_ll)
            assert r.n_observations > 0
