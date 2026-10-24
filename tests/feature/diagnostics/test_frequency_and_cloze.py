"""Frequency predictors, ClozeLM, and the sampled-unigram MC estimator.

With a FrequencySource fed the chain's stationary distribution, the
unigram_surprisal spec (-log2 π) is the exact target for the whole
frequency stack: ``UnigramSurprisal`` exactly; ``LogFrequency`` affinely
with slope -log10 2 (it returns the raw log10 probability); and
``UnigramSurprisalLM`` via its nats ``log_p_observed``. ``ClozeLM`` fed
the true chain probabilities fingerprints as contextual ``surprisal``;
the Hopton et al. (2026b) ``mc_estimate`` on chain samples as ``unigram_surprisal``,
ranked strictly above contextual surprisal (context-blindness).
"""

import numpy as np
import pytest

from replica.core import AnnotationLayer
from replica.feature import (
    LogFrequency,
    PrecomputedFeature,
    Surprisal,
    UnigramSurprisal,
    UnigramSurprisalLM,
    UserTable,
)
from replica.feature.diagnostics import MarkovReferenceLM, fingerprint
from replica.feature.diagnostics.references._closed_forms import stationary_distribution
from replica.feature.surprisal import ClozeLM


@pytest.fixture(scope="module")
def ref():
    return MarkovReferenceLM()


@pytest.fixture(scope="module")
def stationary_table(ref):
    """UserTable mapping each vocab word to its stationary probability."""
    pi = stationary_distribution(ref.transition)
    return UserTable({w: float(pi[i]) for i, w in enumerate(ref.vocab)})


class TestFrequencyStack:
    def test_unigram_surprisal_matches_spec_exactly(self, ref, stationary_table):
        report = fingerprint(UnigramSurprisal(stationary_table), ref)
        assert report.best.name == "unigram_surprisal", report.to_text()
        assert report.best.tier == "exact", report.to_text()

    def test_log_frequency_is_affine_log10_probability(self, ref, stationary_table):
        report = fingerprint(LogFrequency(stationary_table), ref)
        assert report.best.name == "unigram_surprisal", report.to_text()
        assert report.best.tier == "affine", report.to_text()
        assert np.isclose(report.best.slope, -np.log10(2.0), rtol=1e-9)
        assert "log10 probability" in report.verdict

    def test_unigram_lm_round_trips_through_surprisal_extractor(
        self, ref, stationary_table
    ):
        lm = UnigramSurprisalLM(stationary_table)
        stim = ref.probe_stimulus()
        vals = Surprisal()(stim, lm(stim))
        report = fingerprint(vals, ref, stimuli=[stim])
        assert report.best.name == "unigram_surprisal", report.to_text()
        assert report.best.tier == "exact", report.to_text()

    def test_unigram_lm_bits_key_via_precomputed_feature(self, ref, stationary_table):
        lm = UnigramSurprisalLM(stationary_table)
        stim = ref.probe_stimulus()
        vals = PrecomputedFeature("unigram_bits", "unigram_surprisal")(stim, lm(stim))
        report = fingerprint(vals, ref, stimuli=[stim])
        assert report.best.name == "unigram_surprisal", report.to_text()
        assert report.best.tier == "exact", report.to_text()


class TestClozeLM:
    def test_true_cloze_probabilities_fingerprint_as_surprisal(self, ref):
        stim = ref.probe_stimulus()
        dists, observed = ref.predictive_distributions(stim)
        probs = dists[np.arange(len(observed)), observed]
        words = stim.layers["words"]
        stim.add_layer(
            AnnotationLayer(
                name="human_cloze", spans=list(words.spans), values=list(probs)
            )
        )
        vals = Surprisal()(stim, ClozeLM()(stim))
        assert np.isnan(vals[0])  # ClozeLM's no-left-context convention
        report = fingerprint(vals, ref, stimuli=[stim])
        assert report.best.name == "surprisal", report.to_text()
        assert report.best.tier == "exact", report.to_text()


class TestSampledUnigramMC:
    def test_mc_estimator_targets_stationary_unigram(self, ref):
        from replica.feature.frequency.sampled import mc_estimate

        # Short walks attenuate the fitted slope below the approx band
        # (walk-start contamination + the estimator's +n denominator);
        # length-300 walks keep the tier assertion seed-robust.
        samples = [s.text for s in ref.sample_stimuli(300, 300, seed=11)]
        p_hat = {w: mc_estimate(w, samples) for w in ref.vocab}
        assert all(p > 0 for p in p_hat.values())

        probe = ref.probe_stimulus()
        vals = np.array([-np.log2(p_hat[w]) for w in probe.unit_strings()])
        report = fingerprint(vals, ref, stimuli=[probe])
        assert report.best.name == "unigram_surprisal", report.to_text()
        # The estimator's denominator (n + Σ(len+1)) is a constant factor
        # across word types → a small uniform intercept in bits; with MC
        # noise the match lands in affine or approx, never exact.
        assert report.best.tier in ("affine", "approx"), report.to_text()
        # Context-blindness: must beat contextual surprisal decisively.
        by_name = {m.name: m for m in report.matches}
        assert report.best.r2 > by_name["surprisal"].r2 + 0.1, report.to_text()
