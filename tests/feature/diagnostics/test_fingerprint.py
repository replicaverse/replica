"""Acceptance tests for distribution fingerprinting.

Three of these encode known real-world predictor bug classes:

- the constant-output degenerate case (the all-OOV KenLM unigram pickles);
- unit drift (a predictor emitting nats where bits are documented);
- the truncated-sampler MC entropy (a continuation-entropy sampler that
  inherits ``top_k=50``: samples from a truncated distribution, scores under
  the full one).
"""

import numpy as np
import pytest

from replica.feature import (
    Entropy,
    ExpectedProbability,
    Rank,
    RenyiEntropy,
    Surprisal,
)
from replica.feature.diagnostics import (
    MarkovReferenceLM,
    fingerprint,
    simulate_mc_entropy,
)


@pytest.fixture(scope="module")
def ref():
    return MarkovReferenceLM()


def _exact(ref, name):
    from replica.feature.diagnostics import default_specs

    spec = {s.name: s for s in default_specs()}[name]
    return spec.compute(ref, ref.probe_stimulus())


# ── REPLICA extractors fingerprint to their documented specs ─────────────────


class TestReplicaExtractors:
    @pytest.mark.parametrize(
        "feature,expected_spec",
        [
            (Surprisal(), "surprisal"),
            (Entropy(), "entropy"),
            (RenyiEntropy(alpha=0.5), "renyi_entropy_alpha0.5"),
            (RenyiEntropy(alpha=2.0), "renyi_entropy_alpha2.0"),
            (RenyiEntropy(alpha=1.0), "entropy"),  # Shannon limit
            (ExpectedProbability(), "expected_probability"),
            (Rank(), "rank"),
        ],
    )
    def test_extractor_matches_documented_spec(self, ref, feature, expected_spec):
        report = fingerprint(feature, ref)
        assert report.best.name == expected_spec
        assert report.best.tier == "exact"


# ── Bug-class detection ──────────────────────────────────────────────────


class TestUnitAndSignDrift:
    def test_nats_detected_as_affine_ln2(self, ref):
        nats = _exact(ref, "entropy") * np.log(2.0)
        report = fingerprint(nats, ref)
        assert report.best.name == "entropy"
        assert report.best.tier == "affine"
        assert np.isclose(report.best.slope, np.log(2.0), rtol=1e-6)
        assert "nats" in report.verdict

    def test_sign_flip_detected(self, ref):
        log_p = -_exact(ref, "surprisal")  # Inject a sign error.
        report = fingerprint(log_p, ref)
        assert report.best.name == "surprisal"
        assert report.best.tier == "affine"
        assert np.isclose(report.best.slope, -1.0, rtol=1e-6)
        assert "sign flip" in report.verdict

    def test_noisy_nats_estimator_gets_unit_caution(self, ref):
        # The most likely real-world MC case: a correct estimator that is
        # both noisy and in nats. The ln2 rescale happens to land on
        # 'entropy_top4''s range, which can take a tier, so the verdict
        # must surface that the true spec fits far better with a nats
        # slope.
        rng = np.random.default_rng(3)
        ent = _exact(ref, "entropy")
        nats = ent * np.log(2.0) + rng.normal(0, 0.02 * np.std(ent), len(ent))
        report = fingerprint(nats, ref)
        by_name = {m.name: m for m in report.matches}
        assert by_name["entropy"].r2 > report.best.r2 or report.best.name == "entropy"
        assert "nats" in report.verdict, report.verdict

    def test_no_caution_when_best_spec_is_constant_on_probe(self):
        # A spec constant on the probe gives the exact-tier best a NaN R²;
        # the scale caution must treat NaN as "suppress", not "fire".
        from replica.feature.diagnostics.references.markov import make_stimulus

        rng = np.random.default_rng(7)
        v = 8
        T = np.zeros((v, v))
        for i in range(v):  # observed word always P=0.25 -> constant surprisal
            T[i, (i + 1) % v] = 0.25
            cols = [c for c in range(v) if c != (i + 1) % v]
            T[i, cols] = 0.75 * rng.dirichlet(np.ones(v - 1) * (0.3 + i))
        init = np.zeros(v)
        init[0], init[1:] = 0.25, 0.75 * rng.dirichlet(np.ones(v - 1))
        ref = MarkovReferenceLM(transition=T, initial=init)
        words = [ref.vocab[0]] + [ref.vocab[(i + 1) % v] for i in range(20)]
        stim = make_stimulus(words)
        pred = 2.0 + rng.uniform(-1e-7, 1e-7, len(words))  # exact, jittered
        report = fingerprint(pred, ref, stimuli=[stim])
        assert report.best.name == "surprisal"
        assert report.best.tier == "exact"
        assert "CAUTION" not in report.verdict, report.verdict


class TestDegenerate:
    def test_constant_predictor_is_degenerate(self, ref):
        # The all-OOV KenLM bug shape: same value at every position.
        n = len(ref.probe_stimulus().layers["words"])
        report = fingerprint(np.full(n, -22.719), ref)
        assert report.degenerate
        assert "constant" in report.verdict
        assert report.matches == []

    def test_all_nan_predictor_is_degenerate(self, ref):
        n = len(ref.probe_stimulus().layers["words"])
        report = fingerprint(np.full(n, np.nan), ref)
        assert report.degenerate
        assert "no finite values" in report.verdict

    def test_tiny_magnitude_predictor_is_not_degenerate(self, ref):
        # Degeneracy is about relative spread; a rescaled-but-informative
        # predictor must classify as affine, not constant.
        report = fingerprint(_exact(ref, "surprisal") * 1e-13, ref)
        assert not report.degenerate
        assert report.best.name == "surprisal"
        assert report.best.tier == "affine"


class TestCoverage:
    """A predictor that is NaN at most positions must not earn an
    unqualified verdict from the few positions that survive masking."""

    def test_low_coverage_is_flagged_in_verdict(self, ref):
        vals = _exact(ref, "surprisal").copy()
        vals[20:] = np.nan  # 20/65 finite, still enough for a tier
        report = fingerprint(vals, ref)
        assert report.best.tier == "exact"
        assert report.verdict.startswith("LOW COVERAGE")
        assert "45 of 65" in report.verdict

    def test_full_coverage_is_not_flagged(self, ref):
        report = fingerprint(_exact(ref, "surprisal"), ref)
        assert not report.verdict.startswith("LOW COVERAGE")

    def test_too_few_finite_positions_gets_no_tier(self, ref):
        vals = _exact(ref, "surprisal").copy()
        vals[5:] = np.nan  # below the 8-position tier floor
        report = fingerprint(vals, ref)
        assert report.best.tier is None
        assert "insufficient finite positions" in report.verdict


class TestTruncatedSamplerBug:
    """A correct MC continuation-entropy estimator must fingerprint as the
    full-distribution continuation entropy; the buggy variant (top-k
    sampling, full-distribution scoring) must fingerprint as the
    truncated-sampler cross-entropy instead."""

    N = 20_000

    def test_correct_sampler_targets_continuation_entropy(self, ref):
        est = simulate_mc_entropy(
            ref,
            ref.probe_stimulus(),
            L=2,
            n_samples=self.N,
            rng=np.random.default_rng(0),
        )
        report = fingerprint(est, ref)
        assert report.best.name == "continuation_entropy_L2"
        assert report.best.tier == "approx"

    def test_topk_sampler_targets_truncated_xent(self, ref):
        est = simulate_mc_entropy(
            ref,
            ref.probe_stimulus(),
            L=2,
            n_samples=self.N,
            rng=np.random.default_rng(0),
            sample_top_k=4,
        )
        report = fingerprint(est, ref)
        assert report.best.name == "truncated_sampler_xent_top4_L2"
        assert report.best.tier == "approx"
        # The buggy estimator must not be attributed to the true entropy.
        by_name = {m.name: m for m in report.matches}
        assert by_name["continuation_entropy_L2"].tier != "approx"


# ── Input validation ─────────────────────────────────────────────────────


class TestInputs:
    def test_length_mismatch_raises(self, ref):
        with pytest.raises(ValueError, match="units"):
            fingerprint(np.zeros(7), ref)

    def test_unsupported_type_raises(self, ref):
        with pytest.raises(TypeError, match="predictor must be"):
            fingerprint("surprisal", ref)

    def test_2d_array_raises(self, ref):
        n = len(ref.probe_stimulus().layers["words"])
        with pytest.raises(ValueError, match="1-D"):
            fingerprint(np.zeros((n, 1)), ref)

    def test_per_stimulus_length_mismatch_raises(self, ref):
        stims = ref.sample_stimuli(2, 12, seed=0)
        with pytest.raises(ValueError, match="units"):
            fingerprint([np.zeros(11), np.zeros(13)], ref, stimuli=stims)

    def test_list_of_arrays_accepted(self, ref):
        stims = ref.sample_stimuli(2, 12, seed=0)
        from replica.feature.diagnostics import default_specs

        spec = {s.name: s for s in default_specs()}["entropy"]
        arrays = [spec.compute(ref, s) for s in stims]
        report = fingerprint(arrays, ref, stimuli=stims)
        assert report.best.name == "entropy"
        assert report.best.tier == "exact"
