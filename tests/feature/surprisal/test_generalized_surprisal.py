"""Tests for the generalized surprisal / responsivity measures from
Giulianelli et al. (2024a).

Covers:
  * ``ExpectedProbability``, formula correctness against hand-computed
    reference values for known distributions.
  * ``ResponsivityLM`` + ``ResponsivityDistance``, sampling determinism
    under fixed seed, NaN at position 0, finite/non-negative distances
    elsewhere, and that observed-vs-expected distances are not always
    equal (would indicate the two paths share state).
"""

import numpy as np
import pytest

from replica.feature import (
    ExpectedProbability,
    FeatureModelOutput,
    ResponsivityDistance,
)
from tests._stimuli import word_stimulus


def _make_stimulus(words=("a", "b", "c")):
    return word_stimulus(" ".join(words), "s1")


def _outs_with_dists(dists):
    """Build a list of FeatureModelOutput from a list of probability vectors
    (None means no next_log_probs at that position, e.g. position 0 without
    BOS)."""
    out = []
    for d in dists:
        if d is None:
            out.append(FeatureModelOutput(values={}))
        else:
            d_arr = np.asarray(d, dtype=float)
            out.append(
                FeatureModelOutput(
                    values={"next_log_probs": np.log(d_arr)},
                )
            )
    return out


# ---------------------------------------------------------------------------
# ExpectedProbability, formula tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "probs, expected, tol",
    [
        # Σ p² for known distributions; third case is a sanity-check
        # that near-one-hot returns ~1.0 (looser tol because p² of 1-ε
        # picks up O(ε) error).
        ([0.5, 0.5], 0.5, 1e-10),
        ([0.25, 0.25, 0.25, 0.25], 0.25, 1e-10),
        ([1 - 1e-9] + [1e-10] * 10, 1.0, 1e-7),
        ([0.9, 0.1], 0.9**2 + 0.1**2, 1e-10),
    ],
)
def test_expected_probability_formula(probs, expected, tol):
    """Σ p(v)² for hand-computed cases, applied at position 1 (position 0 is
    NaN by convention)."""
    stim = _make_stimulus(("x", "y"))
    outs = _outs_with_dists([None, probs])
    vals = ExpectedProbability()(stim, outs)
    assert np.isnan(vals[0])
    assert abs(vals[1] - expected) < tol


def test_expected_probability_position_zero_nan_when_no_dist():
    """Position 0 must be NaN when next_log_probs is absent (no BOS)."""
    stim = _make_stimulus(("x", "y"))
    outs = _outs_with_dists([None, [0.5, 0.5]])
    vals = ExpectedProbability()(stim, outs)
    assert np.isnan(vals[0])


def test_expected_probability_all_positions_have_dist():
    """When every position has next_log_probs (e.g. with BOS), no NaN."""
    stim = _make_stimulus(("x", "y", "z"))
    outs = _outs_with_dists([[0.5, 0.5], [0.25, 0.25, 0.5], [0.9, 0.1]])
    vals = ExpectedProbability()(stim, outs)
    assert np.all(np.isfinite(vals))


def test_expected_probability_collision_vs_entropy_distinction():
    """A regression guard for the formula: a uniform-over-2 has Σp²=0.5
    while a uniform-over-100 has Σp²=0.01, the same direction as
    H₂=-log(Σp²), distinguishing this measure from Shannon entropy
    (which they share the *direction* of but not the magnitude)."""
    stim = _make_stimulus(("a", "b"))
    flat_2 = [0.5, 0.5]
    flat_100 = [0.01] * 100
    outs = _outs_with_dists([flat_2, flat_100])
    vals = ExpectedProbability()(stim, outs)
    # Position 0 has next_log_probs in this construction, so both are finite.
    assert abs(vals[0] - 0.5) < 1e-10
    assert abs(vals[1] - 0.01) < 1e-10
    assert vals[0] > vals[1]  # peakier dist → higher expected prob


def test_expected_probability_requires_outputs():
    stim = _make_stimulus()
    with pytest.raises(ValueError, match="ExpectedProbability"):
        ExpectedProbability()(stim, None)


def test_expected_probability_name():
    assert ExpectedProbability().name == "expected_probability"


# ---------------------------------------------------------------------------
# ResponsivityDistance extractor, pulls keys from FeatureModelOutput
# ---------------------------------------------------------------------------


def test_responsivity_distance_observed_kind():
    stim = _make_stimulus(("a", "b", "c"))
    outs = [
        FeatureModelOutput(
            values={
                "decontextualised_distance": 0.1,
                "expected_decontextualised_distance": 0.2,
            }
        ),
        FeatureModelOutput(
            values={
                "decontextualised_distance": 0.3,
                "expected_decontextualised_distance": 0.4,
            }
        ),
        FeatureModelOutput(values={}),  # missing → NaN
    ]
    obs = ResponsivityDistance("observed")(stim, outs)
    np.testing.assert_array_almost_equal(obs[:2], [0.1, 0.3])
    assert np.isnan(obs[2])


def test_responsivity_distance_expected_kind():
    stim = _make_stimulus(("a", "b"))
    outs = [
        FeatureModelOutput(
            values={
                "decontextualised_distance": 0.1,
                "expected_decontextualised_distance": 0.2,
            }
        ),
        FeatureModelOutput(
            values={
                "decontextualised_distance": 0.3,
                "expected_decontextualised_distance": 0.4,
            }
        ),
    ]
    exp = ResponsivityDistance("expected")(stim, outs)
    np.testing.assert_array_almost_equal(exp, [0.2, 0.4])


def test_responsivity_distance_invalid_kind():
    with pytest.raises(ValueError, match="kind must be"):
        ResponsivityDistance("nonsense")


# ---------------------------------------------------------------------------
# ResponsivityLM end-to-end (slow, uses GPT-2)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.needs_model
def test_responsivity_lm_basic_invariants():
    """Slow GPT-2 smoke test pinning the basic shape of ResponsivityLM
    outputs: one output per word, position 0 NaN (no add_bos), positions
    ≥1 finite + non-negative (cosine distance ∈ [0, 2])."""
    from replica.feature import ResponsivityLM

    stim = _make_stimulus(("The", "cat", "sat", "on", "the", "mat"))
    lm = ResponsivityLM(model_name="gpt2", n_samples=10, seed=42, device="cpu")
    outs = lm(stim)
    assert len(outs) == len(stim.layers["words"])

    obs = ResponsivityDistance("observed")(stim, outs)
    exp = ResponsivityDistance("expected")(stim, outs)

    # First word: no left context, no distribution to sample from.
    assert np.isnan(obs[0])
    assert np.isnan(exp[0])

    # Rest: cosine distance is in [0, 2]; in practice in [0, 1] for
    # raw embeddings.
    assert np.all(np.isfinite(obs[1:]))
    assert np.all(np.isfinite(exp[1:]))
    assert np.all(obs[1:] >= 0)
    assert np.all(exp[1:] >= 0)


@pytest.mark.slow
@pytest.mark.needs_model
def test_responsivity_lm_seed_reproducibility():
    """Same seed → identical numbers. Catches accidental global-RNG use."""
    from replica.feature import ResponsivityLM

    stim = _make_stimulus(("The", "cat", "sat", "on", "the"))
    lm1 = ResponsivityLM(model_name="gpt2", n_samples=8, seed=7, device="cpu")
    lm2 = ResponsivityLM(model_name="gpt2", n_samples=8, seed=7, device="cpu")
    o1 = ResponsivityDistance("observed")(stim, lm1(stim))
    o2 = ResponsivityDistance("observed")(stim, lm2(stim))
    np.testing.assert_array_almost_equal(o1, o2)


@pytest.mark.slow
@pytest.mark.needs_model
def test_responsivity_lm_observed_vs_expected_differ():
    """The two distance keys must come from different computations, a
    regression that aliased one to the other would silently zero out the
    paper's anticipation/responsivity distinction."""
    from replica.feature import ResponsivityLM

    stim = _make_stimulus(("The", "quick", "brown", "fox"))
    lm = ResponsivityLM(model_name="gpt2", n_samples=20, seed=13, device="cpu")
    outs = lm(stim)
    obs = ResponsivityDistance("observed")(stim, outs)
    exp = ResponsivityDistance("expected")(stim, outs)
    # Skip position 0 (NaN); the rest must not all be equal.
    assert not np.allclose(obs[1:], exp[1:]), (
        "observed and expected responsivity distances are pointwise equal, "
        "they should come from different computations"
    )


@pytest.mark.slow
@pytest.mark.needs_model
def test_responsivity_lm_with_bos_first_word_finite():
    """A start token gives the first word a finite sampling distance."""
    from replica.feature import ResponsivityLM

    stim = _make_stimulus(("The", "cat"))
    lm = ResponsivityLM(
        model_name="gpt2", n_samples=8, seed=0, device="cpu", add_bos=True
    )
    outs = lm(stim)
    obs = ResponsivityDistance("observed")(stim, outs)
    assert np.isfinite(obs[0])
    assert np.isfinite(obs[1])


@pytest.mark.slow
@pytest.mark.needs_model
def test_responsivity_lm_windows_overlength_stimulus():
    """Stimuli longer than the model's context window are scored with a
    50%-overlap sliding window: every word gets a finite distance, and words
    within the first window match a single-pass reference. Truncation would
    shift the alternatives' sampling context."""
    import numpy as np

    from replica.feature import ResponsivityDistance, ResponsivityLM

    long_words = tuple([f"w{i}" for i in range(1400)])  # > gpt2's 1024 window
    stim = _make_stimulus(long_words)
    lm = ResponsivityLM(model_name="gpt2", n_samples=8, seed=0, device="cpu")
    outs = lm(stim)
    vals = ResponsivityDistance(kind="observed")(stim, outs)
    # No silent truncation: every non-initial word scored (position 0 is NaN).
    assert np.all(np.isfinite(vals[1:])), "windowed pass left words unscored"
    # First-window words match a single forward pass over the truncated prefix.
    ref_stim = _make_stimulus(long_words[:400])
    ref = ResponsivityDistance(kind="observed")(
        ref_stim,
        ResponsivityLM(model_name="gpt2", n_samples=8, seed=0, device="cpu")(ref_stim),
    )
    np.testing.assert_allclose(vals[1:300], ref[1:300], rtol=1e-4, atol=1e-4)
