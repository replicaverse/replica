"""Regression gate for the shared info-theory primitives.

``core/infotheory.py`` holds the one definition of the Shannon / Rényi /
collision formulas used by the LM backend and the Feature extractors. This test
pins those functions to the arithmetic used at each real call site. Bitwise
identity is required where a call site actually uses that dtype; mathematically
equivalent float32 paths are compared numerically because converting the max
shift to a Python float changes last-bit reduction rounding across NumPy builds.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from replica.feature.core.outputs.information import (
    collision_prob,
    renyi_bits,
    shannon_bits,
)

_LN2 = float(np.log(2.0))


# --- reference expressions, one per call site ------------------------------
# Shannon, ``-Σ p·ln p / ln2``, in two forms: ``nansum``, which holds for
# distributions with -inf entries (the extractor and token sites), and ``sum``
# for the finite, renormalised distribution of the restricted site. Both forms
# must equal shannon_bits on their valid inputs.


def _ref_shannon_nansum(v):
    p = np.exp(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(p * v) / _LN2


def _ref_shannon_sum(v):  # restricted path (finite v only)
    return -np.sum(np.exp(v) * v) / _LN2


def _ref_renyi_lm(v, a):  # entropy/extractors.py RenyiEntropy
    if a == 1.0:
        return _ref_shannon_nansum(v)
    scaled = a * v
    m = np.max(scaled)
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    return log_sum / ((1.0 - a) * _LN2)


def _ref_renyi_token(v, a):  # huggingface _token_entropy_arrays
    if a == 1.0:
        return _ref_shannon_nansum(v)
    scaled = a * v
    m = float(np.max(scaled))
    log_sum = m + np.log(np.sum(np.exp(scaled - m)))
    return log_sum / ((1.0 - a) * _LN2)


def _ref_renyi_restricted(v, a):  # huggingface _restricted_first_token_entropies
    if a == 1.0:
        return _ref_shannon_sum(v)
    scaled = a * v
    m2 = float(scaled.max())
    log_sum = m2 + float(np.log(np.sum(np.exp(scaled - m2))))
    return log_sum / ((1.0 - a) * _LN2)


def _ref_collision(v):  # entropy/extractors.py ExpectedProbability
    two = 2 * v
    m = np.max(two)
    return float(np.exp(m + np.log(np.sum(np.exp(two - m)))))


# --- test distributions ----------------------------------------------------


def _log_softmax(logits):
    m = np.max(logits)
    return (logits - m) - np.log(np.sum(np.exp(logits - m)))


def _vectors(dtype):
    rng = np.random.default_rng(0)
    out = {}
    # Normalised next-token-style distributions of varied peakedness.
    for scale, tag in ((0.5, "flat"), (3.0, "mid"), (12.0, "peaked")):
        logits = rng.standard_normal(257) * scale
        out[f"norm_{tag}"] = _log_softmax(logits).astype(dtype)
    # Uniform.
    out["uniform"] = _log_softmax(np.zeros(64)).astype(dtype)
    # With explicit zero-probability (-inf) entries, exercises nansum.
    v = _log_softmax(rng.standard_normal(128) * 2.0).astype(dtype)
    v[::7] = -np.inf
    out["with_neg_inf"] = v
    return out


_FINITE_KEYS = ("norm_flat", "norm_mid", "norm_peaked", "uniform")
_ALPHAS = (0.5, 1.0, 2.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shannon_matches_inlined(dtype):
    for key, v in _vectors(dtype).items():
        got = float(shannon_bits(v))
        # nansum form holds for every vector (incl. -inf entries).
        assert got == float(_ref_shannon_nansum(v)), (dtype, key)
        # On finite vectors the restricted (sum) form is identical too.
        if key in _FINITE_KEYS:
            assert got == float(_ref_shannon_sum(v)), (dtype, key)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", _ALPHAS)
def test_renyi_matches_all_three_inlined(dtype, alpha):
    # Each reference is checked at the float dtype its real call site uses:
    #   extractor (entropy/extractors.py) -> next_log_probs is float64
    #   token (huggingface) -> log_probs[...].cpu().numpy()       (float32)
    #   restricted (hf)     -> log_probs[...].cpu().double().numpy() (float64 only)
    # renyi_bits preserves input dtype, so compare the float32 implementation
    # exactly with the token site. The LM and restricted paths only receive
    # float64 in production and are therefore asserted exactly at float64.
    for key, v in _vectors(dtype).items():
        got = float(renyi_bits(v, alpha))
        assert got == float(_ref_renyi_token(v, alpha)), (dtype, alpha, key)
        if dtype is np.float64:
            assert got == float(_ref_renyi_lm(v, alpha)), (dtype, alpha, key)
            if key in _FINITE_KEYS:
                assert got == float(_ref_renyi_restricted(v, alpha)), (
                    dtype,
                    alpha,
                    key,
                )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_collision_matches_inlined(dtype):
    for key, v in _vectors(dtype).items():
        got = float(collision_prob(v))
        expected = _ref_collision(v)
        if dtype is np.float64:
            assert got == expected, (dtype, key)
        else:
            assert np.isclose(got, expected, rtol=2e-7, atol=0.0), (dtype, key)


def test_renyi_alpha_one_is_shannon():
    v = _log_softmax(np.random.default_rng(1).standard_normal(200)).astype(np.float64)
    assert float(renyi_bits(v, 1.0)) == float(shannon_bits(v))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_renyi_order_zero_counts_positive_support(dtype):
    for probabilities, expected in (([1.0], 0.0), ([0.25, 0.75], 1.0)):
        values = np.r_[np.log(probabilities), -np.inf, -np.inf].astype(dtype)
        with np.errstate(invalid="raise"):
            actual = renyi_bits(values, 0.0)
        assert actual == expected
        assert actual.dtype == dtype
    values = np.full(70000, -np.log(70000), dtype=dtype)
    with np.errstate(over="raise"):
        assert renyi_bits(values, 0.0) == dtype(np.log2(70000))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_renyi_near_one_matches_high_precision_definition(dtype):
    values = np.array([np.log(0.25), np.log(0.75), -np.inf], dtype=dtype)
    for alpha in (1 - 1e-4, 1 - 1e-8, 1 + 1e-8, 1 + 1e-4):
        with localcontext() as context:
            context.prec = 60
            order = Decimal.from_float(alpha)
            total = sum(Decimal(p) ** order for p in ("0.25", "0.75"))
            expected = float(total.ln() / ((1 - order) * Decimal(2).ln()))
        with np.errstate(invalid="raise"):
            actual = renyi_bits(values, alpha)
        assert actual == pytest.approx(expected, rel=2 * np.finfo(dtype).eps)
        assert actual.dtype == dtype
        with np.errstate(over="raise", invalid="raise"):
            peaked = renyi_bits(np.array([0.0, -1000, -1e30, -np.inf], dtype), alpha)
        assert peaked == 0


@pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf, -np.inf])
def test_renyi_rejects_unsupported_orders(alpha):
    from replica.feature.entropy.extractors import RenyiEntropy, TokenRenyiEntropy

    for make_feature in (RenyiEntropy, TokenRenyiEntropy):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            make_feature(alpha)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        renyi_bits(np.log([0.25, 0.75]), alpha)


def test_collision_is_renyi2_consistent():
    # Σp² == 2^(-H₂), checked to float tolerance.
    v = _log_softmax(np.random.default_rng(2).standard_normal(300)).astype(np.float64)
    h2 = float(renyi_bits(v, 2.0))
    assert np.isclose(float(collision_prob(v)), 2.0 ** (-h2), rtol=1e-12)
