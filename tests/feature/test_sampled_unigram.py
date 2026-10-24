"""Sampled unigram estimation, checked against the authors' algorithm.

``_authors_mmc_reference`` ports ``unigram-freq``'s ``meco_rep/util.py`` with
this package's position-id and BOS handling. Independent cross-checks live in
``test_mmc_cross_validation.py``.
"""

import importlib.util
import sys

import numpy as np
import pytest

from replica.external import repo_path
from replica.feature.frequency.sampled import (
    mc_estimate,
    mmc_estimate,
    mmc_estimate_batch,
)
from tests._require import require

pytestmark = pytest.mark.needs_model

_AUTHORS = repo_path("unigram-freq", fetch=False) / "meco_rep"
_SAMPLES = [
    "the dog ran across the field",
    "a cat sat on the mat",
    "the the the dog dog",
    "hello world this is a test",
]
_WORDS = ["the", "dog", "cat", "world", "zzz"]
#: Word-initial surfaces: verbatim case, leading space where the passage has one.
_SURFACE_WORDS = [" the", " dog", " The", "In", " antidisestablishment"]

needs_authors = require(
    lambda: (
        (_AUTHORS / "util.py").exists()
        and importlib.util.find_spec("transformers") is not None
    ),
    "unigram-freq authors' code or transformers not available",
)


@needs_authors
def test_estimators_match_authors_code():
    """REPLICA matches the authors' algorithm on gpt2.

    ``mmc`` is compared against ``_authors_mmc_reference``; ``mc`` is model-free
    and compared against upstream directly.
    """

    sys.path.insert(0, str(_AUTHORS))
    import util as ref
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tests.feature.frequency._authors_mmc_reference import (
        mmc_estimate as reference_mmc,
    )

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    for w in _WORDS + _SURFACE_WORDS:
        np.testing.assert_allclose(
            mmc_estimate(w, _SAMPLES, model, tok, batch_size=2),
            reference_mmc(w, _SAMPLES, model, tok, batch_size=2),
            rtol=1e-6,
            atol=1e-12,
            err_msg=f"mmc {w!r}",
        )

    # mc is model-free, so upstream needs no correction to compare against.
    for w in _WORDS:
        np.testing.assert_allclose(
            mc_estimate(w, _SAMPLES),
            ref.mc_estimator(w, _SAMPLES),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"mc {w}",
        )


@needs_authors
def test_mmc_is_batch_size_invariant():
    """``batch_size`` leaves the estimates unchanged up to float32 accumulation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    words = [" the", " dog", " antidisestablishment"]
    reference = mmc_estimate_batch(
        words, _SAMPLES, model, tok, 1, max_continuation_boundaries=None
    )
    for batch_size in (2, 3, 8):
        got = mmc_estimate_batch(
            words, _SAMPLES, model, tok, batch_size, max_continuation_boundaries=None
        )
        for w in words:
            # Agreement is bounded by float32 accumulation over thousands of
            # boundaries.
            np.testing.assert_allclose(
                got[w],
                reference[w],
                rtol=1e-4,
                atol=1e-15,
                err_msg=f"{w!r} moved at batch_size={batch_size}",
            )


@needs_authors
def test_mc_cannot_score_word_initial_surfaces():
    """``mc`` counts against ``sample.split()``, so a space-prefixed surface
    estimates to exactly 0."""
    sys.path.insert(0, str(_AUTHORS))
    import util as ref

    for w in (" the", " dog"):
        assert mc_estimate(w, _SAMPLES) == 0.0
        assert ref.mc_estimator(w, _SAMPLES) == 0.0
    assert mc_estimate("the", _SAMPLES) > 0.0
