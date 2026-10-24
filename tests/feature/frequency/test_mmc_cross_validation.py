"""Cross-checks on the sampled unigram estimator.

:func:`mmc_estimate_batch` is checked against

* a naive implementation written from the definition (:func:`_naive_mmc`);
* the per-word path, and several batch sizes;
* the plain Monte Carlo rate of the same quantity, which it must converge to.
"""

import importlib.util
import math

import numpy as np
import pytest

from replica.feature.frequency.sampled import (
    _encode_samples,
    _is_word_boundary,
    _word_boundary_marker,
    mc_estimate,
    mmc_estimate,
    mmc_estimate_batch,
)
from tests._require import require

pytestmark = pytest.mark.needs_model

needs_transformers = require(
    lambda: importlib.util.find_spec("transformers") is not None,
    "transformers not available",
)

_SAMPLES = [
    "the dog ran across the field",
    "a cat sat on the mat",
    "the the the dog dog",
    "hello world this is a test",
]


def _naive_mmc(word, samples, model, tokenizer):
    """The estimator written from its definition.

    One sample, boundary and continuation token at a time, in float64 and
    unpadded, sharing no batching or masking code with the production path.
    """
    import torch

    marker = _word_boundary_marker(tokenizer)
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        return 0.0

    total_prob = 0.0
    n_boundaries = 0
    for ids in _encode_samples(tokenizer, samples):
        strs = tokenizer.convert_ids_to_tokens(ids)
        for p in range(len(ids)):
            if not _is_word_boundary(strs, p, marker):
                continue
            n_boundaries += 1
            prefix = ids[: p + 1]
            joint = 1.0
            for step, token_id in enumerate(token_ids):
                sequence = prefix + token_ids[:step]
                with torch.no_grad():
                    logits = model(torch.tensor([sequence])).logits
                probs = torch.softmax(logits[0, -1, :].double(), dim=-1)
                joint *= float(probs[token_id])
            total_prob += joint
    return total_prob / (len(samples) + n_boundaries)


@needs_transformers
@pytest.mark.parametrize("word", [" the", " dog", " cat", " world"])
def test_matches_a_naive_implementation(word):
    """Optimised path == the definition, on an unpadded float64 reference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    naive = _naive_mmc(word, _SAMPLES, model, tokenizer)
    fast = mmc_estimate_batch(
        [word], _SAMPLES, model, tokenizer, 4, max_continuation_boundaries=None
    )[word]
    np.testing.assert_allclose(fast, naive, rtol=1e-4, atol=1e-15)


@needs_transformers
def test_per_word_and_batched_paths_agree():
    """``mmc_estimate`` and its batched form compute the same thing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    words = [" the", " dog", " antidisestablishment"]
    batched = mmc_estimate_batch(
        words, _SAMPLES, model, tokenizer, 2, max_continuation_boundaries=None
    )
    for word in words:
        np.testing.assert_allclose(
            batched[word],
            mmc_estimate(word, _SAMPLES, model, tokenizer, batch_size=2),
            rtol=1e-4,
            atol=1e-15,
        )


@needs_transformers
def test_converges_to_the_monte_carlo_rate():
    """MMC and MC estimate the same ``p(w)`` and must agree on many samples.

    Compared in bits, with a tolerance set by MC's sampling error.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    torch.manual_seed(0)
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([[tokenizer.bos_token_id]]),
            do_sample=True,
            top_k=0,
            max_new_tokens=30,
            num_return_sequences=256,
            pad_token_id=tokenizer.eos_token_id,
        )
    samples = [tokenizer.decode(g[1:], skip_special_tokens=True) for g in generated]

    # Frequent words only: MC cannot resolve a rare word from 256 samples.
    for word in ("the", "of"):
        rb = mmc_estimate_batch(
            [" " + word],
            samples,
            model,
            tokenizer,
            64,
            max_continuation_boundaries=None,
        )[" " + word]
        mc = mc_estimate(word, samples)
        assert mc > 0, f"{word!r} never sampled; raise the sample count"
        # MC's relative standard error is ~1/sqrt(count); allow 3 of them.
        count = sum(s.split().count(word) for s in samples)
        tolerance_bits = 3.0 / math.sqrt(count) / math.log(2)
        assert abs(-math.log2(rb) - -math.log2(mc)) < max(tolerance_bits, 0.5), (
            f"{word!r}: rb={-math.log2(rb):.3f} bits vs mc={-math.log2(mc):.3f} bits"
        )
