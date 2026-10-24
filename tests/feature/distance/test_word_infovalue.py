"""The word-level information value port matches the authors' score_infovalue.

Same tiny model, same seed, same draw stream: values must agree bit-for-bit.
"""

import math
import sys

import pytest

from replica.external import repo_path
from replica.feature.distance.word_infovalue import (
    WordInfoValueLM,
    clean_word,
    first_generated_word,
)
from tests._require import require
from tests._stimuli import s0_word_stimulus as _stimulus

_AUTHORS_CODE = repo_path("llm-representations-rt", fetch=False)
needs_authors = require(
    (_AUTHORS_CODE / "score_infovalue.py").exists,
    "llm-representations-rt checkout not available",
)

_MODEL = "sshleifer/tiny-gpt2"
_TEXT = "The quick brown fox jumps over the lazy dog today."


def test_first_generated_word_cuts_at_boundaries():
    assert first_generated_word("  hello, world") == "hello"
    assert first_generated_word("don't stop") == "don't"
    assert first_generated_word("...") is None
    assert first_generated_word("") is None


def test_clean_word_keeps_contractions():
    assert clean_word(" “doesn't”. ") == "doesn't"


@needs_authors
@pytest.mark.needs_model
def test_matches_the_authors_functions_bit_for_bit():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(_AUTHORS_CODE))
    import score_infovalue as ref

    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(_MODEL).eval()

    words = _TEXT.split()
    torch.manual_seed(0)
    theirs = [
        ref.compute_infovalue_for_word(
            model=model,
            tokenizer=tokenizer,
            context_text=" ".join(words[:i]),
            target_word=words[i],
            k=5,
            max_new_tokens=3,
            device=torch.device("cpu"),
            layer_indices=None,
        )
        for i in range(1, len(words))
    ]

    # The reference loads its model with the transformers-default attention, so
    # the port comparison does too; production defaults to eager.
    lm = WordInfoValueLM(
        model_id=_MODEL,
        k=5,
        max_new_tokens=3,
        seed=0,
        device="cpu",
        attn_implementation=None,
    )
    outputs = lm(_stimulus(_TEXT))

    assert all(math.isnan(v) for v in outputs[0].values.values())
    n_layers = lm.layer_count
    for i, ref_vals in enumerate(theirs, start=1):
        mine = [outputs[i].values[f"infovalue_layer{j}"] for j in range(n_layers)]
        if ref_vals is None:
            assert all(math.isnan(v) for v in mine)
        else:
            assert mine == pytest.approx(ref_vals, abs=0.0)
