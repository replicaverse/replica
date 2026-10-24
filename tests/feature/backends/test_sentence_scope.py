"""Sentence-scope scoring equals scoring each sentence as its own stimulus.

The scope axis must change only the context each word sees, so the dispatch
is checked against independent single-sentence calls with the same model.
"""

import numpy as np
import pytest

from replica.context import SENTENCE_LAYER
from replica.core import AnnotationLayer, Stimulus
from tests._stimuli import s0_word_stimulus as _word_stimulus

_MODEL = "sshleifer/tiny-gpt2"


def _attach_ids(stimulus: Stimulus, ids: list[int]) -> Stimulus:
    words = stimulus.layers["words"]
    stimulus.add_layer(
        AnnotationLayer(name=SENTENCE_LAYER, spans=list(words.spans), values=ids)
    )
    return stimulus


@pytest.mark.needs_model
def test_sentence_scope_concatenates_independent_sentences():
    from replica.feature.backends.huggingface import HuggingFaceLM

    first = "The cat sat on the mat."
    second = "A dog barked loudly outside."
    stimulus = _attach_ids(_word_stimulus(f"{first} {second}"), [1] * 6 + [2] * 5)

    scoped = HuggingFaceLM(_MODEL, context_scope="sentence")(stimulus)
    plain = HuggingFaceLM(_MODEL)
    expected = plain(_word_stimulus(first)) + plain(_word_stimulus(second))

    assert len(scoped) == len(expected) == 11
    got = [o.values["log_p_observed"] for o in scoped]
    want = [o.values["log_p_observed"] for o in expected]
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.needs_model
def test_sentence_scope_requires_the_layer():
    from replica.feature.backends.huggingface import HuggingFaceLM

    stimulus = _word_stimulus("No boundaries here.")
    with pytest.raises(ValueError, match=SENTENCE_LAYER):
        HuggingFaceLM(_MODEL, context_scope="sentence")(stimulus)


@pytest.mark.resource_free("Invalid scope is rejected before model resolution")
def test_unknown_scope_rejected_at_construction(monkeypatch):
    from replica.feature.backends.huggingface import HuggingFaceLM

    def unexpected_resolution(*_args, **_kwargs):
        pytest.fail("Invalid scope must fail before model resolution")

    monkeypatch.setattr(
        HuggingFaceLM,
        "_ensure_loaded",
        unexpected_resolution,
    )
    with pytest.raises(ValueError, match="context_scope"):
        HuggingFaceLM(_MODEL, context_scope="paragraph")
