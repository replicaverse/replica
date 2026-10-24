"""A unit of several whitespace words under the entropy estimators: the
closed form reads its first token, the sampler samples all its words."""

import math

import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.backends.huggingface import HuggingFaceLM
from replica.feature.entropy._mc_word import MCWordEntropyLM
from tests._stimuli import word_stimulus

pytestmark = [pytest.mark.needs_model]

_MODEL = "sshleifer/tiny-gpt2"
_TEXT = "we saw of the cat on a mat"


def _fused(text: str, pairs: list[int]) -> Stimulus:
    """Words layer of *text* with each word index in *pairs* fused to its successor."""
    single = word_stimulus(text, "fused")
    spans = list(single.layers["words"].spans)
    fused_spans = []
    skip = False
    for i, span in enumerate(spans):
        if skip:
            skip = False
            continue
        if i in pairs:
            fused_spans.append(Span(span.start, spans[i + 1].end))
            skip = True
        else:
            fused_spans.append(span)
    stimulus = Stimulus(text=text, stimulus_id="fused")
    stimulus.add_layer(
        AnnotationLayer(
            "words", fused_spans, [text[s.start : s.end] for s in fused_spans]
        )
    )
    return stimulus


def _values(outputs, key):
    return [o.values[key] for o in outputs]


def test_closed_form_entropy_reads_the_first_token_of_a_fused_unit():
    lm = HuggingFaceLM(
        _MODEL,
        add_bos=True,
        provide_word_dist=False,
        restricted_first_token_entropy=True,
    )
    single = _values(
        lm(word_stimulus(_TEXT, "single")), "restricted_first_token_entropy"
    )
    fused = _values(lm(_fused(_TEXT, [2, 5])), "restricted_first_token_entropy")
    # Units: we, saw, "of the", cat, "on a", mat: each read before its first word.
    assert fused == [single[0], single[1], single[2], single[4], single[5], single[7]]
    assert all(not math.isnan(v) for v in fused)


def test_monte_carlo_word_entropy_samples_a_fused_unit_as_its_words():
    # A fused unit is sampled where its first word starts, as that many words.
    lm = MCWordEntropyLM(
        _MODEL, n_samples=8, max_iter=3, alpha=1.0, add_bos=True, seed=0
    )
    token_ids, n_single, single, _ = lm._prepare_word_sampling(
        word_stimulus(_TEXT, "single"), "words"
    )
    fused_ids, n_fused, fused, _ = lm._prepare_word_sampling(
        _fused(_TEXT, [2, 5]), "words"
    )
    assert (n_single, n_fused, fused_ids) == (8, 6, token_ids)
    by_unit = dict(single)
    assert dict(fused) == {
        0: by_unit[0],
        1: by_unit[1],
        2: by_unit[2],
        3: by_unit[4],
        4: by_unit[5],
        5: by_unit[7],
    }
    stimulus = _fused(_TEXT, [2, 5])
    assert lm._unit_word_counts(stimulus, "words") == {2: 2, 4: 2}
    outputs = lm(stimulus)
    assert len(outputs) == 6
    assert all(not math.isnan(o.values["mc_word_entropy"]) for o in outputs[1:])
    assert all(len(o.values["log_p_samples_nats"]) == 8 for o in outputs[1:])
