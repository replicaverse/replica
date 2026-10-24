"""Tests for the ModelBlocks word-surprisal backend (postprocess_bi port).

The model-free tests exercise the boundary redistribution + sub-token→word
aggregation directly.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica.feature import FeatureModel, PrecedingContextModel, surprisal_model
from replica.feature.surprisal.modelblocks import ModelBlocksSurprisalLM


def _agg(rows, n_words):
    return ModelBlocksSurprisalLM._postprocess_and_aggregate(rows, n_words)


def test_factory_dispatch():
    lm = surprisal_model("gpt2", method="modelblocks_surprisal")
    assert isinstance(lm, ModelBlocksSurprisalLM)
    assert isinstance(lm, FeatureModel)
    assert lm.vocab is None
    assert PrecedingContextModel(lm).vocab is None


def test_single_token_words_redistribution():
    # 3 single-token words (all "B") + the synthetic EOS row (also "B").
    # rows: (surprisal, bori, bprob, is_eos).
    rows = [
        (5.0, "B", -0.5, False),  # w0
        (6.0, "B", -0.3, False),  # w1
        (7.0, "B", -0.2, False),  # w2
        (1.0, "B", -0.1, True),  # <eos>
    ]
    out = _agg(rows, 3)
    # postprocess moves bprob[i+1] from token i to i+1 when i+1 is "B":
    #   w0 = 5.0 - (-0.3) = 5.3
    #   w1 = 6.0 + (-0.3) - (-0.2) = 5.9
    #   w2 = 7.0 + (-0.2) - (-0.1) = 6.9   (eos bprob folded in, eos dropped)
    np.testing.assert_allclose(out, [5.3, 5.9, 6.9], rtol=0, atol=1e-9)


def test_multitoken_word_aggregation():
    # "The" (1 tok) + "cat" (2 toks: B then I) + EOS.
    rows = [
        (4.0, "B", -0.4, False),  # The
        (8.0, "B", -0.6, False),  # cat-1 (B)
        (3.0, "I", -0.9, False),  # cat-2 (I, continuation)
        (1.0, "B", -0.2, True),  # <eos>
    ]
    out = _agg(rows, 2)
    # i=0: next is B -> The -= bprob[1](-0.6); cat1 += -0.6
    # i=1: next is I -> no change
    # i=2: next is B(eos) -> cat2 -= bprob[3](-0.2); eos += -0.2
    # The = 4.0 - (-0.6) = 4.6
    # cat = (8.0 + (-0.6)) + (3.0 - (-0.2)) = 7.4 + 3.2 = 10.6
    np.testing.assert_allclose(out, [4.6, 10.6], rtol=0, atol=1e-9)


def test_leading_continuation_token_opens_first_word():
    # A document's first token has no leading space -> tagged "I"; it must still
    # open the first word (regression: a None-accumulator crash / dropped word).
    rows = [
        (4.0, "I", -0.4, False),  # "The" (string-initial, no Ġ)
        (9.0, "B", -0.5, False),  # "dog"
        (1.0, "B", -0.1, True),  # <eos>
    ]
    out = _agg(rows, 2)
    # The = 4.0 - (-0.5) = 4.5 ; dog = (9.0 + (-0.5)) - (-0.1) = 8.6
    np.testing.assert_allclose(out, [4.5, 8.6], rtol=0, atol=1e-9)


def test_word_count_mismatch_raises():
    rows = [(5.0, "B", -0.5, False), (6.0, "B", -0.3, False), (1.0, "B", -0.1, True)]
    with np.testing.assert_raises(ValueError):
        _agg(rows, 5)


def test_a_multiword_unit_sums_its_whitespace_words(monkeypatch):
    from replica.core import AnnotationLayer, Span, Stimulus
    from replica.feature.core.outputs.information import LN2

    rows = [
        (5.0, "B", -0.5, False),
        (6.0, "B", -0.3, False),
        (7.0, "B", -0.2, False),
        (1.0, "B", -0.1, True),
    ]
    lm = ModelBlocksSurprisalLM.__new__(ModelBlocksSurprisalLM)
    lm.name = "mb"
    monkeypatch.setattr(lm, "_ensure", lambda: None)
    monkeypatch.setattr(lm, "_subtoken_rows", lambda text: rows)
    text = "of the cat"
    single = Stimulus(text=text, stimulus_id="a")
    single.add_layer(
        AnnotationLayer("words", [Span(0, 2), Span(3, 6), Span(7, 10)], text.split())
    )
    fused = Stimulus(text=text, stimulus_id="b")
    fused.add_layer(
        AnnotationLayer("words", [Span(0, 6), Span(7, 10)], ["of the", "cat"])
    )
    per_word = [o.values["log_p_observed"] for o in lm(single)]
    np.testing.assert_allclose(per_word, -np.array([5.3, 5.9, 6.9]) * LN2)
    per_unit = [o.values["log_p_observed"] for o in lm(fused)]
    np.testing.assert_allclose(per_unit, [per_word[0] + per_word[1], per_word[2]])


class _SpaceTokenizer:
    """``a`` opens the text; every later word is the word-initial ``Ġa``."""

    vocab = {"<bos>": 0, "Ġa": 1, "a": 2}

    def __call__(self, text):
        return {"input_ids": [2] + [1] * (len(text.split()) - 1)}

    def convert_ids_to_tokens(self, ids):
        names = {i: t for t, i in self.vocab.items()}
        return [names[i] for i in ids]


class _RecordingModel:
    def __init__(self, **positions):
        self.config = SimpleNamespace(bos_token_id=0, **positions)
        self.lengths: list[int] = []

    def __call__(self, input_ids):
        self.lengths.append(input_ids.shape[-1])
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[-1], 3))


def _window_lengths(lm, n_words, **positions):
    lm._tokenizer = _SpaceTokenizer()
    lm._model = _RecordingModel(**positions)
    lm._space_ids = torch.tensor([1])
    rows = lm._subtoken_rows(" ".join(["a"] * n_words))
    assert len(rows) == n_words + 1  # every sub-token once, plus one EOS row
    return lm._model.lengths


@pytest.mark.resource_free("Lazy construction with an in-memory model and tokenizer")
def test_windows_use_the_trained_position_count():
    # mGPT declares 2048 positions but was trained at 512.
    config = dict(n_positions=2048, max_position_embeddings=2048)
    assert _window_lengths(
        ModelBlocksSurprisalLM("ai-forever/mGPT"), 700, **config
    ) == [512, 445]
    assert _window_lengths(ModelBlocksSurprisalLM("gpt2"), 700, **config) == [701]
    gpt2 = dict(n_positions=1024, max_position_embeddings=1024)
    assert _window_lengths(ModelBlocksSurprisalLM("gpt2"), 1100, **gpt2) == [1024, 589]


@pytest.mark.resource_free("Lazy construction with an in-memory model and tokenizer")
def test_a_model_without_a_position_limit_is_refused_before_inference():
    lm = ModelBlocksSurprisalLM("new/model")
    with pytest.raises(ValueError, match="no usable context limit"):
        _window_lengths(lm, 700)
    assert lm._model.lengths == []
