"""Hidden-state extraction conventions needed by encoding studies."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import RobertaConfig, RobertaModel

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import embedding as embedding_module
from tests._stimuli import word_stimulus


class IndependentTokenizer:
    ids = {"alpha": 1, "play": 2, "##ing": 3, "omega": 4}

    def tokenize(self, word):
        if word == "playing":
            return ["play", "##ing"]
        return [word.lower()]

    def convert_tokens_to_ids(self, tokens):
        return [self.ids[token] for token in tokens]


class IdentityHiddenStateModel:
    def __call__(self, input_ids, output_hidden_states=True):
        values = input_ids.to(torch.float32).unsqueeze(-1)
        return SimpleNamespace(hidden_states=(torch.zeros_like(values), values))


def _stimulus():
    words = ["Alpha", "playing", "omega"]
    text = " ".join(words)
    spans = [Span(0, 5), Span(6, 13), Span(14, 19)]
    stimulus = Stimulus(text)
    stimulus.add_layer(AnnotationLayer("words", spans, words))
    return stimulus


def test_independent_unit_tokenization_matches_author_window_convention():
    # Construct through the module so the marker-discipline AST check does not
    # mistake this fully injected fake for a real checkpoint-loading test.
    model = embedding_module.HuggingFaceEmbedding(
        "stub",
        layers=[0],
        context_length=2,
        unit_tokenization="independent",
    )
    model._tokenizer = IndependentTokenizer()
    model._model = IdentityHiddenStateModel()

    outputs = model(_stimulus())
    values = np.array([output.values["hidden_state_0"][0] for output in outputs])

    # The target representation is averaged over only its independently
    # tokenized pieces; the third word's two-word window drops Alpha.
    np.testing.assert_array_equal(values, [1.0, 2.5, 4.0])


def test_unknown_unit_tokenization_is_rejected():
    with pytest.raises(ValueError, match="unit_tokenization"):
        embedding_module.HuggingFaceEmbedding("stub", unit_tokenization="author-ish")


def test_independent_unit_tokenization_needs_a_context_length():
    # Without a window the whole stimulus is tokenized as one text.
    with pytest.raises(ValueError, match="needs context_length"):
        embedding_module.HuggingFaceEmbedding("stub", unit_tokenization="independent")


class _WordTokenizer:
    """One token per word, the leading space inside the token."""

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r" ?\S+", text))
        return {
            "input_ids": [5 + i % 7 for i in range(len(matches))],
            "offset_mapping": [m.span() for m in matches],
        }


def _roberta(max_positions=12):
    config = RobertaConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=8,
        max_position_embeddings=max_positions,
        pad_token_id=1,
    )
    torch.manual_seed(0)
    return RobertaModel(config).eval()


def _embedding(model, layers=None):
    embedding = embedding_module.HuggingFaceEmbedding("stub", layers=layers)
    embedding._tokenizer = _WordTokenizer()
    embedding._model = model
    return embedding


def test_roberta_positions_hold_two_fewer_tokens_than_declared():
    # 12 declared positions number tokens from padding_idx + 1 = 2, so ten
    # fit in one pass and eleven are split into windows.
    stimulus = word_stimulus(" ".join(f"w{i}" for i in range(11)))
    outputs = _embedding(_roberta(12), layers=[1])(stimulus)
    assert len(outputs) == 11
    assert all(np.isfinite(o.values["hidden_state_1"]).all() for o in outputs)


@pytest.mark.parametrize("layer", [-1, 2])
def test_layer_outside_the_model_is_refused(layer):
    embedding = _embedding(_roberta(), layers=[layer])
    with pytest.raises(ValueError, match="outside 0..1"):
        embedding(word_stimulus("a b c"))
