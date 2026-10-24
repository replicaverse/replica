"""HuggingFaceLM's word-type distribution, driven by a fixed-vocabulary double.

The tokenizer keeps a word's leading space inside its token and makes a lone
space its own token, so a double space leaves a token that belongs to no word.
The causal model's logits differ at every position, so reading the wrong row
changes every distribution feature.
"""

import re
from types import SimpleNamespace

import numpy as np
import torch

from replica.feature import Entropy, HuggingFaceLM, Rank
from tests._stimuli import word_stimulus

PIECES = ["<bos>", "He", " left.", " ", " Then", " she", " came"]


class _PieceTokenizer:
    bos_token_id = 0
    vocab_size = len(PIECES)

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r" ?\S+| ", text))
        return {
            "input_ids": [PIECES.index(m.group()) for m in matches],
            "offset_mapping": [m.span() for m in matches],
        }

    def decode(self, ids):
        return "".join(PIECES[i] for i in ids)


class _PositionModel:
    config = SimpleNamespace(max_position_embeddings=1024, vocab_size=len(PIECES))

    def __call__(self, input_ids):
        positions = torch.arange(input_ids.shape[-1], dtype=torch.float32)
        vocabulary = torch.arange(len(PIECES), dtype=torch.float32)
        logits = -(vocabulary - positions.unsqueeze(-1)).square() / 4
        return SimpleNamespace(logits=logits.unsqueeze(0))


class _DoubleLM(HuggingFaceLM):
    def _ensure_loaded(self):
        if self._vocab is None:
            self._build_word_vocab()


def _lm(add_bos=True):
    lm = _DoubleLM("stub", device="cpu", add_bos=add_bos)
    lm._tokenizer = _PieceTokenizer()
    lm._model = _PositionModel()
    return lm


def _row(lm, text, row):
    """The word-type distribution at one row of the BOS-prefixed forward pass."""
    ids = [PIECES.index("<bos>"), *_PieceTokenizer()(text)["input_ids"]]
    logits = lm._model(torch.tensor([ids])).logits[0]
    return lm._marginalise_to_word_dist(torch.log_softmax(logits, -1)[row].numpy())


def test_next_word_distribution_is_read_where_the_unit_is_predicted():
    text = "He left.  Then she came"
    lm = _lm()
    outputs = lm(word_stimulus(text), extract=["next_log_probs"])
    # Rows: 0 <bos>, 1 "He", 2 " left.", 3 " ", 4 " Then", 5 " she". Row r
    # predicts the token at r + 1, so "Then" is predicted at row 3, after the
    # lone space that belongs to no word.
    for unit, row in enumerate([0, 1, 3, 4, 5]):
        np.testing.assert_array_equal(
            outputs[unit].values["next_log_probs"], _row(lm, text, row)
        )
    entropy = Entropy()(word_stimulus(text), outputs)
    assert entropy[2] == Entropy()._compute(_row(lm, text, 3))
    assert entropy[2] != Entropy()._compute(_row(lm, text, 2))


def test_contiguous_units_read_the_previous_unit_s_last_row():
    text = "He left. Then she came"
    lm = _lm()
    outputs = lm(word_stimulus(text), extract=["next_log_probs"])
    for unit in range(5):
        np.testing.assert_array_equal(
            outputs[unit].values["next_log_probs"], _row(lm, text, unit)
        )


def test_first_unit_has_no_distribution_without_bos():
    outputs = _lm(add_bos=False)(word_stimulus("He left."), extract=["next_log_probs"])
    assert "next_log_probs" not in outputs[0].values
    assert "next_log_probs" in outputs[1].values


def test_rank_looks_units_up_in_the_shared_word_type_index():
    text = "He left.  Then she came"
    lm = _lm()
    outputs = lm(word_stimulus(text), extract=["next_log_probs"])
    assert all(o.metadata["vocab_index"] is lm._vocab_index for o in outputs)
    ranks = Rank()(word_stimulus(text), outputs)
    # Word types are lower-cased and stripped, so "Then" ranks as "then"; the
    # BOS token gives the first unit a distribution, so it is ranked too.
    for unit, word in enumerate(["he", "left.", "then", "she", "came"]):
        dist = outputs[unit].values["next_log_probs"]
        assert ranks[unit] == 1 + np.sum(dist > dist[lm._vocab.index(word)])


def test_rank_is_nan_at_the_first_unit_without_bos():
    text = "He left."
    outputs = _lm(add_bos=False)(word_stimulus(text), extract=["next_log_probs"])
    ranks = Rank()(word_stimulus(text), outputs)
    assert np.isnan(ranks[0])
    assert np.isfinite(ranks[1])
