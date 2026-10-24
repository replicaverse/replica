"""Model-free unit tests for the similarity-adjusted surprisal helpers."""

import numpy as np
import pytest

from replica.feature.distance.similarity_surprisal import (
    _BaseSimilaritySurprisalLM,
    _neg_log,
    levenshtein,
    orthographic_similarity,
)


def test_levenshtein_known_cases():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("flaw", "lawn") == 2
    assert levenshtein("a", "b") == 1


def test_levenshtein_symmetric():
    assert levenshtein("sunday", "saturday") == levenshtein("saturday", "sunday") == 3


def test_orthographic_similarity_range_and_extremes():
    assert orthographic_similarity("word", "word") == 1.0
    assert orthographic_similarity("", "") == 1.0
    # Completely disjoint strings of equal length: similarity 0.
    assert orthographic_similarity("abc", "xyz") == 0.0
    # Empty vs non-empty: lev = len, denom = len, similarity 0.
    assert orthographic_similarity("", "abc") == 0.0
    s = orthographic_similarity("cat", "cart")
    assert 0.0 < s < 1.0
    assert np.isclose(s, 1.0 - 1.0 / 4.0)


def test_neg_log_zero_is_inf():
    assert _neg_log(0.0) == float("inf")
    assert _neg_log(-1e-12) == float("inf")
    assert np.isclose(_neg_log(np.exp(-2.0)), 2.0)


def test_window_bounds_short_sequence_is_single_window():
    bounds = list(_BaseSimilaritySurprisalLM._window_bounds(10, 1024))
    assert bounds == [(0, 10, 0)]


def test_window_bounds_cover_every_position_exactly_once():
    for n_tokens, max_len in [(2500, 1024), (1024, 1024), (1025, 1024), (700, 128)]:
        covered = np.zeros(n_tokens, dtype=int)
        for start, end, fill_start in _BaseSimilaritySurprisalLM._window_bounds(
            n_tokens, max_len
        ):
            assert start <= fill_start < end <= n_tokens
            assert end - start <= max_len
            covered[fill_start:end] += 1
        assert (covered == 1).all()


def test_window_bounds_min_left_context():
    # Every fresh position past the first window has >= stride tokens of
    # left context within its window.
    max_len = 128
    stride = max_len // 2
    for start, _end, fill_start in _BaseSimilaritySurprisalLM._window_bounds(
        1000, max_len
    ):
        if start > 0:
            assert fill_start - start >= stride


@pytest.mark.resource_free("Lazy constructor and model double; no model loading")
def test_word_sampler_rejects_context_without_generation_budget():
    from types import SimpleNamespace

    from replica.feature import SimilarityAdjustedSurprisalWordLM

    model = SimilarityAdjustedSurprisalWordLM(
        "gpt2", context_window=4, max_iter=3, pos=False
    )
    model._device = "cpu"
    model._model = SimpleNamespace(config=SimpleNamespace(n_positions=1024))
    alignment = SimpleNamespace(mapping={0: [0]})
    with pytest.raises(ValueError, match="two prefix tokens"):
        model._sample_all_words([1, 2], alignment, 1, 1)


@pytest.mark.resource_free("Synthetic vocabulary and logits; no model loading")
@pytest.mark.parametrize("prefix", ["Ġ", "▁"])
def test_word_sampler_requires_supported_boundaries_even_with_eos(prefix):
    from types import SimpleNamespace

    import torch

    from replica.feature import SimilarityAdjustedSurprisalWordLM

    pieces = ["<eos>", prefix + "cat", "tail", prefix + "dog"]
    model = SimilarityAdjustedSurprisalWordLM(
        first_token="word_starter", pos=False, n_samples=3, max_iter=2
    )
    model._device = "cpu"
    model._tokenizer = SimpleNamespace(
        vocab_size=len(pieces),
        eos_token_id=0,
        convert_ids_to_tokens=pieces.__getitem__,
        decode=lambda ids: "".join(pieces[i].replace("Ġ", " ") for i in ids),
    )

    class Model:
        config = SimpleNamespace(vocab_size=len(pieces))

        def __call__(self, input_ids, *, past_key_values, **kwargs):
            # Every continuation draws another word starter, ending this word.
            logits = torch.full((len(input_ids), 1, len(pieces)), -torch.inf)
            logits[:, :, 3] = 0
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    model._model = Model()
    if prefix == "▁":
        with pytest.raises(ValueError, match="word-boundary stop rule is undefined"):
            model._ensure_starter_mask()
        return
    model._ensure_starter_mask()
    assert model._starter_mask.tolist() == [True, True, False, True]
    # The highest-probability token is not a word starter and must be excluded.
    first_logits = torch.tensor([-torch.inf, 0.0, 100.0, -torch.inf])
    samples = model._sample_words((), 0, first_logits, torch.Generator().manual_seed(3))
    assert samples == ["cat"] * 3
