"""Unit tests for the Clark et al. (2025)-style overlapping-window driver in
``MCWordEntropyLM`` and ``ContinuationEntropyLM``.

What's covered here:
  * ``_clark_half_window_starts``: the pure-Python boundary computation
    is unit-tested in isolation (no LM needed). We assert exact
    boundary positions for hand-crafted scenarios that mirror the
    upstream's behavior.
  * The windowed driver: with a mock LM whose ctx_size is tiny (so
    windowing fires on a short stim), the windowed call must (a)
    produce exactly one sample array per REPLICA word boundary, (b) cover
    every word boundary exactly once across windows, and (c) produce
    sample arrays of the correct ``(N,)`` shape. Keyed sampling must preserve
    sample identities across chunks and windows. The context-independent
    model also permits exact comparison with the single-window driver.

Run on CPU in seconds; no HuggingFace download.
"""

import math
import weakref

import numpy as np
import pytest

from replica.core import Stimulus
from replica.feature.entropy.estimators import (
    ContinuationEntropyLM,
    ForwardWordLookaheadEntropyLM,
    MCWordEntropyLM,
    _clark_half_window_starts,
)
from tests._stimuli import word_stimulus
from tests.feature.entropy._entropy_mocks import (
    UniformCausalLM as _UniformLM,
)
from tests.feature.entropy._entropy_mocks import (
    WhitespaceTokenizer as _Tokenizer,
)
from tests.feature.entropy._entropy_mocks import build_estimator

# ── Pure-Python boundary tests (no LM) ─────────────────────────────────────


def test_half_window_starts_single_window():
    """Stim fits comfortably in one window → returns
    ``[0, n_tokens, n_tokens]`` matching the upstream "hacky fix"
    (its comment: 'when the whole story fits in one context window')."""
    # 10 tokens with word_final at positions 2, 5, 9. half_size=20
    # (way larger than n_tokens) → no second start ever emitted.
    starts = _clark_half_window_starts(
        n_tokens=10,
        word_final_positions={2, 5, 9},
        half_size=20,
    )
    assert starts == [0, 10, 10]


def test_half_window_starts_splits_at_word_boundary():
    """When the window-size cap is hit, the new half-window must start
    at the position after the latest word_final seen (upstream line 210:
    ``curr_start_ix = curr_end_ix + 1``). This avoids splitting a word
    across windows."""
    # 12 tokens, word_final at 1, 4, 6, 10. half_size=5.
    #   ix=0, no split, last_word_end stays -1
    #   ix=1, word_final → last_word_end=1
    #   ix=4, word_final → last_word_end=4
    #   ix=5, ix-cur_start=5 → split. cur_start = 4+1 = 5.
    #          → half_starts.append(5)
    #   ix=6, word_final → last_word_end=6
    #   ix=10, ix-cur_start=10-5=5 → split. cur_start = 6+1 = 7.
    #          → half_starts.append(7)
    #   final n_tokens=12 → half_starts.append(12)
    starts = _clark_half_window_starts(
        n_tokens=12,
        word_final_positions={1, 4, 6, 10},
        half_size=5,
    )
    assert starts == [0, 5, 7, 12]


def test_half_window_starts_rejects_zero_half_size():
    with pytest.raises(ValueError, match="half_size must be ≥ 1"):
        _clark_half_window_starts(10, {0}, half_size=0)


def test_half_window_starts_no_word_final_falls_back_to_cur_plus_1():
    """If a half-window's size cap is hit but no ``word_final`` has been
    seen yet (degenerate case, should never trigger on real data, but
    the splitter must not infinite-loop), the new start is just
    ``cur_start + 1``."""
    # 7 tokens, no word_final. half_size=3 → splits at every 3 tokens.
    starts = _clark_half_window_starts(
        n_tokens=7,
        word_final_positions=set(),
        half_size=3,
    )
    # ix=3: no word_final → start = 0 + 1 = 1
    # ix=4: 4-1=3 → start = 1+1 = 2
    # ix=5: 5-2=3 → start = 2+1 = 3
    # ix=6: 6-3=3 → start = 3+1 = 4
    # final: append 7
    assert starts == [0, 1, 2, 3, 4, 7]


def _make_mcword(
    n_samples=4, max_iter=2, alpha=1.0, chunk_size=None, seed=0, n_positions=1024
):
    return build_estimator(
        MCWordEntropyLM,
        _UniformLM(n_positions=n_positions),
        tokenizer=_Tokenizer(),
        name=f"mcword_mock_N{n_samples}_M{max_iter}_a{alpha}",
        n_samples=n_samples,
        seed=seed,
        add_bos=True,
        chunk_size=chunk_size,
        max_iter=max_iter,
        alpha=float(alpha),
    )


def _make_cont(n_samples=4, max_tokens=2, chunk_size=None, seed=0, n_positions=1024):
    return build_estimator(
        ContinuationEntropyLM,
        _UniformLM(n_positions=n_positions),
        tokenizer=_Tokenizer(),
        name=f"cont_mock_N{n_samples}_L{max_tokens}",
        n_samples=n_samples,
        seed=seed,
        add_bos=True,
        chunk_size=chunk_size,
        max_tokens=max_tokens,
    )


def _make_stim(words: list[str]) -> Stimulus:
    return word_stimulus(" ".join(words))


@pytest.mark.parametrize("kind", ["word", "continuation"])
def test_prefix_output_is_released_before_sampling(kind, monkeypatch):
    """Full prefix logits must not remain live while a window is sampled."""
    lm = _make_mcword() if kind == "word" else _make_cont()
    method = "_sample_word_log_p" if kind == "word" else "_sample_continuation_log_p"
    prefix_outputs = []

    def remember_prefix(_model, args, output):
        # The initial forward has no past cache; subsequent forwards do.
        if len(args) == 1 and args[0].shape[1] > 1:
            prefix_outputs.append(weakref.ref(output))

    lm._model.register_forward_hook(remember_prefix)
    sample = getattr(lm, method)

    def checked_sample(*args, **kwargs):
        assert prefix_outputs and prefix_outputs[-1]() is None
        return sample(*args, **kwargs)

    monkeypatch.setattr(lm, method, checked_sample)
    driver = (
        lm._estimate_words_one_pass
        if kind == "word"
        else lm._estimate_continuations_one_pass
    )
    result = driver([0, 1, 1, 1], [(0, 2), (1, 3)], 4)
    assert set(result) == {0, 1}
    assert all(values.shape == (4,) for values in result.values())


# ── Windowed-driver end-to-end tests ────────────────────────────────────────


def test_mcword_windowed_covers_all_words_exactly_once():
    """With n_positions=8 (mock ctx) and max_iter=2, half_size = 8//2 - 2
    = 2. A 6-word stim has 6 word boundaries after BOS-prepending:
    tokens = [BOS, w1, w2, w3, w4, w5, w6] → 7 tokens, with sample
    positions at 0, 1, 2, 3, 4, 5. Windowing must fire and each REPLICA
    word must receive exactly N samples (i.e., exactly one entry in
    log_p_samples_nats with the right shape).
    """
    lm = _make_mcword(n_samples=4, max_iter=2, chunk_size=4, n_positions=8)
    stim = _make_stim(["a", "b", "c", "d", "e", "f"])
    outs = lm(stim)
    assert len(outs) == 6
    for i, fmo in enumerate(outs):
        samples = fmo.values["log_p_samples_nats"]
        h = fmo.values["mc_word_entropy"]
        # Word 0 might be NaN if first sample position lacks left context.
        if math.isnan(h):
            continue
        # Each word has exactly N=4 samples: windowing neither drops nor
        # double-counts it.
        assert samples.shape == (4,), (
            f"word {i}: expected (4,) got {samples.shape}, windowing duplicated or dropped"
        )


def test_mcword_windowed_round_trip_aggregation():
    """Stored ``mc_word_entropy`` must equal the Shannon mean derived
    from the persisted ``log_p_samples_nats``."""
    lm = _make_mcword(n_samples=8, max_iter=2, chunk_size=4, n_positions=8)
    stim = _make_stim(["a", "b", "c", "d", "e"])
    outs = lm(stim)
    for fmo in outs:
        h = fmo.values["mc_word_entropy"]
        samples = fmo.values["log_p_samples_nats"]
        if math.isnan(h):
            continue
        # alpha=1 case, direct Shannon
        expected = float(-samples.mean() / math.log(2))
        assert abs(h - expected) < 1e-9


@pytest.mark.parametrize("chunk_size", [1, 7, 16])
def test_mcword_long_stimulus_preserves_samples_across_chunks(chunk_size):
    """Later sample chunks must not repeat the first chunk in every window."""
    stimulus = _make_stim([f"w{i}" for i in range(20)])
    options = dict(n_samples=32, max_iter=3, seed=7, n_positions=12)
    complete = _make_mcword(**options)(stimulus)
    chunked = _make_mcword(**options, chunk_size=chunk_size)(stimulus)
    expected = np.stack([out.values["log_p_samples_nats"] for out in complete])
    actual = np.stack([out.values["log_p_samples_nats"] for out in chunked])
    assert actual.shape == (20, 32)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual[:, :16], actual[:, 16:])
    np.testing.assert_array_equal(
        [out.values["mc_word_entropy"] for out in chunked],
        [out.values["mc_word_entropy"] for out in complete],
    )


def test_mcword_windowed_offset_matches_single_pass_samples():
    """A context-independent model preserves keyed draws across window layouts."""
    lm = _make_mcword(n_samples=32, max_iter=3, seed=7, n_positions=12)
    stimulus = _make_stim([f"w{i}" for i in range(20)])
    tokens, _, positions, _ = lm._prepare_word_sampling(stimulus, "words")
    complete = lm._estimate_words_one_pass(tokens, positions, 32)
    windowed = lm._estimate_words_one_pass_windowed(
        tokens, positions, 11, 12, sample_offset=13
    )
    assert windowed.keys() == complete.keys()
    for word, samples in windowed.items():
        np.testing.assert_array_equal(samples, complete[word][13:24])


def test_cont_entropy_windowed_covers_all_words_exactly_once():
    """Same invariant for ContinuationEntropyLM."""
    lm = _make_cont(n_samples=4, max_tokens=2, chunk_size=4, n_positions=8)
    stim = _make_stim(["a", "b", "c", "d", "e", "f"])
    outs = lm(stim)
    assert len(outs) == 6
    for _i, fmo in enumerate(outs):
        samples = fmo.values["log_p_samples_nats"]
        h = fmo.values["continuation_entropy"]
        if math.isnan(h):
            continue
        assert samples.shape == (4,)


def test_windowed_handles_long_stim_without_overflow():
    """20-word stim with n_positions=8 (so windowing fires multiple
    times): the driver must complete without crashing and produce
    exactly one (N,)-shaped sample array per word. Regression guard
    against off-by-one errors in the half-window boundary loop."""
    lm = _make_mcword(n_samples=4, max_iter=2, chunk_size=4, n_positions=8)
    words = [f"w{i}" for i in range(20)]
    stim = _make_stim(words)
    outs = lm(stim)
    assert len(outs) == 20
    n_with_samples = 0
    for fmo in outs:
        samples = fmo.values["log_p_samples_nats"]
        if samples.size > 0:
            assert samples.shape == (4,)
            n_with_samples += 1
    # Most words should have samples (only the BOS-context word might
    # be skipped depending on dispatch). With BOS prepended the
    # first-word's predicting position is BOS, so entropy is defined.
    assert n_with_samples >= 19


def test_cont_entropy_windowed_handles_long_stim():
    """ContinuationEntropy: 15-word stim, ctx=8, max_tokens=2 →
    half_size = 2. Multiple half-windows expected."""
    lm = _make_cont(n_samples=4, max_tokens=2, chunk_size=4, n_positions=8)
    words = [f"w{i}" for i in range(15)]
    stim = _make_stim(words)
    outs = lm(stim)
    assert len(outs) == 15
    n_with_samples = sum(1 for fmo in outs if fmo.values["log_p_samples_nats"].size > 0)
    assert n_with_samples >= 14


@pytest.mark.parametrize("cls", [MCWordEntropyLM, ForwardWordLookaheadEntropyLM])
def test_literal_start_token_keeps_word_sampling_rule_without_added_prefix(cls):
    class Tokenizer(_Tokenizer):
        vocab_size = 5
        _id_to_tok = {0: "<bos>", 1: "Ġa", 2: "Ġb", 3: "x", 4: "y"}

        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            result["input_ids"] = [0 if w == "<bos>" else 1 for w in text.split()]
            return result

    outputs = []
    for add_bos, text in ((False, "<bos> b"), (True, "b")):
        model = build_estimator(
            cls,
            _UniformLM(vocab_size=5),
            tokenizer=Tokenizer(),
            name="word_start",
            n_samples=16,
            add_bos=add_bos,
            max_iter=2,
            max_words=1,
            alpha=1.0,
        )
        model.release_rng_stream = True
        outputs.append(model(word_stimulus(text, "start"))[-1])
    assert outputs[0].metadata["start_ids"] == []
    assert outputs[1].metadata["start_ids"] == [0]
    for field, expected in outputs[1].values.items():
        np.testing.assert_array_equal(outputs[0].values[field], expected)
