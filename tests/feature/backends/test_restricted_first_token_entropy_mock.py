"""Fast unit tests for the restricted first-token entropy (Clark et al. (2025)).

These tests use mock components (no HuggingFace download) so they run on
CPU in seconds. They exercise the closed-form computation directly on
synthetic log-prob arrays and assert REPLICA helper produces the same
values as a hand-coded restricted-distribution entropy.
"""

import math

import numpy as np
import pytest
import torch

from replica.feature.backends.huggingface import HuggingFaceLM
from tests._stimuli import word_stimulus as _build_stim
from tests.feature.entropy._entropy_mocks import WhitespaceTokenizer

# ── Minimal tokenizer-and-model stubs ───────────────────────────────────────


class _MockTokenizer(WhitespaceTokenizer):
    """8-token vocab: 1 BOS + 3 subword + 4 word-starter (Ġ-prefix)."""

    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0
    vocab_size = 8
    name_or_path = "mock"

    _id_to_tok = {
        0: "<bos>",
        1: "a",
        2: "b",
        3: "c",  # subwords
        4: "Ġd",
        5: "Ġe",
        6: "Ġf",
        7: "Ġg",  # word starters
    }
    _tok_to_id = {v: k for k, v in _id_to_tok.items()}

    def __call__(self, text, **_):
        # Trivial char-based tokenization for testing: each char maps to
        # an id via the inverse vocab; default to subword id 1 if unknown.
        ids = []
        offsets = []
        i = 0
        for _w_idx, w in enumerate(text.split()):
            # word-starter for non-initial words
            tok = "Ġ" + w[:1]
            tid = self._tok_to_id.get(tok, 4)
            start = text.index(w, i)
            ids.append(tid)
            offsets.append((start, start + len(w)))
            i = start + len(w)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


class _LinearLM:
    """Returns a fixed per-position logit pattern (no params trained).
    Useful for verifying restricted-entropy bookkeeping without a real LM."""

    class _Cfg:
        n_positions = 1024
        max_position_embeddings = 1024
        vocab_size = 8

    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.config = self._Cfg()

    def to(self, *_):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, **_):
        b, t = input_ids.shape
        # Each position gets a deterministic logit pattern based on the
        # input id at that position. Keeps test outputs predictable.
        torch.manual_seed(int(input_ids.sum().item()) + 17)
        logits = torch.randn(b, t, _LinearLM._Cfg.vocab_size)

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.past_key_values = ()
        return out


def _build_lm(*, alpha: float | None = None) -> HuggingFaceLM:
    lm = HuggingFaceLM.__new__(HuggingFaceLM)
    lm.token_entropy_sum = False
    lm.name = "mock"
    lm.aggregation = "sum"
    lm.device = "cpu"
    lm.provide_word_dist = False
    lm.add_bos = True
    lm.token_level_entropy = False
    lm.token_renyi_alphas = []
    lm.restricted_first_token_entropy = alpha is None
    lm.restricted_first_token_renyi_alphas = [] if alpha is None else [float(alpha)]
    lm._model = _LinearLM()
    lm._tokenizer = _MockTokenizer()
    lm._vocab = None
    lm._word_to_token_ids = None
    lm._space_ids_np = None
    lm._subword_ids_np = None
    return lm


# ── Tests ──────────────────────────────────────────────────────────────────


def test_word_starter_masks_partition_vocab_correctly():
    lm = _build_lm(alpha=None)
    lm._ensure_word_starter_masks()
    space = set(lm._space_ids_np.tolist())
    sub = set(lm._subword_ids_np.tolist())
    # 4..7 are Ġ-prefix word starters; 0..3 are subwords.
    assert space == {4, 5, 6, 7}
    assert sub == {0, 1, 2, 3}
    # Disjoint union over the whole vocab.
    assert space.isdisjoint(sub)
    assert space | sub == set(range(_MockTokenizer.vocab_size))


def test_shannon_matches_hand_computed_restricted_entropy():
    """Verify that for a fixed seed + small stim, REPLICA restricted_first_token_entropy
    matches a hand-rolled restricted-Shannon computation that uses the
    same log-probs from the same forward pass."""
    lm = _build_lm(alpha=None)
    stim = _build_stim("apple banana cherry")
    # Need to invoke the LM once so the same log-probs can be read back.
    outs = lm(stim)

    # Reproduce the forward pass independently for verification.
    tokenizer = lm._tokenizer
    enc = tokenizer(stim.text)
    ids = [tokenizer.bos_token_id] + enc["input_ids"]
    with torch.no_grad():
        out = lm._model(torch.tensor([ids]))
    log_probs = torch.log_softmax(out.logits[0], dim=-1).double()
    space_ids = lm._space_ids_np
    subword_ids = lm._subword_ids_np
    ln2 = math.log(2.0)

    def restricted_shannon_bits(pred_pos: int, allowed) -> float:
        lp = log_probs[pred_pos].numpy()
        lp_r = lp[allowed]
        m = float(lp_r.max())
        log_z = m + float(np.log(np.sum(np.exp(lp_r - m))))
        lp_norm = lp_r - log_z
        p_norm = np.exp(lp_norm)
        return float(-np.sum(p_norm * lp_norm) / ln2)

    # Word 0 prefix is just BOS, predicting position is 0 (BOS pos).
    # First content token after BOS is a subword (per the upstream reference): use
    # subword_ids.
    h_word0_expected = restricted_shannon_bits(0, subword_ids)
    # Words 1, 2 predict at first_subword - 1 positions; both are not
    # BOS positions → use space_ids.
    h_word1_expected = restricted_shannon_bits(1, space_ids)
    h_word2_expected = restricted_shannon_bits(2, space_ids)

    assert (
        abs(outs[0].values["restricted_first_token_entropy"] - h_word0_expected) < 1e-9
    )
    assert (
        abs(outs[1].values["restricted_first_token_entropy"] - h_word1_expected) < 1e-9
    )
    assert (
        abs(outs[2].values["restricted_first_token_entropy"] - h_word2_expected) < 1e-9
    )


def test_renyi_half_matches_hand_computed_restricted():
    lm = _build_lm(alpha=0.5)
    stim = _build_stim("apple banana cherry")
    outs = lm(stim)

    tokenizer = lm._tokenizer
    enc = tokenizer(stim.text)
    ids = [tokenizer.bos_token_id] + enc["input_ids"]
    with torch.no_grad():
        out = lm._model(torch.tensor([ids]))
    log_probs = torch.log_softmax(out.logits[0], dim=-1).double()
    space_ids = lm._space_ids_np
    subword_ids = lm._subword_ids_np
    ln2 = math.log(2.0)

    def restricted_renyi_bits(pred_pos: int, allowed, alpha=0.5) -> float:
        lp = log_probs[pred_pos].numpy()
        lp_r = lp[allowed]
        # renormalise
        m = float(lp_r.max())
        log_z = m + float(np.log(np.sum(np.exp(lp_r - m))))
        lp_norm = lp_r - log_z
        scaled = alpha * lp_norm
        m2 = float(scaled.max())
        log_sum = m2 + float(np.log(np.sum(np.exp(scaled - m2))))
        return float(log_sum / ((1.0 - alpha) * ln2))

    expected = [
        restricted_renyi_bits(0, subword_ids),
        restricted_renyi_bits(1, space_ids),
        restricted_renyi_bits(2, space_ids),
    ]
    for i, exp in enumerate(expected):
        got = outs[i].values["restricted_first_token_renyi_entropy_alpha0.5"]
        assert abs(got - exp) < 1e-9, f"word {i}: got={got}, expected={exp}"


def test_renyi_half_greater_than_shannon_per_word():
    """Mathematical identity: Rényi alpha=0.5 ≥ Shannon alpha=1 on the same dist."""
    lm_s = _build_lm(alpha=None)
    lm_r = _build_lm(alpha=0.5)
    stim = _build_stim("apple banana cherry date elder fig")
    outs_s = lm_s(stim)
    outs_r = lm_r(stim)
    for fs, fr in zip(outs_s, outs_r, strict=False):
        h_s = fs.values.get("restricted_first_token_entropy")
        h_r = fr.values.get("restricted_first_token_renyi_entropy_alpha0.5")
        if h_s is None or h_r is None:
            continue
        assert h_r >= h_s - 1e-9, f"Rényi alpha=0.5 ({h_r}) < Shannon ({h_s})"


def test_returns_nan_when_no_left_context():
    """add_bos=False and word 0 has no prefix, so its entropy must be NaN."""
    lm = _build_lm(alpha=None)
    lm.add_bos = False
    stim = _build_stim("apple banana cherry")
    outs = lm(stim)
    # word 0 has pred_pos = -1 → NaN
    assert math.isnan(outs[0].values["restricted_first_token_entropy"])
    # word 1, 2 have left context
    assert math.isfinite(outs[1].values["restricted_first_token_entropy"])
    assert math.isfinite(outs[2].values["restricted_first_token_entropy"])


def test_extract_keys_filter_disables_computation():
    """``extract`` parameter controls which keys are computed. If neither
    restricted_first_token_entropy nor restricted_first_token_renyi_entropy is requested, the
    masks should not be built (lazy)."""
    lm = _build_lm(alpha=None)
    lm.restricted_first_token_entropy = True
    stim = _build_stim("apple banana cherry")
    # extract=[] means "compute nothing extra"
    outs = lm(stim, extract=["log_p_observed"])
    # restricted_first_token_entropy was not in extract → not present in outputs.
    for fmo in outs:
        assert "restricted_first_token_entropy" not in fmo.values
    # masks should have been left None, lazy build
    assert lm._space_ids_np is None
    assert lm._subword_ids_np is None


def test_no_space_prefix_vocab_raises():
    """The mask helper raises with a useful message when the tokenizer
    has no Ġ-prefix tokens in its vocab."""

    class _NoSpaceTok:
        vocab_size = 4

        def convert_ids_to_tokens(self, i):
            return {0: "a", 1: "b", 2: "c", 3: "d"}[int(i)]

    lm = HuggingFaceLM.__new__(HuggingFaceLM)
    lm.token_entropy_sum = False
    lm.name = "stub"
    lm._tokenizer = _NoSpaceTok()
    lm._space_ids_np = None
    lm._subword_ids_np = None
    with pytest.raises(ValueError, match="no 'Ġ'-prefix"):
        lm._ensure_word_starter_masks()
