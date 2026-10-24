"""Verify ``HuggingFaceLM(restricted_first_token_entropy=...)`` against Clark et al. (2025)'s
upstream ``first_token_entropy.py`` (default *restricted* mode).

The upstream reference renormalises each position's next-token logits over
the word-starter ids (``Ġ``-prefix tokens), then computes Shannon /
Rényi-alpha from the renormalised distribution. The position right after
BOS is the special case where the first content token does not carry a
``Ġ`` marker, the upstream reference there restricts to ``subword_ixs`` instead.

What this file tests:
  * Port the relevant slice of the upstream ``main()`` as ``_clark_restricted``
    (single-window, no half-window splitting, applicable for stim that
    fits in the context).
  * Run REPLICA driver on the same small stim with the same model.
  * Assert per-word values match to floating-point precision (same
    logits, same closed-form formula on both sides).

Also tests algorithmic invariants (positive, finite, Rényi ≥ Shannon)
and the BOS branch (the very first word uses ``subword_ids``, all
others use ``space_ids``).

Marked ``@pytest.mark.slow`` because it loads gpt2; runs in GPU CI.
"""

import math

import pytest
import torch

from replica.core import Stimulus
from replica.feature.backends.huggingface import HuggingFaceLM
from tests._stimuli import word_stimulus

# A bare ``pytestmark = X`` after another bare assignment silently overwrites
# the first; both markers must live in one list.
pytestmark = [pytest.mark.needs_model, pytest.mark.slow]


@pytest.fixture(scope="module")
def small_stimulus() -> Stimulus:
    # No trailing punctuation so each REPLICA word maps to a single Ġ-prefix
    # token; that way the upstream reference's per-token output positions and REPLICA
    # per-word outputs are 1-to-1 comparable.
    return word_stimulus("the cat sat on the mat", "test")


# ── Upstream reference port (single-window, restricted) ─────────────────────


def _get_space_subword_idx(tokenizer):
    """Port of the upstream ``get_space_subword_idx``, vocab partition by
    leading ``Ġ``.  Returns ``(space_ids_tensor, subword_ids_tensor)`` on
    CPU as ``int64`` (matches REPLICA helper).
    """
    space_idx, subword_idx = [], []
    for i in range(tokenizer.vocab_size):
        tok = tokenizer.convert_ids_to_tokens(i)
        if tok and tok.startswith("Ġ"):
            space_idx.append(i)
        else:
            subword_idx.append(i)
    return (
        torch.tensor(space_idx, dtype=torch.long),
        torch.tensor(subword_idx, dtype=torch.long),
    )


def _clark_restricted(
    model,
    tokenizer,
    story_text: str,
    *,
    alpha: float = 1.0,
):
    """Faithful port of the relevant body of the upstream
    ``first_token_entropy.main``, restricted-mode branch, single window
    (no half-window split). Returns per-position entropy values *in
    bits* aligned with the tokenized output_ids of the story.

    Output convention: returns one entry per output token. The entry is
    the restricted entropy at that token's *predicting* position. Caller
    extracts per-word values by indexing the first subword of each word.
    """
    space_ixs, subword_ixs = _get_space_subword_idx(tokenizer)
    bos_id = tokenizer.bos_token_id
    enc = tokenizer(story_text, add_special_tokens=False)
    ids = [bos_id] + enc["input_ids"]  # BOS-prepended
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids[:-1]], device=device)  # window input
    # output_ids covers positions 1..end (positions being predicted).
    output_ids = ids[1:]

    with torch.no_grad():
        out = model(input_ids=input_ids)
    # logits[i] predicts ids[i+1]; back to CPU so the CPU vocab-partition
    # index tensors (space_ixs/subword_ixs) can index it on any device.
    logits = out.logits.squeeze(0).double().cpu()  # (T_in, V)

    eps = float(torch.finfo(torch.float64).eps)
    # first window path, position 0 uses subword_ixs (predicts first
    # content token, which is not Ġ-prefix); positions 1..T_in-1 use
    # space_ixs.
    bos_logits = logits[0, subword_ixs]
    bos_probs = torch.softmax(bos_logits, dim=-1) + eps
    if alpha == 1.0:
        bos_entropy = float((-bos_probs * torch.log2(bos_probs)).sum())
    else:
        bos_entropy = float(torch.log2((bos_probs**alpha).sum()) / (1.0 - alpha))

    rest_logits = logits[1:, space_ixs]
    rest_probs = torch.softmax(rest_logits, dim=-1) + eps
    if alpha == 1.0:
        rest_entropies = (-rest_probs * torch.log2(rest_probs)).sum(dim=1)
    else:
        rest_entropies = torch.log2((rest_probs**alpha).sum(dim=1)) / (1.0 - alpha)
    rest_entropies = rest_entropies.tolist()
    return [bos_entropy, *rest_entropies], ids, output_ids


# ── Tests ───────────────────────────────────────────────────────────────────


def _build_lm(*, alpha: float | None = None) -> HuggingFaceLM:
    if alpha is None:
        return HuggingFaceLM(
            model_name="gpt2",
            add_bos=True,
            provide_word_dist=False,
            restricted_first_token_entropy=True,
        )
    return HuggingFaceLM(
        model_name="gpt2",
        add_bos=True,
        provide_word_dist=False,
        restricted_first_token_renyi_alphas=[alpha],
    )


def _clark_per_word_entropy(
    model,
    tokenizer,
    alignment_mapping_first_t,
    bos_offset,
    ids,
    output_ids,
    ref_entropies,
):
    """Index the upstream reference's output by the first subword of each REPLICA word."""
    out = []
    for first_t in alignment_mapping_first_t:
        # REPLICA first_t is the position of the word's first subword in the
        # original (un-BOS-prepended) token stream. Map to ids index by
        # adding bos_offset; the predicting position is one before that.
        pred_pos = first_t + bos_offset - 1
        if pred_pos < 0:
            out.append(None)
        else:
            out.append(float(ref_entropies[pred_pos]))
    return out


@pytest.mark.parametrize(
    "lm_alpha,reference_alpha,key",
    [
        (None, 1.0, "restricted_first_token_entropy"),
        (0.5, 0.5, "restricted_first_token_renyi_entropy_alpha0.5"),
    ],
    ids=["shannon", "renyi_half"],
)
def test_matches_clark_reference(
    small_stimulus: Stimulus, lm_alpha, reference_alpha, key
):
    lm = _build_lm(alpha=lm_alpha)
    # Trigger model load so the tokenizer is available + reference can run.
    outs = lm(small_stimulus)
    model = lm._model
    tokenizer = lm._tokenizer

    # Get REPLICA alignment first-subword positions to align indexing.
    from replica.feature import align_from_tokenizer

    alignment = align_from_tokenizer(
        small_stimulus,
        tokenizer,
        source_layer="words",
    )
    first_ts = [
        alignment.mapping[i][0] if alignment.mapping[i] else None
        for i in range(len(alignment.mapping))
    ]

    ref_entropies, ids, output_ids = _clark_restricted(
        model,
        tokenizer,
        small_stimulus.text,
        alpha=reference_alpha,
    )

    replica_vals = [fmo.values.get(key) for fmo in outs]
    clark_vals = _clark_per_word_entropy(
        model,
        tokenizer,
        first_ts,
        bos_offset=1,
        ids=ids,
        output_ids=output_ids,
        ref_entropies=ref_entropies,
    )

    assert len(replica_vals) == len(clark_vals)
    for i, (g, c) in enumerate(zip(replica_vals, clark_vals, strict=False)):
        if c is None:
            assert g is None or math.isnan(g)
        else:
            assert g is not None and math.isfinite(g)
            # REPLICA computes log_softmax in float32 then upcasts; the upstream
            # reference is float64, so they agree to float32 precision (~1e-6);
            # a distribution/restriction bug would exceed 1e-3 (cf. the
            # space-vs-subword guard in the BOS-branch test).
            assert abs(g - c) < 1e-5, f"word {i}: REPLICA={g}  clark={c}  diff={g - c}"


def test_bos_branch_uses_subword_ids_for_first_word(small_stimulus: Stimulus):
    """The very first word's predicting position is BOS, and the first
    content token after BOS is not a Ġ-prefix token. The upstream reference
    therefore restricts to ``subword_ixs`` for that position. REPLICA
    must match: comparing the word[0] entropy to a restriction
    over ``space_ids`` should disagree (different distributions); against
    ``subword_ids`` it should agree to float tolerance."""
    lm = _build_lm(alpha=None)
    outs = lm(small_stimulus)
    model = lm._model
    tokenizer = lm._tokenizer
    space_ids, subword_ids = _get_space_subword_idx(tokenizer)

    enc = tokenizer(small_stimulus.text, add_special_tokens=False)
    ids = [tokenizer.bos_token_id] + enc["input_ids"]
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids[:-1]], device=device))
    logits0 = out.logits.squeeze(0).double().cpu()[0]  # predicts position 1
    eps = float(torch.finfo(torch.float64).eps)

    def shannon_bits(allowed):
        lg = logits0[allowed]
        p = torch.softmax(lg, dim=-1) + eps
        return float((-p * torch.log2(p)).sum())

    h_subword = shannon_bits(subword_ids)
    h_space = shannon_bits(space_ids)
    replica_first = outs[0].values["restricted_first_token_entropy"]

    # float32 (REPLICA) vs float64 (reference) softmax agree to float32 precision
    # (~1e-6); the space-vs-subword guard below (> 1e-3) checks the branch.
    assert abs(replica_first - h_subword) < 1e-5, (
        "REPLICA first-word entropy must use subword_ids (BOS branch); "
        f"got {replica_first}  vs subword-restricted {h_subword}"
    )
    # And it should not equal the space-restricted version.
    assert abs(replica_first - h_space) > 1e-3, (
        "REPLICA first-word entropy unexpectedly equals space-restricted "
        "entropy; the BOS branch is not active."
    )
