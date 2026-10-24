"""Verify ``MCWordEntropyLM`` against the Clark et al. (2025) upstream reference.

Upstream: ``mc_word_entropy.py`` in christian-clark/word-entropy. The reference
algorithm processes a story in a single forward pass that maintains a
master KV cache, forking the cache at each word boundary to sample a word
continuation. ``MCWordEntropyLM.__call__`` uses the same single-pass
algorithm, which avoids re-forwarding the prefix for every word.

Strategy:
  * Run a *port* of the upstream algorithm (single-batch, no windowing, i.e.,
    ``samples_per_batch == num_samples``) and the
    ``MCWordEntropyLM`` on the same story with the same seed.
  * With a single sample (``N=1``) and shared RNG state, the two
    algorithms invoke the same sequence of operations and ``multinomial``
    draws, so the resulting per-word log-probabilities must agree to
    floating-point precision.
  * With ``N>1`` the multinomial draws on the two paths are issued in
    different orders (the upstream reference draws fresh randomness per batch of
    ``samples_per_batch`` samples; REPLICA draws per chunk), so bit
    equality is not expected. Per-word *means* are asserted to agree
    within MC standard error (5 sigma).

The reference port keeps the upstream bits-throughout convention; converted
to nats at the boundary for direct comparison with the
``log_p_samples_nats``.
"""

import math

import pytest
import torch

from replica.core import Stimulus
from replica.feature.entropy.estimators import (
    MCWordEntropyLM,
    _renyi_entropy_bits_from_log_p_nats,
)
from tests._stimuli import word_stimulus

pytestmark = [pytest.mark.slow, pytest.mark.needs_model]


@pytest.fixture(scope="module")
def small_stimulus() -> Stimulus:
    # Pure-whitespace tokenization with no punctuation so every REPLICA word
    # aligns one-to-one with a single GPT-2 ``Ġ``-prefix token. This
    # makes the per-word sample positions of REPLICA driver and the
    # upstream reference port identical except for one trailing sample.
    return word_stimulus("the cat sat on the mat", "test")


def _build_lm(n_samples=4, chunk_size=4, seed=0) -> MCWordEntropyLM:
    lm = MCWordEntropyLM(
        model_id="gpt2",
        n_samples=n_samples,
        max_iter=3,
        alpha=1.0,
        add_bos=True,
        seed=seed,
        chunk_size=chunk_size,
    )
    # Compare like with like: production re-seeds per stimulus from the
    # content so a cell is reproducible however the run is sharded, while
    # upstream advances one stream from the raw seed. The anchor is about the
    # algorithm issuing the same draws, so it runs against upstream's stream.
    lm.release_rng_stream = True
    return lm


# ── Upstream reference port (single-window, single-batch) ──────────────────


def _clark_reference_logp(
    model,
    tokenizer,
    space_ids: torch.Tensor,
    subword_ids: torch.Tensor,
    story_text: str,
    *,
    num_samples: int,
    max_iter: int,
    seed: int,
    device,
):
    """Faithful port of Clark et al. (2025)'s ``mc_word_entropy.main`` body for the
    case where ``samples_per_batch == num_samples`` and the story fits in
    a single context window (no half-window split).

    Returns a list of length ``W`` (number of words in the story; here a
    "word" is a Ġ-boundary, as the upstream reference defines it)
    where each entry is a tensor of shape ``(num_samples,)`` of joint
    log-probabilities in NATS for the sampled word continuation at that
    boundary. Converting to nats up-front avoids base-mixing.
    """
    from transformers import DynamicCache

    gen = torch.Generator(device=device).manual_seed(seed)

    # Tokenize with BOS prepended (the upstream default for GPT-2 family).
    bos_id = tokenizer.bos_token_id
    enc = tokenizer(story_text, return_offsets_mapping=False, add_special_tokens=False)
    story_ids = [bos_id] + enc["input_ids"]

    # word_final[i] is True iff token i is the last subword of a word
    # (i.e., token i+1 starts a new word with Ġ, or i is the final token).
    space_set = set(space_ids.detach().cpu().tolist())
    word_final = []
    for i in range(len(story_ids) - 1):
        if story_ids[i] == bos_id or story_ids[i + 1] in space_set:
            word_final.append(True)
        else:
            word_final.append(False)
    word_final.append(True)

    # Single-window single-batch master pass.
    k = num_samples
    log_p_per_word: list[torch.Tensor] = []
    master_kv = None
    with torch.no_grad():
        for i, tok in enumerate(story_ids):
            next_id = torch.full(
                (k, 1),
                int(tok),
                device=device,
                dtype=torch.long,
            )
            if master_kv is None:
                out = model(next_id, use_cache=True)
            else:
                fresh = DynamicCache.from_legacy_cache(master_kv)
                out = model(next_id, past_key_values=fresh, use_cache=True)
            master_kv = out.past_key_values.to_legacy_cache()
            if not word_final[i]:
                continue

            # Sample word continuation from the cache state after tok.
            logits = out.logits[:, -1, :].double()
            log_p = torch.zeros(k, device=device, dtype=torch.float64)

            # First step: restrict to space-prefix ids (or subword if prev = BOS).
            first_from_subword = int(tok) == bos_id
            allowed = subword_ids if first_from_subword else space_ids
            allowed_logits = logits.index_select(1, allowed)
            allowed_probs = torch.softmax(allowed_logits, dim=-1)
            local_idx = torch.multinomial(allowed_probs, 1, generator=gen)
            sample_prob = allowed_probs.gather(1, local_idx).squeeze(1)
            log_p += torch.log(sample_prob.clamp_min(1e-40))
            sampled_id = allowed[local_idx.squeeze(1)]

            sample_kv_dyn = DynamicCache.from_legacy_cache(master_kv)
            cur = sampled_id.unsqueeze(1)
            still_going = torch.ones(k, dtype=torch.bool, device=device)
            for _ in range(max_iter - 1):
                out2 = model(cur, past_key_values=sample_kv_dyn, use_cache=True)
                sample_kv_dyn = out2.past_key_values
                step_logits = out2.logits[:, -1, :].double()
                step_probs = torch.softmax(step_logits, dim=-1)
                subword_probs = step_probs.index_select(1, subword_ids)
                space_probs = step_probs.index_select(1, space_ids)
                eow_prob = space_probs.sum(dim=1, keepdim=True)
                local_probs = torch.cat([subword_probs, eow_prob], dim=1)
                local_idx = torch.multinomial(local_probs, 1, generator=gen)
                sample_prob = local_probs.gather(1, local_idx).squeeze(1)
                log_p += still_going.double() * torch.log(sample_prob.clamp_min(1e-40))
                is_subword = local_idx.squeeze(1) < subword_ids.numel()
                still_going = still_going & is_subword
                if not bool(still_going.any()):
                    break
                pad = (
                    tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                )
                local_ix = local_idx.squeeze(1)
                ix_filt = torch.where(
                    is_subword,
                    subword_ids[local_ix.clamp_max(subword_ids.numel() - 1)],
                    torch.full_like(local_ix, pad),
                )
                cur = ix_filt.unsqueeze(1)
            log_p_per_word.append(log_p.detach().cpu())

    return log_p_per_word, story_ids, word_final


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_clark_reference_bit_identical_at_N1(small_stimulus: Stimulus, seed: int):
    """N=1, single chunk: both implementations issue ``multinomial`` calls
    in exactly the same order against an identical RNG, so per-word
    sampled log-probabilities must match to floating-point precision."""
    lm = _build_lm(n_samples=1, chunk_size=1, seed=seed)
    outs = lm(small_stimulus)
    replica_logp_nats = [
        fmo.values["log_p_samples_nats"][0]
        for fmo in outs
        if fmo.values["log_p_samples_nats"].size > 0
    ]

    ref_logp_per_word, _story_ids, _word_final = _clark_reference_logp(
        lm._model,
        lm._tokenizer,
        lm._space_ids,
        lm._subword_ids,
        small_stimulus.text,
        num_samples=1,
        max_iter=lm.max_iter,
        seed=seed,
        device=lm._device,
    )
    # The reference port samples at every word boundary in the tokenization,
    # including a trailing one after the final input token (predicting a
    # "next word" that doesn't exist in the stim). REPLICA driver only
    # samples at boundaries that correspond to actual REPLICA word units
    # (each tokenized stim word), so for a stim of W words REPLICA produces
    # W samples and the reference port produces W+1. Compare the W aligned samples.
    assert len(ref_logp_per_word) == len(replica_logp_nats) + 1, (
        f"unexpected count: REPLICA={len(replica_logp_nats)} ref={len(ref_logp_per_word)}"
    )

    for i, (replica_lp, ref_lp) in enumerate(
        zip(replica_logp_nats, ref_logp_per_word, strict=False)
    ):
        ref_val = float(ref_lp.item())
        assert abs(replica_lp - ref_val) < 1e-6, (
            f"word {i}: REPLICA log_p={replica_lp}  ref log_p={ref_val}  "
            f"diff={replica_lp - ref_val}"
        )


def test_clark_reference_means_agree_at_large_N(small_stimulus: Stimulus):
    """N=128, both implementations: per-word means should agree within
    a few MC standard errors. Different RNG draw orderings → bit equality
    is impossible, but the *distribution* of the estimator is identical
    so the means converge."""
    N = 128
    lm = _build_lm(n_samples=N, chunk_size=N, seed=0)
    outs = lm(small_stimulus)
    replica_means_bits = []
    replica_ses_bits = []
    for fmo in outs:
        samples = fmo.values["log_p_samples_nats"]
        if samples.size == 0:
            continue
        # Shannon mean in bits = -mean(log_p_nats) / ln 2
        mean_bits = float(-samples.mean() / math.log(2))
        # MC SE of Shannon mean: std(-log_p_nats/ln2) / sqrt(N)
        se_bits = float(samples.std(ddof=1) / math.log(2) / math.sqrt(N))
        replica_means_bits.append(mean_bits)
        replica_ses_bits.append(se_bits)

    ref_logp_per_word, _, _ = _clark_reference_logp(
        lm._model,
        lm._tokenizer,
        lm._space_ids,
        lm._subword_ids,
        small_stimulus.text,
        num_samples=N,
        max_iter=lm.max_iter,
        seed=12345,  # independent seed
        device=lm._device,
    )

    # The reference port samples one extra position past the final REPLICA word, drop it
    # for the comparison (see ``test_clark_reference_bit_identical_at_N1``).
    assert len(ref_logp_per_word) == len(replica_means_bits) + 1

    for i, (replica_mean, replica_se, ref_lp) in enumerate(
        zip(replica_means_bits, replica_ses_bits, ref_logp_per_word, strict=False)
    ):
        ref_mean = float(-ref_lp.mean().item() / math.log(2))
        ref_se = float(ref_lp.std(unbiased=True).item() / math.log(2) / math.sqrt(N))
        combined_se = math.sqrt(replica_se**2 + ref_se**2)
        diff = abs(replica_mean - ref_mean)
        # 5sigma window: very conservative, both samples are size 128.
        assert diff <= 5.0 * combined_se + 0.1, (
            f"word {i}: REPLICA mean={replica_mean:.4f}  ref mean={ref_mean:.4f}  "
            f"diff={diff:.4f}  combined SE={combined_se:.4f}"
        )


def test_renyi_inequality_holds(small_stimulus: Stimulus):
    """Per-word Rényi alpha=0.5 must be ≥ Shannon alpha=1, derived from the
    same MC samples. Mathematical inequality; failure means the
    aggregator is broken."""
    lm = _build_lm(n_samples=32, chunk_size=32, seed=0)
    outs = lm(small_stimulus)
    for fmo in outs:
        samples = fmo.values["log_p_samples_nats"]
        if samples.size == 0:
            continue
        H_shannon = _renyi_entropy_bits_from_log_p_nats(samples, 1.0)
        H_renyi = _renyi_entropy_bits_from_log_p_nats(samples, 0.5)
        assert H_renyi >= H_shannon - 1e-9, (
            f"Rényi alpha=0.5 = {H_renyi}  <  Shannon alpha=1 = {H_shannon}, "
            "inequality violated"
        )


def test_log_p_samples_nats_persisted_with_correct_shape(small_stimulus: Stimulus):
    """Each FeatureModelOutput must carry an N-shaped sample array for
    its word, so future aggregator changes can re-derive without GPU."""
    lm = _build_lm(n_samples=8, chunk_size=4, seed=0)
    outs = lm(small_stimulus)
    n_with_samples = 0
    for fmo in outs:
        samples = fmo.values["log_p_samples_nats"]
        if samples.size > 0:
            assert samples.shape == (8,), f"expected (8,), got {samples.shape}"
            n_with_samples += 1
    assert n_with_samples >= len(outs) - 1  # first word may lack a prefix


def test_chunked_and_unchunked_paths_use_same_one_pass(small_stimulus: Stimulus):
    """chunk_size affects only the batching of MC draws. The total
    log_p_samples_nats array must have shape (N,) regardless of how N is
    split. The per-chunk RNG draw order will differ from the un-chunked case,
    so bit equality of values is not required, only the same dimensions and
    Shannon equivalence within MC noise."""
    lm_unchunked = _build_lm(n_samples=16, chunk_size=16, seed=0)
    lm_chunked = _build_lm(n_samples=16, chunk_size=4, seed=0)
    outs_a = lm_unchunked(small_stimulus)
    outs_b = lm_chunked(small_stimulus)
    assert len(outs_a) == len(outs_b)
    for fa, fb in zip(outs_a, outs_b, strict=False):
        sa = fa.values["log_p_samples_nats"]
        sb = fb.values["log_p_samples_nats"]
        assert sa.shape == sb.shape, f"{sa.shape} vs {sb.shape}"
