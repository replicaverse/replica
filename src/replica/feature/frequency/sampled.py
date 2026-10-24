"""Sampled unigram-frequency estimation (Hopton et al., 2026b).

A word's *unigram* (context-free) probability ``p(w)``, estimated from sentences
sampled out of a language model.

:func:`mmc_estimate` is the marginal Monte Carlo estimator: the autoregressive
joint probability of the word's tokens, summed over every word-boundary position
of every sample, over ``n_samples + n_boundaries``. :func:`mc_estimate` is the
plain Monte Carlo rate of the same quantity.

Plugs into :class:`~replica.feature.UnigramSurprisal` and
:class:`~replica.feature.LogFrequency` via the
:class:`~replica.feature.FrequencySource` protocol. ``samples`` may be supplied
directly or generated from ``model_name`` on first use.
"""

import json
import logging
import math
import pickle
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from ...utils.atomic import atomic_pickle
from ...utils.digest import text_sha256
from ..backends.generation import usable_position_limit, validate_context_window
from ..backends.start_state import resolve_start_state, single_token_prefix
from .estimated import EstimatedUnigramSource

log = logging.getLogger(__name__)

#: Seconds between checkpoint writes while a sampled source primes its cache.
PRIMING_WRITE_INTERVAL = 60.0

# Version 3 invalidates checkpoints written before the stop-token and padding
# fixes; explicitly supplied samples remain caller data.
_PRIMING_FORMAT = "replica/mmc-priming/3"

#: The generation contract a sampling request names, so a shard drawn under an
#: earlier contract is never resumed as this one: rows end at the generation
#: config's end tokens, a drawn pad is kept, the prefix is one token.
GENERATION_CONTRACT = "replica/mmc-generate/3"


def _left_pad_batch(
    encoded: list[list[int]], pad_id: int
) -> tuple[list[list[int]], list[list[int]], int]:
    """Left-pad token-id lists to a rectangle; returns (padded, masks, max_len)."""
    max_len = max(len(ids) for ids in encoded)
    padded, masks = [], []
    for ids in encoded:
        pad = max_len - len(ids)
        padded.append([pad_id] * pad + ids)
        masks.append([0] * pad + [1] * len(ids))
    return padded, masks, max_len


def generation_stop_ids(model, tokenizer) -> frozenset[int]:
    """The ids at which ``generate`` ends a row.

    The model's generation config is what ``generate`` reads, so it decides:
    one id, a list of ids, or ``None`` for no end token at all. The
    tokenizer's end token applies only to a model without a generation config.

    Args:
        model: The sampling model.
        tokenizer: The sampling model's tokenizer.
    """
    config = getattr(model, "generation_config", None)
    if config is None:
        return frozenset({tokenizer.eos_token_id} - {None})
    declared = getattr(config, "eos_token_id", None)
    if declared is None:
        return frozenset()
    if isinstance(declared, (list, tuple)):
        return frozenset(int(i) for i in declared)
    return frozenset({int(declared)})


def _drawn_ids(ids: list[int], tokenizer, stop_ids=None) -> list[int]:
    """The tokens a sample consists of: *ids* up to the first end token.

    ``generate`` pads a finished row after its end token, so the end tokens
    are the only stops; a pad id the model drew before one is part of the
    sample, and a model without an end token keeps every drawn id.

    Args:
        ids: Generated ids after the start prefix.
        tokenizer: The sampling model's tokenizer, for its end id.
        stop_ids: Every id generation ends at, see :func:`generation_stop_ids`;
            ``None`` uses the tokenizer's end id alone.
    """
    stop = set(stop_ids) if stop_ids is not None else {tokenizer.eos_token_id} - {None}
    out: list[int] = []
    for i in ids:
        if i in stop:
            break
        out.append(int(i))
    return out


def start_prefix(tokenizer, start_ids: "Sequence[int] | None" = None) -> list[int]:
    """The one-token prefix a sample is generated and scored from.

    Args:
        tokenizer: The sampling model's tokenizer.
        start_ids: An explicit prefix of exactly one id, as the compute layer
            passes the resolved start state of the model; ``None`` keeps the
            tokenizer's own convention, its bos token, else its eos token.

    Raises:
        ValueError: *start_ids* holds other than one non-negative integer id,
            or no explicit prefix is given and the tokenizer declares neither
            a bos nor an eos token.
    """
    if start_ids is not None:
        return list(validate_start_ids(start_ids))
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        return [int(bos)]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError(
            "Sampling requires a start prefix, a bos token or an eos token"
        )
    return [int(eos)]


def validate_start_ids(start_ids: "Sequence[int]") -> tuple[int, ...]:
    """The canonical form of an explicit prefix: exactly one non-negative integer id.

    Args:
        start_ids: The prefix as the caller gave it.

    Raises:
        ValueError: Other than one id, or an id that is not a non-negative
            integer (a float, a bool or a negative number).
    """
    ids = list(start_ids)
    try:
        return single_token_prefix(ids)
    except ValueError:
        raise ValueError(
            "the start prefix must be exactly one non-negative integer token id, "
            f"got {ids!r}"
        ) from None


def _encode_samples(tokenizer, samples, start_ids: "Sequence[int] | None" = None):
    """Encode sample text after exactly the prefix used for generation."""
    prefix = start_prefix(tokenizer, start_ids)
    return [
        prefix + tokenizer.encode(s, add_special_tokens=False) if s else list(prefix)
        for s in samples
    ]


def _model_start_prefix(tokenizer, start_ids=None, model_name=None) -> list[int]:
    """Resolve the model's prefix, with EOS fallback when BOS is unavailable."""
    resolved = resolve_start_state(tokenizer, model_name, start_ids=start_ids).ids
    return start_prefix(tokenizer, resolved or None)


def _context_limit(model, tokenizer, context_window=None) -> int | None:
    """Resolve the complete-context limit from the tokenizer and model config."""
    config = getattr(model, "config", None)
    name = getattr(tokenizer, "name_or_path", "") or getattr(
        config, "_name_or_path", ""
    )
    return usable_position_limit(str(name), config, context_window=context_window)


def _check_context_length(length: int, context_window: int | None) -> None:
    """Refuse a complete context that cannot fit the selected model window."""
    if context_window is not None and length > context_window:
        raise ValueError(
            f"MMC requires a complete context of {length} tokens, exceeding "
            f"context_window={context_window}; choose a larger supported window"
        )


def _padded_logits(model, padded, masks, device, context_window=None):
    """Forward a left-padded batch with mask-derived ``position_ids``.

    Without them the model positions tokens by ``arange`` over the padded
    rectangle, making probabilities depend on batch composition.
    """
    import torch

    _check_context_length(len(padded[0]), context_window)
    input_ids = torch.tensor(padded, device=device)
    attention_mask = torch.tensor(masks, device=device)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    return model(
        input_ids, attention_mask=attention_mask, position_ids=position_ids
    ).logits


def mc_estimate(word: str, samples: list[str]) -> float:
    """Plain Monte Carlo unigram probability: the rate of *word* across samples.

    Args:
        word: Surface form to count, matched against the whitespace tokens of
            each sample.
        samples: Sentences drawn from the language model.

    Returns:
        Occurrences over ``n_samples + n_boundaries``, the denominator
        :func:`mmc_estimate` also divides by; 0.0 when *samples* is empty.
    """
    total_count = 0
    total_prefixes = 0
    n = 0
    for sample in samples:
        tokens = sample.split()
        total_count += tokens.count(word)
        total_prefixes += len(tokens) + 1
        n += 1
    if n + total_prefixes == 0:
        return 0.0
    return total_count / (n + total_prefixes)


def _word_boundary_marker(tokenizer) -> str:
    """The word-initial subword marker for *tokenizer*: ``'Ġ'`` (byte-level
    BPE, GPT-2, Qwen) or ``'▁'`` (SentencePiece, mGPT).
    """
    try:
        toks = tokenizer.tokenize(" the")
    except Exception:
        toks = []
    if toks:
        if toks[0].startswith("Ġ"):
            return "Ġ"
        if toks[0].startswith("▁"):
            return "▁"
    return "Ġ"


def _is_word_boundary(strs: list[str], p: int, marker: str) -> bool:
    """True if token position *p* ends a whitespace word: it is first, last, or
    the next token is word-initial (starts with the BPE/SP *marker*).
    """
    return p == 0 or p == len(strs) - 1 or strs[p + 1].startswith(marker)


def _boundary_probs_per_sample(
    token_id,
    samples,
    model,
    tokenizer,
    batch_size,
    marker,
    start_ids=None,
    context_window=None,
):
    """``P(token_id | prefix)`` at word-boundary positions of every sample.

    A position ``p`` is a boundary when it is first, last, or the next token
    starts with ``marker``. Returns one ``(p, probability)`` list per sample.
    """
    import torch

    device = next(model.parameters()).device
    out = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        encoded = _encode_samples(tokenizer, batch, start_ids)
        padded, masks, max_len = _left_pad_batch(encoded, tokenizer.pad_token_id)
        with torch.no_grad():
            logits = _padded_logits(model, padded, masks, device, context_window)
            tok_probs = torch.softmax(logits, dim=-1)[:, :, token_id]  # (b, seq)
        for j, ids in enumerate(encoded):
            valid = tok_probs[j, max_len - len(ids) :].cpu().tolist()
            strs = tokenizer.convert_ids_to_tokens(ids)
            rec = []
            for p in range(len(ids)):
                if _is_word_boundary(strs, p, marker):
                    rec.append((p, valid[p]))
            out.append(rec)
    return out


def mmc_estimate(
    word: str,
    samples: list[str],
    model,
    tokenizer,
    batch_size: int = 64,
    start_ids: "Sequence[int] | None" = None,
    context_window: int | None = None,
) -> float:
    """Marginal Monte Carlo unigram probability for one word.

    The autoregressive joint of the word's tokens, summed over the word-boundary
    positions of every sample, over ``n_samples + n_boundaries``.

    Args:
        word: Surface form to score, as the tokenizer sees it; a leading space
            makes it word-initial.
        samples: Sentences drawn from *model*, rescored here under the BOS they
            were generated from.
        model: Causal language model supplying the token probabilities.
        tokenizer: Tokenizer paired with *model*; its word-initial marker is
            what locates the boundaries.
        batch_size: Sequences per forward pass, for the sample sweep and for the
            continuation passes alike.
        start_ids: The prefix the samples were generated under; ``None`` uses
            REPLICA's model default, falling back to tokenizer BOS or EOS.
        context_window: Maximum complete forward sequence; ``None`` uses the
            model default. Longer sample or continuation contexts are refused.

    Returns:
        0.0 when *word* encodes to no tokens, or when *samples* is empty.
    """
    import torch

    start_ids = _model_start_prefix(tokenizer, start_ids)
    context_window = _context_limit(model, tokenizer, context_window)
    device = next(model.parameters()).device
    marker = _word_boundary_marker(tokenizer)
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        return 0.0
    total_prob = 0.0
    total_prefixes = 0
    n = 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        encoded = _encode_samples(tokenizer, batch, start_ids)
        boundaries = _boundary_probs_per_sample(
            token_ids[0],
            batch,
            model,
            tokenizer,
            batch_size,
            marker,
            start_ids,
            context_window,
        )
        prefixes, joint_list = [], []
        for j, recs in enumerate(boundaries):
            for p, prob in recs:
                prefixes.append(encoded[j][: p + 1])
                joint_list.append(prob)
        joint = torch.tensor(joint_list, dtype=torch.float64)
        for t_idx in range(1, len(token_ids)):
            tid = token_ids[t_idx]
            word_prefix = token_ids[:t_idx]
            step = []
            for k in range(0, len(prefixes), batch_size):
                chunk = prefixes[k : k + batch_size]
                seqs = [pre + word_prefix for pre in chunk]
                padded, masks, _ = _left_pad_batch(seqs, tokenizer.pad_token_id)
                with torch.no_grad():
                    logits = _padded_logits(
                        model, padded, masks, device, context_window
                    )
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                step.append(probs[:, tid].cpu().double())
            joint *= torch.cat(step) if step else joint
        total_prob += float(joint.sum().item())
        total_prefixes += len(prefixes)
        n += len(batch)
    if n + total_prefixes == 0:
        return 0.0
    return total_prob / (n + total_prefixes)


def _boundary_sweep(
    samples,
    model,
    tokenizer,
    first_ids_t,
    marker,
    batch_size,
    start_ids=None,
    context_window=None,
):
    """Gather P(first_token | prefix) at every word boundary of *samples*.

    Returns ``(prefixes, first_probs_rows, n_samples)``: the boundary
    prefixes (token-id lists), one CPU probability row per boundary
    (restricted to the ``first_ids_t`` columns), and the sample count.
    """
    import torch

    device = next(model.parameters()).device
    prefixes: list[list[int]] = []
    first_probs_rows: list = []
    n_samples = 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        n_samples += len(batch)
        encoded = _encode_samples(tokenizer, batch, start_ids)
        padded, masks, max_len = _left_pad_batch(encoded, tokenizer.pad_token_id)
        with torch.no_grad():
            logits = _padded_logits(model, padded, masks, device, context_window)
            gathered = torch.softmax(logits, dim=-1).index_select(-1, first_ids_t)
        for j, ids in enumerate(encoded):
            start = max_len - len(ids)
            strs = tokenizer.convert_ids_to_tokens(ids)
            for p in range(len(ids)):
                if _is_word_boundary(strs, p, marker):
                    prefixes.append(ids[: p + 1])
                    first_probs_rows.append(gathered[j, start + p].cpu())
        del logits, gathered
    return prefixes, first_probs_rows, n_samples


def mmc_estimate_batch(
    words: list[str],
    samples: list[str],
    model,
    tokenizer,
    batch_size: int = 64,
    prob_floor: float = 0.0,
    max_continuation_boundaries: int | None = None,
    progress: Callable[[dict[str, float]], None] | None = None,
    start_ids: "Sequence[int] | None" = None,
    context_window: int | None = None,
) -> dict[str, float]:
    """Batched :func:`mmc_estimate` over one shared boundary sweep.

    After the sweep, each multi-token word gets its autoregressive
    continuation. ``prob_floor`` and ``max_continuation_boundaries`` default to
    no truncation, which reproduces :func:`mmc_estimate`.

    Args:
        words: Surface forms to score against one sweep of *samples*.
        samples: Sentences drawn from *model*, rescored under the BOS they were
            generated from.
        model: Causal language model supplying the token probabilities.
        tokenizer: Tokenizer paired with *model*; its word-initial marker is
            what locates the boundaries.
        batch_size: Sequences per forward pass.
        prob_floor: A boundary whose first-token probability is at or below this
            contributes nothing to a multi-token word.
        max_continuation_boundaries: Most boundaries a multi-token word runs
            continuation passes at; the survivors are strided and the sum scaled
            back up by the subsampling ratio.
        progress: Called with each group of estimates as it finishes: every
            empty and single-token word at once, then each multi-token word.
            Not called when no word has tokens or no sample has a boundary.
        start_ids: The prefix the samples were generated under; ``None`` uses
            REPLICA's model default, falling back to tokenizer BOS or EOS.
        context_window: Maximum complete forward sequence; ``None`` uses the
            model default. Longer sample or continuation contexts are refused.

    Returns:
        p(word) for every entry of *words*, 0.0 for a word that encodes to no
        tokens and for every word when no sample yields a boundary.
    """
    import torch

    start_ids = _model_start_prefix(tokenizer, start_ids)
    context_window = _context_limit(model, tokenizer, context_window)
    device = next(model.parameters()).device
    marker = _word_boundary_marker(tokenizer)
    word_tok_ids = [tokenizer.encode(w, add_special_tokens=False) for w in words]
    first_ids = sorted({t[0] for t in word_tok_ids if t})
    if not first_ids:
        return {w: 0.0 for w in words}
    first_col = {t: i for i, t in enumerate(first_ids)}
    first_ids_t = torch.tensor(first_ids, device=device, dtype=torch.long)

    # 1) Shared boundary sweep: gather P(first_token | prefix) at every boundary.
    prefixes, first_probs_rows, n_samples = _boundary_sweep(
        samples,
        model,
        tokenizer,
        first_ids_t,
        marker,
        batch_size,
        start_ids,
        context_window,
    )
    if not prefixes:
        return {w: 0.0 for w in words}
    first_probs = torch.stack(first_probs_rows).double()  # (n_boundaries, |first|)
    denom = n_samples + len(prefixes)

    # 2) single-token words: column sum over boundaries.
    out: dict[str, float] = {}
    by_first: dict[int, list[int]] = defaultdict(list)
    for w_idx, toks in enumerate(word_tok_ids):
        if not toks:
            out[words[w_idx]] = 0.0
        elif len(toks) == 1:
            out[words[w_idx]] = (
                float(first_probs[:, first_col[toks[0]]].sum().item()) / denom
            )
        else:
            by_first[toks[0]].append(w_idx)
    if progress is not None:
        # Single-token words are reported together: a resumed call summing only
        # some of them would sum a narrower matrix, whose column sums can differ
        # in their last bits.
        progress(dict(out))

    # 3) multi-token words: autoregressive continuation at boundaries whose
    #    first-token prob clears the floor.
    for w_idx in [i for lst in by_first.values() for i in lst]:
        toks = word_tok_ids[w_idx]
        col = first_col[toks[0]]
        p0 = first_probs[:, col]
        sel_all = (p0 > prob_floor).nonzero(as_tuple=True)[0]
        n_all = sel_all.numel()
        if n_all == 0:
            out[words[w_idx]] = 0.0
            if progress is not None:
                progress({words[w_idx]: 0.0})
            continue
        # Cap the (expensive) continuation passes by *systematically
        # subsampling* the boundaries and scaling the numerator back up
        # an unbiased estimate of the full boundary sum (no systematic bias,
        # unlike a top-K truncation which would under-count and re-inflate
        # surprisal). No-op when n_all <= cap, so small inputs stay exact.
        if max_continuation_boundaries and n_all > max_continuation_boundaries:
            stride = (
                n_all + max_continuation_boundaries - 1
            ) // max_continuation_boundaries
            sel_t = sel_all[::stride]
        else:
            sel_t = sel_all
        scale = n_all / sel_t.numel()
        sel = sel_t.tolist()
        joint = p0[sel].clone()
        for t_idx in range(1, len(toks)):
            tid = toks[t_idx]
            word_prefix = toks[:t_idx]
            step = []
            for k in range(0, len(sel), batch_size):
                chunk = sel[k : k + batch_size]
                seqs = [prefixes[b] + word_prefix for b in chunk]
                padded, masks, _ = _left_pad_batch(seqs, tokenizer.pad_token_id)
                with torch.no_grad():
                    logits = _padded_logits(
                        model, padded, masks, device, context_window
                    )
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                step.append(probs[:, tid].cpu().double())
            joint = joint * torch.cat(step)
        out[words[w_idx]] = scale * float(joint.sum().item()) / denom
        if progress is not None:
            progress({words[w_idx]: out[words[w_idx]]})
    return out


def _scorer_key(device: str) -> dict[str, str | None]:
    """The hardware and kernel build that estimates on *device* are computed with.

    GPU models, CUDA builds and CPU instruction sets select different kernels,
    so the same contexts can give estimates that differ in their last digits.
    """
    import torch

    target = torch.device(device)
    if target.type == "cuda":
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(target),
            "cuda": torch.version.cuda,
        }
    if target.type == "cpu":
        return {"device": "cpu", "capability": torch.backends.cpu.get_cpu_capability()}
    return {"device": target.type}


class SampledUnigramSource(EstimatedUnigramSource):
    """:class:`FrequencySource` estimating unigram p(w) from LM samples.

    Args:
        samples: list of sampled sentences. If None, ``n_samples`` are drawn
            from ``model_name`` (ancestral sampling, ``max_len`` new tokens).
        model_name: LM to sample from and score against.
        n_samples: sentences to draw when ``samples`` is None.
        max_len: new tokens per drawn sentence.
        seed: seed for the sampling run.
        batch_size: sentences scored per forward pass.
        gen_batch_size: sentences drawn per ``generate`` call, which bounds the
            generation KV-cache peak. ``None`` draws all ``n_samples`` at once.
        device: torch device to sample and score on.
        prob_floor: skip continuation passes below this first-token
            probability. The default ``0`` is exact.
        max_continuation_boundaries: cap the continuation passes per
            multi-token word by systematic subsampling. ``None`` is exact.
        language: stored on the source for the FrequencySource protocol.
        start_ids: the one-token prefix samples are drawn and scored from;
            ``None`` uses REPLICA's model default, falling back to tokenizer
            BOS or EOS. Overrides of model defaults emit a warning.
        context_window: Maximum complete forward sequence; ``None`` uses the
            model default. Samples and continuations are never truncated.

    ``probability(word)`` returns p(w), NaN when the estimate is 0. Estimates
    are cached per word; :meth:`estimate_words` can save the cache with the
    sampled contexts to a checkpoint file and resume from it.
    """

    def __init__(
        self,
        samples: list[str] | None = None,
        model_name: str = "ai-forever/mGPT",
        n_samples: int = 500,
        max_len: int = 50,
        language: str = "en",
        device: str = "cpu",
        seed: int = 0,
        batch_size: int = 64,
        gen_batch_size: int | None = None,
        prob_floor: float = 0.0,
        max_continuation_boundaries: int | None = None,
        start_ids: "Sequence[int] | None" = None,
        context_window: int | None = None,
    ):
        """Record the sampling and scoring settings; the model loads on first use."""
        self.context_window = validate_context_window(context_window)
        self._config = None
        self._resolved_context_window: int | None = None
        self._context_resolved = False
        self.prob_floor = float(prob_floor)
        self.max_continuation_boundaries = (
            int(max_continuation_boundaries) if max_continuation_boundaries else None
        )
        self.language = language
        self.model_name = model_name
        self.n_samples = int(n_samples)
        self.max_len = int(max_len)
        self.device = device
        self.seed = int(seed)
        self.start_ids = None if start_ids is None else validate_start_ids(start_ids)
        self._resolved_start_ids: tuple[int, ...] | None = None
        self.batch_size = int(batch_size)
        # Generate samples in chunks to bound the generation KV-cache peak.
        # Seeded once before the loop, so settings are deterministic.
        self.gen_batch_size = int(gen_batch_size) if gen_batch_size else None
        self._samples = list(samples) if samples is not None else None
        self._sample_ids: list[list[int]] | None = None
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, float] = {}

    def _ensure_tokenizer(self):
        """Load the tokenizer; the priming fingerprint needs it before the model."""
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # gpt2 and similar ship without a pad token; padded positions are
        # masked out and given no position id, so aliasing it to eos is
        # inert.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _start_prefix(self) -> list[int]:
        """Resolve one prefix shared by generation, scoring and checkpoints."""
        if self._resolved_start_ids is None:
            tok = self._ensure_tokenizer()
            self._resolved_start_ids = tuple(
                _model_start_prefix(tok, self.start_ids, self.model_name)
            )
        return list(self._resolved_start_ids)

    def _model_config(self):
        """Load model metadata without loading weights."""
        if self._model is not None:
            return getattr(self._model, "config", None)
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_name)
        return self._config

    def _context_limit(self) -> int | None:
        """Resolve the complete-context limit once for this source."""
        if not self._context_resolved:
            self._resolved_context_window = usable_position_limit(
                self.model_name,
                self._model_config(),
                context_window=self.context_window,
            )
            self._context_resolved = True
        return self._resolved_context_window

    def _ensure(self):
        if self._samples is not None and self._model is not None:
            return
        from transformers import AutoModelForCausalLM

        self._ensure_tokenizer()
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device).eval()
        if self._samples is None:
            self._samples = self._generate_samples()

    def _generate_samples(self) -> list[str]:
        import torch

        tok = self._tokenizer
        prefix = self._start_prefix()
        _check_context_length(len(prefix) + self.max_len, self._context_limit())
        input_ids = torch.tensor([prefix], device=self.device)
        gb = self.gen_batch_size or self.n_samples
        samples: list[str] = []
        sample_ids: list[list[int]] = []
        stop_ids = generation_stop_ids(self._model, tok)
        torch.manual_seed(self.seed)
        with torch.no_grad():
            remaining = self.n_samples
            while remaining > 0:
                k = min(gb, remaining)
                gen = self._model.generate(
                    input_ids,
                    do_sample=True,
                    top_k=0,
                    max_new_tokens=self.max_len,
                    num_return_sequences=k,
                    pad_token_id=(
                        tok.pad_token_id
                        if tok.pad_token_id is not None
                        else tok.eos_token_id
                    ),
                )
                for g in gen:
                    drawn = _drawn_ids(g[len(prefix) :].tolist(), tok, stop_ids)
                    sample_ids.append(drawn)
                    samples.append(tok.decode(drawn, skip_special_tokens=True))
                remaining -= k
        self._sample_ids = sample_ids
        return samples

    def estimate(self, word: str) -> float:
        """Estimated unigram probability p(word).

        Args:
            word: Surface form to score, as the tokenizer sees it; the estimate
                is cached under this exact string.
        """
        if word in self._cache:
            return self._cache[word]
        # Route through the batched estimator so estimate() and
        # estimate_words() apply the same boundary cap / floor and agree.
        p = self._estimate_batch([word])[word]
        self._cache[word] = p
        return p

    def estimate_words(
        self,
        words: list[str],
        *,
        checkpoint: Path | None = None,
        owner: object = None,
    ) -> dict[str, float]:
        """Batched :meth:`estimate` for many words; populates the cache.

        One shared boundary sweep serves every word. Cached words are skipped.

        Args:
            words: Surface forms to score; duplicates collapse into one estimate.
            checkpoint: File the sampled contexts and finished estimates are
                saved to, and resumed from when it was written for the same
                *owner* and settings; estimates from other hardware or another
                CUDA build are recomputed on its contexts. ``None`` keeps them
                in memory only.
            owner: JSON-serializable identity of the computation, which must
                name the model and tokenizer by content. Required with
                *checkpoint*.

        Returns:
            p(word) keyed by every entry of *words*.

        Raises:
            ValueError: When *checkpoint* is given without *owner*.
        """
        if checkpoint is not None and owner is None:
            raise ValueError("a priming checkpoint needs the owner it belongs to")
        todo = [w for w in dict.fromkeys(words) if w not in self._cache]
        if not todo:
            return {w: self._cache[w] for w in words}
        if checkpoint is None:
            self._cache.update(self._estimate_batch(todo))
        else:
            self._prime_with_checkpoint(todo, Path(checkpoint), owner)
        return {w: self._cache[w] for w in words}

    def _estimate_batch(
        self,
        words: list[str],
        progress: Callable[[dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """:func:`mmc_estimate_batch` over this source's samples and settings."""
        self._ensure()
        return mmc_estimate_batch(
            words,
            self._samples,
            self._model,
            self._tokenizer,
            self.batch_size,
            prob_floor=self.prob_floor,
            max_continuation_boundaries=self.max_continuation_boundaries,
            progress=progress,
            start_ids=self._start_prefix(),
            context_window=self._context_limit(),
        )

    def _prime_with_checkpoint(self, todo: list[str], path: Path, owner) -> None:
        """Estimate *todo* in one batched call, saving progress to *path*.

        The contexts are written before the sweep, so a job killed before any
        word finishes still resumes against them.
        """
        header = {
            "fingerprint": self._priming_fingerprint(owner),
            "scorer": _scorer_key(self.device),
        }
        self._resume_priming(path, header)
        todo = [w for w in todo if w not in self._cache]
        if not todo:
            return
        self._ensure()
        self._write_priming(path, header)
        last_write = time.monotonic()

        def record(group: dict[str, float]) -> None:
            nonlocal last_write
            self._cache.update(group)
            if time.monotonic() - last_write >= PRIMING_WRITE_INTERVAL:
                self._write_priming(path, header)
                last_write = time.monotonic()

        self._cache.update(self._estimate_batch(todo, progress=record))
        self._write_priming(path, header)

    def _priming_fingerprint(self, owner) -> dict:
        """The settings a saved estimate depends on besides its contexts and
        scorer; the owner identifies the model and tokenizer.
        """
        return {
            "owner": text_sha256(json.dumps(owner, sort_keys=True)),
            "start": self._start_prefix(),
            "context_window": self._context_limit(),
            "n_samples": self.n_samples,
            "max_len": self.max_len,
            "seed": self.seed,
            "gen_batch_size": self.gen_batch_size,
            "batch_size": self.batch_size,
            "prob_floor": self.prob_floor,
            "max_continuation_boundaries": self.max_continuation_boundaries,
        }

    def _resume_priming(self, path: Path, header: dict) -> None:
        """Adopt the contexts of a resumable file at *path*, and its estimates
        when this scorer computed them, so a cell never mixes two scorers' estimates.
        """
        try:
            with open(path, "rb") as stream:
                payload = pickle.load(stream)
            problem = self._priming_problem(payload, header["fingerprint"])
        except FileNotFoundError:
            return
        except Exception as error:
            problem = f"unreadable ({type(error).__name__})"
        if problem is not None:
            log.warning("[prime] %s ignored: %s; starting fresh", path.name, problem)
            return
        if self._samples is None:
            self._samples = list(payload["contexts"])
        if payload.get("scorer") != header["scorer"]:
            log.warning(
                "[prime] %s was computed on %s and this run scores on %s; "
                "reusing its contexts and recomputing its %d estimates",
                path.name,
                payload.get("scorer"),
                header["scorer"],
                len(payload["estimates"]),
            )
            return
        self._cache.update(payload["estimates"])
        log.info(
            "[prime] resumed %d word estimates from %s",
            len(payload["estimates"]),
            path.name,
        )

    def _priming_problem(self, payload, fingerprint: dict) -> str | None:
        """Why *payload* cannot be resumed from, or ``None`` when it can."""
        if not isinstance(payload, dict) or payload.get("format") != _PRIMING_FORMAT:
            return "not a priming checkpoint"
        saved = payload.get("fingerprint")
        if saved != fingerprint:
            saved = saved if isinstance(saved, dict) else {}
            differing = [
                k for k, v in fingerprint.items() if k not in saved or saved[k] != v
            ]
            return f"settings differ ({', '.join(differing) or 'extra keys'})"
        contexts = payload.get("contexts")
        if not isinstance(contexts, list) or not all(
            isinstance(c, str) for c in contexts
        ):
            return "contexts are not a list of strings"
        if self._samples is not None and contexts != self._samples:
            return "contexts differ from the ones this source holds"
        if self._samples is None and len(contexts) != self.n_samples:
            return f"{len(contexts)} contexts, expected {self.n_samples}"
        # The estimator returns NaN where the model's probabilities were NaN.
        estimates = payload.get("estimates")
        if not isinstance(estimates, dict) or not all(
            isinstance(word, str)
            and isinstance(p, float)
            and (math.isnan(p) or 0 <= p < math.inf)
            for word, p in estimates.items()
        ):
            return "estimates are not words mapped to non-negative floats or NaN"
        return None

    def _write_priming(self, path: Path, header: dict) -> None:
        """Save the contexts and every cached estimate to *path* under *header*.

        A failed write is logged, and the next write tries again.
        """
        payload = {
            "format": _PRIMING_FORMAT,
            **header,
            "contexts": list(self._samples),
            "estimates": dict(self._cache),
        }
        try:
            atomic_pickle(payload, path, fsync=True)
        except OSError as error:
            log.warning("[prime] could not write %s: %s", path.name, error)
