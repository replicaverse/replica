"""MC word-entropy estimator from Clark et al. (2025).

Sample |S| word-bounded continuations per word position and estimate
H ≈ -(1/|S|) Σ log p_LM(s | ctx). Supports Rényi alpha.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from replica.feature.backends.hub import hub_id

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput
from ._base import _BaseEntropyEstimatorLM
from .utils import _clark_windows, _renyi_entropy_bits_from_log_p_nats

if TYPE_CHECKING:
    import torch


class MCWordEntropyLM(_BaseEntropyEstimatorLM):
    """MC estimate of word-level entropy from Clark et al. (2025).

    Each MC sample is one word: a token sequence sampled from the LM
    that terminates as soon as the next token would be the start of a
    new word (a whitespace-prefixed token in GPT-2-style BPE, those
    starting with the ``Ġ`` proxy character; ``Ċ`` and ``ĉ`` for
    newline/tab are also treated as boundaries).

    For each word position t, draws |S| samples and estimates

        H_alpha(W_t | ctx) ≈ (1/(1-alpha)) · log₂ ( (1/|S|) Σ_s P(s)^(alpha-1) )

    which reduces to Shannon entropy at alpha = 1:

        H₁ ≈ -(1/|S|) Σ_s log₂ P(s).

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_identity: Verified Hub source/revision or local bundle ``sha256``.
            Assertions are checked against loaded files; omitted identities resolve
            from the model source.
        model_id: GPT-style byte-level BPE tokenizer required.
        n_samples: |S|. Clark et al. (2025) use 512; 64 is a speed/variance
            tradeoff.
        max_iter: Max subword tokens per sampled word; stops at first
            whitespace-prefix token.
        alpha: 1.0 = Shannon.
        chunk_size: Draw samples in batches; Rényi-alpha aggregates once over
            the full collected ``log_p`` array, so the estimate is
            mathematically identical to the un-chunked path.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is MCWordEntropyLM

    def __init__(
        self,
        model_id: str = "gpt2",
        *,
        n_samples: int = 64,
        max_iter: int = 10,
        alpha: float = 1.0,
        add_bos: bool = True,
        seed: int | None = None,
        model_identity: dict | None = None,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
        chunk_size: int | None = None,
        word_starter_chars: tuple[str, ...] = ("Ġ", "Ċ", "ĉ"),
    ):
        super().__init__(
            model_id,
            add_bos=add_bos,
            seed=seed,
            model_identity=model_identity,
            start_ids=start_ids,
            context_window=context_window,
        )
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be ≥ 1, got {max_iter}")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be ≥ 1 if given, got {chunk_size}")
        self.n_samples = int(n_samples)
        self.max_iter = int(max_iter)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        if alpha < 0:
            raise ValueError(f"Rényi alpha must be ≥ 0, got {alpha}")
        self.alpha = float(alpha)
        # Which BPE prefixes start a new word (= terminate the current one).
        # Default Ġ/Ċ/ĉ also breaks at newline/control bytes; the upstream
        # ``mc_word_entropy.py`` uses Ġ only, which changes the EOW marginal
        # and thus the tail-sensitive Rényi-alpha estimate.
        self._word_starter_chars = tuple(word_starter_chars)
        tag = (
            ""
            if self._word_starter_chars == ("Ġ", "Ċ", "ĉ")
            else "_ws" + "".join(self._word_starter_chars)
        )
        self.name = (
            f"mcword_entropy_{hub_id(model_id)}_N{n_samples}_M{max_iter}_a{alpha}{tag}"
        )

        # Pre-compute word-starter vs subword token masks for the sampler.
        self._space_ids, self._subword_ids = self._compute_token_masks()
        if self._space_ids.numel() == 0:
            raise ValueError(
                f"MCWordEntropyLM: tokenizer for {model_id!r} has no "
                "whitespace-prefix tokens (no 'Ġ'/'Ċ'/'ĉ' in vocab). "
                "Word-boundary stopping is undefined; the MC sampler "
                "would never terminate at a word boundary. This LM is "
                "not applicable to non-whitespace scripts (Chinese, "
                "Japanese, Thai, character-level corpora). For word units "
                "use ContinuationEntropyLM or RenyiEntropy; CharBeamLM "
                "scores per-character units (surprisal)."
            )

    def _compute_token_masks(self):
        """Partition vocab into word-starter ids vs subwords."""
        import torch

        starters = self._word_starter_chars
        space_ids: list[int] = []
        subword_ids: list[int] = []
        vocab_size = self._tokenizer.vocab_size
        for i in range(vocab_size):
            tok = self._tokenizer.convert_ids_to_tokens(i)
            if tok and tok[0] in starters:
                space_ids.append(i)
            else:
                subword_ids.append(i)
        return (
            torch.tensor(space_ids, device=self._device, dtype=torch.long),
            torch.tensor(subword_ids, device=self._device, dtype=torch.long),
        )

    def _empty_values(self) -> dict:
        return {"mc_word_entropy": np.nan}

    def _wrap(self, value: float) -> dict:
        return {"mc_word_entropy": value}

    @property
    def _provenance(self) -> dict:
        """The settings that decide which draws were taken.

        The seed, word-starter set and device move the sample, and so does the
        batch size under the advancing-stream sampler. ``name`` and the compute
        cache tag record a non-default starter set, the cache tag also records
        the seed, and the batch size and device reach neither. Computed on
        access from the current settings.
        """
        return {**super()._provenance, "word_starter_chars": self._word_starter_chars}

    def _check_stimulus_compat(self, stimulus: Stimulus, target_layer: str) -> None:
        # The word-boundary stop rule requires whitespace-separated units; refuse
        # character-level and no-space scripts (Chinese, Japanese, Thai).
        if target_layer == "characters":
            raise ValueError(
                "MCWordEntropyLM requires a word-level target_layer; got "
                "'characters'. The word-boundary stopping rule is "
                "meaningless on per-character units. For character-level "
                "corpora (e.g. MECO_char), CharBeamLM scores per-character "
                "units (surprisal)."
            )
        layer = stimulus.layers[target_layer]
        text = stimulus.text
        if len(layer) > 1 and " " not in text and "\t" not in text and "\n" not in text:
            raise ValueError(
                f"MCWordEntropyLM: stimulus_id={stimulus.stimulus_id!r} "
                "has multiple word units but no whitespace in its text. "
                "Likely a no-space script (Chinese/Japanese/Thai): the "
                "word-boundary stopping rule cannot match the stimulus's "
                "unit boundaries. Use ContinuationEntropyLM or "
                "RenyiEntropy instead."
            )

    def _estimate_with_samples(
        self,
        prefix_ids: list[int],
    ) -> tuple[float, np.ndarray]:
        """Sample ``n_samples`` words at one prefix and return its entropy.

        Returns ``(H_bits, log_p_samples_nats)``; samples are persisted so the
        pickle can be re-aggregated. With ``chunk_size`` set, the N samples are
        drawn in batches of K and Rényi-alpha aggregation runs once over the
        full collected array.
        """
        chunk = self.chunk_size or self.n_samples
        log_p_all_nats: list[float] = []
        remaining = self.n_samples
        while remaining > 0:
            k = min(chunk, remaining)
            log_p_all_nats.extend(
                self._sample_log_p_chunk(
                    prefix_ids, k, sample_offset=self.n_samples - remaining
                )
            )
            remaining -= k

        samples = np.asarray(log_p_all_nats, dtype=np.float64)
        H_bits = _renyi_entropy_bits_from_log_p_nats(samples, self.alpha)
        return H_bits, samples

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One-pass driver for Clark et al. (2025)'s word-boundary sampling rule.

        Once per MC chunk, one master cache steps through the tokens and is
        forked, sharing tensors, at each unit boundary to sample a unit
        continuation of as many words as the unit holds. A stimulus with
        ``max_prefix + 1 + longest_unit * max_iter > ctx_size`` is split into
        50%-overlap windows, dropping left context beyond it.

        Args:
            stimulus: Text whose word boundaries are scored.
            extract: Part of the shared FeatureModel signature; the entropy and
                the per-sample log probabilities are always returned.
            target_layer: Annotation layer whose units are the words scored.
        """
        token_ids, n_units, sample_positions, is_empty = self._prepare_word_sampling(
            stimulus, target_layer
        )
        per_unit_entropy = np.full(n_units, np.nan)
        per_unit_samples: list[np.ndarray | None] = [None] * n_units

        if is_empty or not sample_positions:
            return self._outputs_with_samples(per_unit_entropy, per_unit_samples)

        unit_words = self._unit_word_counts(stimulus, target_layer)
        if unit_words and self.max_iter < 2:
            raise ValueError(
                f"{type(self).__name__}: stimulus_id={stimulus.stimulus_id!r} has "
                f"{len(unit_words)} units of several whitespace words, which need "
                "max_iter of at least 2 to open each word after the first"
            )
        ctx_size = self._model_max_ctx_size()
        max_prefix = max(p for _, p in sample_positions)
        longest = max(unit_words.values(), default=1)

        # Single window if the longest needed prefix fits; else 50%-overlap
        # windowing (each word sampled once; see the windowed driver's
        # docstring for the left-context floor).
        single_window = max_prefix + 1 + longest * self.max_iter <= ctx_size

        def dispatch(k, offset):
            if single_window:
                return self._estimate_words_one_pass(
                    token_ids,
                    sample_positions,
                    k,
                    sample_offset=offset,
                    unit_words=unit_words,
                )
            return self._estimate_words_one_pass_windowed(
                token_ids,
                sample_positions,
                k,
                ctx_size,
                sample_offset=offset,
                unit_words=unit_words,
            )

        for wi, samples in self._collect_sample_chunks(dispatch).items():
            per_unit_samples[wi] = samples
            per_unit_entropy[wi] = _renyi_entropy_bits_from_log_p_nats(
                samples,
                self.alpha,
            )

        return self._outputs_with_samples(per_unit_entropy, per_unit_samples)

    def _estimate_words_one_pass(
        self,
        token_ids: list[int],
        sample_positions: list[tuple[int, int]],
        k: int,
        *,
        sample_offset: int = 0,
        unit_words: dict[int, int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Forward ``token_ids`` once with ``k`` parallel sample streams,
        forking the KV-cache at each ``(word_i, last_prefix_token)``
        position and sampling a unit continuation from the fork: one word,
        or as many words as ``unit_words`` names for the unit.

        Returns ``{word_i: log_p_samples_nats[k]}`` for every entry in
        ``sample_positions``.
        """
        unit_words = unit_words or {}
        import torch

        gen = self._rng
        device = self._device
        bos_id = self._word_start_token_id()

        # Sort by position so the master pass steps in order.
        sorted_positions = sorted(sample_positions, key=lambda x: x[1])
        word_log_probs: dict[int, np.ndarray] = {}

        # Initial forward: process [0..first_sample_pos] in one batch.
        first_pos = sorted_positions[0][1]
        init_ids = token_ids[: first_pos + 1]
        ctx = torch.tensor([init_ids], device=device, dtype=torch.long)
        ctx_b = ctx.repeat(k, 1)

        with torch.no_grad():
            out = self._model(ctx_b, use_cache=True)
            # Keep master_kv as a legacy tuple: forks wrap a fresh DynamicCache
            # without copying tensors.
            master_kv_dyn = out.past_key_values
            master_kv = master_kv_dyn.to_legacy_cache()
            last_logits = out.logits[:, -1, :].double()
        # Only the final logits and the current cache are needed below. Keeping
        # the initial output alive retains all prefix logits and an obsolete
        # cache throughout the window, adding gigabytes for long prefixes.
        del out, master_kv_dyn

        # Sample the unit continuation for the first position.
        prev_tok = init_ids[-1]
        first_word_i = sorted_positions[0][0]
        log_p = self._sample_unit_log_p(
            last_logits,
            master_kv,
            k,
            gen,
            prev_tok,
            bos_id,
            n_words=unit_words.get(first_word_i, 1),
            word_index=first_word_i,
            sample_offset=sample_offset,
        )
        word_log_probs[first_word_i] = log_p.detach().cpu().numpy()

        # Continue: step through remaining positions one token at a time.
        cur_pos = first_pos + 1
        for word_i, target_pos in sorted_positions[1:]:
            cur_pos, master_kv, last_logits = self._advance_master_cache(
                token_ids, cur_pos, target_pos, k, master_kv, last_logits
            )

            prev_tok = token_ids[target_pos]
            log_p = self._sample_unit_log_p(
                last_logits,
                master_kv,
                k,
                gen,
                prev_tok,
                bos_id,
                n_words=unit_words.get(word_i, 1),
                word_index=word_i,
                sample_offset=sample_offset,
            )
            word_log_probs[word_i] = log_p.detach().cpu().numpy()

        return word_log_probs

    def _sample_unit_log_p(
        self,
        logits_after_prev: torch.Tensor,
        master_kv: tuple,
        k: int,
        gen: torch.Generator | None,
        prev_tok: int,
        bos_id: int | None,
        *,
        n_words: int,
        word_index: int,
        sample_offset: int,
    ) -> torch.Tensor:
        """Sample ``k`` continuations of ``n_words`` whole words; joint log p (nats).

        Each later word opens with a starter drawn from the logits that closed
        the previous one; a word cut at ``max_iter - 1`` draws has no
        end-of-word term and the next word opens after its last subword.
        """
        if n_words == 1:
            return self._sample_word_log_p(
                logits_after_prev,
                master_kv,
                k,
                gen,
                prev_tok,
                bos_id,
                word_index=word_index,
                sample_offset=sample_offset,
            )
        import torch
        from transformers import DynamicCache

        device = self._device
        n_sub = self._subword_ids.numel()
        per_word = self.max_iter - 1
        # A column per continuation draw, plus one per cut word's opening step.
        steps = n_words * per_word + n_words - 1
        u_tok = (
            None
            if self.release_rng_stream
            else self._uniform_block(("wu", word_index), sample_offset, k, 1 + steps)
        )
        u_start = (
            None
            if self.release_rng_stream
            else self._uniform_block(("wus", word_index), sample_offset, k, n_words - 1)
        )
        log_p = torch.zeros(k, device=device, dtype=torch.float64)
        first_from_subword = bos_id is not None and prev_tok == bos_id
        allowed = self._subword_ids if first_from_subword else self._space_ids
        probs = torch.softmax(logits_after_prev.index_select(1, allowed), dim=-1)
        local_idx = (
            torch.multinomial(probs, 1, generator=gen)
            if u_tok is None
            else self._inverse_cdf(probs, u_tok[:, 0])
        )
        log_p += torch.log(probs.gather(1, local_idx).squeeze(1).clamp_min(1e-40))
        cur = allowed[local_idx.squeeze(1)].unsqueeze(1)

        pad = self._tokenizer.pad_token_id
        pad = 0 if pad is None else pad
        arange = torch.arange(k, device=device)
        word = torch.zeros(k, dtype=torch.long, device=device)
        draws = torch.zeros(k, dtype=torch.long, device=device)
        done = torch.zeros(k, dtype=torch.bool, device=device)
        # Streams whose cut word opens the next one at the coming step.
        pending = torch.zeros(k, dtype=torch.bool, device=device)
        kv = DynamicCache.from_legacy_cache(master_kv)
        with torch.no_grad():
            for step in range(1, 1 + steps):
                out = self._model(cur, past_key_values=kv, use_cache=True)
                kv = out.past_key_values
                logits = out.logits[:, -1, :].double()
                step_probs = torch.softmax(logits, dim=-1)
                sub_probs = step_probs.index_select(1, self._subword_ids)
                space_probs = step_probs.index_select(1, self._space_ids)
                local = torch.cat([sub_probs, space_probs.sum(dim=1, keepdim=True)], 1)
                local_idx = (
                    torch.multinomial(local, 1, generator=gen)
                    if u_tok is None
                    else self._inverse_cdf(local, u_tok[:, step])
                ).squeeze(1)
                opening = pending & ~done
                active = ~done & ~opening
                prob = local.gather(1, local_idx.unsqueeze(1)).squeeze(1)
                log_p += active.double() * torch.log(prob.clamp_min(1e-40))
                is_sub = local_idx < n_sub
                draws = draws + active.long()
                # A word ends on an end-of-word draw, or is cut at its cap.
                ended = active & (~is_sub | (draws >= per_word))
                last = ended & (word == n_words - 1)
                done = done | last
                cut = ended & ~last & is_sub
                opens = (ended & ~last & ~is_sub) | opening
                sub_tok = self._subword_ids[local_idx.clamp_max(n_sub - 1)]
                nxt = torch.where(is_sub, sub_tok, torch.full_like(sub_tok, pad))
                if bool(opens.any()):
                    starter_probs = torch.softmax(
                        logits.index_select(1, self._space_ids), dim=-1
                    )
                    starter_idx = (
                        torch.multinomial(starter_probs, 1, generator=gen).squeeze(1)
                        if u_start is None
                        else self._inverse_cdf(
                            starter_probs, u_start[arange, word.clamp_max(n_words - 2)]
                        ).squeeze(1)
                    )
                    starter_prob = starter_probs.gather(
                        1, starter_idx.unsqueeze(1)
                    ).squeeze(1)
                    log_p += opens.double() * torch.log(starter_prob.clamp_min(1e-40))
                    nxt = torch.where(opens, self._space_ids[starter_idx], nxt)
                    word = word + opens.long()
                    draws = torch.where(opens, torch.zeros_like(draws), draws)
                pending = cut
                if bool(done.all()):
                    break
                cur = nxt.unsqueeze(1)
        return log_p

    def _estimate_words_one_pass_windowed(
        self,
        token_ids: list[int],
        sample_positions: list[tuple[int, int]],
        k: int,
        ctx_size: int,
        *,
        sample_offset: int = 0,
        unit_words: dict[int, int] | None = None,
    ) -> dict[int, np.ndarray]:
        """50%-overlap windowed driver for stimuli longer than ctx.

        Half-windows of ``ctx_size//2 - longest * max_iter`` tokens, for the
        longest unit's word count; every unit is sampled once with at least
        that much left context, except in the first window.
        """
        longest = max((unit_words or {}).values(), default=1)
        half_size = ctx_size // 2 - longest * self.max_iter
        if half_size < 1:
            raise ValueError(
                f"_estimate_words_one_pass_windowed: ctx_size={ctx_size} "
                f"too small to fit max_iter={self.max_iter} for {longest} words"
            )

        word_log_probs: dict[int, np.ndarray] = {}
        for window_tokens, window_samples in _clark_windows(
            token_ids, sample_positions, half_size
        ):
            sub = self._estimate_words_one_pass(
                window_tokens,
                window_samples,
                k,
                sample_offset=sample_offset,
                unit_words=unit_words,
            )
            word_log_probs.update(sub)

        return word_log_probs

    def _sample_word_log_p(
        self,
        logits_after_prev: torch.Tensor,
        master_kv: tuple,
        k: int,
        gen: torch.Generator | None,
        prev_tok: int,
        bos_id: int | None,
        *,
        word_index: int = 0,
        sample_offset: int = 0,
    ) -> torch.Tensor:
        """Sample ``k`` word continuations from the LM state and return log_p (nats).

        Termination: emit a word-starter token (Ġ/Ċ/ĉ). Step 1 restricts to
        word-starter ids (subword if prev = BOS); subsequent steps draw from
        ``[subword | EOW]`` with ``P(EOW) = Σ P(word-starter)``. Mirrors
        the upstream Clark et al. (2025) ``mc_word_entropy.py``.
        """
        import torch
        from transformers import DynamicCache

        device = self._device
        log_p = torch.zeros(k, device=device, dtype=torch.float64)

        first_from_subword = bos_id is not None and prev_tok == bos_id
        allowed_ids = self._subword_ids if first_from_subword else self._space_ids
        allowed_logits = logits_after_prev.index_select(1, allowed_ids)
        allowed_probs = torch.softmax(allowed_logits, dim=-1)
        unif = (
            None
            if self.release_rng_stream
            else self._uniform_block(("w", word_index), sample_offset, k, self.max_iter)
        )
        local_idx = (
            torch.multinomial(allowed_probs, 1, generator=gen)
            if unif is None
            else self._inverse_cdf(allowed_probs, unif[:, 0])
        )
        sample_prob = allowed_probs.gather(1, local_idx).squeeze(1)
        log_p += torch.log(sample_prob.clamp_min(1e-40))
        sampled_id = allowed_ids[local_idx.squeeze(1)]

        # Fork master_kv: tensors are shared; appends land in the fork's tuple.
        sample_kv_dyn = DynamicCache.from_legacy_cache(master_kv)
        cur = sampled_id.unsqueeze(1)
        still_going = torch.ones(k, dtype=torch.bool, device=device)

        with torch.no_grad():
            for step in range(1, self.max_iter):
                out = self._model(
                    cur,
                    past_key_values=sample_kv_dyn,
                    use_cache=True,
                )
                sample_kv_dyn = out.past_key_values
                step_logits = out.logits[:, -1, :].double()
                step_probs = torch.softmax(step_logits, dim=-1)

                subword_probs = step_probs.index_select(1, self._subword_ids)
                space_probs = step_probs.index_select(1, self._space_ids)
                eow_prob = space_probs.sum(dim=1, keepdim=True)
                local_probs = torch.cat([subword_probs, eow_prob], dim=1)
                local_idx = (
                    torch.multinomial(local_probs, 1, generator=gen)
                    if unif is None
                    else self._inverse_cdf(local_probs, unif[:, step])
                )
                sample_prob = local_probs.gather(1, local_idx).squeeze(1)

                log_p += still_going.double() * torch.log(sample_prob.clamp_min(1e-40))

                is_subword = local_idx.squeeze(1) < self._subword_ids.numel()
                still_going = still_going & is_subword

                if not bool(still_going.any()):
                    break

                pad = (
                    self._tokenizer.pad_token_id
                    if self._tokenizer.pad_token_id is not None
                    else 0
                )
                local_ix = local_idx.squeeze(1)
                ix_filt = torch.where(
                    is_subword,
                    self._subword_ids[
                        local_ix.clamp_max(self._subword_ids.numel() - 1)
                    ],
                    torch.full_like(local_ix, pad),
                )
                cur = ix_filt.unsqueeze(1)

        return log_p

    def _sample_log_p_chunk(
        self, prefix_ids: list[int], k: int, *, sample_offset: int = 0
    ) -> list[float]:
        """Draw ``k`` MC samples and return their joint log-probs (nats)."""
        import torch

        gen = self._rng
        # Keyed on the prefix: a given context always draws the same samples.
        unif = (
            None
            if self.release_rng_stream
            else self._uniform_block(
                ("p", tuple(prefix_ids)), sample_offset, k, self.max_iter
            )
        )
        ctx = torch.tensor([prefix_ids], device=self._device, dtype=torch.long)
        ctx_b = ctx.repeat(k, 1)

        log_p = torch.zeros(k, device=self._device, dtype=torch.float64)

        bos_id = self._word_start_token_id()
        last_id = prefix_ids[-1]
        # Right after BOS, the first content token does not carry the Ġ prefix.
        first_from_subword = last_id == bos_id

        with torch.no_grad():
            out = self._model(ctx_b, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].double()

            allowed_ids = self._subword_ids if first_from_subword else self._space_ids
            allowed_logits = logits.index_select(1, allowed_ids)
            allowed_probs = torch.softmax(allowed_logits, dim=-1)
            local_idx = (
                torch.multinomial(allowed_probs, 1, generator=gen)
                if unif is None
                else self._inverse_cdf(allowed_probs, unif[:, 0])
            )
            sample_prob = allowed_probs.gather(1, local_idx).squeeze(1)
            log_p += torch.log(sample_prob.clamp_min(1e-40))
            sampled_id = allowed_ids[local_idx.squeeze(1)]

            # Track in-progress samples (not yet terminated by EOW).
            still_going = torch.ones(k, dtype=torch.bool, device=self._device)

            cur = sampled_id.unsqueeze(1)
            for step in range(1, self.max_iter):
                out = self._model(
                    cur,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                step_logits = out.logits[:, -1, :].double()
                step_probs = torch.softmax(step_logits, dim=-1)

                subword_probs = step_probs.index_select(1, self._subword_ids)
                space_probs = step_probs.index_select(1, self._space_ids)
                # EOW = marginal over word-starter tokens; [subword | EOW] vocab.
                eow_prob = space_probs.sum(dim=1, keepdim=True)
                local_probs = torch.cat([subword_probs, eow_prob], dim=1)
                local_idx = (
                    torch.multinomial(local_probs, 1, generator=gen)
                    if unif is None
                    else self._inverse_cdf(local_probs, unif[:, step])
                )
                sample_prob = local_probs.gather(1, local_idx).squeeze(1)

                # Terminated samples keep their accumulated log P unchanged.
                log_p += still_going.double() * torch.log(sample_prob.clamp_min(1e-40))

                is_subword = local_idx.squeeze(1) < self._subword_ids.numel()
                still_going = still_going & is_subword

                if not bool(still_going.any()):
                    break

                # Next input: subword ids for still-going samples; pad for terminated.
                pad = (
                    self._tokenizer.pad_token_id
                    if self._tokenizer.pad_token_id is not None
                    else 0
                )
                ix_filt = local_idx.squeeze(1).clone()
                ix_filt = torch.where(
                    is_subword,
                    self._subword_ids[ix_filt.clamp_max(self._subword_ids.numel() - 1)],
                    torch.full_like(ix_filt, pad),
                )
                cur = ix_filt.unsqueeze(1)

        return log_p.detach().cpu().tolist()
