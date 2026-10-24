"""Continuation-entropy estimator from Giulianelli et al. (2024a).

Sample N continuations of up to L tokens and estimate H ≈ -(1/N) Σ log p_LM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from replica.feature.backends.hub import hub_id
from replica.feature.core.annotation.alignment import TEXT_OFFSET, tokenize_spans

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput
from ._base import _BaseEntropyEstimatorLM
from .utils import _clark_windows, _renyi_entropy_bits_from_log_p_nats

if TYPE_CHECKING:
    import torch


class ContinuationEntropyLM(_BaseEntropyEstimatorLM):
    """MC estimate of the paper's sequence-level entropy.

    Sample ``n_samples`` continuations of length ``max_tokens`` by manual
    ancestral sampling, compute their joint log-probabilities, and return

        H ≈ -(1/N) Σᵢ log p_LM(c⁽ⁱ⁾ | ctx)   in bits.

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_identity: Verified Hub source/revision or local bundle ``sha256``.
            Assertions are checked against loaded files; omitted identities resolve
            from the model source.
        n_samples: N, the number of continuations drawn per unit; more draws
            lower the variance of the estimate at a proportional cost.
        max_tokens: L, the length of each continuation in tokens.
        chunk_size: Split ``n_samples`` into batches of this size so the
            GPU only holds K samples' worth of KV cache. Under the default
            sampler each sample keeps its own uniforms whatever the batch, so
            only the memory peak changes; under the advancing-stream sampler
            the draw order changes too.

    Samples directly from ``softmax(raw_logits)`` step-by-step (no
    ``model.generate``), unlike the paper's ``measures.py``, which reshapes
    instead of transposing and inherits ``top_k=50``. Output
    therefore disagrees with the paper's ``sequence_entropy_*`` CSV column;
    to reproduce paper numbers as-is, regress on that column via
    ``Giulianelli2024Generalized.evaluate_swap``.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is ContinuationEntropyLM

    def __init__(
        self,
        model_id: str = "gpt2",
        *,
        n_samples: int = 64,
        max_tokens: int = 10,
        add_bos: bool = True,
        seed: int | None = None,
        model_identity: dict | None = None,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
        chunk_size: int | None = None,
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
        if int(max_tokens) < 1:
            raise ValueError(f"max_tokens must be ≥ 1, got {max_tokens}")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be ≥ 1 if given, got {chunk_size}")
        self.n_samples = int(n_samples)
        self.max_tokens = int(max_tokens)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.name = f"cont_entropy_{hub_id(model_id)}_N{n_samples}_L{max_tokens}"

    def _empty_values(self) -> dict:
        return {"continuation_entropy": np.nan}

    def _wrap(self, value: float) -> dict:
        return {"continuation_entropy": value}

    def _estimate_with_samples(
        self,
        prefix_ids: list[int],
    ) -> tuple[float, np.ndarray]:
        """Manual step-by-step ancestral sampling.

        Deliberately does not use ``model.generate``: ``generate`` returns
        *post-processed* scores (logits processors can mask tokens to
        ``-inf``), so reconstructing ``log p(sample)`` from them gives a
        biased estimate. This forwards the prefix, takes raw logits, and
        samples one token at a time so per-sample log-prob is identifiable.

        Returns ``(H_bits, log_p_samples_nats)``. Storing samples in the
        pickle means any future fix to the aggregator can be applied
        without a fresh MC run.
        """
        chunk = self.chunk_size or self.n_samples
        pieces: list[float] = []
        remaining = self.n_samples
        while remaining > 0:
            k = min(chunk, remaining)
            pieces.extend(
                self._sample_log_p_chunk(prefix_ids, k, self.n_samples - remaining)
            )
            remaining -= k
        samples = np.asarray(pieces, dtype=np.float64)
        # All samples are equally weighted: H ≈ -(1/N) Σ log p (nats → bits).
        H_bits = _renyi_entropy_bits_from_log_p_nats(samples, 1.0)
        return H_bits, samples

    def _sample_log_p_chunk(
        self, prefix_ids: list[int], k: int, offset: int = 0
    ) -> list[float]:
        """Samples ``k`` continuations in one batch and returns their joint
        log-probs (nats).
        """
        import torch

        gen = self._rng
        L = self.max_tokens
        # The prefix length identifies the word within the stimulus.
        unif = (
            None
            if self.release_rng_stream
            else self._uniform_block(("p", len(prefix_ids)), offset, k, L)
        )
        ctx = torch.tensor([prefix_ids], device=self._device, dtype=torch.long)
        ctx_b = ctx.repeat(k, 1)

        log_p = torch.zeros(k, device=self._device, dtype=torch.float64)
        with torch.no_grad():
            out = self._model(ctx_b, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].double()
            probs = torch.softmax(logits, dim=-1)
            sample = (
                torch.multinomial(probs, 1, generator=gen)
                if unif is None
                else self._inverse_cdf(probs, unif[:, 0])
            )
            sample_p = probs.gather(1, sample).squeeze(1).clamp_min(1e-40)
            log_p += torch.log(sample_p)
            cur = sample
            for step in range(1, L):
                out = self._model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].double()
                probs = torch.softmax(logits, dim=-1)
                sample = (
                    torch.multinomial(probs, 1, generator=gen)
                    if unif is None
                    else self._inverse_cdf(probs, unif[:, step])
                )
                sample_p = probs.gather(1, sample).squeeze(1).clamp_min(1e-40)
                log_p += torch.log(sample_p)
                cur = sample
        return log_p.detach().cpu().tolist()

    # ── One-pass stimulus-level driver (shared prefix) ──
    #
    # Mirror of :meth:`MCWordEntropyLM.__call__` adapted for fixed-length
    # continuations. At each REPLICA word boundary, fork the KV cache and
    # sample L tokens from the full vocab (no word-boundary stopping), which
    # shares one prefix forward across every word boundary in a stimulus.

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One-pass driver for continuation entropy.

        The master cache is built once per chunk of samples and forked at each
        word boundary to draw ``max_tokens``-long continuations; a stimulus with
        ``max_prefix + 1 + max_tokens > ctx_size`` is split into 50%-overlap
        windows instead, dropping left context beyond ``ctx_size``.

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

        for wi, samples in self._sample_positions(token_ids, sample_positions).items():
            per_unit_samples[wi] = samples
            # Shannon mean: -E[log p_nats] / ln 2 → bits.
            per_unit_entropy[wi] = _renyi_entropy_bits_from_log_p_nats(samples, 1.0)

        return self._outputs_with_samples(per_unit_entropy, per_unit_samples)

    def score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
    ) -> list[FeatureModelOutput]:
        """Continuation entropy at every model token of *stimulus*, in text order.

        The draws are seeded per stimulus from the seed, the model digest and
        the text, and keyed by token position, so they do not depend on the word
        layer or on what was scored before. The first token scored without a
        beginning-of-sequence token has no context and is NaN.

        Args:
            stimulus: Text to score.
            extract: Part of the shared signature; both keys are returned.

        Returns:
            One record per token, carrying its index, character span, id and
            decoded string, the entropy estimated at the position that predicts
            it and the per-sample joint log probabilities.
        """
        ids, spans = tokenize_spans(stimulus.text, self._tokenizer)
        n_tokens = len(ids)
        self._seed_for_stimulus(stimulus, None)
        prefix = self._resolved_start_ids() if self.add_bos else []
        token_ids = [*prefix, *ids]
        bos_offset = len(prefix)
        positions = [
            (t, t + bos_offset - 1) for t in range(n_tokens) if t + bos_offset - 1 >= 0
        ]
        entropy = np.full(n_tokens, np.nan)
        samples: list[np.ndarray | None] = [None] * n_tokens
        if positions:
            for t, draws in self._sample_positions(token_ids, positions).items():
                samples[t] = draws
                entropy[t] = _renyi_entropy_bits_from_log_p_nats(draws, 1.0)
        offset = stimulus.metadata.get(TEXT_OFFSET, 0)
        outputs = self._outputs_with_samples(entropy, samples)
        for t, output in enumerate(outputs):
            output.metadata.update(
                token_index=t,
                span=(spans[t].start + offset, spans[t].end + offset),
                token_id=int(ids[t]),
                token=self._tokenizer.decode([ids[t]]),
            )
        return outputs

    def window(self) -> dict:
        """The context rule of the one-pass driver, for provenance.

        Returns:
            ``rule``, ``max_positions``, ``half_size`` and ``continuation``. A
            stimulus whose sampled prefixes plus the continuation budget fit in
            ``max_positions`` is scored in one window. A longer one is split
            into half windows of ``half_size`` tokens, a window being two
            consecutive halves; the positions of the second half are scored in
            that window and the first window scores both of its halves.
        """
        ctx_size = self._model_max_ctx_size()
        return {
            "rule": "half_windows",
            "max_positions": ctx_size,
            "half_size": ctx_size // 2 - self.max_tokens,
            "continuation": self.max_tokens,
        }

    def sampling(self) -> dict:
        """How the per-token table is estimated, for provenance."""
        return {
            "positions": "all",
            "seed": self._seed,
            "seeding": "per stimulus from the seed, the model digest and the "
            "text; uniforms keyed by token position and sample index",
            "scheme": self._provenance["seed_scheme"],
            "chunk_size": self.chunk_size,
        }

    def _sample_positions(
        self, token_ids: list[int], sample_positions: list[tuple[int, int]]
    ) -> dict[int, np.ndarray]:
        """Joint log probabilities of the draws at every ``(unit, prefix end)``.

        Same window dispatch as ``MCWordEntropyLM``: the single-window
        driver when the whole stimulus fits in ``ctx_size - max_tokens``,
        otherwise 50%-overlap half-windows of size ``ctx_size // 2 -
        max_tokens`` (a full window is two consecutive halves).
        """
        ctx_size = self._model_max_ctx_size()
        max_prefix = max(p for _, p in sample_positions)
        single_window = max_prefix + 1 + self.max_tokens <= ctx_size

        def dispatch(k, offset):
            if single_window:
                return self._estimate_continuations_one_pass(
                    token_ids,
                    sample_positions,
                    k,
                    offset,
                )
            return self._estimate_continuations_one_pass_windowed(
                token_ids,
                sample_positions,
                k,
                ctx_size,
                offset,
            )

        return self._collect_sample_chunks(dispatch)

    def _estimate_continuations_one_pass(
        self,
        token_ids: list[int],
        sample_positions: list[tuple[int, int]],
        k: int,
        offset: int = 0,
    ) -> dict[int, np.ndarray]:
        """Single pass through ``token_ids`` with ``k`` parallel sample
        streams. At each ``(word_i, last_prefix_token)`` position the KV
        cache is forked and ``max_tokens`` tokens are sampled from the
        full vocabulary; the per-sample joint log-prob (nats) is recorded
        for ``word_i``.
        """
        import torch

        gen = self._rng
        device = self._device

        sorted_positions = sorted(sample_positions, key=lambda x: x[1])
        word_log_probs: dict[int, np.ndarray] = {}

        # Initial batched forward of the prefix up to and including the
        # first sampling position.
        first_pos = sorted_positions[0][1]
        init_ids = token_ids[: first_pos + 1]
        ctx = torch.tensor([init_ids], device=device, dtype=torch.long)
        ctx_b = ctx.repeat(k, 1)
        with torch.no_grad():
            out = self._model(ctx_b, use_cache=True)
            master_kv = out.past_key_values.to_legacy_cache()
            last_logits = out.logits[:, -1, :].double()
        # Release the full prefix logits and the initial cache container; the
        # final logits and current cache are retained explicitly above.
        del out

        # Sample the first word's continuation.
        first_word_i = sorted_positions[0][0]
        log_p = self._sample_continuation_log_p(
            last_logits,
            master_kv,
            k,
            gen,
            first_word_i,
            offset,
        )
        word_log_probs[first_word_i] = log_p.detach().cpu().numpy()

        cur_pos = first_pos + 1
        for word_i, target_pos in sorted_positions[1:]:
            cur_pos, master_kv, last_logits = self._advance_master_cache(
                token_ids, cur_pos, target_pos, k, master_kv, last_logits
            )
            log_p = self._sample_continuation_log_p(
                last_logits,
                master_kv,
                k,
                gen,
                word_i,
                offset,
            )
            word_log_probs[word_i] = log_p.detach().cpu().numpy()

        return word_log_probs

    def _estimate_continuations_one_pass_windowed(
        self,
        token_ids: list[int],
        sample_positions: list[tuple[int, int]],
        k: int,
        ctx_size: int,
        offset: int = 0,
    ) -> dict[int, np.ndarray]:
        """50%-overlap windowed driver for stimuli that exceed the LM's
        positional budget.

        Identical structure to
        :meth:`MCWordEntropyLM._estimate_words_one_pass_windowed`,
        but with ``max_tokens`` (fixed continuation
        length) in place of ``max_iter`` (variable-length word
        sampling). Each window's master pass goes through
        :meth:`_estimate_continuations_one_pass`, so each window runs the
        same code path as the un-windowed driver (on its truncated
        left context).
        """
        half_size = ctx_size // 2 - self.max_tokens
        if half_size < 1:
            raise ValueError(
                f"_estimate_continuations_one_pass_windowed: ctx_size="
                f"{ctx_size} too small to fit max_tokens={self.max_tokens}"
            )

        word_log_probs: dict[int, np.ndarray] = {}
        for window_tokens, window_samples in _clark_windows(
            token_ids, sample_positions, half_size
        ):
            sub = self._estimate_continuations_one_pass(
                window_tokens,
                window_samples,
                k,
                offset,
            )
            word_log_probs.update(sub)

        return word_log_probs

    def _sample_continuation_log_p(
        self,
        logits_after_prev: torch.Tensor,
        master_kv: tuple,
        k: int,
        gen: torch.Generator | None,
        word_index: int = 0,
        offset: int = 0,
    ) -> torch.Tensor:
        """Sample ``k`` length-``max_tokens`` continuations from the LM
        state described by ``logits_after_prev`` + ``master_kv``. Returns
        joint log-probabilities (nats), shape ``[k]``.

        Same estimator as the paper's single-position sequence entropy:
        ancestral sampling from the full-vocabulary softmax, summed over
        ``max_tokens`` steps. Differs from the per-prefix path only in that
        the prefix forward is amortised across word boundaries in a stimulus.
        """
        import torch
        from transformers import DynamicCache

        device = self._device
        log_p = torch.zeros(k, device=device, dtype=torch.float64)
        # Keyed on the word index: _clark_windows re-indexes positions per
        # window, so two words can share a window-local position.
        unif = (
            None
            if self.release_rng_stream
            else self._uniform_block(("c", word_index), offset, k, self.max_tokens)
        )

        # First sampled token: use the logits at the prefix's last
        # position.
        probs = torch.softmax(logits_after_prev, dim=-1)
        sample = (
            torch.multinomial(probs, 1, generator=gen)
            if unif is None
            else self._inverse_cdf(probs, unif[:, 0])
        )
        sample_p = probs.gather(1, sample).squeeze(1).clamp_min(1e-40)
        log_p += torch.log(sample_p)

        sample_kv_dyn = DynamicCache.from_legacy_cache(master_kv)
        cur = sample
        with torch.no_grad():
            for step in range(1, self.max_tokens):
                out = self._model(
                    cur,
                    past_key_values=sample_kv_dyn,
                    use_cache=True,
                )
                sample_kv_dyn = out.past_key_values
                step_logits = out.logits[:, -1, :].double()
                probs = torch.softmax(step_logits, dim=-1)
                sample = (
                    torch.multinomial(probs, 1, generator=gen)
                    if unif is None
                    else self._inverse_cdf(probs, unif[:, step])
                )
                sample_p = probs.gather(1, sample).squeeze(1).clamp_min(1e-40)
                log_p += torch.log(sample_p)
                cur = sample
        return log_p
