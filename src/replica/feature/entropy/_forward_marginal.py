"""Monte Carlo marginal and conditional entropy profiles over future tokens.

For each word's context, sample continuations and average their next-token
probability distributions at each depth. The entropy of this mixture estimates
the marginal entropy at that depth, integrating over the intervening tokens.
Outputs contain one array per word, indexed by future token depth from one:

* ``forward_marginal_entropy``: entropy of the sampled mixture.
* ``free_step_entropy``: mean entropy of the conditional distributions.
* ``joint_entropy``: mean sampled continuation surprisal through each depth.

All values are in bits. Sampling uses the configured start token, context window
and random seed. Probability accumulators use memory proportional to the number
of words, look-ahead depth and vocabulary size.
"""

import numpy as np

from replica.feature.backends.hub import hub_id
from replica.feature.core.outputs.information import LN2

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput
from ._base import _BaseEntropyEstimatorLM
from .utils import _clark_windows


class ForwardMarginalEntropyLM(_BaseEntropyEstimatorLM):
    """MC forward marginal look-ahead entropy (see module docstring).

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_identity: Verified Hub source/revision or local bundle ``sha256``.
            Assertions are checked against loaded files; omitted identities resolve
            from the model source.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is ForwardMarginalEntropyLM

    def __init__(
        self,
        model_id: str = "gpt2",
        *,
        n_samples: int = 512,
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
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if int(max_tokens) < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be >= 1 if given, got {chunk_size}")
        self.n_samples = int(n_samples)
        self.max_tokens = int(max_tokens)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.name = f"fwd_marg_entropy_{hub_id(model_id)}_N{n_samples}_L{max_tokens}"

    def _empty_values(self) -> dict:
        nan = np.full(self.max_tokens, np.nan)
        return {
            "forward_marginal_entropy": nan.copy(),
            "free_step_entropy": nan.copy(),
            "joint_entropy": nan.copy(),
        }

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Forward marginal, free-step and joint entropy profiles per unit.

        Args:
            stimulus: Text whose word boundaries are scored.
            extract: Part of the shared FeatureModel signature; every profile
                is returned.
            target_layer: Annotation layer whose units are the words scored.
        """
        token_ids, n_units, sample_positions, _ = self._prepare_word_sampling(
            stimulus, target_layer
        )
        per_unit: dict[int, dict] = {}
        if sample_positions:
            ctx_size = self._model_max_ctx_size()
            max_prefix = max(p for _, p in sample_positions)
            single_window = max_prefix + 1 + self.max_tokens <= ctx_size
            if single_window:
                per_unit = self._run_window(token_ids, sample_positions)
            else:
                per_unit = self._run_windowed(token_ids, sample_positions, ctx_size)

        return self._outputs_from_mapping(n_units, per_unit)

    # ------------------------------------------------------------------ #
    def _run_windowed(self, token_ids, sample_positions, ctx_size) -> dict[int, dict]:
        """50%-overlap windowing for over-context stimuli (mirrors
        :meth:`ContinuationEntropyLM._estimate_continuations_one_pass_windowed`).
        """
        half_size = ctx_size // 2 - self.max_tokens
        if half_size < 1:
            raise ValueError(
                f"ctx_size={ctx_size} too small for max_tokens={self.max_tokens}"
            )
        out: dict[int, dict] = {}
        for window_tokens, window_positions in _clark_windows(
            token_ids, sample_positions, half_size
        ):
            out.update(self._run_window(window_tokens, window_positions))
        return out

    def _run_window(self, token_ids, sample_positions) -> dict[int, dict]:
        """Accumulate, across sample chunks, the summed softmax / conditional
        entropy / cumulative log-prob for every word in this window, then
        reduce to the three depth profiles. Prefix forwards are shared across
        words within a chunk (one master KV pass, forked per word).
        """
        import torch

        device = self._device
        L, V = self.max_tokens, self._model.config.vocab_size
        words = [wi for wi, _ in sample_positions]
        widx = {wi: j for j, wi in enumerate(words)}
        W = len(words)
        # GPU accumulators over the window: summed softmax (fp32, for the
        # mixture), summed per-stream conditional entropy (bits), summed
        # cumulative log-prob (nats, for the joint).
        sum_soft = torch.zeros((W, L, V), device=device, dtype=torch.float32)
        sum_cond = torch.zeros((W, L), device=device, dtype=torch.float64)
        sum_cumlp = torch.zeros((W, L), device=device, dtype=torch.float64)

        chunk = self.chunk_size or self.n_samples
        remaining = self.n_samples
        while remaining > 0:
            k = min(chunk, remaining)
            self._accumulate_chunk(
                token_ids,
                sample_positions,
                k,
                self.n_samples - remaining,
                widx,
                sum_soft,
                sum_cond,
                sum_cumlp,
            )
            remaining -= k

        N = float(self.n_samples)
        mixture = (sum_soft / N).double()
        marg = -(mixture * mixture.clamp_min(1e-40).log()).sum(-1) / LN2  # [W,L]
        free = sum_cond / N
        joint = -(sum_cumlp / N) / LN2
        marg, free, joint = marg.cpu().numpy(), free.cpu().numpy(), joint.cpu().numpy()

        out: dict[int, dict] = {}
        for wi in words:
            j = widx[wi]
            out[wi] = {
                "forward_marginal_entropy": marg[j].astype(np.float64),
                "free_step_entropy": free[j].astype(np.float64),
                "joint_entropy": joint[j].astype(np.float64),
            }
        return out

    def _accumulate_chunk(
        self,
        token_ids,
        sample_positions,
        k,
        offset,
        widx,
        sum_soft,
        sum_cond,
        sum_cumlp,
    ) -> None:
        """One master pass through ``token_ids`` with ``k`` sample streams,
        forking at each word boundary to sample ``max_tokens`` tokens and
        accumulate into the window accumulators in place.
        """
        import torch

        gen, device = self._rng, self._device
        ordered = sorted(sample_positions, key=lambda x: x[1])

        first_pos = ordered[0][1]
        ctx = torch.tensor(
            [token_ids[: first_pos + 1]], device=device, dtype=torch.long
        )
        with torch.no_grad():
            out = self._model(ctx.repeat(k, 1), use_cache=True)
            master_kv = out.past_key_values.to_legacy_cache()
            last_logits = out.logits[:, -1, :].double()

        self._sample_accumulate(
            last_logits,
            master_kv,
            k,
            offset,
            gen,
            widx[ordered[0][0]],
            ordered[0][0],
            sum_soft,
            sum_cond,
            sum_cumlp,
        )

        cur_pos = first_pos + 1
        for word_i, target_pos in ordered[1:]:
            cur_pos, master_kv, last_logits = self._advance_master_cache(
                token_ids, cur_pos, target_pos, k, master_kv, last_logits
            )
            self._sample_accumulate(
                last_logits,
                master_kv,
                k,
                offset,
                gen,
                widx[word_i],
                word_i,
                sum_soft,
                sum_cond,
                sum_cumlp,
            )

    def _sample_accumulate(
        self,
        logits_after_prev,
        master_kv,
        k,
        offset,
        gen,
        j,
        word_index,
        sum_soft,
        sum_cond,
        sum_cumlp,
    ) -> None:
        """Sample ``k`` length-L continuations from the given LM state and add
        their per-step softmax / conditional entropy / cumulative log-prob into
        accumulator row ``j`` (one word).
        """
        import torch
        from transformers import DynamicCache

        cum = torch.zeros(k, device=self._device, dtype=torch.float64)
        logits = logits_after_prev
        kv = None
        cur = None  # sampled at d==0 below, before the d>0 branch reads it
        # Keyed on the word index: window-local positions are not unique.
        unif = (
            None
            if self.release_rng_stream
            else self._uniform_block(("m", word_index), offset, k, self.max_tokens)
        )
        with torch.no_grad():
            for d in range(self.max_tokens):
                if d > 0:
                    out = self._model(cur, past_key_values=kv, use_cache=True)
                    kv = out.past_key_values
                    logits = out.logits[:, -1, :].double()
                probs = torch.softmax(logits, dim=-1)
                sum_soft[j, d] += probs.sum(0).float()
                ent = -(probs * probs.clamp_min(1e-40).log()).sum(-1) / LN2
                sum_cond[j, d] += ent.sum()
                cur = (
                    torch.multinomial(probs, 1, generator=gen)
                    if unif is None
                    else self._inverse_cdf(probs, unif[:, d])
                )
                p = probs.gather(1, cur).squeeze(1).clamp_min(1e-40)
                cum = cum + torch.log(p)
                sum_cumlp[j, d] += cum.sum()
                if d == 0:
                    kv = DynamicCache.from_legacy_cache(master_kv)
