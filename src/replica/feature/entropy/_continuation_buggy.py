"""Continuation-entropy estimator as released by Giulianelli et al. (2024a).

Reproduces the released ``src/measures.py`` in rycolab/generalized-surprisal
as is, so the multiverse grid can contrast the released and corrected
predictors. The corrected estimator
is :class:`ContinuationEntropyLM` in :mod:`._continuation`.
"""

import math

import numpy as np

from replica.feature.backends.hub import hub_id

from ..core.reproduction import warn_reproduction_only
from ._base import _BaseEntropyEstimatorLM


class ContinuationEntropyLMBuggy(_BaseEntropyEstimatorLM):
    """MC sequence-level entropy as the released ``measures.py`` computes it.

    Like :class:`ContinuationEntropyLM`, draws ``n_samples`` continuations of
    ``max_tokens`` tokens at each word boundary and returns
    ``H ≈ -(1/N) Σᵢ log p_LM(c⁽ⁱ⁾ | ctx)`` in bits. Unlike the corrected
    estimator, it samples with ``model.generate`` and reconstructs the joint
    log-probabilities exactly as the released code does, which gives the
    estimate two properties:

    1. ``do_sample=True`` with no explicit ``top_k`` takes the
       ``GenerationConfig`` default ``top_k=50``, so samples come from the truncated
       top-50 distribution, while the joint log-prob is taken over the full
       vocabulary (``measures.py`` lines ~251-266).
    2. ``torch.reshape`` of the stacked ``(max_new_tokens, n_samples, vocab)``
       logits to ``(n_samples, max_new_tokens, vocab)`` reinterprets the buffer
       instead of transposing axes, so the ``(sample, step)`` a logit row is
       read as follows the buffer order (``measures.py`` line 265).

    These operations do not compute full-distribution continuation entropy.
    This class is for reproduction only; use :class:`ContinuationEntropyLM`
    for new analyses.

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_identity: Verified Hub source/revision or local bundle ``sha256``.
            Assertions are checked against loaded files; omitted identities resolve
            from the model source.
        allow_reproduction: Required acknowledgement for this reproduction-only estimator.
        n_samples: N continuations per word boundary.
        max_tokens: L tokens per continuation.
        chunk_size: Accepted for compute-driver signature parity but not used.
            The reshape reorders rows within a single ``generate`` batch of N
            samples, so batching would change the estimate; it is therefore
            always drawn in one batch.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is ContinuationEntropyLMBuggy

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
        allow_reproduction: bool = False,
    ):
        warn_reproduction_only(
            "continuation_entropy_buggy", allow_reproduction=allow_reproduction
        )
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
        self.n_samples = int(n_samples)
        self.max_tokens = int(max_tokens)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        self.name = f"cont_entropy_buggy_{hub_id(model_id)}_N{n_samples}_L{max_tokens}"

    @property
    def _provenance(self) -> dict:
        """Record the released generator's global RNG and per-input seed reset."""
        return {
            **super()._provenance,
            "sampler": "transformers.generate",
            "seed_scheme": "global-per-stimulus"
            if self._seed is not None
            else "global-unseeded",
        }

    def _empty_values(self) -> dict:
        return {"continuation_entropy_buggy": np.nan}

    def _wrap(self, value: float) -> dict:
        return {"continuation_entropy_buggy": value}

    def __call__(self, stimulus, extract=None, target_layer="words"):
        """Continuation entropy per unit, seeding torch's global RNG per stimulus.

        Args:
            stimulus: Text whose word boundaries are scored.
            extract: Part of the shared FeatureModel signature, passed on to the
                base estimator.
            target_layer: Annotation layer whose units are the words scored.
        """
        # Seed the global RNG once per stimulus, as measures.py does per input
        # (``model.generate`` draws from the global RNG, not ``self._rng``).
        if self._seed is not None:
            import torch

            torch.manual_seed(int(self._seed))
        return super().__call__(stimulus, extract=extract, target_layer=target_layer)

    def _estimate_with_samples(
        self,
        prefix_ids: list[int],
    ) -> tuple[float, np.ndarray]:
        import torch

        L = self.max_tokens
        N = self.n_samples
        # Keep prefix + continuation within the positional budget by keeping
        # the prefix's last tokens, which drops the BOS. measures.py
        # re-tokenizes the text in windows of max_seq_len - 1 tokens behind a
        # fresh BOS, a later window restarting at the text after the
        # 200th-from-last token of the previous one.
        ctx_size = self._model_max_ctx_size()
        if L >= ctx_size:
            raise ValueError("context_window must exceed max_tokens")
        if len(prefix_ids) + L > ctx_size:
            prefix_ids = prefix_ids[-(ctx_size - L) :]
        ctx_len = len(prefix_ids)

        ctx = torch.tensor([prefix_ids], device=self._device, dtype=torch.long)
        attn = torch.ones_like(ctx)
        with torch.no_grad():
            out = self._model.generate(
                ctx,
                attention_mask=attn,
                max_new_tokens=L,
                pad_token_id=self._tokenizer.eos_token_id,
                do_sample=True,  # property 1: GenerationConfig default top_k=50
                num_return_sequences=N,
                return_dict_in_generate=True,
                output_logits=True,
                temperature=1.0,
            )
            # property 2: reshape reinterprets the (L, N, V) buffer as (N, L, V)
            # rather than transposing, so each row's logits do not match
            # its (sample, step). Verbatim from measures.py:264-266.
            logits = torch.stack(out.logits)
            logits = torch.reshape(logits, (N, L, logits.shape[2]))
            logprobs_all = torch.nn.functional.log_softmax(logits, dim=2)
            idx = out.sequences[:, ctx_len : ctx_len + L]
            logprobs = torch.gather(
                logprobs_all,
                dim=2,
                index=idx.unsqueeze(-1),
            ).squeeze(-1)
            # Joint log p (nats) per sampled continuation; mean surprisal in
            # bits over samples is the released entropy estimate.
            joint_nats = logprobs.double().sum(dim=1)

        samples = joint_nats.detach().cpu().numpy().astype(np.float64)
        H_bits = float(-samples.mean() / math.log(2))
        return H_bits, samples
