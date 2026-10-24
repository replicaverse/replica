"""Generalized responsivity measures from Giulianelli et al. (2024a).

Reimplements the *decontextualised* responsivity distances from
*Generalized Measures of Anticipation and Responsivity in Online Language
Processing*.

For each word ``w``: sample ``n_samples`` alternative tokens from the LM
at the position preceding the first subword of ``w``, look up input-embedding
representations of the observed token and each alternative, then return:

  * ``decontextualised_distance``, mean cosine distance from observed to
    alternatives (= responsivity: how much ``w`` differs from expectation).
  * ``expected_decontextualised_distance``, mean pairwise cosine distance
    among alternatives only (= expected responsivity: spread of the LM's
    predictive distribution in embedding space).

"Decontextualised" because the embedding lookup uses no transformer layers
(input embeddings only). ``importance_temp != 1.0`` samples from a tempered
proposal and reweights the draws; see ``expected_is_correction``. Distances use
the first subword of each word.

Notes:
    * The first word's distances are NaN (no left context to sample from).
    * Sampling is seeded per stimulus so reruns are reproducible.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from replica.feature.backends.generation import (
    usable_position_limit,
    validate_context_window,
)
from replica.feature.backends.hub import hub_id
from replica.feature.backends.start_state import start_ids as resolve_start_ids
from replica.feature.backends.start_state import validate_start_ids
from replica.feature.core.annotation.alignment import (
    TEXT_OFFSET,
    aligned_token_ids,
    tokenize_spans,
)

from ..core.protocols import FeatureModelOutput

if TYPE_CHECKING:
    from ...core import Stimulus


class ResponsivityLM:
    """Decontextualised responsivity FeatureModel.

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_name: Hugging Face causal LM supplying both the next-token
            distribution and the input-embedding matrix.
        n_samples: Alternative-token samples per word position. The authors'
            released code defaults to 4; their cross-validation run used 512.
        importance_temp: Proposal-distribution temperature. ``1.0`` draws
            from the model's own distribution and avoids importance
            reweighting; >1 flattens the proposal (more diverse) at the
            cost of an importance correction; <1 sharpens it.
        device: Torch device the model and the distance arithmetic run on.
        seed: Seed of the per-call sampling generator, so two calls with the
            same arguments return the same numbers.
        expected_is_correction: How the *expected* distance reweights its two
            importance-sampled draws (only matters when ``importance_temp !=
            1``). ``"unbiased"`` (default) weights both draws by ``w = p/q``,
            the unbiased estimator of the paper's estimand ``E_{a,a'~p}[d]``
            (proposal-invariant: same limit as ``importance_temp=1``).
            ``"reference"`` weights only one draw, mirroring the authors'
            ``measures.py`` weighting; that estimator is biased toward the
            mixed limit ``Σ p(v) q(v') d(v, v')``.
        add_bos: Prepend the model's start token so the first word also
            gets a defined distance. Default False (first word is NaN).
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is ResponsivityLM

    def __init__(
        self,
        model_name: str = "gpt2",
        n_samples: int = 100,
        importance_temp: float = 1.0,
        device: str = "cpu",
        seed: int = 0,
        add_bos: bool = False,
        expected_is_correction: str = "unbiased",
        *,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
    ):
        """Record the sampling settings; the model loads on first call.

        The class docstring describes the parameters.

        Raises:
            ValueError: When *expected_is_correction* is neither ``unbiased``
                nor ``reference``.
        """
        if expected_is_correction not in ("unbiased", "reference"):
            raise ValueError(
                "expected_is_correction must be 'unbiased' (weight both IS "
                "draws) or 'reference' (weight one draw, as in the authors' "
                "released estimator)."
            )
        self.name = f"{hub_id(model_name)}_responsivity"
        self.model_name = model_name
        self.n_samples = int(n_samples)
        self.importance_temp = float(importance_temp)
        self.device = device
        self.seed = int(seed)
        self.add_bos = add_bos
        self.context_window = validate_context_window(context_window)
        self._requested_start_ids = validate_start_ids(start_ids, add_bos=add_bos)
        self.expected_is_correction = expected_is_correction
        if expected_is_correction == "reference" and self.importance_temp != 1.0:
            warnings.warn(
                "ResponsivityLM(expected_is_correction='reference') weights only "
                "one importance-sampling draw, so the expected distance is biased "
                "toward the mixed limit sum p(v) q(v') d(v, v') instead of the "
                "paper's E_{a,a'~p}[d]. Use 'unbiased' for the proposal-invariant "
                "estimand.",
                stacklevel=2,
            )
        self._model = None
        self._tokenizer = None
        # Input-embedding matrix (vocab, embed); position-0 positional embedding.
        # The paper computes hidden_states[0] of a single-token forward pass =
        # wte(t) + wpe(0); RoPE-style models get a zero vector.
        self._token_embeds = None
        self._wpe0 = None

    @property
    def vocab(self):
        """Always ``None``; the outputs are distances, which carry no vocabulary."""
        return None  # FeatureModel protocol stub.

    def _start_prefix(self) -> list[int]:
        """Resolve and cache the prefix used for alignment and scoring."""
        if not hasattr(self, "_start_prefix_ids"):
            self._start_prefix_ids = tuple(
                resolve_start_ids(
                    self._tokenizer,
                    model_name=self.model_name,
                    override=getattr(self, "_requested_start_ids", None),
                    add_bos=self.add_bos,
                )
            )
        return list(self._start_prefix_ids)

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._max_ctx()
        self._start_prefix()

        # Embedding matrix (vocab_size, embed_dim).
        with torch.no_grad():
            self._token_embeds = self._model.get_input_embeddings().weight.detach()
            self._wpe0 = self._lookup_position0_embedding(torch)

    def _lookup_position0_embedding(self, torch):
        """Position-0 positional embedding ``(1, embed)``; zeros for RoPE-style models."""
        wpe = None
        for path in ("transformer.wpe", "model.wpe", "wpe"):
            obj = self._model
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "weight"):
                wpe = obj
                break
        if wpe is None:
            embed_dim = self._token_embeds.shape[1]
            return torch.zeros(1, embed_dim, device=self.device)
        return wpe.weight[0:1].detach()

    def _max_ctx(self) -> int:
        """Resolve and cache the effective token budget."""
        if not hasattr(self, "_context_limit"):
            self._context_limit = usable_position_limit(
                self.model_name,
                self._model.config,
                context_window=getattr(self, "context_window", None),
            )
        return self._context_limit

    def _all_logits(self, input_ids, n_tokens: int):
        """(n_tokens, vocab) next-token logits, sliding-window for long inputs.

        Window k>0 scores the positions from the previous window's end on, so
        the first window covering a position scores it, with at least half a
        window of left context (matches HuggingFaceLM).
        """
        import torch

        max_len = self._max_ctx()
        with torch.no_grad():
            if n_tokens <= max_len:
                return self._model(input_ids).logits[0]
            stride = max_len // 2
            vocab = self._model.config.vocab_size
            logits = torch.zeros(n_tokens, vocab, device=self.device)
            filled = torch.zeros(n_tokens, dtype=torch.bool)
            for start in range(0, n_tokens, stride):
                end = min(start + max_len, n_tokens)
                chunk = self._model(input_ids[:, start:end]).logits[0]
                fill_start = start if start == 0 else start + max_len - stride
                mask = ~filled[fill_start:end]
                logits[fill_start:end][mask] = chunk[
                    (fill_start - start) : (end - start)
                ][mask]
                filled[fill_start:end] = True
                if end == n_tokens:
                    break
            return logits

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Both decontextualised distances at every unit of *stimulus*.

        Args:
            stimulus: Passage to score.
            extract: Part of the shared FeatureModel signature; both distances
                are returned regardless.
            target_layer: Annotation layer whose units the distances align to;
                each unit is represented by its first subword.

        Returns:
            One output per unit, carrying ``decontextualised_distance`` and
            ``expected_decontextualised_distance``. A unit with no predicting
            position, the first one without ``add_bos``, is NaN in both.
        """
        import torch

        self._ensure_loaded()

        token_ids, alignment, bos_offset, n_units = aligned_token_ids(
            stimulus,
            self._tokenizer,
            target_layer,
            self.add_bos,
            start_ids=self._start_prefix(),
        )

        if len(token_ids) == bos_offset:
            return [
                FeatureModelOutput(
                    values={
                        "decontextualised_distance": np.nan,
                        "expected_decontextualised_distance": np.nan,
                    },
                    metadata={
                        "model_name": self.name,
                        "start_ids": self._start_prefix(),
                        "context_window": self._max_ctx(),
                    },
                )
                for _ in range(n_units)
            ]

        input_ids = torch.tensor([token_ids], device=self.device)

        # Next-token logits at every position; 50%-overlap sliding windows for
        # stimuli over the context window (each position scored from the window
        # giving it the most left context), mirroring HuggingFaceLM.
        logits = self._all_logits(input_ids, len(token_ids))

        # Per-word distances.
        decontextualised = np.full(n_units, np.nan)
        expected_decontextualised = np.full(n_units, np.nan)

        # Seeded generator so two calls with the same args return the same
        # numbers (without globally seeding torch).
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        for w_idx in range(n_units):
            tok_indices = alignment.mapping[w_idx]
            if not tok_indices:
                continue
            first_tok_idx = tok_indices[0] + bos_offset

            # The distribution that predicts the first subword of word w_idx
            # comes from position (first_tok_idx - 1). Need ≥0.
            pred_pos = first_tok_idx - 1
            if pred_pos < 0:
                continue  # first word, no BOS → NaN
            decontextualised[w_idx], expected_decontextualised[w_idx] = self._distances(
                logits[pred_pos], int(input_ids[0, first_tok_idx]), gen
            )

        return [
            FeatureModelOutput(
                values={
                    "decontextualised_distance": float(decontextualised[w]),
                    "expected_decontextualised_distance": float(
                        expected_decontextualised[w]
                    ),
                },
                metadata={
                    "model_name": self.name,
                    "start_ids": self._start_prefix(),
                    "context_window": self._max_ctx(),
                },
            )
            for w in range(n_units)
        ]

    def score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
    ) -> list[FeatureModelOutput]:
        """Both decontextualised distances at every model token of *stimulus*.

        The generator is reseeded with the seed at every stimulus and draws once
        per scored position in text order. The first token scored without a
        beginning-of-sequence token has no predicting position and is NaN.

        Args:
            stimulus: Text to score.
            extract: Part of the shared signature; both distances are returned.

        Returns:
            One record per token, carrying its index, character span, id and
            decoded string.
        """
        import torch

        self._ensure_loaded()
        ids, spans = tokenize_spans(stimulus.text, self._tokenizer)
        if not ids:
            return []
        prefix = self._start_prefix()
        token_ids = [*prefix, *ids]
        bos_offset = len(prefix)
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self._all_logits(input_ids, len(token_ids))
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        n_tokens = len(ids)
        decontextualised = np.full(n_tokens, np.nan)
        expected = np.full(n_tokens, np.nan)
        for t in range(n_tokens):
            pred_pos = t + bos_offset - 1
            if pred_pos < 0:
                continue
            decontextualised[t], expected[t] = self._distances(
                logits[pred_pos], int(token_ids[t + bos_offset]), gen
            )
        offset = stimulus.metadata.get(TEXT_OFFSET, 0)
        return [
            FeatureModelOutput(
                values={
                    "decontextualised_distance": float(decontextualised[t]),
                    "expected_decontextualised_distance": float(expected[t]),
                },
                metadata={
                    "model_name": self.name,
                    "start_ids": self._start_prefix(),
                    "context_window": self._max_ctx(),
                    "token_index": t,
                    "span": (spans[t].start + offset, spans[t].end + offset),
                    "token_id": int(ids[t]),
                    "token": self._tokenizer.decode([ids[t]]),
                },
            )
            for t in range(n_tokens)
        ]

    def window(self) -> dict:
        """The context-window rule the forward pass applies, for provenance.

        Windows of ``max_positions`` tokens advance by ``stride``; a later
        window scores the positions from the previous window's end on.
        """
        self._ensure_loaded()
        max_len = self._max_ctx()
        return {
            "rule": "sliding",
            "max_positions": max_len,
            "stride": max_len // 2,
            "fresh_from": "start + max_positions - stride",
        }

    def sampling(self) -> dict:
        """How the per-token table is estimated, for provenance."""
        return {
            "positions": "all",
            "seed": self.seed,
            "seeding": "reseeded with the seed at every stimulus; one draw of "
            "n_samples alternatives per scored position, in text order",
        }

    def _distances(self, row, target_id: int, gen) -> tuple[float, float]:
        """Observed and expected distance at one predicting position.

        Args:
            row: Next-token logits at the predicting position.
            target_id: Id of the token observed at the predicted position.
            gen: Generator the alternatives are drawn with.
        """
        import torch

        # Sample n_samples alternative tokens from this distribution
        # (with optional temperature).
        standard_log_probs = torch.log_softmax(row, dim=-1)
        if self.importance_temp == 1.0:
            proposal_probs = torch.softmax(row, dim=-1)
            samples = torch.multinomial(
                proposal_probs,
                num_samples=self.n_samples,
                replacement=True,
                generator=gen,
            )
            importance_weights = None
        else:
            proposal_log_probs = torch.log_softmax(row / self.importance_temp, dim=-1)
            proposal_probs = torch.exp(proposal_log_probs)
            samples = torch.multinomial(
                proposal_probs,
                num_samples=self.n_samples,
                replacement=True,
                generator=gen,
            )
            importance_weights = torch.exp(
                standard_log_probs[samples] - proposal_log_probs[samples]
            )

        # Look up "decontextualised" embeddings: input embedding +
        # position-0 positional embedding (matches the paper's
        # hidden_states[0] of a single-token forward pass).
        target_emb = self._token_embeds[target_id : target_id + 1] + self._wpe0
        alt_embs = self._token_embeds[samples] + self._wpe0  # (n_samples, embed)

        # 1) Mean cosine distance between target and alternatives.
        target_norm = target_emb / target_emb.norm(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        alt_norm = alt_embs / alt_embs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos_target_alt = (alt_norm * target_norm).sum(dim=-1)  # (n_samples,)
        target_distances = 1.0 - cos_target_alt
        if importance_weights is not None:
            target_distances = target_distances * importance_weights
        observed = float(target_distances.mean().item())

        # 2) Mean pairwise cosine distance among alternatives, excluding
        # self-pairs (the diagonal). cdist-style: D = 1 - A A^T.
        sim = alt_norm @ alt_norm.T  # (n, n) cosine similarities
        dist = 1.0 - sim
        mask = 1.0 - torch.eye(self.n_samples, device=self.device)
        if importance_weights is not None:
            # Both members of each alternative pair come from the proposal q:
            # "unbiased" weights each pair by w_i*w_j, "reference" by w_i
            # alone (see ``expected_is_correction``).
            w = importance_weights
            denom = mask.sum(dim=1).clamp_min(1.0)
            if self.expected_is_correction == "unbiased":
                row_means = (dist * mask * w.unsqueeze(0)).sum(dim=1) / denom
            else:
                row_means = (dist * mask).sum(dim=1) / denom
            return observed, float((row_means * w).mean().item())
        return observed, float(((dist * mask).sum() / mask.sum()).item())
