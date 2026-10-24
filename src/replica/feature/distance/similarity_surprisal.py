r"""Similarity-adjusted surprisal and next-word information value.

Implements the predictors of Meister et al. (2024),
"Towards a Similarity-adjusted Surprisal Theory". For a word
:math:`w_t` in context, with a similarity function :math:`z(w, w') \\in [0, 1]`
and next-token distribution :math:`p`:

* similarity-adjusted surprisal
  :math:`s_z(w_t) = -\\log \\sum_{w'} z(w_t, w')\\, p(w' | ctx)`
* next-word information value
  :math:`i(w_t) = \\sum_{w'} (1 - z(w_t, w'))\\, p(w' | ctx)`

Both derive from the same expected similarity
:math:`m = \\mathbb{E}_{w' \\sim p}[z(w_t, w')]` as ``-log(m)`` and ``1 - m``.
The paper's four similarity functions map onto two feature models here:

* :class:`SimilarityAdjustedSurprisalEmbeddingLM`: normalized cosine
  ``z = (cos + 1) / 2`` between LM token embeddings, at the subword level.
  Non-contextual (input-embedding) similarity computes ``m`` exactly over
  the full subword vocabulary; contextual (final-layer hidden state)
  similarity uses a Monte Carlo estimate over sampled next tokens.
  Word-level values sum over the word's subwords (the paper's headline
  aggregation; per-subword arrays are stored so the first-subword variant
  can be re-derived).
* :class:`SimilarityAdjustedSurprisalWordLM`: part-of-speech identity
  (flair ``pos-english-fast``) and normalized Levenshtein similarity between the
  observed word and Monte Carlo sampled full-word continuations (subword
  sampling until a whitespace-marker token or EOS).

The authors' release (``cimeister/diverse-surprisal``) supplies its estimator
and regression pipeline plus archived fold outputs. This module implements the
paper's specification without importing that code, so the sampling, token
aggregation, and feature-backend choices above are REPLICA's own. Per-sample
similarities (and per-alpha expected similarities for the exact path) are
stored in the outputs so the paper's temperature sweep ``z^alpha``
re-aggregates from the pickle without a fresh model run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from replica.feature.backends.hub import hub_id
from replica.feature.backends.start_state import start_ids as resolve_start_ids
from replica.feature.backends.start_state import validate_start_ids
from replica.feature.core.annotation.alignment import (
    TEXT_OFFSET,
    aligned_token_ids,
    tokenize_spans,
)

from ..backends.device import resolve_device
from ..backends.generation import usable_position_limit, validate_context_window
from ..core.protocols import FeatureModelOutput

if TYPE_CHECKING:
    from ...core import Stimulus

# Alpha grid for the similarity-temperature sweep: z^alpha converges to
# the identity similarity (standard surprisal) as alpha grows.
PAPER_ALPHAS = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0)


def alpha_sweep_from_expected(
    expected_similarity_tokens: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-alpha word predictors re-derived from a stored expected similarity.

    Mirrors ``_build_outputs``: a word whose column holds any non-finite value
    (or no subwords at all) is NaN, and an expected similarity of zero gives
    ``+inf`` surprisal, kept unclipped for the caller to handle.

    Args:
        expected_similarity_tokens: One word's stored
            ``expected_similarity_tokens``, the expected ``z^alpha`` per subword
            and per alpha.

    Returns:
        ``similarity_adjusted_surprisal`` and ``next_word_information_value``,
        one value per column of the ``(n_subwords, n_alphas)`` input.

    Raises:
        ValueError: When the input is not two-dimensional.
    """
    m = np.asarray(expected_similarity_tokens, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected a (n_subwords, n_alphas) array, got {m.shape}")
    n_alphas = m.shape[1]
    surprisal = np.full(n_alphas, np.nan)
    infovalue = np.full(n_alphas, np.nan)
    for j in range(n_alphas):
        col = m[:, j]
        if col.size and np.all(np.isfinite(col)):
            surprisal[j] = float(sum(_neg_log(v) for v in col))
            infovalue[j] = float(np.sum(1.0 - col))
    return {
        "similarity_adjusted_surprisal": surprisal,
        "next_word_information_value": infovalue,
    }


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insert/delete/substitute, unit costs).

    Args:
        a: First string to compare.
        b: Second string to compare; the order does not change the result.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def orthographic_similarity(a: str, b: str) -> float:
    """Normalized edit similarity ``1 - lev(a, b) / max(|a|, |b|)``.

    Two empty strings score 1.0.

    Args:
        a: First string to compare.
        b: Second string to compare; the order does not change the result.
    """
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / denom


def _neg_log(m: float) -> float:
    """-log(m) with -log(0) = inf (an all-dissimilar MC draw), not a warning."""
    if m <= 0.0:
        return float("inf")
    return float(-np.log(m))


def similarity_unit_values(
    expected_similarity_tokens: np.ndarray, alphas: tuple[float, ...]
) -> dict[str, float]:
    """The four scalar predictors of one unit from its stacked expectations.

    Args:
        expected_similarity_tokens: ``(n_subwords, n_alphas)`` expected
            ``z^alpha`` per subword of the unit and per alpha.
        alphas: Alpha of every column; the ``1.0`` column gives the headline
            predictors.

    Returns:
        ``similarity_adjusted_surprisal`` (the sum over subwords of ``-log m``,
        nats), ``next_word_information_value`` (the sum of ``1 - m``) and
        their first-subword variants. A unit with no subword, or with a
        non-finite expectation at any subword, is NaN throughout.
    """
    m = np.asarray(expected_similarity_tokens, dtype=np.float64)
    a1 = tuple(float(alpha) for alpha in alphas).index(1.0)
    m1 = m[:, a1] if m.size else np.empty(0)
    has = m1.size > 0 and np.all(np.isfinite(m1))
    return {
        "similarity_adjusted_surprisal": (
            float(sum(_neg_log(v) for v in m1)) if has else np.nan
        ),
        "next_word_information_value": (float(np.sum(1.0 - m1)) if has else np.nan),
        "similarity_adjusted_surprisal_first": (_neg_log(m1[0]) if has else np.nan),
        "next_word_information_value_first": (float(1.0 - m1[0]) if has else np.nan),
    }


class _BaseSimilaritySurprisalLM:
    """Shared model/tokenizer loading and alignment for the two variants."""

    vocab = None  # FeatureModel protocol stub.

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None,
        seed: int,
        add_bos: bool,
        revision: str | None = None,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.seed = int(seed)
        self.add_bos = add_bos
        self.context_window = validate_context_window(context_window)
        self._requested_start_ids = validate_start_ids(start_ids, add_bos=add_bos)
        self.revision = revision
        self._model = None
        self._tokenizer = None

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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        pinned = {"revision": self.revision} if self.revision is not None else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **pinned)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, low_cpu_mem_usage=True, **pinned
        )
        self._device = self.device or resolve_device()
        self._model = self._model.to(self._device)
        self._model.eval()
        self._max_ctx()
        self._start_prefix()

    def _max_ctx(self) -> int:
        """Resolve and cache the effective token budget."""
        if not hasattr(self, "_context_limit"):
            self._context_limit = usable_position_limit(
                self.model_name,
                self._model.config,
                context_window=getattr(self, "context_window", None),
            )
        return self._context_limit

    @staticmethod
    def _window_bounds(n_tokens: int, max_len: int):
        """50%-overlap sliding windows with first-covering-window-wins fill.

        Yields ``(start, end, fill_start)``: positions ``fill_start..end-1``
        are the freshly-evaluated region. Window 0 evaluates ``[0, max_len)``;
        window k>0 fills from ``start + stride``, which for an even
        ``max_len`` is the previous window's end, the position
        ``HuggingFaceLM._sliding_window_logits`` fills from.
        """
        stride = max_len // 2
        for start in range(0, n_tokens, stride):
            end = min(start + max_len, n_tokens)
            fill_start = 0 if start == 0 else start + stride
            if fill_start < end:
                yield start, end, fill_start
            if end == n_tokens:
                break


class SimilarityAdjustedSurprisalEmbeddingLM(_BaseSimilaritySurprisalLM):
    """Embedding-similarity variant of Meister et al. (2024), subword level.

    Non-contextual (``contextual=False``): the similarity space is the input
    embedding matrix and the expected similarity ``m`` is computed exactly
    over the full subword vocabulary, as in the paper. Contextual
    (``contextual=True``): the similarity space is the final hidden layer
    (post-``ln_f`` for GPT-2) and ``m`` is a Monte Carlo mean over
    ``n_samples`` next tokens drawn from the model's own distribution, each
    embedded in the current context by a one-step forward pass.

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_name: Hugging Face causal LM supplying both the next-token
            distribution and the similarity space.
        contextual: Similarity space, final hidden layer vs input embeddings.
        n_samples: MC draws per subword position (contextual only). Paper: 50.
        alphas: Similarity temperatures; both paths store an expected
            ``z^alpha`` per alpha, and the MC path additionally keeps the raw
            per-sample similarities. Always includes 1.0.
        include_wpe0: Add the position-0 positional embedding to the input
            embeddings (the ``hidden_states[0]`` idiom of the released
            responsivity code; see ``ResponsivityLM``). Default False, the
            plain embedding matrix: adding wpe0 puts a large shared
            component into every vector, which raises all pairwise
            cosines toward 1. Non-contextual path only.
        device: Torch device; ``None`` resolves one automatically.
        seed: Seed of the Monte Carlo generator (contextual path only).
        position_chunk: Token positions handled per chunk on the exact path,
            bounding the size of the ``(chunk, vocabulary)`` cosine matrix.
        add_bos: Prepend the model's start token for the first word's context.
        revision: Optional checkpoint revision, shared by model and tokenizer.

    Output values per unit (word):

    * ``similarity_adjusted_surprisal``: sum over subwords of ``-log m``, nats.
    * ``next_word_information_value``: sum over subwords of ``1 - m``.
    * ``similarity_adjusted_surprisal_first`` / ``next_word_information_value_first``:
      first-subword variants.
    * ``expected_similarity_tokens``: ``(n_subwords, n_alphas)`` expected
      ``z^alpha`` per alpha; exact over the vocabulary (non-contextual) or
      the MC mean of the per-sample ``z^alpha`` (contextual).
    * ``similarity_samples_tokens``: ``(n_subwords, n_samples)`` per-sample
      similarities (contextual only; empty array on the exact path).

    Positions with no left context (first word without ``add_bos``) are NaN.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is SimilarityAdjustedSurprisalEmbeddingLM

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        contextual: bool = False,
        n_samples: int = 50,
        alphas: tuple[float, ...] = PAPER_ALPHAS,
        include_wpe0: bool = False,
        device: str | None = None,
        seed: int = 0,
        add_bos: bool = True,
        position_chunk: int = 64,
        revision: str | None = None,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
    ):
        """Record the similarity space and its sampling settings.

        The class docstring describes the parameters. ``1.0`` is prepended to
        *alphas* when absent, since the headline predictors read that column.

        Raises:
            ValueError: When *n_samples* is below 1.
        """
        super().__init__(
            model_name,
            device=device,
            seed=seed,
            add_bos=add_bos,
            revision=revision,
            start_ids=start_ids,
            context_window=context_window,
        )
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self.contextual = bool(contextual)
        self.n_samples = int(n_samples)
        alphas = tuple(float(a) for a in alphas)
        if 1.0 not in alphas:
            alphas = (1.0, *alphas)
        self.alphas = alphas
        self.include_wpe0 = bool(include_wpe0)
        self.position_chunk = int(position_chunk)
        kind = "contextual" if contextual else "noncontextual"
        self.name = f"{hub_id(model_name)}_simadj_{kind}"
        self._emb_norm = None

    # ------------------------------------------------------------------
    # Embedding spaces
    # ------------------------------------------------------------------

    def _ensure_input_embeddings(self):
        """Row-normalized input-embedding matrix ``(V, d)``."""
        if self._emb_norm is not None:
            return
        import torch

        with torch.no_grad():
            emb = self._model.get_input_embeddings().weight.detach()
            if self.include_wpe0:
                wpe = getattr(getattr(self._model, "transformer", None), "wpe", None)
                if wpe is not None:
                    emb = emb + wpe.weight[0:1].detach()
            self._emb_norm = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Expected embedding similarity at every subword, summed per unit.

        Args:
            stimulus: Passage to score.
            extract: Part of the shared FeatureModel signature; every value
                listed in the class docstring is returned regardless.
            target_layer: Annotation layer whose units the subword values are
                summed over.

        Returns:
            One output per unit. A unit touching token position 0, which has no
            context, is NaN in the scalar values.
        """
        self._ensure_loaded()
        token_ids, alignment, bos_offset, n_units = aligned_token_ids(
            stimulus,
            self._tokenizer,
            target_layer,
            self.add_bos,
            start_ids=self._start_prefix(),
        )

        n_alphas = len(self.alphas)
        if not token_ids:
            return self._build_outputs(
                n_units, [np.empty((0, n_alphas))] * n_units, [None] * n_units
            )
        m_tok, sim_samples_tok = self._token_expectations(token_ids)

        # Aggregate subword values to units.
        per_unit_m: list[np.ndarray] = []
        per_unit_samples: list[np.ndarray | None] = []
        for i in range(n_units):
            tok_positions = [bos_offset + p for p in alignment.mapping[i]]
            # Position 0 has no context; a word touching it has no defined
            # value (a partial subword sum would silently change the measure).
            valid = tok_positions if tok_positions and tok_positions[0] >= 1 else []
            per_unit_m.append(m_tok[valid] if valid else np.empty((0, n_alphas)))
            if sim_samples_tok is None:
                per_unit_samples.append(None)
            else:
                per_unit_samples.append(
                    sim_samples_tok[valid] if valid else np.empty((0, self.n_samples))
                )
        return self._build_outputs(n_units, per_unit_m, per_unit_samples)

    def score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
    ) -> list[FeatureModelOutput]:
        """One output per model token of *stimulus*, in text order.

        The unit-level scalars derive from a unit's stacked expectations. The
        first token scored without a beginning-of-sequence token has no context
        and is NaN.

        Args:
            stimulus: Text to score.
            extract: Value keys to keep; ``None`` keeps every per-token key.

        Returns:
            One record per token, carrying its index, character span, id and
            decoded string, its expected similarity per alpha (and per-sample
            similarities on the contextual path) and the two scalar predictors
            of that position alone.
        """
        self._ensure_loaded()
        ids, spans = tokenize_spans(stimulus.text, self._tokenizer)
        if not ids:
            return []
        prefix = self._start_prefix()
        token_ids = [*prefix, *ids]
        bos_offset = len(prefix)
        m_tok, sim_samples_tok = self._token_expectations(token_ids)
        offset = stimulus.metadata.get(TEXT_OFFSET, 0)
        outputs: list[FeatureModelOutput] = []
        for t, (token_id, span) in enumerate(zip(ids, spans, strict=True)):
            position = t + bos_offset
            scalars = similarity_unit_values(
                m_tok[position : position + 1], self.alphas
            )
            values: dict = {
                "similarity_adjusted_surprisal": scalars[
                    "similarity_adjusted_surprisal"
                ],
                "next_word_information_value": scalars["next_word_information_value"],
                "expected_similarity_tokens": m_tok[position].copy(),
            }
            if sim_samples_tok is not None:
                values["similarity_samples_tokens"] = sim_samples_tok[position].copy()
            if extract is not None:
                values = {k: v for k, v in values.items() if k in extract}
            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={
                        "model_name": self.name,
                        "start_ids": self._start_prefix(),
                        "context_window": self._max_ctx(),
                        "alphas": self.alphas,
                        "token_index": t,
                        "span": (span.start + offset, span.end + offset),
                        "token_id": int(token_id),
                        "token": self._tokenizer.decode([token_id]),
                    },
                )
            )
        return outputs

    def window(self) -> dict:
        """The context-window rule the forward pass applies, for provenance.

        Windows of ``max_positions`` tokens advance by ``stride``; a later
        window scores the positions from ``start + stride`` on.
        """
        self._ensure_loaded()
        max_len = self._max_ctx()
        return {
            "rule": "sliding",
            "max_positions": max_len,
            "stride": max_len // 2,
            "fresh_from": "start + stride",
        }

    def sampling(self) -> dict:
        """How the per-token table is estimated, for provenance.

        Every token position is estimated. The contextual path reseeds its
        generator with the seed at every stimulus and draws in text order;
        the exact path draws nothing.
        """
        if not self.contextual:
            return {"positions": "all", "seed": None, "seeding": "exact, no draws"}
        return {
            "positions": "all",
            "seed": self.seed,
            "seeding": "reseeded with the seed at every stimulus; one draw of "
            "n_samples per scored position, in text order",
        }

    def _token_expectations(
        self, token_ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Expected similarity per position and alpha, and the MC similarities.

        Position 0 has no context and stays NaN.
        """
        import torch

        n_tokens = len(token_ids)
        n_alphas = len(self.alphas)
        ids_t = torch.tensor(token_ids, device=self._device, dtype=torch.long)
        max_len = self._max_ctx()
        m_tok = np.full((n_tokens, n_alphas), np.nan)
        sim_samples_tok = (
            np.full((n_tokens, self.n_samples), np.nan) if self.contextual else None
        )

        gen = torch.Generator(device=self._device)
        gen.manual_seed(self.seed)

        for start, end, fill_start in self._window_bounds(n_tokens, max_len):
            self._process_window(
                ids_t[start:end],
                eval_from=max(fill_start, 1) - start,
                out_offset=start,
                m_tok=m_tok,
                sim_samples_tok=sim_samples_tok,
                gen=gen,
            )
        return m_tok, sim_samples_tok

    def _process_window(
        self,
        window_ids,
        *,
        eval_from: int,
        out_offset: int,
        m_tok: np.ndarray,
        sim_samples_tok: np.ndarray | None,
        gen,
    ) -> None:
        """Fill ``m_tok`` (and MC similarities) for window positions >= eval_from."""
        import torch

        with torch.no_grad():
            out = self._model(
                window_ids.unsqueeze(0),
                use_cache=self.contextual,
                output_hidden_states=self.contextual,
            )
            logits = out.logits[0].double()  # (W, V)

            if not self.contextual:
                self._exact_window(window_ids, logits, eval_from, out_offset, m_tok)
                return

            from transformers import DynamicCache

            h_last = out.hidden_states[-1][0]  # (W, d), post-ln_f
            h_norm = h_last / h_last.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            legacy = out.past_key_values.to_legacy_cache()

            for t in range(eval_from, len(window_ids)):
                probs = torch.softmax(logits[t - 1], dim=-1)
                samples = torch.multinomial(
                    probs, self.n_samples, replacement=True, generator=gen
                )
                # Contextual embeddings of the sampled tokens: one batched
                # step over the cache sliced to the t-token prefix (batch dim
                # expands as a view, no copy of the master cache).
                sliced = tuple(
                    (
                        k[:, :, :t].expand(self.n_samples, -1, -1, -1),
                        v[:, :, :t].expand(self.n_samples, -1, -1, -1),
                    )
                    for k, v in legacy
                )
                step = self._model(
                    samples.unsqueeze(1),
                    past_key_values=DynamicCache.from_legacy_cache(sliced),
                    use_cache=False,
                    output_hidden_states=True,
                )
                h_smp = step.hidden_states[-1][:, -1]  # (n_samples, d)
                h_smp = h_smp / h_smp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cos = (h_smp * h_norm[t]).sum(dim=-1)
                sims = (0.5 * (cos + 1.0)).double().cpu().numpy()
                sim_samples_tok[out_offset + t] = sims
                for a_i, alpha in enumerate(self.alphas):
                    m_tok[out_offset + t, a_i] = float(np.mean(sims**alpha))

    def _exact_window(
        self, window_ids, logits, eval_from: int, out_offset: int, m_tok: np.ndarray
    ) -> None:
        """Exact ``m_alpha = sum_v z(w_t, v)^alpha p(v)`` over the vocabulary."""
        import torch

        self._ensure_input_embeddings()
        n = len(window_ids)
        for c0 in range(eval_from, n, self.position_chunk):
            c1 = min(c0 + self.position_chunk, n)
            obs = self._emb_norm[window_ids[c0:c1]]  # (C, d)
            cos = obs @ self._emb_norm.T  # (C, V)
            sims = (0.5 * (cos + 1.0)).double()
            probs = torch.softmax(logits[c0 - 1 : c1 - 1], dim=-1)  # (C, V)
            for a_i, alpha in enumerate(self.alphas):
                m = (sims.pow(alpha) * probs).sum(dim=-1)
                m_tok[out_offset + c0 : out_offset + c1, a_i] = m.cpu().numpy()

    def _build_outputs(
        self,
        n_units: int,
        per_unit_m: list[np.ndarray],
        per_unit_samples: list[np.ndarray | None],
    ) -> list[FeatureModelOutput]:
        outs = []
        for i in range(n_units):
            m = np.asarray(per_unit_m[i], dtype=np.float64)
            values = {
                **similarity_unit_values(m, self.alphas),
                "expected_similarity_tokens": m,
                "similarity_samples_tokens": (
                    np.empty((0, self.n_samples))
                    if per_unit_samples[i] is None
                    else np.asarray(per_unit_samples[i], dtype=np.float64)
                ),
            }
            outs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={
                        "model_name": self.name,
                        "alphas": self.alphas,
                        "start_ids": self._start_prefix(),
                        "context_window": self._max_ctx(),
                    },
                )
            )
        return outs


class SimilarityAdjustedSurprisalWordLM(_BaseSimilaritySurprisalLM):
    """Full-word-similarity variant of Meister et al. (2024): POS + orthographic.

    At each word position, samples ``n_samples`` full-word continuations from
    the LM (ancestral subword sampling; a word ends when the next draw is a
    whitespace-marker token or EOS, capped at ``max_iter`` subwords) and
    estimates the expected similarity of the observed word to the samples
    under two similarity functions:

    * POS identity: 1 if the flair ``pos-english-fast`` tag of the sampled word
      equals the tag of the observed word, else 0. Both are tagged in
      context; since the paper does not specify the tagging context, the
      trailing ``pos_context_words`` corpus words are used.
    * Orthographic: normalized Levenshtein similarity
      ``1 - lev / max(len)`` on the surface strings.

    An all-dissimilar draw gives expected similarity 0 and similarity-adjusted
    surprisal ``inf`` (stored as-is; per-sample values are kept so any
    smoothing choice can be applied downstream).

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Token budget; ``None`` uses the model default.
        model_name: Hugging Face causal LM the word continuations are sampled
            from.
        n_samples: Word continuations drawn per word position.
        max_iter: Subword cap on one sampled word.
        first_token: ``"unrestricted"`` samples the first subword from the
            full vocabulary (the paper's procedure read literally);
            ``"word_starter"`` restricts it to whitespace-marker tokens
            (a new whitespace-delimited word, as in the Clark et al. (2025) sampler).
        pos: Compute the POS variant (loads flair lazily on first use).
        pos_context_words: Corpus words of preceding context a word is tagged
            in, observed and sampled alike.
        pos_batch_size: Sentences per flair prediction batch.
        device: Torch device; ``None`` resolves one automatically.
        seed: Seed of the sampling generator.
        add_bos: Prepend the model's start token for the first word's context.
        revision: Optional checkpoint revision, shared by model and tokenizer.
        tagger_revision: Optional immutable revision of the POS checkpoint.

    Output values per unit: ``similarity_adjusted_surprisal_pos`` / ``_orth``,
    ``next_word_information_value_pos`` / ``_orth``, per-sample arrays
    ``pos_match_samples`` / ``orth_similarity_samples``, ``sampled_words``
    (object array of strings), and ``observed_pos``.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is SimilarityAdjustedSurprisalWordLM

    _WORD_STARTERS = ("Ġ", "Ċ", "ĉ")

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        n_samples: int = 50,
        max_iter: int = 10,
        first_token: str = "unrestricted",
        pos: bool = True,
        pos_context_words: int = 20,
        pos_batch_size: int = 256,
        device: str | None = None,
        seed: int = 0,
        add_bos: bool = True,
        revision: str | None = None,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
        tagger_revision: str | None = None,
    ):
        """Record the sampling and tagging settings.

        The class docstring describes the parameters.

        Raises:
            ValueError: When *n_samples* or *max_iter* is below 1, or
                *first_token* is neither ``unrestricted`` nor ``word_starter``.
        """
        super().__init__(
            model_name,
            device=device,
            seed=seed,
            add_bos=add_bos,
            revision=revision,
            start_ids=start_ids,
            context_window=context_window,
        )
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if first_token not in ("unrestricted", "word_starter"):
            raise ValueError(
                "first_token must be 'unrestricted' or 'word_starter', "
                f"got {first_token!r}"
            )
        self.n_samples = int(n_samples)
        self.max_iter = int(max_iter)
        self.first_token = first_token
        self.pos = bool(pos)
        self.pos_context_words = int(pos_context_words)
        self.pos_batch_size = int(pos_batch_size)
        self.name = f"{hub_id(model_name)}_simadj_word"
        self._starter_mask = None
        self._tagger = None
        self.tagger_revision = tagger_revision

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _ensure_starter_mask(self):
        """Boolean vocab mask of word-starter (whitespace-prefix) tokens + EOS."""
        if self._starter_mask is not None:
            return
        import torch

        vocab_size = self._model.config.vocab_size
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=self._device)
        for i in range(self._tokenizer.vocab_size):
            tok = self._tokenizer.convert_ids_to_tokens(i)
            if tok and tok[0] in self._WORD_STARTERS:
                mask[i] = True
        if not bool(mask.any()):
            raise ValueError(
                f"SimilarityAdjustedSurprisalWordLM: tokenizer for "
                f"{self.model_name!r} has no byte-level whitespace-prefix "
                "tokens (Ġ, Ċ, ĉ); the word-boundary stop rule is undefined."
            )
        eos = self._tokenizer.eos_token_id
        if eos is not None:
            mask[eos] = True
        self._starter_mask = mask

    def _sample_words(self, legacy, prefix_len: int, first_logits, gen) -> list[str]:
        """Sample ``n_samples`` word strings from the LM state at one position.

        ``legacy`` is the window's full KV cache (legacy tuple); the batch
        dimension expands as a view over the ``prefix_len``-token slice.
        """
        import torch
        from transformers import DynamicCache

        k = self.n_samples
        probs = torch.softmax(first_logits, dim=-1)
        if self.first_token == "word_starter":
            probs = probs * self._starter_mask.float()
            probs = probs / probs.sum().clamp_min(1e-40)
        first = torch.multinomial(probs, k, replacement=True, generator=gen)

        eos = self._tokenizer.eos_token_id
        # EOS as the very first draw terminates the word immediately (empty).
        alive = ~(first == eos) if eos is not None else torch.ones_like(first).bool()

        sliced = tuple(
            (
                kk[:, :, :prefix_len].expand(k, -1, -1, -1),
                vv[:, :, :prefix_len].expand(k, -1, -1, -1),
            )
            for kk, vv in legacy
        )
        cache = DynamicCache.from_legacy_cache(sliced)

        sampled: list[list[int]] = [
            [int(t)] if a else [] for t, a in zip(first, alive, strict=True)
        ]
        cur = first.unsqueeze(1)
        with torch.no_grad():
            for _ in range(self.max_iter - 1):
                if not bool(alive.any()):
                    break
                out = self._model(cur, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                step_probs = torch.softmax(out.logits[:, -1, :].double(), dim=-1)
                nxt = torch.multinomial(step_probs, 1, generator=gen).squeeze(1)
                stops = self._starter_mask[nxt]
                for s_i in range(k):
                    if bool(alive[s_i]) and not bool(stops[s_i]):
                        sampled[s_i].append(int(nxt[s_i]))
                alive = alive & ~stops
                cur = nxt.unsqueeze(1)

        return [self._tokenizer.decode(ids).lstrip() if ids else "" for ids in sampled]

    # ------------------------------------------------------------------
    # POS tagging
    # ------------------------------------------------------------------

    def _ensure_tagger(self):
        if self._tagger is not None:
            return
        from flair.models import SequenceTagger

        checkpoint = "flair/pos-english-fast"
        if self.tagger_revision is not None:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                checkpoint, "pytorch_model.bin", revision=self.tagger_revision
            )
        self._tagger = SequenceTagger.load(checkpoint)

    def _tag_last_words(self, contexts: list[list[str]]) -> list[str | None]:
        """Tag each pre-tokenized context and return the final word's POS tag."""
        from flair.data import Sentence

        sentences = []
        keep: list[int] = []
        for idx, words in enumerate(contexts):
            target = words[-1]
            if not target:
                continue  # empty sampled word, no tag
            clean = [*(w for w in words[:-1] if w), target]
            sentences.append(Sentence(clean))
            keep.append(idx)
        if sentences:
            self._tagger.predict(sentences, mini_batch_size=self.pos_batch_size)
        tags: list[str | None] = [None] * len(contexts)
        for idx, sent in zip(keep, sentences, strict=True):
            labels = sent[len(sent) - 1].labels
            tags[idx] = labels[0].value if labels else None
        return tags

    def _sample_all_words(
        self, token_ids: list[int], alignment, bos_offset: int, n_units: int
    ) -> list[list[str] | None]:
        """Sample ``n_samples`` word strings at every word boundary.

        Windows shrink by ``max_iter`` so a sampled continuation never
        exceeds the model's positional budget; word i is sampled in the
        window where its predicting logit row (``prefix_len - 1``) is fresh,
        so every prefix is sampled exactly once.
        """
        import torch

        sampled_words: list[list[str] | None] = [None] * n_units
        if not token_ids:
            return sampled_words

        ids_t = torch.tensor(token_ids, device=self._device, dtype=torch.long)
        gen = torch.Generator(device=self._device)
        gen.manual_seed(self.seed)

        # Word positions needing samples: unit index -> prefix length.
        prefix_of: dict[int, int] = {}
        for i in range(n_units):
            positions = alignment.mapping[i]
            if not positions:
                continue
            first_pos = bos_offset + positions[0]
            if first_pos >= 1:
                prefix_of[i] = first_pos

        eff_max = self._max_ctx() - self.max_iter
        if eff_max < 2:
            raise ValueError(
                "context_window must allow at least two prefix tokens plus "
                f"max_iter={self.max_iter}"
            )
        for start, end, fill_start in self._window_bounds(len(ids_t), eff_max):
            in_window = {
                i: p for i, p in prefix_of.items() if fill_start <= p - 1 < end
            }
            if not in_window:
                continue
            with torch.no_grad():
                out = self._model(ids_t[start:end].unsqueeze(0), use_cache=True)
            legacy = out.past_key_values.to_legacy_cache()
            logits = out.logits[0].double()
            for i, p in in_window.items():
                local = p - start
                sampled_words[i] = self._sample_words(
                    legacy, local, logits[local - 1], gen
                )
        return sampled_words

    def _pos_matches(
        self, unit_words: list[str], sampled_words: list[list[str] | None]
    ) -> tuple[list[np.ndarray | None], list[str | None]]:
        """Tag observed and (deduplicated) sampled words in trailing context."""
        n_units = len(unit_words)
        pos_match: list[np.ndarray | None] = [None] * n_units
        observed_pos: list[str | None] = [None] * n_units
        self._ensure_tagger()

        contexts: list[list[str]] = []
        owners: list[tuple[int, str | None]] = []  # (unit, sampled word or None)
        for i, samples in enumerate(sampled_words):
            left = unit_words[max(0, i - self.pos_context_words) : i]
            contexts.append([*left, unit_words[i]])
            owners.append((i, None))
            if samples is None:
                continue
            for w in sorted(set(samples)):
                contexts.append([*left, w])
                owners.append((i, w))

        tags = self._tag_last_words(contexts)
        tag_of: dict[tuple[int, str], str | None] = {}
        for (i, w), tag in zip(owners, tags, strict=True):
            if w is None:
                observed_pos[i] = tag
            else:
                tag_of[(i, w)] = tag
        for i, samples in enumerate(sampled_words):
            if samples is None:
                continue
            obs = observed_pos[i]
            pos_match[i] = np.array(
                [float(obs is not None and tag_of.get((i, s)) == obs) for s in samples]
            )
        return pos_match, observed_pos

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Sample full words at every unit and score both word similarities.

        Args:
            stimulus: Passage to score.
            extract: Part of the shared FeatureModel signature; every value
                listed in the class docstring is returned regardless.
            target_layer: Annotation layer whose units are sampled at and whose
                character spans supply the observed surface forms.

        Returns:
            One output per unit. A unit with no predicting position, or one the
            POS variant was switched off for, is NaN in the affected scalars.
        """
        self._ensure_loaded()
        self._ensure_starter_mask()
        token_ids, alignment, bos_offset, n_units = aligned_token_ids(
            stimulus,
            self._tokenizer,
            target_layer,
            self.add_bos,
            start_ids=self._start_prefix(),
        )
        # Observed surface forms come from the char spans: layer values are
        # ints in some pipelines.
        unit_words = [
            stimulus.text[span.start : span.end]
            for span, _ in stimulus.layers[target_layer]
        ]

        sampled_words = self._sample_all_words(
            token_ids, alignment, bos_offset, n_units
        )

        orth_sims: list[np.ndarray | None] = [None] * n_units
        for i, samples in enumerate(sampled_words):
            if samples is None:
                continue
            orth_sims[i] = np.array(
                [orthographic_similarity(unit_words[i], s) for s in samples]
            )

        if self.pos:
            pos_match, observed_pos = self._pos_matches(unit_words, sampled_words)
        else:
            pos_match = [None] * n_units
            observed_pos = [None] * n_units

        outs = []
        for i in range(n_units):
            values: dict = {}
            for key, sims in (("orth", orth_sims[i]), ("pos", pos_match[i])):
                if sims is None:
                    values[f"similarity_adjusted_surprisal_{key}"] = np.nan
                    values[f"next_word_information_value_{key}"] = np.nan
                else:
                    m = float(np.mean(sims))
                    values[f"similarity_adjusted_surprisal_{key}"] = _neg_log(m)
                    values[f"next_word_information_value_{key}"] = 1.0 - m
            values["orth_similarity_samples"] = (
                np.empty(0) if orth_sims[i] is None else orth_sims[i]
            )
            values["pos_match_samples"] = (
                np.empty(0) if pos_match[i] is None else pos_match[i]
            )
            values["sampled_words"] = np.array(
                sampled_words[i] if sampled_words[i] is not None else [],
                dtype=object,
            )
            values["observed_pos"] = observed_pos[i]
            outs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={
                        "model_name": self.name,
                        "start_ids": self._start_prefix(),
                        "context_window": self._max_ctx(),
                    },
                )
            )
        return outs
