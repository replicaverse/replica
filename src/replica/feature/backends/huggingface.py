"""Causal language-model scores and token-distribution statistics.

Aligns token outputs to stimulus units and uses sliding windows for sequences
longer than the configured context limit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from replica.feature.backends.hub import hub_id
from replica.feature.backends.start_state import start_ids as resolve_start_ids
from replica.feature.backends.start_state import validate_start_ids
from replica.feature.core.annotation.alignment import (
    TEXT_OFFSET,
    aligned_token_ids,
    tokenize_spans,
)
from replica.feature.core.annotation.tokens import token_outputs
from replica.feature.core.outputs.information import renyi_bits, shannon_bits

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput
from ..core.reproduction import warn_reproduction_only
from .device import loading_options, resolve_device, validate_device_map, validate_dtype
from .generation import usable_position_limit, validate_context_window


@dataclass(frozen=True)
class _FeatureRequest:
    compute_log_p: bool
    compute_nlp: bool
    compute_tok_ent: bool
    renyi_alphas: list[float]
    compute_renyi: bool
    compute_rft: bool
    rft_alphas: list[float]
    compute_rft_renyi: bool
    compute_tok_ent_sum: bool = False
    compute_tok_renyi_sum: bool = False


@dataclass(frozen=True)
class _PreparedInput:
    alignment: object
    token_ids: list[int]
    bos_offset: int
    n_units: int


class HuggingFaceLM:
    """Causal language model from HuggingFace Transformers.

    ``log_p_observed`` is the sum of subword token log-probs for each word.
    ``next_log_probs`` pools the distribution predicting a unit's first token
    by surface form: the vocab is the set of unique decoded tokens,
    lower-cased and whitespace-stripped, so case and leading-space variants
    merge and word pieces count as types.

    Args:
        start_ids: Optional single-token prefix; ``None`` uses the backend default.
        context_window: Maximum tokens per forward pass; ``None`` uses the model default.
        dtype: Parameter precision the weights load in, ``float32``,
            ``float16`` or ``bfloat16``; ``None`` keeps single precision.
        device_map: ``auto`` shards the weights over every visible device
            through Accelerate; ``None`` places the whole model on ``device``.
        model_name: HuggingFace checkpoint id or local snapshot directory to
            load. The model's ``name`` in analysis results is its Hub id.
        aggregation: ``"sum"`` writes the unit log probability as
            ``log_p_observed``. ``"mean"`` writes the mean subtoken log
            probability as ``mean_token_log_p`` (nats), a normalized score
            rather than a unit log probability.
        device: ``None`` (default) auto-selects CUDA when available, else CPU.
            An explicit ``"cpu"`` or ``"cuda"`` selects that device. Floating-point
            results can differ across devices.
        provide_word_dist: Compute and return next_log_probs (marginalised
            to word types). Disable if only surprisal is needed.
        add_bos: Prepend the resolved start token so the first word has a
            defined surprisal. Set False to get NaN for the first word of each
            context.
        token_level_entropy: Compute Shannon entropy over the full token
            vocabulary and store the first subword token's value per word as
            ``token_entropy``.
        token_renyi_alphas: Rényi-alpha exponents. For each alpha, computes
            H_alpha over the first-subword token distribution, stored as
            ``token_renyi_entropy_alpha{alpha}``.
        restricted_first_token_entropy: Shannon entropy of the next-token distribution
            renormalised over word-starter (``Ġ``-prefix) tokens, stored as
            ``restricted_first_token_entropy``; after Clark et al. (2025).
        restricted_first_token_renyi_alphas: Rényi-alpha exponents for the same restricted
            distribution, stored as ``restricted_first_token_renyi_entropy_alpha{alpha}``.
        token_entropy_sum: Also store the sum of per-subtoken Shannon
            entropies per word as ``token_entropy_sum``, reproducing the
            summed aggregation in the Pimentel et al. (2023) release.
        token_renyi_entropy_sum: Same summed aggregation for the first
            configured Rényi alpha, stored as ``token_renyi_entropy_sum``.
        allow_reproduction: Required when requesting summed entropy variants.
        trust_remote_code: Let the checkpoint bring its own modelling code,
            which released models built on an out-of-tree architecture need.
        context_scope: ``"stimulus"`` scores the whole stimulus as one
            context; ``"sentence"`` scores each sentence independently,
            using the boundaries in the stimulus's ``sentences`` layer.
    """

    # Class-level default so partially-constructed instances (test doubles
    # built via ``__new__``) dispatch like a stimulus-scope model.
    context_scope = "stimulus"

    @property
    def supports_output_union(self) -> bool:
        """This deterministic producer preserves outputs when requests are combined."""
        return type(self) is HuggingFaceLM

    def __init__(
        self,
        model_name: str = "gpt2",
        aggregation: str = "sum",
        device: str | None = None,
        provide_word_dist: bool = True,
        add_bos: bool = True,
        token_level_entropy: bool = False,
        token_renyi_alphas: list[float] | None = None,
        restricted_first_token_entropy: bool = False,
        restricted_first_token_renyi_alphas: list[float] | None = None,
        token_entropy_sum: bool = False,
        token_renyi_entropy_sum: bool = False,
        trust_remote_code: bool = False,
        context_scope: str = "stimulus",
        *,
        allow_reproduction: bool = False,
        start_ids: tuple[int, ...] | None = None,
        context_window: int | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
    ):
        """Record the checkpoint and which quantities it is asked to compute.

        The model and tokenizer load on first use, so constructing one costs
        nothing and touches no network.
        """
        for enabled, method in (
            (token_entropy_sum, "first_token_entropy_buggy"),
            (token_renyi_entropy_sum, "renyi_entropy_buggy"),
        ):
            if enabled:
                warn_reproduction_only(method, allow_reproduction=allow_reproduction)
        if context_scope not in ("stimulus", "sentence"):
            raise ValueError(f"unknown context_scope {context_scope!r}")
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
            )
        self.context_scope = context_scope
        # A pinned snapshot path loads as given; only the name maps to the id.
        self._checkpoint = model_name
        self.name = hub_id(model_name)
        self.aggregation = aggregation
        self.device = device
        self.dtype = validate_dtype(dtype)
        self.device_map = validate_device_map(device_map)
        self.trust_remote_code = bool(trust_remote_code)
        self.provide_word_dist = provide_word_dist
        self.add_bos = add_bos
        self._requested_start_ids = validate_start_ids(start_ids, add_bos=add_bos)
        self._start_ids: list[int] | None = None
        self.context_window = validate_context_window(context_window)
        self.token_level_entropy = token_level_entropy
        self.token_entropy_sum = bool(token_entropy_sum)
        self.token_renyi_entropy_sum = bool(token_renyi_entropy_sum)
        self.token_renyi_alphas: list[float] = (
            [float(a) for a in token_renyi_alphas] if token_renyi_alphas else []
        )
        self.restricted_first_token_entropy = bool(restricted_first_token_entropy)
        self.restricted_first_token_renyi_alphas: list[float] = (
            [float(a) for a in restricted_first_token_renyi_alphas]
            if restricted_first_token_renyi_alphas
            else []
        )
        self._model = None
        self._tokenizer = None
        # vocab: sorted list of unique surface-form word types
        self._vocab: list[str] | None = None
        # _vocab_index[word] = position of word in _vocab, shared by all outputs
        self._vocab_index: dict[str, int] | None = None
        # _word_to_token_ids[word] = list of token IDs that decode to that word
        self._word_to_token_ids: dict[str, list[int]] | None = None
        # Lazily built token-id masks for the restricted first-token entropies.
        self._space_ids_np: np.ndarray | None = None
        self._subword_ids_np: np.ndarray | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._checkpoint, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._checkpoint,
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code,
            **loading_options(self.dtype, self.device_map),
        )
        if self.device_map is None:
            self.device = resolve_device(self.device)
            self._model.to(self.device)
        else:
            # Sharded weights stay where the loader placed them; inputs go to
            # the device of the first shard.
            self.device = self._model.device
        self._model.eval()

        if self.provide_word_dist:
            self._build_word_vocab()

    def _build_word_vocab(self):
        """Build word-type -> token-ID mapping by decoding every token."""
        vocab_size = self._tokenizer.vocab_size
        word_to_ids: dict[str, list[int]] = {}

        for token_id in range(vocab_size):
            decoded = self._tokenizer.decode([token_id]).strip().lower()
            if not decoded:
                continue
            word_to_ids.setdefault(decoded, []).append(token_id)

        self._word_to_token_ids = word_to_ids
        self._vocab = sorted(word_to_ids.keys())
        self._vocab_index = {word: i for i, word in enumerate(self._vocab)}

    @property
    def vocab(self) -> list[str] | None:
        """Word-type vocabulary; None if provide_word_dist=False."""
        if not self.provide_word_dist:
            return None
        self._ensure_loaded()
        return self._vocab

    def _ensure_word_starter_masks(self) -> None:
        """Partition vocab into Ġ-prefix word-starters vs subwords.

        ``Ċ`` (newline) and ``ĉ`` (tab) do not count as starters here, unlike
        ``MCWordEntropyLM``, which folds them in for paragraph breaks.
        """
        if self._space_ids_np is not None and self._subword_ids_np is not None:
            return
        space_ids: list[int] = []
        subword_ids: list[int] = []
        vocab_size = self._tokenizer.vocab_size
        for i in range(vocab_size):
            tok = self._tokenizer.convert_ids_to_tokens(i)
            if tok and tok.startswith("Ġ"):
                space_ids.append(i)
            else:
                subword_ids.append(i)
        if not space_ids:
            raise ValueError(
                f"HuggingFaceLM(restricted_first_token_entropy): tokenizer {self.name!r} "
                "has no 'Ġ'-prefix tokens; restricted first-token entropy is "
                "undefined for non-byte-level tokenizers. Use "
                "token_level_entropy instead."
            )
        self._space_ids_np = np.asarray(space_ids, dtype=np.int64)
        self._subword_ids_np = np.asarray(subword_ids, dtype=np.int64)

    def _marginalise_to_word_dist(self, token_log_probs_np: np.ndarray) -> np.ndarray:
        """Group token log-probs by word type via logsumexp.

        Args:
            token_log_probs_np: shape (vocab_size,) log-probs over tokens.
        Returns:
            log-probs over word types, shape (len(self._vocab),).
        """
        word_lps = np.full(len(self._vocab), -np.inf)
        for i, word in enumerate(self._vocab):
            ids = self._word_to_token_ids[word]
            lps = token_log_probs_np[ids]
            word_lps[i] = logsumexp(lps)
        log_z = logsumexp(word_lps)
        if not np.isfinite(log_z):
            raise ValueError("The decoded-token support has no finite probability mass")
        return word_lps - log_z

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One output per unit of ``stimulus.layers[target_layer]``.

        Args:
            stimulus: Text to score, carrying the annotation layers.
            extract: Value keys to compute, e.g. ``["log_p_observed",
                "next_log_probs"]``. ``None`` computes every key the
                constructor enabled.
            target_layer: Name of the annotation layer whose units the outputs
                align to, one per unit.

        Returns:
            One record per unit, in layer order. Under sentence context scope
            each sentence is scored on its own, so no word conditions on an
            earlier sentence.
        """
        if self.context_scope == "sentence":
            return self._call_per_sentence(stimulus, extract, target_layer)
        return self._call_stimulus(stimulus, extract, target_layer)

    def _call_stimulus(
        self,
        stimulus: Stimulus,
        extract: list[str] | None,
        target_layer: str,
    ) -> list[FeatureModelOutput]:
        import torch

        if extract is not None:
            unavailable = (
                "log_p_observed" if self.aggregation == "mean" else "mean_token_log_p"
            )
            if unavailable in extract:
                raise ValueError(
                    f"aggregation={self.aggregation!r} supplies {self._unit_log_p_key!r}, "
                    f"not {unavailable!r}; select the aggregation matching the quantity"
                )
        self._ensure_loaded()

        request = self._feature_request(extract)
        if request.compute_rft or request.compute_rft_renyi:
            self._ensure_word_starter_masks()

        prepared = self._prepare_input(stimulus, target_layer)
        if not prepared.token_ids:
            return self._empty_outputs(prepared.n_units)

        input_ids = torch.tensor([prepared.token_ids], device=self.device)
        logits = self._forward_logits(input_ids, len(prepared.token_ids))

        # log_probs[t] = log P(next_token | token_{0..t})
        log_probs = torch.log_softmax(logits[0], dim=-1)
        n_original_tokens = len(prepared.token_ids) - prepared.bos_offset

        token_log_p = self._token_log_probabilities(
            log_probs,
            prepared.token_ids,
            prepared.bos_offset,
            n_original_tokens,
            request.compute_log_p,
        )
        unit_log_p = prepared.alignment.aggregate(token_log_p, method=self.aggregation)
        if not (self.add_bos and prepared.bos_offset > 0):
            unit_log_p[0] = np.nan

        token_entropy, token_renyi = self._token_entropy_arrays(
            log_probs,
            n_original_tokens,
            prepared.bos_offset,
            request.compute_tok_ent or request.compute_tok_ent_sum,
            request.renyi_alphas,
            request.compute_renyi or request.compute_tok_renyi_sum,
        )
        unit_tok_ent = self._first_subword_array(
            token_entropy, prepared.alignment.mapping, prepared.n_units
        )
        unit_tok_ent_sum = self._sum_subword_array(
            token_entropy, prepared.alignment.mapping, prepared.n_units
        )
        unit_tok_renyi = self._first_subword_renyi(
            token_renyi, prepared.alignment.mapping, prepared.n_units
        )
        unit_tok_renyi_sum = self._sum_subword_renyi(
            token_renyi, prepared.alignment.mapping, prepared.n_units
        )
        unit_rft, unit_rft_renyi = self._restricted_first_token_entropies(
            log_probs,
            prepared.alignment.mapping,
            prepared.bos_offset,
            prepared.n_units,
            request.compute_rft,
            request.rft_alphas,
            request.compute_rft_renyi,
        )
        return self._build_outputs(
            request,
            prepared.alignment.mapping,
            log_probs,
            prepared.bos_offset,
            unit_log_p,
            unit_tok_ent,
            unit_tok_renyi,
            unit_rft,
            unit_rft_renyi,
            prepared.n_units,
            unit_tok_ent_sum,
            unit_tok_renyi_sum,
        )

    def _feature_request(self, extract: list[str] | None) -> _FeatureRequest:
        compute_log_p = extract is None or bool(
            {"log_p_observed", "mean_token_log_p"}.intersection(extract)
        )
        compute_nlp = (
            extract is None or "next_log_probs" in extract
        ) and self.provide_word_dist
        compute_tok_ent = (
            extract is None or "token_entropy" in extract
        ) and self.token_level_entropy
        compute_tok_ent_sum = (
            extract is None or "token_entropy_sum" in extract
        ) and self.token_entropy_sum
        # Subword-level Rényi-alpha (one per alpha); shares the per-position log_probs.
        renyi_alphas = list(self.token_renyi_alphas) if self.token_renyi_alphas else []
        compute_renyi = bool(renyi_alphas) and (
            extract is None
            or any(f"token_renyi_entropy_alpha{a}" in extract for a in renyi_alphas)
        )
        # The summed variant needs a configured alpha and uses the first one.
        compute_tok_renyi_sum = (
            bool(renyi_alphas)
            and self.token_renyi_entropy_sum
            and (extract is None or "token_renyi_entropy_sum" in extract)
        )
        # Restricted next-word entropy on word-starter tokens.
        compute_rft = (
            extract is None or "restricted_first_token_entropy" in extract
        ) and self.restricted_first_token_entropy
        rft_alphas = (
            list(self.restricted_first_token_renyi_alphas)
            if self.restricted_first_token_renyi_alphas
            else []
        )
        compute_rft_renyi = bool(rft_alphas) and (
            extract is None
            or any(
                f"restricted_first_token_renyi_entropy_alpha{a}" in extract
                for a in rft_alphas
            )
        )
        return _FeatureRequest(
            compute_log_p=compute_log_p,
            compute_nlp=compute_nlp,
            compute_tok_ent=compute_tok_ent,
            renyi_alphas=renyi_alphas,
            compute_renyi=compute_renyi,
            compute_rft=compute_rft,
            rft_alphas=rft_alphas,
            compute_rft_renyi=compute_rft_renyi,
            compute_tok_ent_sum=compute_tok_ent_sum,
            compute_tok_renyi_sum=compute_tok_renyi_sum,
        )

    #: The resolved prefix, cached after the tokenizer loads.
    _start_ids: list[int] | None = None

    def _start_prefix(self) -> list[int]:
        """The prefix the empty context is scored from; empty without ``add_bos``."""
        if self._start_ids is None:
            self._start_ids = resolve_start_ids(
                self._tokenizer,
                self.name,
                override=getattr(self, "_requested_start_ids", None),
                add_bos=self.add_bos,
            )
        return list(self._start_ids)

    def _prepare_input(self, stimulus: Stimulus, target_layer: str) -> _PreparedInput:
        token_ids, alignment, bos_offset, n_units = aligned_token_ids(
            stimulus, self._tokenizer, target_layer, self.add_bos, self._start_prefix()
        )
        return _PreparedInput(
            alignment=alignment,
            token_ids=token_ids,
            bos_offset=bos_offset,
            n_units=n_units,
        )

    @property
    def _unit_log_p_key(self) -> str:
        return "log_p_observed" if self.aggregation == "sum" else "mean_token_log_p"

    def _output_metadata(self) -> dict:
        """Resolved scoring choices shared by unit-level records."""
        return {
            "model_name": self.name,
            "start_ids": self._start_prefix(),
            "context_window": self._context_length(),
            "aggregation": self.aggregation,
            "context_scope": self.context_scope,
            "next_log_probs_support": (
                "decoded_tokens_strip_lower_nonempty"
                if self.provide_word_dist
                else None
            ),
        }

    def _empty_outputs(self, n_units: int) -> list[FeatureModelOutput]:
        return [
            FeatureModelOutput(
                values={self._unit_log_p_key: np.nan},
                metadata=self._output_metadata(),
            )
            for _ in range(n_units)
        ]

    def _call_per_sentence(
        self,
        stimulus: Stimulus,
        extract: list[str] | None,
        target_layer: str,
    ) -> list[FeatureModelOutput]:
        from replica.feature.core.annotation.sentences import per_sentence_outputs

        return per_sentence_outputs(
            lambda sub: self._call_stimulus(sub, extract, target_layer),
            stimulus,
            target_layer,
        )

    def window(self) -> dict:
        """The context-window rule the forward pass applies, for provenance.

        ``max_positions`` is the resolved position limit and ``stride`` the
        sliding-window step.
        """
        self._ensure_loaded()
        max_len = self._context_length()
        return {
            "rule": "sliding",
            "max_positions": max_len,
            "stride": max_len // 2,
        }

    def score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
    ) -> list[FeatureModelOutput]:
        """One output per model token of *stimulus*, in text order.

        BOS handling and the sliding window are those of the unit-level call;
        under sentence scope each sentence is tokenized and scored on its own,
        with spans offset to the full text. The first token scored without a
        beginning-of-sequence token has no predicting position and is NaN.

        Args:
            stimulus: Text to score.
            extract: Value keys to compute; ``None`` computes every per-token
                key the constructor enabled. A key that is not per token raises.

        Returns:
            One record per token. Its metadata carries the token's index,
            character span in the stimulus text, id and decoded string; its
            values hold the token's log probability given the preceding context
            (nats) and the Shannon or Rényi entropy of the distribution that
            predicted it (bits), as enabled on the constructor and selected by
            *extract*.
        """
        return token_outputs(
            lambda sub: self._score_tokens(sub, extract), stimulus, self.context_scope
        )

    def _token_request(self, extract: list[str] | None) -> _FeatureRequest:
        request = self._feature_request(extract)
        if extract is not None:
            allowed = {
                "log_p_observed",
                "token_entropy",
                *(f"token_renyi_entropy_alpha{a}" for a in self.token_renyi_alphas),
            }
            unsupported = sorted(set(extract) - allowed)
            if unsupported:
                raise ValueError(
                    f"token scoring cannot supply {unsupported}; the per-token "
                    f"keys are {sorted(allowed)}"
                )
        return request

    def _score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None,
    ) -> list[FeatureModelOutput]:
        import torch

        self._ensure_loaded()
        request = self._token_request(extract)
        ids, spans = tokenize_spans(stimulus.text, self._tokenizer)
        if not ids:
            return []
        prefix = self._start_prefix()
        token_ids = [*prefix, *ids]
        bos_offset = len(prefix)
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self._forward_logits(input_ids, len(token_ids))
        log_probs = torch.log_softmax(logits[0], dim=-1)
        n_tokens = len(ids)
        token_log_p = self._token_log_probabilities(
            log_probs, token_ids, bos_offset, n_tokens, request.compute_log_p
        )
        token_entropy, token_renyi = self._token_entropy_arrays(
            log_probs,
            n_tokens,
            bos_offset,
            request.compute_tok_ent,
            request.renyi_alphas,
            request.compute_renyi,
        )
        offset = stimulus.metadata.get(TEXT_OFFSET, 0)
        outputs: list[FeatureModelOutput] = []
        for t in range(n_tokens):
            values: dict = {}
            if request.compute_log_p:
                values["log_p_observed"] = float(token_log_p[t])
            if request.compute_tok_ent and token_entropy is not None:
                values["token_entropy"] = float(token_entropy[t])
            if request.compute_renyi:
                for a in request.renyi_alphas:
                    values[f"token_renyi_entropy_alpha{a}"] = float(token_renyi[a][t])
            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={
                        "model_name": self.name,
                        "start_ids": self._start_prefix(),
                        "context_window": self._context_length(),
                        "token_index": t,
                        "span": (spans[t].start + offset, spans[t].end + offset),
                        "token_id": int(ids[t]),
                        "token": self._tokenizer.decode([ids[t]]),
                    },
                )
            )
        return outputs

    def _context_length(self) -> int:
        if not hasattr(self, "_resolved_context_window"):
            self._resolved_context_window = usable_position_limit(
                str(self.name),
                self._model.config,
                context_window=getattr(self, "context_window", None),
            )
        return self._resolved_context_window

    def _forward_logits(self, input_ids, n_tokens: int):
        import torch

        max_len = self._context_length()
        with torch.no_grad():
            if n_tokens <= max_len:
                model_out = self._model(input_ids)
                return model_out.logits  # (1, seq_len, vocab_size)
            return self._sliding_window_logits(input_ids, n_tokens, max_len)

    def _sliding_window_logits(self, input_ids, n_tokens: int, max_len: int):
        import torch

        stride = max_len // 2
        vocab_size = self._model.config.vocab_size
        logits = torch.zeros(1, n_tokens, vocab_size, device=self.device)
        filled = torch.zeros(n_tokens, dtype=torch.bool)
        for start in range(0, n_tokens, stride):
            end = min(start + max_len, n_tokens)
            chunk_ids = input_ids[:, start:end]
            chunk_out = self._model(chunk_ids)
            # A later window scores from the previous window's end.
            fill_start = start if start == 0 else start + max_len - stride
            fill_end = end
            mask = ~filled[fill_start:fill_end]
            logits[0, fill_start:fill_end][mask] = chunk_out.logits[
                0, (fill_start - start) : (fill_end - start)
            ][mask]
            filled[fill_start:fill_end] = True
            if end == n_tokens:
                break
        return logits

    def _token_log_probabilities(
        self,
        log_probs,
        token_ids: list[int],
        bos_offset: int,
        n_original_tokens: int,
        compute_log_p: bool,
    ) -> np.ndarray:
        # Per-token log P(token_t | token_{<t}); BOS shifts originals by bos_offset.
        token_log_p = np.zeros(n_original_tokens)
        if not compute_log_p:
            return token_log_p
        if self.add_bos and bos_offset > 0:
            for t in range(n_original_tokens):
                src = t + bos_offset
                token_log_p[t] = log_probs[src - 1, token_ids[src]].item()
            return token_log_p
        token_log_p[0] = np.nan
        for t in range(1, n_original_tokens):
            token_log_p[t] = log_probs[t - 1, token_ids[t]].item()
        return token_log_p

    def _token_entropy_arrays(
        self,
        log_probs,
        n_original_tokens: int,
        bos_offset: int,
        compute_tok_ent: bool,
        renyi_alphas: list[float],
        compute_renyi: bool,
    ) -> tuple[np.ndarray | None, dict[float, np.ndarray]]:
        # Token-level Shannon entropy (full token vocab).
        token_entropy = np.full(n_original_tokens, np.nan) if compute_tok_ent else None
        # token_renyi[alpha][t] = subword Rényi-alpha entropy at position t.
        token_renyi: dict[float, np.ndarray] = {}
        if compute_renyi:
            for a in renyi_alphas:
                token_renyi[a] = np.full(n_original_tokens, np.nan)
        if compute_tok_ent or compute_renyi:
            for t in range(n_original_tokens):
                # Predict-position for token t.
                pred_pos = (
                    t + bos_offset - 1 if self.add_bos and bos_offset > 0 else t - 1
                )
                if pred_pos < 0:
                    continue
                # float32 as emitted by the forward pass; only the restricted
                # next-word path runs in float64.
                lp = log_probs[pred_pos].cpu().numpy()
                if compute_tok_ent:
                    token_entropy[t] = float(shannon_bits(lp))
                for a in renyi_alphas if compute_renyi else ():
                    token_renyi[a][t] = float(renyi_bits(lp, a))
        return token_entropy, token_renyi

    @staticmethod
    def _first_subword_array(
        token_values: np.ndarray | None,
        mapping: list[list[int]],
        n_units: int,
    ) -> np.ndarray | None:
        # Per-unit token-level entropy: first subword's entropy.
        if token_values is None:
            return None
        unit_values = np.full(n_units, np.nan)
        for w_idx in range(n_units):
            tok_indices = mapping[w_idx]
            if tok_indices:
                unit_values[w_idx] = token_values[tok_indices[0]]
        return unit_values

    @staticmethod
    def _sum_subword_array(
        token_values: np.ndarray | None,
        mapping: list[list[int]],
        n_units: int,
    ) -> np.ndarray | None:
        # A plain sum propagates NaN like the first-subtoken rule does at a
        # no-context (no-BOS first-word) position.
        if token_values is None:
            return None
        unit_values = np.full(n_units, np.nan)
        for w_idx in range(n_units):
            tok_indices = mapping[w_idx]
            if tok_indices:
                unit_values[w_idx] = float(np.sum(token_values[tok_indices]))
        return unit_values

    def _first_subword_renyi(
        self,
        token_renyi: dict[float, np.ndarray],
        mapping: list[list[int]],
        n_units: int,
    ) -> dict[float, np.ndarray]:
        # Per-alpha subword Rényi entropy per unit (first-subword rule).
        unit_tok_renyi: dict[float, np.ndarray] = {}
        for a, values in token_renyi.items():
            arr = self._first_subword_array(values, mapping, n_units)
            if arr is not None:
                unit_tok_renyi[a] = arr
        return unit_tok_renyi

    def _sum_subword_renyi(
        self,
        token_renyi: dict[float, np.ndarray],
        mapping: list[list[int]],
        n_units: int,
    ) -> dict[float, np.ndarray]:
        unit_tok_renyi_sum: dict[float, np.ndarray] = {}
        for a, values in token_renyi.items():
            arr = self._sum_subword_array(values, mapping, n_units)
            if arr is not None:
                unit_tok_renyi_sum[a] = arr
        return unit_tok_renyi_sum

    def _restricted_first_token_entropies(
        self,
        log_probs,
        mapping: list[list[int]],
        bos_offset: int,
        n_units: int,
        compute_rft: bool,
        rft_alphas: list[float],
        compute_rft_renyi: bool,
    ) -> tuple[np.ndarray | None, dict[float, np.ndarray]]:
        # Restrict logits[pred_pos] to word-starter ids (subword ids right
        # after BOS), renormalise, then compute Shannon / Rényi.
        unit_rft = np.full(n_units, np.nan) if compute_rft else None
        unit_rft_renyi: dict[float, np.ndarray] = {}
        if compute_rft_renyi:
            for a in rft_alphas:
                unit_rft_renyi[a] = np.full(n_units, np.nan)
        if compute_rft or compute_rft_renyi:
            space_ids = self._space_ids_np
            subword_ids = self._subword_ids_np
            for w_idx in range(n_units):
                tok_indices = mapping[w_idx]
                if not tok_indices:
                    continue
                first_t = tok_indices[0]
                pred_pos = first_t + bos_offset - 1
                if pred_pos < 0:
                    continue
                # At the BOS position, restrict to subword ids.
                is_bos_pred = (
                    self.add_bos and bos_offset > 0 and pred_pos == bos_offset - 1
                )
                allowed = subword_ids if is_bos_pred else space_ids
                # float64 to match the reference implementation.
                lp_full = log_probs[pred_pos].cpu().double().numpy()
                lp_restricted = lp_full[allowed]
                m = float(lp_restricted.max())
                log_z = m + float(np.log(np.sum(np.exp(lp_restricted - m))))
                lp_norm = lp_restricted - log_z
                if compute_rft:
                    unit_rft[w_idx] = float(shannon_bits(lp_norm))
                for a in rft_alphas:
                    unit_rft_renyi[a][w_idx] = float(renyi_bits(lp_norm, a))
        return unit_rft, unit_rft_renyi

    def _build_outputs(
        self,
        request: _FeatureRequest,
        mapping: list[list[int]],
        log_probs,
        bos_offset: int,
        unit_log_p: np.ndarray,
        unit_tok_ent: np.ndarray | None,
        unit_tok_renyi: dict[float, np.ndarray],
        unit_rft: np.ndarray | None,
        unit_rft_renyi: dict[float, np.ndarray],
        n_units: int,
        unit_tok_ent_sum: np.ndarray | None = None,
        unit_tok_renyi_sum: dict[float, np.ndarray] | None = None,
    ) -> list[FeatureModelOutput]:
        outputs: list[FeatureModelOutput] = []
        for w_idx in range(n_units):
            values: dict = {}

            if request.compute_log_p:
                values[self._unit_log_p_key] = float(unit_log_p[w_idx])

            if request.compute_nlp:
                self._add_next_log_probs(values, w_idx, mapping, log_probs, bos_offset)

            if request.compute_tok_ent and unit_tok_ent is not None:
                values["token_entropy"] = float(unit_tok_ent[w_idx])

            if request.compute_tok_ent_sum and unit_tok_ent_sum is not None:
                values["token_entropy_sum"] = float(unit_tok_ent_sum[w_idx])

            if request.compute_renyi:
                for a in request.renyi_alphas:
                    values[f"token_renyi_entropy_alpha{a}"] = float(
                        unit_tok_renyi[a][w_idx]
                    )

            if request.compute_tok_renyi_sum and unit_tok_renyi_sum:
                a = request.renyi_alphas[0]
                values["token_renyi_entropy_sum"] = float(unit_tok_renyi_sum[a][w_idx])

            if request.compute_rft and unit_rft is not None:
                values["restricted_first_token_entropy"] = float(unit_rft[w_idx])

            if request.compute_rft_renyi:
                for a in request.rft_alphas:
                    values[f"restricted_first_token_renyi_entropy_alpha{a}"] = float(
                        unit_rft_renyi[a][w_idx]
                    )

            metadata = self._output_metadata()
            if request.compute_nlp:
                metadata["vocab_index"] = self._vocab_index
            outputs.append(FeatureModelOutput(values=values, metadata=metadata))

        return outputs

    def _add_next_log_probs(
        self,
        values: dict,
        w_idx: int,
        mapping: list[list[int]],
        log_probs,
        bos_offset: int,
    ) -> None:
        # The row that predicts the unit's first token, as for the entropies;
        # a token between two units stays in the context.
        tok_indices = mapping[w_idx]
        if not tok_indices:
            return
        pred_pos = tok_indices[0] + bos_offset - 1
        if pred_pos < 0:
            return
        nlp_np = log_probs[pred_pos].cpu().numpy()
        values["next_log_probs"] = self._marginalise_to_word_dist(nlp_np)
