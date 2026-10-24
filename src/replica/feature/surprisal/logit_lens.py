"""Internal-layer next-word surprisal via the logit lens, Kuribayashi et al. (2025).

Projects an intermediate hidden state to the vocabulary with the model's
own final layer norm + unembedding (nostalgebraist 2020)::

    p(w_t | w_<t ; h_{l,t}) = softmax(W_U · LayerNorm(h_{l,t}))[id(w_t)]

Per-word surprisal is the sum of its subword surprisals. Follows the recipe in
the paper's ``src/run_gpt2.py`` (``<bos> w_1 … w_n`` with leading-space
tokenisation); anchored against the authors' released ``surprisal.json``, to
the tolerances the reproduction test asserts.

The companion extractor :class:`~replica.feature.LayerSurprisal`
selects a layer's surprisal in bits. Layer 0 is the embedding layer
(excluded from paper analysis); the final layer reproduces standard surprisal.
"""

from dataclasses import dataclass

import numpy as np

from replica.feature.core.annotation.alignment import TEXT_OFFSET
from replica.feature.core.annotation.tokens import token_outputs

from ...core import Span, Stimulus
from ..backends.generation import usable_position_limit
from ..core.protocols import FeatureModelOutput

_EPS = 1e-8  # the epsilon the anchored released predictors are consistent with


@dataclass(frozen=True)
class _LensPass:
    """One forward pass over the words of a stimulus joined by single spaces."""

    words: list[str]
    token_ids: list[int]
    offsets: np.ndarray
    surprisal: dict[int, np.ndarray]
    layer_ids: list[int]
    final: int


def _resolve_lens_components(model):
    """Return ``(final_layer_norm, unembedding)`` for a causal LM.

    The logit lens needs the model's *final* layer norm (applied to every
    internal hidden state before projection) and its unembedding matrix
    (the output ``lm_head``). Attribute paths cover the paper's
    architectures; a generic fallback
    handles the head via ``get_output_embeddings``.
    """
    # GPT-2 / GPT-J / Falcon-style: transformer.ln_f + lm_head
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f, model.lm_head
    # GPT-NeoX (Pythia): gpt_neox.final_layer_norm + embed_out
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm, model.embed_out
    # OPT: model.decoder.final_layer_norm + lm_head
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and getattr(model.model.decoder, "final_layer_norm", None) is not None
    ):
        return model.model.decoder.final_layer_norm, model.lm_head
    # XGLM: model.layer_norm + lm_head
    if hasattr(model, "model") and getattr(model.model, "layer_norm", None) is not None:
        return model.model.layer_norm, model.lm_head
    # Llama / Qwen / Mistral-style: model.norm (RMSNorm) + lm_head
    if hasattr(model, "model") and getattr(model.model, "norm", None) is not None:
        return model.model.norm, model.get_output_embeddings()
    raise ValueError(
        f"LogitLensLM: cannot locate final layer norm / unembedding for "
        f"model class {model.__class__.__name__!r}. Supported families: "
        "GPT-2, GPT-NeoX (Pythia), OPT, XGLM, Llama/Qwen."
    )


class LogitLensLM:
    """Per-layer next-word surprisal from a HuggingFace causal LM via logit lens.

    For each requested layer ``l`` and each word, the summed subword
    surprisal ``-log p(w | context; h_{l})`` is computed and stored (as the
    log-prob, in nats, to match the ``log_p_observed`` convention)
    under the key ``"logit_lens_log_p_L{l}"``. The final layer is also
    mirrored to ``"log_p_observed"`` so the standard
    :class:`~replica.feature.extractors.Surprisal` extractor reads it
    directly.

    Each ``Stimulus`` is processed as a single sequence with one prepended
    BOS token, so the first word of the stimulus receives a defined
    surprisal. The paper resets context at every sentence;
    ``context_scope="sentence"`` reproduces that.

    Args:
        model_name: HuggingFace id of the causal LM to load, also the name
            reported in AnalysisResult results.
        layers: ``None`` computes all layers ``0..n_blocks`` (0 = embeddings,
            ``n_blocks`` = output). Pass a list to restrict (negative
            indices count from the output, e.g. ``-1`` is the output).
        device: ``"auto"`` selects CUDA when available, otherwise CPU. An
            explicit device such as ``"cpu"`` is always respected.
        assert_final_matches: Assert that the logit-lens projection of the
            final hidden state matches the model's own output logits
            (``atol=1e-4``). Skipped for quantized models.
        context_scope: ``"stimulus"`` scores the whole stimulus as one
            BOS-prefixed sequence; ``"sentence"`` scores each sentence
            independently via the stimulus's ``sentences`` layer, the
            paper's convention for document corpora.
        revision: Immutable Hugging Face revision for model and tokenizer;
            ``None`` uses the repository default.
        device_map: Optional Transformers placement strategy or module mapping.
            Dispatched models retain float32 weights and are not moved wholesale
            to ``device``. Disk-offloaded input embeddings are unsupported.
        max_memory: Optional per-device loading budgets, with ``device_map``.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is LogitLensLM

    vocab = None
    _lens = "logit_lens"  # storage-key prefix; overridden by TunedLensLM

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        layers: list[int] | None = None,
        device: str = "auto",
        assert_final_matches: bool = False,
        context_scope: str = "stimulus",
        revision: str | None = None,
        device_map: str | dict[str, int | str] | None = None,
        max_memory: dict[int | str, int | str] | None = None,
    ):
        """Record the lens settings; weights load lazily on the first call."""
        if context_scope not in ("stimulus", "sentence"):
            raise ValueError(f"unknown context_scope {context_scope!r}")
        if max_memory is not None and device_map is None:
            raise ValueError("max_memory requires a device_map")
        self.context_scope = context_scope
        self.name = model_name
        self.revision = revision
        self._requested_layers = list(layers) if layers is not None else None
        self.device = device
        self.device_map = device_map
        self.max_memory = max_memory
        self.assert_final_matches = assert_final_matches
        self._model = None
        self._tokenizer = None
        self._final_norm = None
        self._lm_head = None
        self._bos_id: int | None = None
        self._n_blocks: int | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, revision=self.revision
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        placement = {}
        if self.device_map is not None:
            placement = {
                "device_map": self.device_map,
                "max_memory": self.max_memory,
                "torch_dtype": torch.float32,
            }
        self._model = AutoModelForCausalLM.from_pretrained(
            self.name, revision=self.revision, low_cpu_mem_usage=True, **placement
        )
        if self.device_map is None:
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
        else:
            input_device = self._model.get_input_embeddings().weight.device
            if input_device.type == "meta":
                raise ValueError("Offloaded input embeddings are unsupported")
            self.device = str(input_device)
        self._model.eval()
        self._final_norm, self._lm_head = _resolve_lens_components(self._model)
        # BOS id giving the first word left context. Prefer the model config's
        # id (what the recipe decodes) and fall back to the tokenizer's.
        self._bos_id = self._model.config.bos_token_id
        if self._bos_id is None:
            self._bos_id = self._tokenizer.bos_token_id
        if self._bos_id is None:
            raise ValueError(
                f"LogitLensLM: model/tokenizer {self.name!r} has no bos_token_id; "
                "the recipe prepends a BOS token for left context."
            )

    @property
    def layers(self) -> list[int]:
        """Resolved (non-negative) list of layer indices this model emits."""
        self._ensure_loaded()
        n = self._n_blocks
        if n is None:
            # Determined on first forward pass; fall back to config count.
            n = getattr(self._model.config, "num_hidden_layers", None) or getattr(
                self._model.config, "n_layer", None
            )
        all_layers = list(range(n + 1))
        if self._requested_layers is None:
            return all_layers
        return [
            (n + 1 + layer) if layer < 0 else layer for layer in self._requested_layers
        ]

    def _project_hidden_state(self, hidden, *, normalize):
        """Project on each module's device without moving dispatched weights.

        Args:
            hidden: Hidden states from one transformer layer.
            normalize: Apply the final norm before the vocabulary projection.
        """
        if normalize:
            if self.device_map is not None:
                hidden = hidden.to(next(self._final_norm.parameters()).device)
            hidden = self._final_norm(hidden)
        if self.device_map is not None:
            hidden = hidden.to(next(self._lm_head.parameters()).device)
        return self._lm_head(hidden)

    def _layer_logits(self, hidden_states, layer_id, out_logits):
        """Vocabulary logits from internal hidden state ``layer_id``.

        Logit lens: the model's final layer norm then unembedding,
        ``lm_head(ln_f(h))``. The output layer uses the model's own head.
        """
        import torch

        h = hidden_states[layer_id]
        if layer_id == self._n_blocks:
            logits = self._project_hidden_state(h, normalize=False)
            if self.assert_final_matches and not hasattr(self._model, "hf_quantizer"):
                assert torch.allclose(
                    logits, out_logits.to(logits.device), atol=1e-4
                ), "logit-lens final layer != model output logits"
            return logits
        return self._project_hidden_state(h, normalize=True)

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One output per unit, carrying a log-prob in nats for every requested layer.

        Args:
            stimulus: Text to score, carrying the annotation layers.
            extract: Value keys to keep; ``None`` keeps every layer key plus
                ``log_p_observed``.
            target_layer: Name of the annotation layer whose units the outputs
                align to, one per unit.
        """
        if self.context_scope == "sentence":
            from replica.feature.core.annotation.sentences import per_sentence_outputs

            return per_sentence_outputs(
                lambda sub: self._call_stimulus(sub, extract, target_layer),
                stimulus,
                target_layer,
            )
        return self._call_stimulus(stimulus, extract, target_layer)

    def _call_stimulus(
        self,
        stimulus: Stimulus,
        extract: list[str] | None,
        target_layer: str,
    ) -> list[FeatureModelOutput]:
        scored = self._score(stimulus, target_layer)
        if scored is None:
            return []
        outputs: list[FeatureModelOutput] = []
        for i in range(len(scored.words)):
            lo, hi = int(scored.offsets[i]), int(scored.offsets[i + 1])
            values: dict = {}
            for layer_id in scored.layer_ids:
                # The word's log-prob (nats) is minus its summed subword surprisal.
                values[f"{self._lens}_log_p_L{layer_id}"] = -float(
                    scored.surprisal[layer_id][lo:hi].sum()
                )
            if scored.final in scored.surprisal:
                values["log_p_observed"] = values[f"{self._lens}_log_p_L{scored.final}"]
            if extract is not None:
                values = {k: v for k, v in values.items() if k in extract}
            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={"model_name": self.name, "n_layers": scored.final},
                )
            )
        return outputs

    def score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
    ) -> list[FeatureModelOutput]:
        """One output per subword of the joined words of *stimulus*, in text order.

        The lens scores the words joined by single spaces after a
        beginning-of-sequence token, so every token lies inside one word and has
        a predicting position. A token that consists of the leading space marker
        alone shares the span of the word's first character.

        Args:
            stimulus: Text to score, carrying the ``words`` layer.
            extract: Value keys to keep; ``None`` keeps every layer key plus
                ``log_p_observed``.

        Returns:
            One record per token, carrying its index, character span in the
            stimulus text, id and decoded string, and its log probability
            (nats) under every requested layer.
        """
        return token_outputs(
            lambda sub: self._score_tokens(sub, extract), stimulus, self.context_scope
        )

    def window(self) -> dict:
        """The context-window rule the forward pass applies, for provenance.

        Windows of ``max_positions`` tokens advance by ``stride``; a later
        window scores the positions from the previous window's end on.
        """
        self._ensure_loaded()
        max_pos = usable_position_limit(str(self.name), self._model.config)
        return {
            "rule": "sliding",
            "max_positions": max_pos,
            "stride": max_pos // 2,
            "fresh_from": "start + max_positions - stride",
        }

    def _score_tokens(
        self,
        stimulus: Stimulus,
        extract: list[str] | None,
    ) -> list[FeatureModelOutput]:
        scored = self._score(stimulus, "words")
        if scored is None:
            return []
        spans = self._token_spans(scored, stimulus)
        offset = stimulus.metadata.get(TEXT_OFFSET, 0)
        outputs: list[FeatureModelOutput] = []
        for t, token_id in enumerate(scored.token_ids):
            values: dict = {
                f"{self._lens}_log_p_L{layer_id}": -float(scored.surprisal[layer_id][t])
                for layer_id in scored.layer_ids
            }
            if scored.final in scored.surprisal:
                values["log_p_observed"] = values[f"{self._lens}_log_p_L{scored.final}"]
            if extract is not None:
                values = {k: v for k, v in values.items() if k in extract}
            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={
                        "model_name": self.name,
                        "n_layers": scored.final,
                        "token_index": t,
                        "span": (spans[t].start + offset, spans[t].end + offset),
                        "token_id": int(token_id),
                        "token": self._tokenizer.decode([token_id]),
                    },
                )
            )
        return outputs

    def _token_spans(self, scored: _LensPass, stimulus: Stimulus) -> list[Span]:
        """Character span of every subword, from the per-word encodings.

        Raises:
            ValueError: When a word is empty, or its own encoding differs from
                the joined encoding the pass scored.
        """
        tok = self._tokenizer
        spans: list[Span] = []
        for i, (word, unit) in enumerate(
            zip(scored.words, stimulus.layers["words"].spans, strict=True)
        ):
            if unit.end <= unit.start:
                raise ValueError(
                    f"{stimulus.stimulus_id}: word {i} is empty; the lens token "
                    "table has no span for it"
                )
            encoded = tok(
                " " + word, add_special_tokens=False, return_offsets_mapping=True
            )
            lo, hi = int(scored.offsets[i]), int(scored.offsets[i + 1])
            if list(encoded["input_ids"]) != scored.token_ids[lo:hi]:
                raise ValueError(
                    f"{stimulus.stimulus_id}: word {i} {word!r} encodes differently "
                    "on its own than inside the joined text"
                )
            for start, end in encoded["offset_mapping"]:
                # Offsets index " " + word; the leading space precedes the unit.
                piece = Span(unit.start + max(start, 1) - 1, unit.start + end - 1)
                if piece.end <= piece.start:
                    piece = Span(unit.start, unit.start + 1)
                spans.append(piece)
        return spans

    def _score(self, stimulus: Stimulus, target_layer: str) -> _LensPass | None:
        """Per-position surprisal under every requested layer, or ``None``
        when the stimulus has no unit.
        """
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        words = stimulus.layers[target_layer].extract_strings(stimulus.text)
        n_units = len(words)
        if n_units == 0:
            return None

        # Per-word subword counts using a leading space (paper recipe).
        tok_ls = [
            len(tok(" " + w, add_special_tokens=False)["input_ids"]) for w in words
        ]
        bos_string = tok.decode(self._bos_id)
        text = bos_string + " " + " ".join(words)
        encoded = tok(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(self.device)

        n_tokens = int(encoded.shape[1])
        n_subwords = n_tokens - 1  # positions predicted (drop BOS pos)
        offsets = np.cumsum([0, *tok_ls])
        # leading-space per-word tokenization must reconstruct joined
        # encoding for the cumulative split to be correct. This holds for
        # GPT-2-style byte-level BPE (the paper's models) but can fail for
        # other tokenizers (e.g. SentencePiece).
        if offsets[-1] != n_subwords:
            raise ValueError(
                f"LogitLensLM({self.name}): leading-space tokenization "
                f"({int(offsets[-1])} subwords) does not match the joined "
                f"encoding ({n_subwords}); per-word alignment is undefined for "
                "this tokenizer/stimulus."
            )

        # Long stimuli use the store's sliding-window convention (the
        # HuggingFaceLM backend): windows of the model's trained context,
        # stride = half the window, each position scored by the first window
        # that covers it, with at least half a window of left context.
        max_pos = usable_position_limit(str(self.name), self._model.config)
        if n_tokens <= max_pos:
            windows = [(0, n_tokens)]
            stride = None
        else:
            stride = max_pos // 2
            windows = []
            for start in range(0, n_tokens, stride):
                end = min(start + max_pos, n_tokens)
                windows.append((start, end))
                if end == n_tokens:
                    break

        with torch.no_grad():
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            layer_ids = self.layers
            # pos_surp[layer_id][t] = surprisal (nats) of token t+1 given
            # its window context; every target position is filled exactly
            # once, by the first window that covers it.
            pos_surp: dict[int, np.ndarray] = {
                layer_id: np.full(n_subwords, np.nan) for layer_id in layer_ids
            }
            filled = np.zeros(n_subwords, dtype=bool)
            for w_index, (start, end) in enumerate(windows):
                out = self._model(encoded[:, start:end], output_hidden_states=True)
                hidden_states = out.hidden_states  # tuple, len = n_blocks + 1
                self._n_blocks = len(hidden_states) - 1
                fill_start = start if w_index == 0 else start + max_pos - stride
                fill_end = min(end, n_subwords)
                if fill_start >= fill_end:
                    continue
                local = slice(fill_start - start, fill_end - start)
                targets = encoded[:, fill_start + 1 : fill_end + 1]
                mask = ~filled[fill_start:fill_end]
                for layer_id in layer_ids:
                    logits = self._layer_logits(hidden_states, layer_id, out.logits)[
                        :, local
                    ]
                    surp = (
                        loss_fct(logits.transpose(1, 2), targets.to(logits.device))[0]
                        + _EPS
                    )
                    segment = pos_surp[layer_id][fill_start:fill_end]
                    segment[mask] = surp.detach().float().cpu().numpy()[mask]
                filled[fill_start:fill_end] = True

        return _LensPass(
            words=words,
            token_ids=encoded[0, 1:].tolist(),
            offsets=offsets,
            surprisal=pos_surp,
            layer_ids=layer_ids,
            final=self._n_blocks,
        )
