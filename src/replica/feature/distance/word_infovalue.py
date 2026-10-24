"""Word-level, per-layer information value from Tsipidi et al. (2026).

Faithful port of ``score_infovalue.py`` in ``rycolab/llm-representations-rt``:
for each whitespace word after the first, sample ``k`` continuations of the
document-so-far context, cut each at its first word, and report per layer the
mean cosine distance between the contextualised embedding of the target word
and those of the candidates (mean over the word's tokens; a NaN cosine
counts as zero distance; layer 0 is the embedding layer). ``model.generate(do_sample=True)`` inherits the checkpoint's
sampling defaults, which for mGPT and GPT-2 truncate to ``top_k=50``; that is
the authors' convention, and it is reproduced.

Differences from the reference: the first word and failed words are NaN
rows rather than omitted, and sampling is seeded per stimulus instead of once per process. The reference imposes no context
cap, so mGPT inputs beyond 512 tokens exceed its documented training window.

Two opt-in extensions carry the estimator to corpora the reference cannot
run on, each inert where the reference applies. ``window_policy="cap"``
trims the context to its longest whole-word suffix fitting the model's
position window (documents longer than the window make the reference
error out); when everything fits, values equal the reference's exactly.
``segmentation="layer"`` iterates the corpus's own word layer and joins
words to their context without an inserted space, for scripts written
without word spacing; it refuses corpora where whitespace splitting
already yields the layer's units. The candidate cut
(:func:`first_generated_word`) is unchanged in both modes.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from replica.feature.backends.hub import hub_id

from ..backends.generation import NonFiniteLogitsError, usable_position_limit
from ..core.protocols import FeatureModelOutput

if TYPE_CHECKING:
    from ...core import Stimulus

#: The authors' word-boundary set: ASCII punctuation without the apostrophe,
#: plus curly quotes, guillemets, dashes, and the ellipsis.
PUNCTUATION_FOR_WORD_BOUNDARY = string.punctuation.replace("'", "") + "“”‘’«»—–…"


def clean_word(word: str) -> str:
    """Strip surrounding whitespace and boundary punctuation.

    Args:
        word: A surface form, observed or sampled.
    """
    return word.strip().strip(PUNCTUATION_FOR_WORD_BOUNDARY)


def first_generated_word(generated_text: str) -> str | None:
    """First word of a continuation, cut at whitespace or boundary punctuation.

    Args:
        generated_text: The decoded continuation, leading whitespace included.

    Returns:
        The first word, or ``None`` when the text holds no word characters
        before the cut.
    """
    if not generated_text:
        return None
    i, n = 0, len(generated_text)
    while i < n and generated_text[i].isspace():
        i += 1
    if i >= n:
        return None
    chars = []
    while i < n:
        ch = generated_text[i]
        if ch.isspace() or ch in PUNCTUATION_FOR_WORD_BOUNDARY:
            break
        chars.append(ch)
        i += 1
    word = "".join(chars).strip(PUNCTUATION_FOR_WORD_BOUNDARY)
    return word or None


class WordInfoValueLM:
    """Per-word, per-layer information value over sampled continuations.

    Args:
        model_id: Causal LM used for both sampling and embeddings.
        k: Candidate continuations per word.
        max_new_tokens: Generation budget per candidate before the first-word
            cut.
        seed: Torch seed, applied per stimulus.
        device: ``None`` picks cuda when available.
        attn_implementation: Attention backend; the default ``eager`` avoids
            SDPA kernels that intermittently emit non-finite logits under
            memory pressure. ``None`` keeps the transformers default.
        window_policy: ``reference`` passes the whole document-so-far and errors
            past the model window; ``cap`` trims to the longest whole-word
            suffix that fits.
        segmentation: ``whitespace`` iterates the text's whitespace words, as
            the reference does; ``layer`` iterates the corpus's own word layer
            and joins without an inserted space, for unspaced scripts.
    """

    @property
    def supports_output_union(self) -> bool:
        """Combining requests preserves the same computed or sampled values."""
        return type(self) is WordInfoValueLM

    def __init__(
        self,
        model_id: str = "ai-forever/mGPT",
        k: int = 5,
        max_new_tokens: int = 3,
        seed: int = 0,
        device: str | None = None,
        attn_implementation: str | None = "eager",
        window_policy: str = "reference",
        segmentation: str = "whitespace",
    ):
        """Record the sampling and context settings; the model loads on demand.

        The class docstring describes the parameters.

        Raises:
            ValueError: When *window_policy* or *segmentation* is outside its
                two accepted values.
        """
        if window_policy not in ("reference", "cap"):
            raise ValueError(f"unknown window_policy {window_policy!r}")
        if segmentation not in ("whitespace", "layer"):
            raise ValueError(f"unknown segmentation {segmentation!r}")
        self.model_id = model_id
        self.k = int(k)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = int(seed)
        self.window_policy = window_policy
        self.segmentation = segmentation
        self.name = f"word_infoval_{hub_id(model_id)}"
        self._device = device
        self._attn_implementation = attn_implementation
        self._model = None
        self._tokenizer = None
        self._n_layers = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if (
            self._tokenizer.pad_token_id is None
            and self._tokenizer.eos_token_id is not None
        ):
            self._tokenizer.pad_token = self._tokenizer.eos_token
        kwargs = {}
        if self._attn_implementation is not None:
            kwargs["attn_implementation"] = self._attn_implementation
        self._model = (
            AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            .eval()
            .to(self._device)
        )
        config = self._model.config
        self._n_layers = getattr(config, "n_layer", None) or config.num_hidden_layers

    @property
    def layer_count(self) -> int:
        """Transformer layers plus the embedding layer."""
        self._ensure_model()
        return self._n_layers + 1

    def _generate_candidates(self, context_text: str) -> list[str]:
        from ..backends.logits import FiniteLogitsProcessor

        inputs = self._tokenizer(context_text or "", return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.k,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                logits_processor=[FiniteLogitsProcessor()],
            )
        prompt_len = input_ids.shape[1]
        words = []
        for seq in outputs:
            text = self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            word = first_generated_word(text)
            if word:
                words.append(word)
        return words

    def _separator(self, stimulus: Stimulus, target_layer: str) -> str:
        """A space, or nothing under layer segmentation of an unspaced script."""
        if self.segmentation == "whitespace" or self._spaced(stimulus, target_layer):
            return " "
        return ""

    @staticmethod
    def _spaced(stimulus: Stimulus, target_layer: str) -> bool:
        """Whether whitespace separates the units of *target_layer* in the text."""
        spans = stimulus.layers[target_layer].spans
        return any(
            character.isspace()
            for before, after in zip(spans, spans[1:], strict=False)
            for character in stimulus.text[before.end : after.start]
        )

    def _word_embedding_layers(
        self, context_text: str, word: str, separator: str
    ) -> torch.Tensor | None:
        word = word.strip()
        if not word:
            return None
        context_ids = self._tokenizer(
            context_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self._device)
        full_text = context_text + (separator if context_text else "") + word
        full_ids = self._tokenizer(
            full_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self._device)
        context_len = context_ids.shape[1]
        total_len = full_ids.shape[1]
        if total_len <= context_len:
            return None
        if self.window_policy == "cap":
            limit = usable_position_limit(self.model_id, self._model.config)
            if total_len > limit:
                # A candidate can re-encode slightly longer than the trim
                # margin allowed for; drop it rather than fail the word.
                return None
        else:
            limit = getattr(self._model.config, "n_positions", None)
            if limit is not None and total_len > limit:
                raise ValueError(
                    f"context of {total_len} tokens exceeds the model window "
                    f"({limit}); the reference computation has no truncation"
                )
        with torch.no_grad():
            outputs = self._model(input_ids=full_ids, output_hidden_states=True)
        layers = []
        for layer_h in outputs.hidden_states:
            word_tokens = layer_h[0, context_len:total_len, :]
            if word_tokens.numel() == 0:
                return None
            layers.append(word_tokens.mean(dim=0))
        return torch.stack(layers, dim=0)

    def _infovalue_for_word(
        self, context_text: str, target_word: str, separator: str
    ) -> list[float] | None:
        target_word = clean_word(target_word)
        if not target_word:
            return None
        try:
            candidates = self._generate_candidates(context_text)
        except NonFiniteLogitsError:
            return None
        if not candidates:
            return None
        target_emb = self._word_embedding_layers(context_text, target_word, separator)
        if target_emb is None:
            return None
        cand_embs = []
        for cand in candidates:
            cand_clean = clean_word(cand)
            if not cand_clean:
                continue
            emb = self._word_embedding_layers(context_text, cand_clean, separator)
            if emb is not None:
                cand_embs.append(emb)
        if not cand_embs:
            return None
        cand_stack = torch.stack(cand_embs, dim=0)
        values = []
        for layer_idx in range(target_emb.shape[0]):
            cos_sim = F.cosine_similarity(
                cand_stack[:, layer_idx, :], target_emb[layer_idx].unsqueeze(0), dim=-1
            )
            cos_dist = torch.nan_to_num(1.0 - cos_sim, nan=0.0)
            values.append(float(cos_dist.mean().item()))
        return values

    def _unit_words(self, stimulus: Stimulus, target_layer: str) -> list[str]:
        """The unit surfaces the loop iterates: whitespace words, or the layer's units."""
        layer = stimulus.layers[target_layer]
        if self.segmentation == "whitespace":
            words = stimulus.text.split()
            if len(words) != len(layer):
                raise ValueError(
                    f"WordInfoValueLM: stimulus's {target_layer!r} layer has "
                    f"{len(layer)} units but the text whitespace-splits into "
                    f"{len(words)} words; the reference iterates whitespace "
                    "words (use segmentation='layer' to score the units)."
                )
            return words
        return [stimulus.text[s.start : s.end] for s in layer.spans]

    def _token_len(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _trimmed_context(
        self, stimulus: Stimulus, words: list[str], target_layer: str, i: int
    ) -> str | None:
        """Context for word *i*: the reference's full document-so-far, or the
        longest whole-word suffix fitting the model window under the ``cap``
        policy. ``None`` when even a one-word context cannot fit.
        """
        spans = stimulus.layers[target_layer].spans
        sep = self._separator(stimulus, target_layer)

        def context(lo: int) -> str:
            if sep:
                return " ".join(words[lo:i])
            return stimulus.text[spans[lo].start : spans[i].start].rstrip()

        if self.window_policy == "reference":
            return context(0)
        # Fit the context and target within the model's resolved window.
        limit = usable_position_limit(self.model_id, self._model.config)
        target = clean_word(words[i])

        def fits(lo: int) -> bool:
            ctx = context(lo)
            full = ctx + (sep if ctx else "") + target
            return (
                self._token_len(full) <= limit
                and self._token_len(ctx) + self.max_new_tokens + 2 <= limit
            )

        if fits(0):
            return context(0)
        if i == 1 or not fits(i - 1):
            return None
        low, high = 1, i - 1
        while low < high:
            mid = (low + high) // 2
            if fits(mid):
                high = mid
            else:
                low = mid + 1
        return context(low)

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Per-layer information value at every word of *stimulus*.

        Args:
            stimulus: Passage to score; the words are iterated in text order.
            extract: Part of the shared FeatureModel signature; every layer's
                value is returned regardless.
            target_layer: Annotation layer the outputs align to, and, under
                ``segmentation="layer"``, the source of the unit surfaces.

        Returns:
            One output per unit, keyed ``infovalue_layer{i}`` for layer 0 (the
            embeddings) up to the last transformer layer. The first word, and
            any word whose sampling or embedding failed, is an all-NaN row.

        Raises:
            ValueError: When the stimulus has no *target_layer*, or the
                whitespace mode meets a text that does not split into the
                layer's units.
        """
        if target_layer not in stimulus.layers:
            raise ValueError(
                f"WordInfoValueLM expects a {target_layer!r} layer; "
                f"available: {list(stimulus.layers)}"
            )
        words = self._unit_words(stimulus, target_layer)
        separator = self._separator(stimulus, target_layer)
        self._ensure_model()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        nan_row = {f"infovalue_layer{i}": float("nan") for i in range(self.layer_count)}
        outputs = [
            FeatureModelOutput(values=dict(nan_row), metadata={"model_name": self.name})
        ]
        for i in range(1, len(words)):
            context_text = self._trimmed_context(stimulus, words, target_layer, i)
            values = (
                self._infovalue_for_word(context_text, words[i], separator)
                if context_text is not None
                else None
            )
            row = dict(nan_row)
            if values is not None:
                row.update({f"infovalue_layer{j}": v for j, v in enumerate(values)})
            outputs.append(
                FeatureModelOutput(values=row, metadata={"model_name": self.name})
            )
        return outputs
