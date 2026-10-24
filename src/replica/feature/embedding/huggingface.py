"""HuggingFace embedding extraction for continuous-response encoding models.

Extracts per-word, per-layer hidden state vectors from a HuggingFace encoder
or decoder model loaded with ``AutoModel``; an encoder-decoder model, whose
forward pass needs decoder inputs, cannot be run. Satisfies the FeatureModel
protocol.
"""

import numpy as np

from replica.feature.backends.hub import hub_id
from replica.feature.core.annotation.alignment import (
    TokenAlignment,
    align_from_tokenizer,
)

from ...core import Stimulus
from ..backends.generation import usable_position_limit
from ..core.protocols import FeatureModelOutput


class HuggingFaceEmbedding:
    """Extract per-word hidden states from a HuggingFace model.

    For each word, averages the hidden state vectors of its constituent
    subword tokens.

    Returns FeatureModelOutput per word with
    ``values["hidden_state_{layer}"] = np.ndarray`` of shape ``(embed_dim,)``.

    Args:
        model_name: HuggingFace model id (e.g. "gpt2", "bert-base-uncased") or
            local snapshot directory to load; ``name`` reports its Hub id.
        layers: Which layers to extract. None = all layers.
        context_length: Sliding window context in words. None = full stimulus.
        unit_tokenization: ``"contextual"`` tokenizes each complete window.
            ``"independent"`` tokenizes every unit separately before joining
            token IDs into a window, and needs ``context_length``.
        device: PyTorch device string.
    """

    vocab = None

    @property
    def supports_output_union(self) -> bool:
        """This deterministic producer preserves outputs when requests are combined."""
        return type(self) is HuggingFaceEmbedding

    def __init__(
        self,
        model_name: str = "gpt2",
        layers: list[int] | None = None,
        context_length: int | None = None,
        unit_tokenization: str = "contextual",
        device: str = "cpu",
    ):
        """Record the extraction settings; the model loads on first call."""
        if unit_tokenization not in {"contextual", "independent"}:
            raise ValueError("unit_tokenization must be 'contextual' or 'independent'")
        if unit_tokenization == "independent" and context_length is None:
            raise ValueError(
                "unit_tokenization='independent' needs context_length; without "
                "it the whole stimulus is tokenized as one text"
            )
        # A pinned snapshot path loads as given; only the name maps to the id.
        self._checkpoint = model_name
        self.name = hub_id(model_name)
        self._layers = layers
        self._context_length = context_length
        self._unit_tokenization = unit_tokenization
        self.device = device
        self.metadata = {
            "context_length": context_length,
            "layers": list(layers) if layers is not None else None,
            "unit_tokenization": unit_tokenization,
        }
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        self._model = AutoModel.from_pretrained(self._checkpoint)
        self._model.to(self.device)
        self._model.eval()

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Return one FeatureModelOutput per unit with hidden state vectors.

        Args:
            stimulus: Must have the ``target_layer`` annotation layer.
            extract: Optional list of keys like ["hidden_state_0", "hidden_state_6"].
                If provided, only those layers are extracted.
            target_layer: Annotation layer to align outputs to (default "words").

        Returns:
            List of FeatureModelOutput, one per unit. Each output has
            values["hidden_state_{layer}"] = ndarray of shape (embed_dim,).
        """
        self._ensure_loaded()

        requested_layers = self._layers
        if extract is not None:
            parsed = []
            for key in extract:
                if key.startswith("hidden_state_"):
                    parsed.append(int(key.split("_")[-1]))
            if parsed:
                requested_layers = parsed

        layer = stimulus.layers[target_layer]
        unit_strings = layer.extract_strings(stimulus.text)
        n_units = len(unit_strings)

        if self._context_length is not None:
            return self._extract_sliding_window(
                stimulus,
                unit_strings,
                n_units,
                requested_layers,
                self._context_length,
                target_layer,
            )

        # Build the word/token alignment once and reuse it for both the
        # max-length check and the extraction helper (it re-tokenizes the
        # full text, so calling it per-helper would tokenize twice).
        alignment = align_from_tokenizer(
            stimulus,
            self._tokenizer,
            source_layer=target_layer,
        )

        # Check if full text exceeds the model's max sequence length.
        max_len = self._position_limit()
        if max_len is not None:
            token_layer = stimulus.layers[alignment.target_layer]
            n_tokens = len([v for _, v in token_layer])
            if n_tokens > max_len:
                return self._extract_chunked(
                    stimulus,
                    alignment,
                    n_units,
                    requested_layers,
                    max_len,
                )

        return self._extract_full(stimulus, alignment, n_units, requested_layers)

    def _position_limit(self) -> int:
        """Tokens one forward pass can hold; requires a known position limit.

        RoBERTa-style embeddings, which expose ``padding_idx``, number the
        positions from ``padding_idx + 1``, so they hold that many fewer.
        """
        limit = usable_position_limit(self.name, self._model.config)
        embeddings = getattr(self._model, "embeddings", None)
        padding_idx = getattr(embeddings, "padding_idx", None)
        if padding_idx is not None:
            limit -= padding_idx + 1
        return limit

    def _extract_full(
        self,
        stimulus: Stimulus,
        alignment: TokenAlignment,
        n_words: int,
        requested_layers: list[int] | None,
    ) -> list[FeatureModelOutput]:
        """Extract hidden states using the full stimulus text."""
        import torch

        token_layer = stimulus.layers[alignment.target_layer]
        token_ids = [v for _, v in token_layer]

        if not token_ids:
            return [
                FeatureModelOutput(metadata={"model_name": self.name})
                for _ in range(n_words)
            ]

        input_ids = torch.tensor([token_ids], device=self.device)

        with torch.no_grad():
            model_out = self._model(input_ids, output_hidden_states=True)

        hidden_states = model_out.hidden_states  # tuple of (1, seq_len, dim)

        layer_indices = _layer_indices(requested_layers, len(hidden_states) - 1)

        # Average subword token embeddings per word
        outputs: list[FeatureModelOutput] = []
        for w_idx in range(n_words):
            values: dict = {}
            token_indices = alignment.mapping[w_idx]

            for layer_idx in layer_indices:
                # hidden_states[0] is embedding, [1] is layer 0, etc.
                hs = hidden_states[layer_idx + 1]
                if token_indices:
                    vecs = hs[0, token_indices, :].cpu().numpy()
                    values[f"hidden_state_{layer_idx}"] = np.mean(vecs, axis=0)
                else:
                    dim = hs.shape[-1]
                    values[f"hidden_state_{layer_idx}"] = np.zeros(dim)

            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={"model_name": self.name},
                )
            )

        return outputs

    def _extract_chunked(
        self,
        stimulus: Stimulus,
        alignment: TokenAlignment,
        n_words: int,
        requested_layers: list[int] | None,
        max_len: int,
    ) -> list[FeatureModelOutput]:
        """Extract hidden states by processing text in overlapping chunks.

        For texts longer than the model's max sequence length, splits into
        chunks of max_len tokens with 50% overlap. For tokens in the overlap
        region, uses the representation from the chunk where the token is
        further from the boundary (i.e., has more context).
        """
        import torch

        token_layer = stimulus.layers[alignment.target_layer]
        all_token_ids = [v for _, v in token_layer]
        n_tokens = len(all_token_ids)

        if not all_token_ids:
            return [
                FeatureModelOutput(metadata={"model_name": self.name})
                for _ in range(n_words)
            ]

        with torch.no_grad():
            # Quick probe to get number of layers
            probe_ids = torch.tensor([[all_token_ids[0]]], device=self.device)
            probe_out = self._model(probe_ids, output_hidden_states=True)
            n_layers = len(probe_out.hidden_states) - 1

        layer_indices = _layer_indices(requested_layers, n_layers)

        embed_dim = probe_out.hidden_states[0].shape[-1]

        # Per-token hidden states: accumulate from chunks
        # For each token, store the hidden state from the chunk where
        # the token has the most context (furthest from chunk boundary)
        token_hidden = {
            layer_idx: np.zeros((n_tokens, embed_dim)) for layer_idx in layer_indices
        }
        token_dist_from_edge = np.full(n_tokens, -1, dtype=int)

        stride = max_len // 2  # 50% overlap
        chunk_starts = list(range(0, n_tokens, stride))

        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + max_len, n_tokens)
            chunk_ids = all_token_ids[chunk_start:chunk_end]
            chunk_len = len(chunk_ids)

            input_ids = torch.tensor([chunk_ids], device=self.device)
            with torch.no_grad():
                model_out = self._model(input_ids, output_hidden_states=True)

            # Extract all requested layers for this chunk
            chunk_hidden = {
                layer_idx: model_out.hidden_states[layer_idx + 1][0].cpu().numpy()
                for layer_idx in layer_indices
            }

            for pos_in_chunk in range(chunk_len):
                global_pos = chunk_start + pos_in_chunk
                # Distance from nearest chunk edge
                dist = min(pos_in_chunk, chunk_len - 1 - pos_in_chunk)
                if dist > token_dist_from_edge[global_pos]:
                    token_dist_from_edge[global_pos] = dist
                    for layer_idx in layer_indices:
                        token_hidden[layer_idx][global_pos] = chunk_hidden[layer_idx][
                            pos_in_chunk
                        ]

            if chunk_end >= n_tokens:
                break

        # Average subword token embeddings per word
        outputs: list[FeatureModelOutput] = []
        for w_idx in range(n_words):
            values: dict = {}
            token_indices = alignment.mapping[w_idx]

            for layer_idx in layer_indices:
                if token_indices:
                    vecs = token_hidden[layer_idx][token_indices]
                    values[f"hidden_state_{layer_idx}"] = np.mean(vecs, axis=0)
                else:
                    values[f"hidden_state_{layer_idx}"] = np.zeros(embed_dim)

            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={"model_name": self.name},
                )
            )

        return outputs

    def _extract_sliding_window(
        self,
        stimulus: Stimulus,
        words: list[str],
        n_words: int,
        requested_layers: list[int] | None,
        context_length: int | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Extract hidden states using a sliding context window.

        For each word, tokenizes a window of up to context_length words
        ending at that word and extracts the hidden state for the last word
        in the window.
        """
        import torch

        seq_len = context_length if context_length is not None else self._context_length
        if seq_len is None:
            raise ValueError("sliding-window extraction requires context_length")

        # process word-by-word with context windows
        outputs: list[FeatureModelOutput] = []

        for word_idx in range(n_words):
            if word_idx < seq_len:
                window_words = words[: word_idx + 1]
                target_in_window = word_idx
            else:
                start = word_idx - seq_len + 1
                window_words = words[start : word_idx + 1]
                target_in_window = seq_len - 1

            if self._unit_tokenization == "independent":
                token_groups = [self._tokenizer.tokenize(word) for word in window_words]
                tokens = [token for group in token_groups for token in group]
                token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
                target_start = sum(
                    len(group) for group in token_groups[:target_in_window]
                )
                target_token_indices = list(
                    range(
                        target_start,
                        target_start + len(token_groups[target_in_window]),
                    )
                )
            else:
                window_text = " ".join(window_words)
                encoding = self._tokenizer(
                    window_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                token_ids = encoding["input_ids"]
                offsets = encoding["offset_mapping"]

                # Find which tokens correspond to the target word.
                word_starts = []
                pos = 0
                for word in window_words:
                    word_starts.append(pos)
                    pos += len(word) + 1

                target_start = word_starts[target_in_window]
                target_end = target_start + len(window_words[target_in_window])
                target_token_indices = []
                for token_index, (token_start, token_end) in enumerate(offsets):
                    if token_start == token_end:
                        continue
                    midpoint = (token_start + token_end) / 2
                    if target_start <= midpoint < target_end:
                        target_token_indices.append(token_index)

            if not token_ids:
                outputs.append(FeatureModelOutput(metadata={"model_name": self.name}))
                continue

            if not target_token_indices:
                target_token_indices = [len(token_ids) - 1]

            input_ids = torch.tensor([token_ids], device=self.device)

            with torch.no_grad():
                model_out = self._model(input_ids, output_hidden_states=True)

            hidden_states = model_out.hidden_states

            layer_indices = _layer_indices(requested_layers, len(hidden_states) - 1)

            values: dict = {}
            for layer_idx in layer_indices:
                hs = hidden_states[layer_idx + 1]
                vecs = hs[0, target_token_indices, :].cpu().numpy()
                values[f"hidden_state_{layer_idx}"] = np.mean(vecs, axis=0)

            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={
                        "model_name": self.name,
                        "context_length": seq_len,
                        "unit_tokenization": self._unit_tokenization,
                    },
                )
            )

        return outputs


def _layer_indices(requested: list[int] | None, n_layers: int) -> list[int]:
    """The requested block indices, every block when none are requested.

    Raises:
        ValueError: When an index is outside ``0 .. n_layers - 1``; index
            ``-1`` would read the embedding output under a block's key.
    """
    if requested is None:
        return list(range(n_layers))
    outside = [i for i in requested if not 0 <= i < n_layers]
    if outside:
        raise ValueError(
            f"layers {outside} are outside 0..{n_layers - 1} for a model with "
            f"{n_layers} blocks"
        )
    return list(requested)
