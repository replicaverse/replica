"""Hidden-state feature for continuous-response encoding models.

Extracts a specific layer's hidden state vector from FeatureModelOutput.
Returns a matrix (n_words, embed_dim). Bind it with ``ModelFeature`` and use
``evaluate()`` with ``EncodingConfig`` for sampled-response encoding.
"""

from collections.abc import Sequence

import numpy as np

from replica.feature.core.protocols import ModelOutputRecord

from ...core import Stimulus


class HiddenState:
    """Extract hidden state vector for a specific layer.

    Returns shape (n_words, embed_dim), one vector per word.
    Use ``ModelFeature(model, HiddenState(layer))`` as a predictor with
    ``EncodingConfig``; scalar analyses require scalar features.

    Args:
        layer: 0-indexed transformer block whose output is read; the feature
            reads ``hidden_state_{layer}`` off each record and takes its name
            from the same string.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"key": self._key}

    value_dims = ("embedding",)

    @property
    def encoding_layer(self) -> int:
        """Transformer layer represented by the value axis."""
        return self.layer

    name: str

    def __init__(self, layer: int):
        """Bind the feature to one transformer block's hidden state."""
        self.layer = layer
        self.name = f"hidden_state_{layer}"
        self._key = f"hidden_state_{layer}"

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return (self._key,)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Extract hidden states, shape (n_units, embed_dim).

        Args:
            stimulus: Text the records came from; the vectors are read off
                ``outputs`` alone.
            outputs: Records from a model that writes ``hidden_state_{layer}``,
                such as ``HuggingFaceEmbedding``.
            target_layer: Annotation layer the records are aligned to.

        Returns:
            One row per unit, holding that unit's layer vector.
        """
        if outputs is None:
            raise ValueError("HiddenState requires model outputs")

        vecs = []
        for out in outputs:
            val = out.values.get(self._key)
            if val is None:
                raise ValueError(
                    f"Output missing '{self._key}'. "
                    f"Ensure the feature model produces hidden state outputs."
                )
            vecs.append(val)

        return np.array(vecs)
