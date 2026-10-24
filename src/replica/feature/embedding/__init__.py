"""Embedding / hidden-state backends.

These produce per-unit dense vectors (hidden states or pooled
embeddings) for use as predictors in continuous-response encoding analyses.

- :class:`HuggingFaceEmbedding`, per-word, per-layer hidden states
  from a HuggingFace model (averaging subword tokens per word).
- :class:`HiddenState`, extracts one layer's hidden-state matrix from
  the model outputs.
"""

from .hidden_state import HiddenState
from .huggingface import HuggingFaceEmbedding

__all__ = ["HiddenState", "HuggingFaceEmbedding"]
