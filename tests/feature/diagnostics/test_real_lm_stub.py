"""Reference formulas evaluated against a local model double."""

import re
from types import SimpleNamespace

import numpy as np
import torch

from replica.feature.diagnostics import make_stimulus
from replica.feature.diagnostics.references import causal_lm as real_lm


class _WordTokenizer:
    """One token per word, the leading space inside the token."""

    bos_token_id = 0

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r" ?\S+", text))
        return {
            "input_ids": [1 + i % 5 for i in range(len(matches))],
            "offset_mapping": [m.span() for m in matches],
        }


class _SkewedModel:
    def __call__(self, input_ids):
        positions = torch.arange(input_ids.shape[-1], dtype=torch.float32)
        vocabulary = torch.arange(6, dtype=torch.float32)
        logits = -(vocabulary - positions.unsqueeze(-1)).square() / 3
        return SimpleNamespace(logits=logits.unsqueeze(0))


def test_renyi_order_one_is_the_shannon_entropy(monkeypatch):
    monkeypatch.setattr(
        real_lm, "_load", lambda model_id, device: (_WordTokenizer(), _SkewedModel())
    )
    out = real_lm.exact_token_quantities(
        "stub", make_stimulus(["the", "cat", "sat"]), renyi_alphas=(1.0, 0.5)
    )
    assert np.isfinite(out["token_entropy"]).all()
    np.testing.assert_array_equal(
        out["token_renyi_entropy_alpha1.0"], out["token_entropy"]
    )
    assert (out["token_renyi_entropy_alpha0.5"] > out["token_entropy"]).all()
