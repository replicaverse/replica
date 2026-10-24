"""Deterministic sampling doubles with distinct BOS, EOS and PAD tokens."""

from types import SimpleNamespace

import torch


class _Tokenizer:
    """A stub whose decode is the identity over small ints."""

    name_or_path = "stub"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)

    def encode(self, text, add_special_tokens=True):
        return [int(w[1:]) for w in text.split()]


class _Model:
    """A stub whose generate returns fixed rows: prefix, tokens, end, padding."""

    config = SimpleNamespace(n_positions=2048)

    def generate(self, input_ids, **kwargs):
        assert kwargs["pad_token_id"] == 0
        prefix = input_ids[0].tolist()
        rows = [
            prefix + [5, 6, 2, 0, 0],
            prefix + [7, 2, 0, 0, 0],
            prefix + [8, 9, 10, 11, 12],
        ]
        return torch.tensor(rows[: kwargs["num_return_sequences"]])
