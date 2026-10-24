"""A deterministic tokenizer and causal LM for the pooled MMC tests.

The LM's logits are a fixed function of the unpadded prefix, so a change in
batch size, padding width or chunking cannot move a number. That isolates the
pooled estimator's own arithmetic from float32 GPU noise.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

from replica.feature.frequency import pooled as P

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def build_vocab(marker: str) -> list[str]:
    """Byte-level or SentencePiece style vocabulary over lowercase text."""
    return (
        ["<bos>", marker, f"{marker}th"]
        + [f"{marker}{c}" for c in _LETTERS]
        + list(_LETTERS)
    )


class StubTokenizer:
    """Greedy longest-match tokenizer with a word-initial marker."""

    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self, marker: str = "Ġ"):
        self.marker = marker
        self.vocab = build_vocab(marker)
        self._ids = {t: i for i, t in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def tokenize(self, text):
        marked = text.replace(" ", self.marker)
        out, i = [], 0
        while i < len(marked):
            for length in range(min(8, len(marked) - i), 0, -1):
                piece = marked[i : i + length]
                if piece in self._ids and piece != "<bos>":
                    out.append(piece)
                    i += length
                    break
            else:
                raise ValueError(f"unknown piece at {marked[i:]!r}")
        return out

    def encode(self, text, add_special_tokens=False):
        return [self._ids[t] for t in self.tokenize(text)]

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]

    def decode(self, ids, skip_special_tokens=True):
        pieces = [self.vocab[i] for i in ids]
        if skip_special_tokens:
            pieces = [p for p in pieces if p != "<bos>"]
        return "".join(pieces).replace(self.marker, " ")


class _Output:
    def __init__(self, logits):
        self.logits = logits


class StubLM:
    """Causal LM whose logits depend only on the unpadded prefix."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(n_positions=2048)
        self._param = torch.zeros(1)

    def parameters(self):
        yield self._param

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        rows, seq = input_ids.shape
        logits = torch.zeros(rows, seq, self.vocab_size, dtype=torch.float32)
        for i in range(rows):
            start = 0 if attention_mask is None else int((attention_mask[i] == 0).sum())
            key = 17
            for offset, token in enumerate(input_ids[i, start:].tolist()):
                key = (key * 131 + int(token) * 7 + offset) % 100003
                logits[i, start + offset] = torch.tensor(
                    [
                        math.sin(key * 0.0013 + col * 0.37) * 2.0
                        for col in range(self.vocab_size)
                    ],
                    dtype=torch.float32,
                )
        return _Output(logits)


def stub_pair(marker: str = "Ġ"):
    """A matched tokenizer and model."""
    tok = StubTokenizer(marker)
    return tok, StubLM(len(tok.vocab))


# --- an enumerable pair ----------------------------------------------------

TINY_VOCAB = ["<bos>", "Ġa", "Ġb", "c", "<eos>"]
#: Logits per last token, so the whole string distribution is enumerable.
TINY_ROWS = [
    [-30.0, 0.40, 0.20, 0.10, -1.20],
    [-30.0, 0.10, 0.30, 0.80, -0.90],
    [-30.0, 0.50, -0.20, 0.60, -0.70],
    [-30.0, 0.30, 0.25, -0.10, -1.00],
    [-30.0, 0.00, 0.00, 0.00, 0.00],
]


class TinyTokenizer:
    """Five tokens, so every string the model can emit can be enumerated."""

    bos_token_id = 0
    eos_token_id = 4
    pad_token_id = 0
    name_or_path = "tiny"
    all_special_ids = (0, 4)

    def __len__(self):
        return len(TINY_VOCAB)

    def tokenize(self, text):
        out, i, marked = [], 0, text.replace(" ", "Ġ")
        while i < len(marked):
            for piece in ("Ġa", "Ġb", "c"):
                if marked.startswith(piece, i):
                    out.append(piece)
                    i += len(piece)
                    break
            else:
                raise ValueError(f"unknown piece at {marked[i:]!r}")
        return out

    def encode(self, text, add_special_tokens=False):
        return [TINY_VOCAB.index(p) for p in self.tokenize(text)]

    def convert_ids_to_tokens(self, ids):
        return [TINY_VOCAB[i] for i in ids]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        pieces = [TINY_VOCAB[i] for i in ids]
        if skip_special_tokens:
            pieces = [p for p in pieces if p not in ("<bos>", "<eos>")]
        return "".join(pieces).replace("Ġ", " ")

    def batch_decode(self, batches, **kwargs):
        return [self.decode(b, **kwargs) for b in batches]


class TinyLM:
    """Logits that depend only on the last token."""

    vocab_size = len(TINY_VOCAB)
    config = SimpleNamespace(n_positions=2048)

    def parameters(self):
        yield torch.zeros(1)

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        # float64, so a gate can be tightened past float32's 1e-7.
        table = torch.tensor(TINY_ROWS, dtype=torch.float64)
        return _Output(table[input_ids.long()])


def tiny_pair():
    """A matched enumerable tokenizer and model."""
    return TinyTokenizer(), TinyLM()


# --- an enumerable pair with two ways to spell one unit --------------------

AMBIG_VOCAB = ["<bos>", "Ġa", "Ġab", "b", "c", "<eos>"]
#: ``' ab'`` is either ``[Ġab]`` or ``[Ġa, b]``, so residual mass is nonzero.
AMBIG_ROWS = [
    [-30.0, 0.55, 0.20, 0.05, 0.30, -1.10],
    [-30.0, 0.10, 0.35, 0.70, 0.40, -0.85],
    [-30.0, 0.45, -0.15, 0.30, 0.55, -0.70],
    [-30.0, 0.25, 0.30, -0.20, 0.35, -0.95],
    [-30.0, 0.35, 0.15, 0.25, -0.10, -0.80],
    [-30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]


class AmbigTokenizer:
    """Six tokens, greedy longest match, one unit with two tokenizations."""

    bos_token_id = 0
    eos_token_id = 5
    pad_token_id = 0
    name_or_path = "ambig"
    all_special_ids = (0, 5)

    def __len__(self):
        return len(AMBIG_VOCAB)

    def tokenize(self, text):
        out, i, marked = [], 0, text.replace(" ", "Ġ")
        while i < len(marked):
            for piece in ("Ġab", "Ġa", "b", "c"):
                if marked.startswith(piece, i):
                    out.append(piece)
                    i += len(piece)
                    break
            else:
                raise ValueError(f"unknown piece at {marked[i:]!r}")
        return out

    def encode(self, text, add_special_tokens=False):
        return [AMBIG_VOCAB.index(p) for p in self.tokenize(text)]

    def convert_ids_to_tokens(self, ids):
        return [AMBIG_VOCAB[i] for i in ids]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        pieces = [AMBIG_VOCAB[i] for i in ids]
        if skip_special_tokens:
            pieces = [p for p in pieces if p not in ("<bos>", "<eos>")]
        return "".join(pieces).replace("Ġ", " ")

    def batch_decode(self, batches, **kwargs):
        return [self.decode(b, **kwargs) for b in batches]


class AmbigLM:
    """Logits that depend only on the last token."""

    vocab_size = len(AMBIG_VOCAB)

    def parameters(self):
        yield torch.zeros(1)

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        table = torch.tensor(AMBIG_ROWS, dtype=torch.float64)
        return _Output(table[input_ids.long()])


def ambig_pair():
    """A matched enumerable pair whose vocabulary spells one unit two ways."""
    return AmbigTokenizer(), AmbigLM()


_SAMPLES = [
    "the dog ran across the field",
    "a cat sat on the mat",
    "the the the dog dog",
    "hello world this is a test",
]
_MORE_SAMPLES = [
    "a dog and a cat",
    "the field is a test",
    "this is the world",
    "the cat ran across the mat",
]
_STUB_WORDS = [" the", " dog", " cat", " world", " a", " zz"]


def stub_selection(
    shards, *, stratification, k_store=16, rng_seed=7, replicates=2, **kwargs
):
    """A pooled selection over *shards*, built with the stub model."""
    tok, model = stub_pair()
    token_ids = [tok.encode(w) for w in _STUB_WORDS]
    first_ids = np.array(sorted({t[0] for t in token_ids}), dtype=np.int64)
    mt_ids = first_ids
    store = np.full(
        (mt_ids.size, len(stratification.labels), replicates), k_store, dtype=np.int64
    )
    selection = P.build_pool_selection(
        shards,
        model,
        tok,
        first_ids=first_ids,
        mt_first_ids=mt_ids,
        k_store=store,
        rng_seed=rng_seed,
        stratification=stratification,
        **kwargs,
    )
    return selection, tok, model


def _counts():
    """Pooled occurrence counts over the stub pool."""
    pool = _SAMPLES + _MORE_SAMPLES
    return P.unit_counts(_STUB_WORDS, pool)


def _take_store(shards=None):
    """A take store over the same shards the selection swept."""
    tok, _ = stub_pair()
    first_ids = sorted({tok.encode(w)[0] for w in _STUB_WORDS})
    return P.take_store_from_shards(
        shards if shards is not None else [_SAMPLES, _MORE_SAMPLES], tok, first_ids
    )
