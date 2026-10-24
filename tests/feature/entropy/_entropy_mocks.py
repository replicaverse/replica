"""Reusable lightweight model and tokenizer doubles for entropy tests."""

import torch


def build_estimator(
    cls,
    model,
    *,
    name: str,
    n_samples: int,
    tokenizer=None,
    model_id: str = "mock",
    model_identity: dict | None = None,
    add_bos: bool = False,
    seed: int = 0,
    chunk_size: int | None = None,
    **attributes,
):
    """Build a Monte Carlo entropy estimator on a model double.

    Sets the state the estimator's constructor sets, without loading a
    checkpoint. ``attributes`` holds the class's own settings (``max_tokens``,
    or ``max_iter`` and ``alpha``), ``tokenizer`` stays unset when None, and a
    word-level estimator also gets the default word-starter characters and the
    token masks computed from them.
    """
    lm = cls.__new__(cls)
    lm.model_id = model_id
    lm._model_identity = (
        {"sha256": "a" * 64} if model_identity is None else model_identity
    )
    lm.add_bos = add_bos
    lm._seed = seed
    lm.n_samples = n_samples
    lm.chunk_size = chunk_size
    lm.name = name
    for key, value in attributes.items():
        setattr(lm, key, value)
    lm._model = model
    if tokenizer is not None:
        lm._tokenizer = tokenizer
    lm._device = torch.device("cpu")
    lm._rng = torch.Generator(device=lm._device)
    lm._rng.manual_seed(seed)
    if hasattr(lm, "_compute_token_masks"):
        lm._word_starter_chars = ("Ġ", "Ċ", "ĉ")
        lm._space_ids, lm._subword_ids = lm._compute_token_masks()
    return lm


class UniformCausalLM(torch.nn.Module):
    """Causal-LM double returning uniform logits and an empty cache."""

    def __init__(self, vocab_size: int = 4, n_positions: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = type(
            "Cfg",
            (),
            {
                "max_position_embeddings": n_positions,
                "n_positions": n_positions,
                "n_ctx": n_positions,
            },
        )()

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, length = input_ids.shape
        logits = torch.zeros(
            batch_size,
            length,
            self.vocab_size,
            device=input_ids.device,
        )

        class _StubCache(tuple):
            def to_legacy_cache(self):
                return tuple(self)

        return type(
            "Out",
            (),
            {"logits": logits, "past_key_values": _StubCache()},
        )()

    def eval(self):
        return self


class WhitespaceTokenizer:
    """Four-token tokenizer mapping each whitespace word to one token."""

    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0
    vocab_size = 4
    name_or_path = "mock"
    _id_to_tok = {0: "<bos>", 1: "Ġa", 2: "b", 3: "Ġc"}
    _tok_to_id = {token: token_id for token_id, token in _id_to_tok.items()}

    def get_vocab(self):
        return dict(self._tok_to_id)

    @property
    def vocab(self):
        return dict(self._tok_to_id)

    def convert_ids_to_tokens(self, token_id):
        if isinstance(token_id, (list, tuple)):
            return [self._id_to_tok[int(item)] for item in token_id]
        return self._id_to_tok[int(token_id)]

    def __call__(self, text, **_):
        ids = []
        offsets = []
        cursor = 0
        for word in text.split():
            start = text.index(word, cursor)
            ids.append(1)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }
