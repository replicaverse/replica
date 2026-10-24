"""Torch reference models and tokenizers for testing production MC samplers.

Markov chains supply known distributions; the token chain includes multi-token
words for boundary and alignment checks. Import this module explicitly when
Torch is available.
"""

import re
from types import SimpleNamespace

import numpy as np
import torch

from replica.feature.core.annotation.alignment import align_from_tokenizer
from replica.feature.diagnostics.references._closed_forms import (
    continuation_entropy,
    h_bits,
    mc_word_outcome_distribution,
    renyi_bits,
)
from replica.feature.diagnostics.references.markov import make_stimulus

from ....core import Stimulus

__all__ = [
    "MarkovTokenizer",
    "MarkovTorchLM",
    "TOKEN_TRANSITION",
    "TOKEN_VOCAB",
    "TokenChainReference",
]


class _StubCache(tuple):
    """Quacks like both a legacy KV-cache tuple and a DynamicCache.

    Always empty: valid only for models whose prediction ignores the past
    (a backend that derived sequence length from cache length would
    miscompute against it).
    """

    def to_legacy_cache(self):
        return tuple(self)


class MarkovTorchLM(torch.nn.Module):
    """Causal-LM stand-in whose next-token distribution is a Markov chain.

    ``forward(input_ids)`` returns logits ``log T[input_ids]`` at every
    position, so the predictive distribution depends only on the most
    recent token, a KV cache is unnecessary and the stub cache from
    REPLICA's mock-LM tests is returned instead.
    """

    def __init__(self, transition: np.ndarray):
        """Store the chain as log-transition logits and a transformers-style config.

        Args:
            transition: row-stochastic ``(vocab, vocab)`` matrix; row ``i`` is
                the next-token distribution after token ``i``.
        """
        super().__init__()
        transition = np.asarray(transition, dtype=np.float64)
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("transition rows must each sum to 1")
        self.vocab_size = transition.shape[0]
        # transformers-style config; REPLICA's backends read positional
        # budgets from it (2048 = same value their no-config fallback uses).
        # vocab_size is read by ForwardMarginalEntropyLM's window accumulator.
        self.config = SimpleNamespace(n_positions=2048, vocab_size=transition.shape[0])
        log_t = torch.from_numpy(np.log(transition)).to(torch.float64)
        self.register_buffer("_log_transition", log_t)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        """Logits at every position, plus an always-empty stub cache.

        Args:
            input_ids: token ids, ``(seq,)`` or ``(batch, seq)``.
            past_key_values: ignored, since prediction reads only the current
                token.
            use_cache: ignored, for the same reason.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        logits = self._log_transition[input_ids]  # (batch, seq, vocab)
        return type("Out", (), {"logits": logits, "past_key_values": _StubCache()})()


# ── Token-level chain: multi-token words, Ġ word-starters ────────────────

TOKEN_VOCAB = ["<bos>", "Ġba", "Ġda", "Ġga", "Ġka", "x", "y"]

# Rows over (bos, Ġba, Ġda, Ġga, Ġka, x, y). Subword rows keep enough
# word-starter mass that sampled words terminate well within max_iter.
TOKEN_TRANSITION = np.array(
    [
        [0.01, 0.30, 0.25, 0.20, 0.14, 0.06, 0.04],  # <bos>
        [0.02, 0.10, 0.10, 0.08, 0.10, 0.40, 0.20],  # Ġba → often continues
        [0.02, 0.20, 0.18, 0.15, 0.15, 0.15, 0.15],  # Ġda
        [0.01, 0.04, 0.05, 0.05, 0.05, 0.10, 0.70],  # Ġga → heavy 'y' tail
        [0.02, 0.30, 0.28, 0.20, 0.10, 0.05, 0.05],  # Ġka → mostly ends
        [0.02, 0.18, 0.15, 0.15, 0.10, 0.25, 0.15],  # x
        [0.02, 0.40, 0.20, 0.18, 0.10, 0.05, 0.05],  # y
    ],
    dtype=np.float64,
)


class MarkovTokenizer:
    """GPT-2-shaped tokenizer over a fixed mixed starter/subword vocab.

    Greedy longest-match: each whitespace-separated word starts with one
    ``Ġ``-token (matched against the word's leading characters) followed by
    plain subword tokens. Implements the interfaces REPLICA touches:
    ``__call__`` with ``return_offsets_mapping`` (for
    ``align_from_tokenizer``), ``get_vocab`` / ``convert_ids_to_tokens``
    (for ``MCWordEntropyLM``'s starter/subword masks), and the special-id
    attributes.
    """

    def __init__(self, vocab: list[str] | None = None):
        """Index the vocabulary and split it into word-starters and subwords.

        Args:
            vocab: token strings, position ``i`` being token id ``i``; the
                ``Ġ``-prefixed ones are the word-starters. ``None`` takes
                :data:`TOKEN_VOCAB`.
        """
        self._id_to_tok = dict(enumerate(TOKEN_VOCAB if vocab is None else vocab))
        self.vocab_size = len(self._id_to_tok)
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.name_or_path = "markov_token_chain"
        self._starters = [
            (i, t[1:]) for i, t in self._id_to_tok.items() if t.startswith("Ġ")
        ]
        self._subwords = [
            (i, t)
            for i, t in self._id_to_tok.items()
            if not t.startswith("Ġ") and i != self.bos_token_id
        ]

    def get_vocab(self) -> dict[str, int]:
        """Token string to token id, the mapping ``get_vocab`` callers expect."""
        return {t: i for i, t in self._id_to_tok.items()}

    def convert_ids_to_tokens(self, i):
        """Token string for one id, or a list of them for a sequence of ids.

        Args:
            i: a single token id, or a list or tuple of them.
        """
        if isinstance(i, (list, tuple)):
            return [self._id_to_tok[int(x)] for x in i]
        return self._id_to_tok[int(i)]

    def _tokenize_word(self, word: str, offset: int):
        out = []
        first = max(
            (s for s in self._starters if word.startswith(s[1])),
            key=lambda s: len(s[1]),
            default=None,
        )
        if first is None:
            raise ValueError(f"no word-starter token matches {word!r}")
        out.append((first[0], offset, offset + len(first[1])))
        pos = len(first[1])
        while pos < len(word):
            nxt = max(
                (s for s in self._subwords if word.startswith(s[1], pos)),
                key=lambda s: len(s[1]),
                default=None,
            )
            if nxt is None:
                raise ValueError(f"cannot tokenize {word!r} at position {pos}")
            out.append((nxt[0], offset + pos, offset + pos + len(nxt[1])))
            pos += len(nxt[1])
        return out

    def __call__(self, text: str, **_):
        """Tokenize greedily, returning ``input_ids`` and an offset mapping.

        Args:
            text: the string to tokenize; whitespace-separated runs are words,
                and the offsets index into ``text``.
        """
        ids, offsets = [], []
        for m in re.finditer(r"\S+", text):
            for tok_id, start, end in self._tokenize_word(m.group(), m.start()):
                ids.append(tok_id)
                offsets.append((start, end))
        return {"input_ids": ids, "offset_mapping": offsets}


class TokenChainReference:
    """Exactly-solvable token-level reference for the MC backends.

    Bundles the token chain, its torch module + tokenizer, and the exact
    target values of both production samplers at each word position of a
    stimulus.
    """

    def __init__(self, transition: np.ndarray | None = None):
        """Build the chain, its torch module and tokenizer, and the token masks.

        Args:
            transition: row-stochastic ``(vocab, vocab)`` token-level matrix
                indexed by :data:`TOKEN_VOCAB`. ``None`` takes
                :data:`TOKEN_TRANSITION`.
        """
        self.transition = np.asarray(
            TOKEN_TRANSITION if transition is None else transition,
            dtype=np.float64,
        )
        self.tokenizer = MarkovTokenizer()
        self.model = MarkovTorchLM(self.transition)
        vocab = self.tokenizer.get_vocab()
        # Same partition rule as MCWordEntropyLM._compute_token_masks:
        # Ġ/Ċ/ĉ-prefixed = word starter, everything else (incl. BOS) subword.
        self.starter_ids = np.array(
            sorted(i for t, i in vocab.items() if t[0] in ("Ġ", "Ċ", "ĉ"))
        )
        self.subword_ids = np.array(
            sorted(i for t, i in vocab.items() if t[0] not in ("Ġ", "Ċ", "ĉ"))
        )

    def make_stimulus(
        self, words: list[str], stimulus_id: str = "token_probe"
    ) -> Stimulus:
        """Build a probe stimulus whose words this chain's tokenizer can split.

        Args:
            words: the words in reading order.
            stimulus_id: identifier carried on the returned stimulus.
        """
        return make_stimulus(words, stimulus_id=stimulus_id)

    def word_token_ids(self, stimulus: Stimulus) -> list[list[int]]:
        """Token ids of each word, under the production aligner.

        Going through ``align_from_tokenizer`` makes the exact targets
        condition on the same word-to-token assignment the samplers see.

        Args:
            stimulus: probe to align and tokenize.
        """
        alignment = align_from_tokenizer(stimulus, self.tokenizer, source_layer="words")
        token_ids = [v for _, v in stimulus.layers[alignment.target_layer]]
        out = []
        for i, positions in enumerate(alignment.mapping):
            if not positions:
                raise ValueError(f"word {i} was assigned no tokens")
            out.append([token_ids[p] for p in positions])
        return out

    def _prefix_states(self, stimulus: Stimulus) -> np.ndarray:
        """Per word position, the token id of the last prefix token (the
        chain state the samplers condition on); -1 at position 0.
        """
        words = self.word_token_ids(stimulus)
        states = np.full(len(words), -1, dtype=np.intp)
        states[1:] = [w[-1] for w in words[:-1]]
        return states

    def exact_mc_word_entropy(
        self, stimulus: Stimulus, max_iter: int, alpha: float = 1.0
    ) -> np.ndarray:
        """Exact limit of ``MCWordEntropyLM`` per word position (NaN at 0).

        Args:
            stimulus: probe whose word positions are scored.
            max_iter: token budget the sampler is allowed per word, which caps
                the outcome distribution the entropy is taken over.
            alpha: entropy order, ``1.0`` for Shannon and anything else for
                Rényi.
        """
        out = np.full(len(stimulus.layers["words"]), np.nan)
        for t, state in enumerate(self._prefix_states(stimulus)):
            if state < 0:
                continue
            q = mc_word_outcome_distribution(
                self.transition,
                self.starter_ids,
                self.subword_ids,
                int(state),
                max_iter,
            )
            out[t] = h_bits(q) if alpha == 1.0 else renyi_bits(q, alpha)
        return out

    def exact_continuation_entropy(
        self, stimulus: Stimulus, max_tokens: int
    ) -> np.ndarray:
        """Exact limit of ``ContinuationEntropyLM`` per word position.

        Args:
            stimulus: probe whose word positions are scored.
            max_tokens: continuation length in tokens.
        """
        out = np.full(len(stimulus.layers["words"]), np.nan)
        for t, state in enumerate(self._prefix_states(stimulus)):
            if state < 0:
                continue
            d = self.transition[int(state)][None]
            out[t] = continuation_entropy(self.transition, d, max_tokens)[0]
        return out

    def exact_forward_marginal(
        self, stimulus: Stimulus, max_tokens: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact limit of ``ForwardMarginalEntropyLM`` per word position.

        Full-vocab ancestral sampling on this chain makes all three profiles
        closed-form matrix algebra of the transition ``T`` from the prefix
        state ``s``.

        Args:
            stimulus: probe whose word positions are scored.
            max_tokens: deepest continuation depth L, one column per depth.

        Returns:
            ``(marginal, free, joint)``, each ``(n_words, max_tokens)`` with
            depth k=1..L at column ``k-1`` and NaN at position 0. With
            ``mu_0 = onehot(s)``, ``mu_k = mu_{k-1} @ T`` and
            ``h_rows = h_bits(T)``:

            * marginal[k-1] = H(c_{i+k}|ctx_i) = ``h_bits(mu_k)``
            * free[k-1]     = E_path[H(c_{i+k}|c_<k)] = ``mu_{k-1} @ h_rows``
            * joint[k-1]    = H(c_{i+1..i+k}|ctx_i) = continuation entropy at depth k
        """
        T = self.transition
        h_rows = h_bits(T)
        n = len(stimulus.layers["words"])
        L = int(max_tokens)
        marg = np.full((n, L), np.nan)
        free = np.full((n, L), np.nan)
        joint = np.full((n, L), np.nan)
        for t, state in enumerate(self._prefix_states(stimulus)):
            if state < 0:
                continue
            s = int(state)
            mu_prev = np.zeros(T.shape[0])
            mu_prev[s] = 1.0  # mu_0 = point mass at the prefix state
            for k in range(L):
                free[t, k] = float(mu_prev @ h_rows)
                mu_prev = mu_prev @ T  # mu_{k+1}
                marg[t, k] = float(h_bits(mu_prev))
            # joint via the continuation-entropy chain rule, exposed per depth
            d0 = T[s]
            Hj = float(h_bits(d0))
            mj = d0.copy()
            joint[t, 0] = Hj
            for k in range(1, L):
                Hj = Hj + float(mj @ h_rows)
                mj = mj @ T
                joint[t, k] = Hj
        return marg, free, joint
