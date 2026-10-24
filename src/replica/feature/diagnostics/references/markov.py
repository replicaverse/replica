"""Markov reference models with exactly computable predictive quantities.

The default transition matrix separates candidate targets on the probe.
Custom matrices and probes can be checked with assert_identifiable().
"""

import numpy as np

from replica.feature.diagnostics.references._closed_forms import (
    continuation_entropy,
    h_bits,
    renyi_bits,
)

from ....core import AnnotationLayer, Span, Stimulus
from ...core.protocols import FeatureModelOutput

__all__ = [
    "DEFAULT_INITIAL",
    "DEFAULT_TRANSITION",
    "DEFAULT_VOCAB",
    "MarkovReferenceLM",
    "make_stimulus",
]


DEFAULT_VOCAB = ["ba", "da", "ga", "ka", "ma", "na", "pa", "ta"]

# One row shape per state, each chosen to stress a different regime of the
# entropy functionals (full-vs-truncated, Shannon-vs-Rényi, peakedness).
# No row may tie at the top-k boundary (4th vs 5th largest value): the
# truncated-sampler spec propagates the kept indices through the chain, so
# a boundary tie would make its exact value depend on argsort order
# (guarded by tests/feature/diagnostics/test_specs.py).
_ROW_SHAPES = [
    [0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12],  # near-uniform
    [0.82, 0.06, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01],  # peaked
    [0.48, 0.48, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005],  # two-mass
    [0.40, 0.20, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],  # heavy tail
    [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0078125],
    [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],  # moderate tail
    [0.35, 0.06, 0.35, 0.06, 0.05, 0.05, 0.04, 0.04],  # bimodal
    [0.955, 0.015, 0.01, 0.006, 0.004, 0.004, 0.003, 0.003],  # near-det.
]

# Roll row i by i so probability mass is not column-aligned and the chain
# mixes over all states.
DEFAULT_TRANSITION = np.array(
    [np.roll(row, i) for i, row in enumerate(_ROW_SHAPES)], dtype=np.float64
)

DEFAULT_INITIAL = np.array(
    [0.18, 0.16, 0.14, 0.12, 0.10, 0.10, 0.10, 0.10], dtype=np.float64
)


def make_stimulus(words: list[str], stimulus_id: str = "probe") -> Stimulus:
    """Build a single-layer Stimulus from a list of word strings.

    Args:
        words: the words in reading order; they are joined by single spaces and
            annotated as one ``words`` layer.
        stimulus_id: identifier carried on the returned stimulus.
    """
    text = " ".join(words)
    spans, off = [], 0
    for w in words:
        spans.append(Span(off, off + len(w)))
        off += len(w) + 1
    stim = Stimulus(text=text, stimulus_id=stimulus_id)
    stim.add_layer(AnnotationLayer(name="words", spans=spans, values=list(words)))
    return stim


def _de_bruijn(k: int, n: int) -> list[int]:
    """De Bruijn sequence B(k, n) as a list of symbol indices (cyclic)."""
    a = [0] * (k * n)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


class MarkovReferenceLM:
    """First-order Markov chain over a word vocabulary, as a FeatureModel.

    The distribution predicting unit ``t`` is ``initial`` at ``t = 0`` and
    ``transition[words[t-1]]`` otherwise (the analogue of ``add_bos=True``).
    Outputs follow REPLICA conventions: ``log_p_observed`` in nats,
    ``next_log_probs`` as a full word-type log-distribution,
    ``metadata["vocab_index"]`` for rank extractors.

    Every word is a single token, so token- and word-level distributions
    coincide; outputs also carry the reader-extractor keys with their exact
    target values (``token_entropy``, ``token_renyi_entropy_alpha{a}``,
    ``continuation_entropy`` at ``continuation_length``, and
    ``mc_word_entropy`` = next-word entropy).
    """

    def __init__(
        self,
        transition: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        vocab: list[str] | None = None,
        name: str = "markov_reference",
        continuation_length: int = 2,
        token_renyi_alphas: tuple[float, ...] = (0.5, 2.0),
    ):
        """Build the chain and check that its parameters are valid distributions.

        Args:
            transition: row-stochastic ``(vocab, vocab)`` matrix, row ``i``
                being the distribution after word ``i``.
            initial: distribution predicting the first word, one entry per
                word type.
            vocab: word types, in the order the matrix rows index them.
            name: model name reported in each output's metadata.
            continuation_length: number of following words the
                ``continuation_entropy`` output covers.
            token_renyi_alphas: orders to emit a
                ``token_renyi_entropy_alpha{a}`` output for.
        """
        self.transition = np.asarray(
            DEFAULT_TRANSITION if transition is None else transition,
            dtype=np.float64,
        )
        self.initial = np.asarray(
            DEFAULT_INITIAL if initial is None else initial, dtype=np.float64
        )
        self.vocab = list(DEFAULT_VOCAB if vocab is None else vocab)
        self.name = name
        self.continuation_length = int(continuation_length)
        self.token_renyi_alphas = tuple(float(a) for a in token_renyi_alphas)

        v = len(self.vocab)
        if self.transition.shape != (v, v):
            raise ValueError(
                f"transition must be ({v}, {v}) for a {v}-word vocab, "
                f"got {self.transition.shape}"
            )
        if self.initial.shape != (v,):
            raise ValueError(f"initial must be ({v},), got {self.initial.shape}")
        if not np.allclose(self.transition.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("transition rows must each sum to 1")
        if not np.isclose(self.initial.sum(), 1.0, atol=1e-9):
            raise ValueError("initial must sum to 1")
        self._index = {w: i for i, w in enumerate(self.vocab)}

    # ── FeatureModel protocol ────────────────────────────────────────────

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Exact per-unit outputs for one stimulus, in the FeatureModel format.

        Args:
            stimulus: probe to score; every word must be in the vocabulary.
            extract: ignored, since every value is closed-form and cheap, so
                the full set is always returned.
            target_layer: annotation layer defining the unit granularity.
        """
        # ``extract`` is ignored: every value is closed-form and cheap.
        dists, observed = self.predictive_distributions(stimulus, target_layer)
        word_h = h_bits(dists)
        cont_h = continuation_entropy(self.transition, dists, self.continuation_length)
        outputs = []
        for t in range(len(observed)):
            log_dist = np.log(dists[t])
            values = {
                "log_p_observed": float(log_dist[observed[t]]),
                "next_log_probs": log_dist,
                # Single-token words: token-level == word-level quantities.
                "token_entropy": float(word_h[t]),
                "mc_word_entropy": float(word_h[t]),
                "continuation_entropy": float(cont_h[t]),
            }
            for a in self.token_renyi_alphas:
                values[f"token_renyi_entropy_alpha{a}"] = float(renyi_bits(dists[t], a))
            outputs.append(
                FeatureModelOutput(
                    values=values,
                    metadata={
                        "model_name": self.name,
                        "vocab_index": self._index,
                    },
                )
            )
        return outputs

    # ── Exact ground truth ───────────────────────────────────────────────

    def state_indices(
        self, stimulus: Stimulus, target_layer: str = "words"
    ) -> np.ndarray:
        """Vocabulary index of each unit, raising KeyError on an unknown word.

        Args:
            stimulus: probe whose units are looked up.
            target_layer: annotation layer defining the unit granularity.
        """
        words = stimulus.unit_strings(target_layer)
        try:
            return np.array([self._index[w] for w in words], dtype=np.intp)
        except KeyError as e:
            raise KeyError(
                f"word {e.args[0]!r} not in reference vocab {self.vocab}"
            ) from None

    def predictive_distributions(
        self, stimulus: Stimulus, target_layer: str = "words"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact predictive distribution at every position.

        Args:
            stimulus: probe to score; every word must be in the vocabulary.
            target_layer: annotation layer defining the unit granularity.

        Returns:
            ``(dists, observed)`` where ``dists[t]`` is the distribution
            predicting unit ``t`` (shape ``(n_units, vocab)``) and
            ``observed[t]`` is the vocab index of the word actually at ``t``.
        """
        observed = self.state_indices(stimulus, target_layer)
        dists = np.empty((len(observed), len(self.vocab)), dtype=np.float64)
        if len(observed):
            dists[0] = self.initial
            dists[1:] = self.transition[observed[:-1]]
        return dists, observed

    # ── Probe stimuli ────────────────────────────────────────────────────

    def probe_stimulus(self) -> Stimulus:
        """Deterministic probe covering every bigram (de Bruijn order 2).

        Every state occurs as left context for every possible next word, so
        the probe exercises all distinct predictive distributions and all
        (context, observed) surprisal values.
        """
        idx = _de_bruijn(len(self.vocab), 2)
        idx = idx + idx[:1]  # linearize the cyclic sequence
        return make_stimulus([self.vocab[i] for i in idx], stimulus_id="de_bruijn_2")

    def sample_stimuli(self, n: int, length: int, seed: int = 0) -> list[Stimulus]:
        """Random walks from the chain, for larger / noisier probe sets.

        Args:
            n: how many walks to draw, one stimulus each.
            length: words per walk.
            seed: seed of the generator drawing every walk.
        """
        rng = np.random.default_rng(seed)
        v = len(self.vocab)
        stims = []
        for s in range(n):
            idx = [int(rng.choice(v, p=self.initial))]
            for _ in range(length - 1):
                idx.append(int(rng.choice(v, p=self.transition[idx[-1]])))
            stims.append(
                make_stimulus([self.vocab[i] for i in idx], stimulus_id=f"walk_{s}")
            )
        return stims
