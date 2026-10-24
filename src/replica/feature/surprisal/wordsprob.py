"""Pimentel & Meister (2024) word-level surprisal via ``wordsprobability``.

Reference: "How to Compute the Probability of a Word"
  https://github.com/tpimentelms/probability-of-a-word

The paper identifies BOW/BOS/EOW double-counting in standard
sum-of-subword-log-probs surprisal. This wrapper exposes both columns:

  * ``correction="corrected"`` (default), the paper's fix.
  * ``correction="buggy"``, standard sum-of-subword-log-probs, equal to
    ``HuggingFaceLM(model_id, add_bos=True)`` aggregated to whitespace
    words when both use the same start prefix.

Returns ``log_p_observed`` (signed log-prob per unit) so ``Surprisal``
works unchanged. Operates on the stimulus's ``"words"`` layer (word =
whitespace-split); use ``HuggingFaceLM`` + ``Segmenter`` or ``CharBeamLM``
for finer units.
"""

import numpy as np

from replica.feature.backends.start_state import single_token_prefix
from replica.feature.core.annotation.alignment import aggregate_whitespace_words

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput

_SUPPORTED_MODELS = (
    "gpt2-small",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-14b",
    "pythia-28b",
    "pythia-69b",
    "pythia-120b",
    # The surprisal factory maps ``ai-forever/mGPT`` to this upstream id.
    "mgpt",
)


class WordsProbabilityLM:
    """Word-level surprisal via Pimentel & Meister (2024)'s correction.

    Args:
        model_id: One of the upstream ``wordsprobability`` ids in
            ``_SUPPORTED_MODELS``; ``surprisal_model()`` maps Hugging Face ids
            onto them.
        correction: ``"corrected"`` (default) for the paper's BOW/BOS/EOW
            fix; ``"buggy"`` for the standard sum-of-subword-log-probs
            convention.
        model_source: Model and tokenizer location, such as an immutable
            snapshot directory; omitted to use the upstream model name.
        start_token_id: Prefix token id; omitted to keep the upstream BOS.
    """

    vocab = None

    def __init__(
        self,
        model_id: str = "gpt2-small",
        correction: str = "corrected",
        *,
        model_source: str | None = None,
        start_token_id: int | None = None,
    ):
        """Validate the model and correction; the model itself loads lazily."""
        if model_id not in _SUPPORTED_MODELS:
            raise ValueError(
                f"WordsProbabilityLM: model_id={model_id!r} is not supported "
                f"by upstream wordsprobability. Choose one of {_SUPPORTED_MODELS}."
            )
        if correction not in ("corrected", "buggy"):
            raise ValueError(
                f"correction must be 'corrected' or 'buggy', got {correction!r}"
            )
        if start_token_id is not None:
            try:
                (start_token_id,) = single_token_prefix((start_token_id,))
            except ValueError:
                raise ValueError(
                    "start_token_id must be a nonnegative integer"
                ) from None
        self.model_id = model_id
        self.correction = correction
        self.model_source = model_source
        self.start_token_id = start_token_id
        self.name = f"wordsprob_{model_id}_{correction}"
        # Lazy import + lazy model construction (these models are large).
        self._wp_model = None

    def _ensure_model(self):
        if self._wp_model is not None:
            return self._wp_model
        from wordsprobability.models import MODELS, get_model

        if self.model_source is None:
            model = get_model(self.model_id)
        else:
            # Each instance binds its own loader source; the upstream registry
            # and model classes retain their defaults.
            upstream = MODELS[self.model_id]
            source = self.model_source

            class ResolvedModel(upstream):
                model_name = source

            model = ResolvedModel()
        if self.start_token_id is not None:
            model.bos_token_id = self.start_token_id
        self._wp_model = model
        return self._wp_model

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One output per unit, carrying the word's ``log_p_observed`` in nats.

        Args:
            stimulus: Text to score; upstream re-splits the raw text on
                whitespace, and a unit of ``target_layer`` holding several
                whitespace words takes the sum of their log probabilities.
            extract: Ignored, since the only value this model can supply is
                ``log_p_observed``.
            target_layer: Name of the annotation layer whose units the outputs
                align to, one per unit.
        """
        if target_layer not in stimulus.layers:
            raise ValueError(
                f"WordsProbabilityLM expects a {target_layer!r} layer; "
                f"available: {list(stimulus.layers)}"
            )
        # wordsprobability runs on a single string and re-splits on
        # whitespace internally, so the stimulus text is passed as-is.
        df = self._compute_per_word(stimulus.text)

        # The dataframe has one row per whitespace word, in left-to-right
        # order. Sanity-check alignment with the text:
        ws_words = stimulus.text.split()
        if len(df) != len(ws_words):
            raise ValueError(
                f"WordsProbabilityLM: upstream returned {len(df)} words for "
                f"a stimulus whose text whitespace-splits into {len(ws_words)}. "
                "This usually means stimulus.text contains multiple lines or "
                "non-trivial punctuation that wordsprobability tokenises "
                "differently. Pre-segment to a single line before passing in."
            )
        col = "surprisal_buggy" if self.correction == "buggy" else "surprisal"
        surprisal_nats = df[col].to_numpy(dtype=np.float64)
        # FeatureModelOutput stores log_p_observed (= -surprisal in nats)
        log_p = -surprisal_nats

        rows = [
            FeatureModelOutput(
                values={"log_p_observed": float(log_p[i])},
                metadata={"model_name": self.name, "correction": self.correction},
            )
            for i in range(len(ws_words))
        ]
        # A unit of several words takes the sum of their log probabilities.
        return aggregate_whitespace_words(stimulus, target_layer, rows, "sum")

    def _compute_per_word(self, text: str):
        """One stimulus's per-word surprisals, on the cached model.

        This is upstream's ``get_surprisal_per_word`` inlined over a single
        line of text. Calling that entry point instead would go through
        ``get_model``, which returns ``model_cls()`` and so builds a fresh
        model onto the GPU per stimulus, freed only when the garbage
        collector gets to it, which accumulates until the card is exhausted.
        """
        import pandas as pd
        from wordsprobability.main import agg_surprisal_per_word

        model = self._ensure_model()
        results, offsets = model.get_predictions(text.strip())
        df = pd.DataFrame(results)
        df["text_id"] = 0
        df["offsets"] = offsets
        # Always request both 'surprisal' (corrected) and 'surprisal_buggy'
        # so the same call serves either correction mode.
        df = agg_surprisal_per_word(df, self.model_id, return_buggy_surprisals=True)
        # df is indexed by (text_id, word_id); flatten to row order.
        return df.reset_index(drop=True)
