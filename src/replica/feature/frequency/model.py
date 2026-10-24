"""The frequency FeatureModel: unigram surprisal emitted per unit."""

from pathlib import Path

import numpy as np

from replica.feature.core.outputs.information import LOG2_10

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput, FrequencySource
from .sampled import SampledUnigramSource


class UnigramSurprisalLM:
    """A :class:`FeatureModel` that emits unigram surprisal per unit.

    Each :class:`FeatureModelOutput` carries ``unigram_surprisal`` in bits
    and ``log_p_observed`` in nats.

    Args:
        source: a :class:`FrequencySource` (e.g.
            :class:`~replica.feature.frequency.WordfreqSource`,
            :class:`~replica.feature.frequency.UserTable`). Required.
        lookup: Which surface a unit is looked up under. ``table`` splits a
            fused interest area into whitespace tokens and sums their
            surprisals, the form a lexicon is keyed by; ``lm_surface`` keeps the
            unit as one string with the leading space it carries in the passage,
            the form an LM-derived source is keyed by.

    Example::

        from replica.feature import (
            UnigramSurprisalLM, WordfreqSource,
        )
        lm = UnigramSurprisalLM(WordfreqSource(language="en"))
        outputs = lm(stimulus)  # list[FeatureModelOutput]
    """

    vocab = None  # not a token-level distribution

    def __init__(self, source: FrequencySource, *, lookup: str = "table"):
        """Bind the source and lookup convention."""
        if source is None:
            raise ValueError("UnigramSurprisalLM requires a FrequencySource")
        if lookup not in ("table", "lm_surface"):
            raise ValueError("lookup must be 'table' or 'lm_surface'")
        self._source = source
        self.lookup = lookup
        suffix = "" if lookup == "table" else f"_{lookup}"
        self.name = f"unigram_surprisal_{source.language}{suffix}"

    def _table_keys(self, s: str) -> list[str]:
        """Split joined units; each source normalizes its own lookup keys."""
        return [t for t in s.replace("_", " ").split() if t]

    @staticmethod
    def _lm_surface(text: str, span, s: str) -> list[str]:
        """One lookup surface, with underscores replaced by spaces.

        Include a leading space when one precedes the span in the passage.
        Empty units return no keys and are scored as NaN.
        """
        word = s.replace("_", " ").strip()
        if not word:
            return []
        lead = span.start > 0 and text[span.start - 1] == " "
        return [f" {word}" if lead else word]

    def _surfaces(self, stimulus: Stimulus, target_layer: str) -> list[list[str]]:
        """Lookup keys per unit, one list each.

        Shared by ``__call__`` and :meth:`prepopulate_cache` so both request the
        same keys.
        """
        layer = stimulus.layers[target_layer]
        strings = layer.extract_strings(stimulus.text)
        if self.lookup == "table":
            return [self._table_keys(s) for s in strings]
        return [
            self._lm_surface(stimulus.text, span, s)
            for span, s in zip(layer.spans, strings, strict=True)
        ]

    def prepopulate_cache(
        self,
        stimuli: list[Stimulus],
        target_layer: str = "words",
        *,
        checkpoint: Path | None = None,
        owner: object = None,
    ) -> None:
        r"""Batch-estimate every word the upcoming :meth:`__call__`\\ s will
        look up, if the source supports it. No-op for sources without
        ``estimate_words`` (e.g. ``WordfreqSource``, ``KenLMSource``).

        Args:
            stimuli: Passages about to be scored; their units are pooled and
                deduplicated into one batch request.
            target_layer: Annotation layer whose spans define the units, and
                which the later calls have to pass again for the cache to hit.
            checkpoint: File a :class:`SampledUnigramSource` saves its contexts
                and finished estimates to and resumes from; other sources
                ignore it.
            owner: Identity of the computation *checkpoint* belongs to.
        """
        if not hasattr(self._source, "estimate_words"):
            return
        words: set[str] = set()
        for stim in stimuli:
            for surfaces in self._surfaces(stim, target_layer):
                words.update(surfaces)
        if not words:
            return
        if isinstance(self._source, SampledUnigramSource):
            self._source.estimate_words(
                sorted(words), checkpoint=checkpoint, owner=owner
            )
        else:
            self._source.estimate_words(sorted(words))

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One output per unit, carrying that unit's unigram surprisal.

        Args:
            stimulus: Passage to score, carrying the annotation layers.
            extract: Value keys the caller wants, unused: this model supplies
                both of its keys on every call.
            target_layer: Annotation layer whose spans define the units the
                outputs align to.

        Returns:
            One :class:`FeatureModelOutput` per unit, with
            ``unigram_surprisal`` in bits and ``log_p_observed`` in nats; both
            are NaN for an empty unit.
        """
        surfaces = self._surfaces(stimulus, target_layer)
        log10_p: np.ndarray = np.empty(len(surfaces), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, tokens in enumerate(surfaces):
                if not tokens:
                    log10_p[i] = float("nan")
                    continue
                log10_p[i] = float(
                    np.sum([np.log10(self._source.probability(t)) for t in tokens])
                )
        surprisal_bits = -log10_p * LOG2_10
        # Convert to nats for the conventional ``log_p_observed`` slot,
        # so downstream code that reads that slot works uniformly:
        # log_p_nats = log10_p · ln(10) = log10_p / log10(e).
        log_p_nats = log10_p / np.log10(np.e)
        outs: list[FeatureModelOutput] = []
        for s, lp in zip(surprisal_bits, log_p_nats, strict=False):
            outs.append(
                FeatureModelOutput(
                    values={
                        "unigram_surprisal": float(s),
                        "log_p_observed": float(lp),
                    },
                    metadata={"model_name": self.name},
                )
            )
        return outs
