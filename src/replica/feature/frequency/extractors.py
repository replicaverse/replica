"""Frequency Feature extractors: log-frequency and unigram surprisal."""

from collections.abc import Sequence

import numpy as np

from replica.feature.core.annotation._text import _layer_strings
from replica.feature.core.binding import resolve_stimulus_feature
from replica.feature.core.measurement import FeatureMeasurement
from replica.feature.core.outputs.information import LOG2_10
from replica.feature.core.protocols import ModelOutputRecord

from ...core import Stimulus
from ..core.protocols import FrequencySource


def _probabilities(source: FrequencySource, strings: list[str]) -> np.ndarray:
    """Look up each literal surface; the source owns its normalization."""
    return np.array([source.probability(s) for s in strings], dtype=float)


class LogFrequency:
    """log10 frequency: ``log₁₀ p(word)`` from an external FrequencySource.

    The source returns the base-free probability p(word); this feature applies
    the log10 base it reports in. Returns NaN for unknown units or if no source
    is provided. Each unit is looked up as one whole surface form.
    :class:`~replica.feature.frequency.model.UnigramSurprisalLM` with
    ``lookup="table"`` instead splits a unit and sums its parts' surprisals.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    dependencies: tuple = ()
    resolve = resolve_stimulus_feature

    measurement = FeatureMeasurement("log_probability", "hartleys")
    required_outputs: tuple[str, ...] = ()

    def __init__(self, source: FrequencySource | None = None):
        """Bind the source and name the feature after its language.

        Args:
            source: Supplier of the base-free ``p(word)``. ``None`` makes every
                unit NaN and leaves the feature name without a language suffix.
        """
        self._source = source
        suffix = f"_{source.language}" if source is not None else ""
        self.name = f"log_frequency{suffix}"

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """log10 frequency of every unit of one stimulus.

        Args:
            stimulus: Passage supplying the surface forms.
            outputs: Per-unit model outputs, unused: a unigram frequency depends
                on the surface form alone.
            target_layer: Annotation layer whose spans define the units to
                score, ``words`` or ``characters``.

        Returns:
            One value per unit of the layer, NaN where the source reports no
            probability.
        """
        strings = _layer_strings(stimulus, target_layer)
        if self._source is None:
            return np.full(len(strings), np.nan)
        probs = _probabilities(self._source, strings)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(probs)


class UnigramSurprisal:
    """Unigram surprisal in bits: ``-log₂ p(word)``.

    The standard psycholinguistic control predictor "frequency effect".
    A :class:`FrequencySource` returns the base-free probability ``p(word)``;
    this feature takes ``log₂`` and negates it (surprisal is customarily
    reported in bits), i.e. ``-log₂(10) · LogFrequency(source)``. The lookup
    convention is :class:`LogFrequency`'s: one whole surface form per unit,
    normalized by the source.

    Args:
        source: a :class:`FrequencySource` (e.g.
            :class:`~replica.feature.frequency.WordfreqSource`,
            :class:`~replica.feature.frequency.UserTable`).
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    dependencies: tuple = ()
    resolve = resolve_stimulus_feature

    measurement = FeatureMeasurement("surprisal", "bits", {"context": "unigram"})
    required_outputs: tuple[str, ...] = ()

    def __init__(self, source: FrequencySource | None = None):
        """Bind the source and name the feature after its language."""
        self._source = source
        suffix = f"_{source.language}" if source is not None else ""
        self.name = f"unigram_surprisal{suffix}"

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Unigram surprisal in bits for every unit of one stimulus.

        Args:
            stimulus: Passage supplying the surface forms.
            outputs: Per-unit model outputs, unused: a unigram surprisal depends
                on the surface form alone.
            target_layer: Annotation layer whose spans define the units to
                score, ``words`` or ``characters``.

        Returns:
            One value per unit of the layer, NaN where the source reports no
            probability.
        """
        strings = _layer_strings(stimulus, target_layer)
        if self._source is None:
            return np.full(len(strings), np.nan)
        probs = _probabilities(self._source, strings)
        with np.errstate(divide="ignore", invalid="ignore"):
            log10_p = np.log10(probs)
        # -log₂ p = -log₁₀ p · log₂ 10
        return -log10_p * LOG2_10
