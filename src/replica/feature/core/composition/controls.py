"""Text-derived predictors: word length, position."""

from collections.abc import Sequence

import numpy as np

from replica.core import Stimulus
from replica.feature.core.annotation._text import _layer_strings
from replica.feature.core.binding import resolve_stimulus_feature
from replica.feature.core.measurement import FeatureMeasurement
from replica.feature.core.protocols import ModelOutputRecord

from .values import _shifted


class UnitLength:
    """Number of characters in each unit."""

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    dependencies: tuple = ()
    resolve = resolve_stimulus_feature

    name = "unit_length"
    measurement = FeatureMeasurement("length", "characters")
    required_outputs: tuple[str, ...] = ()

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Character count of every unit, shape (n_units,).

        Args:
            stimulus: Text whose units are measured.
            outputs: Unused; the count is read off the text alone.
            target_layer: Annotation layer whose units are measured.
        """
        return np.array(
            [len(s) for s in _layer_strings(stimulus, target_layer)], dtype=float
        )


class UnitPosition:
    """0-indexed position of each unit in the stimulus."""

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    dependencies: tuple = ()
    resolve = resolve_stimulus_feature

    name = "unit_position"
    measurement = FeatureMeasurement("position", "annotation_units", {"index_base": 0})
    required_outputs: tuple[str, ...] = ()

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Running 0-indexed unit position, shape (n_units,).

        Args:
            stimulus: Text whose units are numbered.
            outputs: Unused; the numbering depends only on the layer length.
            target_layer: Annotation layer whose units are numbered.
        """
        return np.arange(len(stimulus.layers[target_layer]), dtype=float)


class PrevUnitLength:
    """Number of characters in the preceding unit. NaN at position 0."""

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    @property
    def dependencies(self) -> tuple:
        """The native unit lengths to shift."""
        if not hasattr(self, "_source_feature"):
            self._source_feature = UnitLength()
        return (self._source_feature,)

    def resolve(self, context, inputs):
        """Shift length values with their native observation contributors.

        Args:
            context: Selected materials and annotation layer.
            inputs: Resolved unit lengths.
        """
        return _shifted(inputs[0], 1, self.measurement)

    name = "prev_unit_length"
    measurement = FeatureMeasurement("length", "characters", {"lag": 1})
    required_outputs: tuple[str, ...] = ()

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Character count of the preceding unit, NaN at position 0.

        Args:
            stimulus: Text whose units are measured.
            outputs: Unused; the count is read off the text alone.
            target_layer: Annotation layer whose units are measured.
        """
        lengths = np.array(
            [len(s) for s in _layer_strings(stimulus, target_layer)], dtype=float
        )
        n = len(lengths)
        vals = np.empty(n)
        if n == 0:
            return vals
        vals[0] = np.nan
        vals[1:] = lengths[:-1]
        return vals
