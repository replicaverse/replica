"""Surprisal Feature extractors: surprisal, layer surprisal, rank, reduction."""

from collections.abc import Mapping, Sequence

import numpy as np

from replica.feature.core.annotation._text import _layer_strings
from replica.feature.core.measurement import FeatureMeasurement
from replica.feature.core.outputs._access import (
    _get_log_p_observed,
    _get_next_log_probs,
)
from replica.feature.core.outputs.information import LN2
from replica.feature.core.protocols import ModelOutputRecord

from ...core import Stimulus
from ..core.protocols import FeatureNotAvailable
from ..entropy.extractors import Entropy


class Surprisal:
    """-log2 P(unit_t | unit_{<t}) per unit.

    Position 0 is NaN when the model has no left context (no BOS token).
    When a BOS token is used (e.g. ``HuggingFaceLM(add_bos=True)``, the
    default), position 0 will have a defined value. A record that carries a
    ``surprisal`` value, in bits, is read as is; otherwise ``log_p_observed``,
    in nats, is converted to bits via / log(2). Raises FeatureNotAvailable
    when no record carries either key.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    name = "surprisal"
    measurement = FeatureMeasurement("surprisal", "bits")

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return ("log_p_observed",)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Per-unit surprisal in bits, NaN where the model reports no log-prob.

        Args:
            stimulus: Text whose units the values are indexed by.
            outputs: Feature-model records for those same units, supplying
                ``log_p_observed``.
            target_layer: Name of the annotation layer that fixes the unit
                granularity.
        """
        if outputs is None:
            raise ValueError("Surprisal requires model outputs")
        if outputs and not any(
            "log_p_observed" in o.values or "surprisal" in o.values for o in outputs
        ):
            model_name = outputs[0].metadata.get("model_name", "")
            raise FeatureNotAvailable(self.name, model_name, missing="log_p_observed")
        n = len(outputs)
        vals = np.full(n, np.nan)
        for t in range(n):
            bits = _stored_surprisal(outputs[t])
            if bits is not None:
                vals[t] = bits
                continue
            lp = _get_log_p_observed(outputs[t])
            if not np.isnan(lp):
                vals[t] = -lp / LN2
        return vals


def _stored_surprisal(output: ModelOutputRecord) -> float | None:
    """The record's ``surprisal`` value in bits, or ``None`` when it has none."""
    values = getattr(output, "values", None)
    if not isinstance(values, Mapping) or values.get("surprisal") is None:
        return None
    return float(values["surprisal"])


class LayerSurprisal:
    """-log2 P(unit_t | unit_{<t}) from a specific internal layer.

    Reads ``{lens}_log_p_L{layer}`` (log-prob, nats; ``lens`` selects
    ``logit_lens`` or ``tuned_lens``) produced by
    :class:`~replica.feature.surprisal.logit_lens.LogitLensLM` and
    converts to surprisal in bits. This is the internal-layer predictor of
    Kuribayashi et al. (2025); ``layer`` selects the depth (0 = embeddings,
    ``n_blocks`` = the model's output, identical to :class:`Surprisal`).

    Position 0 is defined when the LM prepended a BOS token (the default for
    ``LogitLensLM``); otherwise NaN.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {"layer": self.layer, "lens": self.lens}

    def __init__(self, layer: int, lens: str = "logit_lens"):
        """Fix which decoded layer to read and name the feature after it.

        Args:
            layer: Depth to read, 0 being the embeddings and ``n_blocks`` the
                model's own output.
            lens: Decoder whose log-probs to read, ``logit_lens`` or
                ``tuned_lens``.
        """
        self.layer = int(layer)
        self.lens = lens  # "logit_lens" (default) or "tuned_lens"
        self.name = f"{lens}_surprisal_L{self.layer}"

    @property
    def measurement(self) -> FeatureMeasurement:
        """The selected layer and decoder, reported in bits."""
        return FeatureMeasurement(
            "surprisal", "bits", {"layer": self.layer, "lens": self.lens}
        )

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return (f"{self.lens}_log_p_L{self.layer}",)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Per-unit surprisal in bits decoded at ``self.layer``.

        Raises FeatureNotAvailable when the LM did not decode that layer.

        Args:
            stimulus: Text whose units the values are indexed by.
            outputs: Feature-model records for those same units, supplying the
                per-layer log-probs.
            target_layer: Name of the annotation layer that fixes the unit
                granularity.
        """
        if outputs is None:
            raise ValueError("LayerSurprisal requires model outputs")
        key = f"{self.lens}_log_p_L{self.layer}"
        n = len(outputs)
        vals = np.full(n, np.nan)
        for t in range(n):
            if key not in outputs[t].values:
                # Layer not computed by ``LogitLensLM(layers=...)``, fail loudly.
                model_name = outputs[t].metadata.get("model_name", "")
                raise FeatureNotAvailable(
                    self.name,
                    f"{model_name} (layer {self.layer} not computed; "
                    "include it in LogitLensLM(layers=...))",
                    missing=key,
                )
            lp = outputs[t].values[key]
            if not np.isnan(lp):
                vals[t] = -lp / LN2
        return vals


class SurprisalReduction:
    """Entropy(t) - Surprisal(t): the uncertainty resolved upon seeing t.

    Entropy comes from the word-type ``next_log_probs``; surprisal from the
    subword-sum ``log_p_observed``, so the two terms use different supports.
    Position 0 is NaN only when the model prepends no BOS.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    name = "surprisal_reduction"
    measurement = FeatureMeasurement("entropy_minus_surprisal", "bits")

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return ("log_p_observed", "next_log_probs")

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Per-unit entropy minus surprisal, in bits.

        Args:
            stimulus: Text whose units the values are indexed by.
            outputs: Feature-model records for those same units, supplying both
                ``next_log_probs`` and ``log_p_observed``.
            target_layer: Name of the annotation layer that fixes the unit
                granularity.
        """
        if outputs is None:
            raise ValueError("SurprisalReduction requires model outputs")
        ent = Entropy()(stimulus, outputs, target_layer)
        sur = Surprisal()(stimulus, outputs, target_layer)
        return ent - sur


class Rank:
    """Rank (1-indexed) of the observed unit in the predictive distribution.

    Lower rank means more expected. Position 0 is scored when the model
    supplies a distribution for it, as :class:`HuggingFaceLM` does with its
    default BOS prefix, and is NaN otherwise. Later positions raise
    FeatureNotAvailable when
    ``next_log_probs`` is None. Every position yields NaN when the output
    carries no ``vocab_index`` and ``len(vocab) + 1`` for a unit absent from the
    vocabulary (ranked just past the known vocab).
    The unit is looked up whitespace-stripped and lower-cased, the form of
    :class:`HuggingFaceLM`'s word types.
    """

    def configuration(self) -> dict:
        """Return computation settings without the column alias."""
        return {}

    name = "rank"
    measurement = FeatureMeasurement("rank", "dimensionless", {"index_base": 1})

    @property
    def required_outputs(self) -> tuple[str, ...]:
        """Model value keys consumed by this feature."""
        return ("next_log_probs",)

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Per-unit 1-indexed rank of the observed unit.

        Args:
            stimulus: Text whose unit strings are looked up in the vocabulary.
            outputs: Feature-model records for those same units, supplying
                ``next_log_probs`` and a ``vocab_index``.
            target_layer: Name of the annotation layer that fixes the unit
                granularity.
        """
        if outputs is None:
            raise ValueError("Rank requires model outputs")
        n = len(outputs)
        vals = np.full(n, np.nan)
        unit_strings = _layer_strings(stimulus, target_layer)
        for t in range(n):
            nlp = _get_next_log_probs(outputs[t])
            if nlp is None:
                if t == 0:
                    continue  # No BOS: the first unit has no distribution.
                model_name = outputs[t].metadata.get("model_name", "")
                raise FeatureNotAvailable("rank", model_name)
            vocab_index: dict[str, int] | None = outputs[t].metadata.get("vocab_index")
            unit = unit_strings[t].strip().lower()
            if vocab_index is not None:
                idx = vocab_index.get(unit)
                if idx is None:
                    vals[t] = len(nlp) + 1
                    continue
                unit_lp = nlp[idx]
            else:
                vals[t] = np.nan
                continue
            vals[t] = float(np.sum(nlp > unit_lp) + 1)
        return vals
