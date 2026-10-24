"""Protocols for feature models, output records, extractors, and frequency sources.

Models return per-unit records; extractors convert them to numeric arrays.
Frequency sources expose probabilities independently of a logarithm base.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ...core import Stimulus


class ModelOutputRecord(Protocol):
    """Read-only fields consumed by per-unit feature extractors."""

    @property
    def values(self) -> Mapping:
        """Named model quantities for this unit."""
        ...

    @property
    def metadata(self) -> Mapping:
        """Source annotations for this unit."""
        ...


@dataclass
class FeatureModelOutput:
    """Per-unit output from a feature model.

    ``values`` is a dict of named outputs. The canonical keys read by the
    built-in features (quantity-specific models add their own further keys):
        - ``"log_p_observed"``: log P(w_t | w_{<t}) in nats.
            NaN at t=0 when no left context or configured start prefix is available.
        - ``"next_log_probs"``: log P(vocab | w_{<t}) over the model's type
            vocabulary (for ``HuggingFaceLM``, first-subword tokens pooled by
            surface form). Shape (vocab_size,); exponentiated values sum to one.
            None if the model cannot provide a full type distribution.

    ``metadata`` holds model-specific extras (e.g. subword token boundaries).
    """

    values: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class FeatureModel(Protocol):
    """Anything that maps a Stimulus to per-unit outputs.

    ``name`` identifies the model in AnalysisResult results.
    ``vocab`` is the word-type vocabulary for next_log_probs indexing;
    None if the model does not provide next_log_probs.

    Declare ``supports_output_union=True`` only when combining requested keys
    preserves each output, its sampled realization, and state changes that can
    affect subsequent outputs. Custom subclasses must establish this guarantee
    before sharing differing request sets.

    Independently, declare ``supports_stimulus_selection=True`` only when
    omitting calls for other stimuli preserves retained outputs and source
    state affecting subsequent outputs. This optional capability defaults to
    false; custom subclasses must establish the guarantee before opting in.
    Extraction can skip stimuli only when the model and its extractors opt in.
    """

    @property
    def name(self) -> str:
        """Display name of the model."""
        ...

    @property
    def vocab(self) -> list[str] | None:
        """Optional vocabulary indexing next_log_probs."""
        ...

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """Return one FeatureModelOutput per unit in stimulus.layers[target_layer].

        The first observed log probability needs left context or a configured
        start prefix; otherwise it is NaN.
        len(outputs) == len(stimulus.layers[target_layer]) always.

        Args:
            stimulus: Text to score, carrying the annotation layers.
            extract: Value keys to compute, e.g. ``["log_p_observed",
                "next_log_probs"]``. ``None`` computes every key the model can
                supply.
            target_layer: Name of the annotation layer whose units the outputs
                align to, one per unit.
        """
        ...


class FeatureNotAvailable(RuntimeError):
    """Raised when a feature requires a model output the model did not provide."""

    def __init__(
        self,
        feature_name: str,
        model_name: str = "",
        missing: str = "next_log_probs",
    ):
        """Compose the message naming the feature and the absent output key.

        Args:
            feature_name: Feature that asked for the output.
            model_name: Feature model that failed to supply it; an empty string
                leaves the model unnamed in the message.
            missing: Key the feature needed from ``FeatureModelOutput.values``.
        """
        msg = f"Feature '{feature_name}' requires model output '{missing}'"
        if model_name:
            msg += f", but model '{model_name}' did not provide it."
        super().__init__(msg)
        self.feature_name = feature_name
        self.model_name = model_name
        self.missing = missing


@runtime_checkable
class FeatureExtractor(Protocol):
    """Extract per-unit scalar or explicitly dimensioned values from model outputs.

    Bind an extractor to a source with ``ModelFeature``. Optional measurement
    descriptions document units; configuration declares computation settings.

    Declare ``supports_stimulus_selection=True`` only when omitting calls for
    other stimuli preserves retained values and state affecting subsequent
    outputs. This optional capability defaults to false, independently of the
    model's guarantee; custom subclasses must establish it before opting in.
    """

    @property
    def name(self) -> str:
        """Name used for the design column and comparison term."""
        ...

    def __call__(
        self,
        stimulus: Stimulus,
        outputs: Sequence[ModelOutputRecord] | None,
        target_layer: str = "words",
    ) -> np.ndarray:
        """Values whose first axis indexes units; remaining axes are declared.

        Shared output records are read-only; missing values are NaN.

        Args:
            stimulus: Text whose units the values are indexed by.
            outputs: Feature-model records for those same units, ``None`` for
                data-derived features that need no model.
            target_layer: Name of the annotation layer that fixes the unit
                granularity.
        """
        ...


@runtime_checkable
class OutputRequirements(Protocol):
    """Optional capability declaring the raw outputs a feature consumes."""

    @property
    def required_outputs(self) -> Sequence[str] | None:
        """Required keys, or None when the feature may read any output."""
        ...


@runtime_checkable
class FrequencySource(Protocol):
    """Unigram frequency table for a given language.

    Base-free: returns the unigram probability p(*unit*) in the linear domain.
    The feature layer chooses how to report it (``LogFrequency`` -> log10,
    ``UnigramSurprisal`` -> bits), so no log base is baked into the source.
    """

    language: str

    def probability(self, unit: str) -> float:
        """Unigram probability p(*unit*) (relative frequency); NaN if unknown.

        Args:
            unit: Literal surface form. The source applies any normalization
                required by its table or corpus.
        """
        ...
