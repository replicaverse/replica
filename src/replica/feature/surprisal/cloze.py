"""Human cloze probabilities supplied by a dataset's stimulus annotation layer.

PROVO and RastrOS provide human_cloze values for observed words in their left
contexts. This model exposes those observed-word probabilities.
"""

import numpy as np

from ...core import Stimulus
from ..core.protocols import FeatureModelOutput


class ClozeLM:
    """Human cloze FeatureModel.

    Reads cloze probabilities from ``stimulus.layers["human_cloze"]``.
    The layer is expected to contain one value per word: the cloze
    probability (0-1) for the observed word given the left context.

    ``next_log_probs`` is not provided: the cloze norm records only the
    probability of the *observed* word.

    The target layer must have the same spans as the cloze layer. Other
    segmentations require cloze probabilities collected for those units.

    Args:
        name: Identifier used in AnalysisResult results.
    """

    def __init__(self, name: str = "human_cloze"):
        """Name the model; the cloze values come from the stimulus."""
        self.name = name
        self.vocab: list[str] | None = None

    def __call__(
        self,
        stimulus: Stimulus,
        extract: list[str] | None = None,
        target_layer: str = "words",
    ) -> list[FeatureModelOutput]:
        """One output per unit, carrying the log cloze probability in nats.

        Position 0 is NaN by the no-left-context convention, as is any word the
        cloze layer leaves unrated. A rated probability is floored at 1e-10 so
        the log stays finite.

        Args:
            stimulus: Text to score; it must carry a ``human_cloze`` layer.
            extract: Ignored, since the only value this model can supply is
                ``log_p_observed``.
            target_layer: Name of the annotation layer whose units the outputs
                align to, one per unit.

        Raises:
            ValueError: Cloze spans differ from the target units, or the layer
                does not supply one scalar probability per unit.
        """
        if "human_cloze" not in stimulus.layers:
            raise KeyError(
                f"Stimulus {stimulus.stimulus_id!r} has no 'human_cloze' layer. "
                "Load a dataset with human cloze norms before calling ClozeLM."
            )

        cloze_layer = stimulus.layers["human_cloze"]
        target = stimulus.layers[target_layer]
        if cloze_layer.spans != target.spans or len(cloze_layer.values) != len(target):
            raise ValueError(
                f"ClozeLM requires human_cloze spans and values to align with "
                f"target layer {target_layer!r} in stimulus {stimulus.stimulus_id!r}"
            )
        outputs: list[FeatureModelOutput] = []

        for cloze_p in cloze_layer.values:
            if cloze_p is not None and np.ndim(cloze_p) != 0:
                raise ValueError("human_cloze requires one scalar probability per unit")
            if cloze_p is None or np.isnan(float(cloze_p)):
                log_p = np.nan
            else:
                log_p = np.log(max(float(cloze_p), 1e-10))

            outputs.append(
                FeatureModelOutput(
                    values={"log_p_observed": log_p},
                    metadata={"model_name": self.name},
                )
            )

        # First word has no left context by convention
        if outputs:
            outputs[0] = FeatureModelOutput(
                values={"log_p_observed": np.nan},
                metadata={"model_name": self.name},
            )

        return outputs
