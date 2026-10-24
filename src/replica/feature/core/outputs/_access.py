"""Read observed-token and next-token log probabilities from model outputs."""

import numpy as np

from replica.feature.core.protocols import ModelOutputRecord


def _get_log_p_observed(output: ModelOutputRecord) -> float:
    """Extract log_p_observed from a ModelOutputRecord."""
    return output.values.get("log_p_observed", np.nan)


def _get_next_log_probs(output: ModelOutputRecord) -> np.ndarray | None:
    """Extract next_log_probs from a ModelOutputRecord."""
    return output.values.get("next_log_probs")
