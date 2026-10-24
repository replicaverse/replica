"""Context dispatch and stimulus-wide indexing for token-level model outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from replica.core import Stimulus
    from replica.feature.core.protocols import FeatureModelOutput


def token_outputs(
    score: Callable[[Stimulus], list[FeatureModelOutput]],
    stimulus: Stimulus,
    context_scope: str,
) -> list[FeatureModelOutput]:
    """Score token contexts and assign consecutive indices across the stimulus.

    Args:
        score: Backend scorer for one context; preserves its own token spans,
            value units and beginning-of-sequence convention.
        stimulus: Text and annotation layers to score.
        context_scope: Sentence scope splits on supplied word boundaries;
            other scopes pass the original stimulus to the backend.

    Returns:
        Backend outputs in text order with stimulus-wide token indices.
    """
    if context_scope == "sentence":
        from replica.feature.core.annotation.sentences import per_sentence_outputs

        outputs = per_sentence_outputs(score, stimulus, "words")
    else:
        outputs = score(stimulus)
    for index, output in enumerate(outputs):
        output.metadata["token_index"] = index
    return outputs
