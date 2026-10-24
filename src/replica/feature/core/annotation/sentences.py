"""Score caller-annotated sentences as independent model contexts."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from replica.context import SENTENCE_LAYER

if TYPE_CHECKING:
    from replica.core import Stimulus


def per_sentence_outputs(
    score: Callable[[Stimulus], list],
    stimulus: Stimulus,
    target_layer: str = "words",
) -> list:
    """Score each sentence of *stimulus* as its own context, in word order.

    Boundaries come from the ``sentences`` layer parallel to the word layer.

    Args:
        score: Maps a single-sentence stimulus to its list of per-unit outputs.
        stimulus: Text carrying the unit layer and a parallel ``sentences``
            layer; missing boundaries raise ``ValueError``.
        target_layer: Annotation layer whose units are regrouped by sentence.

    Returns:
        The per-sentence outputs concatenated back into unit order. Each
        single-sentence stimulus records its character offset within the
        full text under the ``text_offset`` metadata key.
    """
    from itertools import groupby

    from replica.core import AnnotationLayer, Span, Stimulus
    from replica.feature.core.annotation.alignment import TEXT_OFFSET

    if SENTENCE_LAYER not in stimulus.layers:
        raise ValueError(
            f"context_scope='sentence' needs a {SENTENCE_LAYER!r} layer on "
            f"{stimulus.stimulus_id}; attach the corpus boundaries first"
        )
    words = stimulus.layers[target_layer]
    sentence_ids = stimulus.layers[SENTENCE_LAYER].values
    if len(sentence_ids) != len(words.spans):
        raise ValueError(
            f"{stimulus.stimulus_id}: {len(sentence_ids)} sentence ids for "
            f"{len(words.spans)} words"
        )
    outputs: list = []
    indexed = list(enumerate(sentence_ids))
    for _, group in groupby(indexed, key=lambda pair: pair[1]):
        idxs = [i for i, _ in group]
        spans = [words.spans[i] for i in idxs]
        start = spans[0].start
        sub = Stimulus(
            text=stimulus.text[start : spans[-1].end],
            stimulus_id=f"{stimulus.stimulus_id}#sent",
            metadata={TEXT_OFFSET: start},
        )
        sub.add_layer(
            AnnotationLayer(
                name=target_layer,
                spans=[Span(s.start - start, s.end - start) for s in spans],
                values=[words.values[i] if words.values else None for i in idxs],
            )
        )
        outputs.extend(score(sub))
    return outputs
