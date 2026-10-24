"""Fingerprint loaded corpus inputs with their recorded segmentation and context."""

from copy import deepcopy

from replica.context import corpus_digest
from replica.core import Stimulus


def corpus_fingerprint(
    dataset: str, stimuli: list[Stimulus], unit: str, context_scope: str
) -> str:
    """Digest the loader's corpus with the recorded segmentation and context.

    Args:
        dataset: Registry name used for sentence segmentation.
        stimuli: Complete corpus; the caller's stimuli are left unchanged.
        unit: Recorded output grain.
        context_scope: Recorded stimulus, sentence or story context.
    """
    stimuli = deepcopy(stimuli)
    if unit == "characters":
        from replica.feature.core.annotation.characters import attach_characters

        attach_characters(stimuli)
    elif unit not in {"words", "tokens"}:
        from replica.segmentation import SEGMENTERS, attach_segmentation

        if unit in SEGMENTERS:
            attach_segmentation(unit, stimuli)
        if any(unit not in stimulus.layers for stimulus in stimuli):
            raise ValueError(f"the {dataset} loader has no {unit!r} layer")
    if context_scope == "story":
        from replica.context import attach_preceding_context

        attach_preceding_context(stimuli)
    elif context_scope == "sentence":
        from replica.dataset.catalog.sentences import attach_sentences

        attach_sentences(dataset, stimuli)
    elif context_scope != "stimulus":
        raise ValueError(f"unknown context_scope {context_scope!r}")
    return corpus_digest(stimuli, unit, context_scope)
