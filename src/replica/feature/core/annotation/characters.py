"""Character annotation for whitespace-normalized stimulus text.

Each character, including whitespace, occupies one span. Text outside this
normal form is rejected so character indices match the annotation geometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from replica.core import Stimulus

CHARACTER_LAYER = "characters"


def attach_characters(stimuli: list[Stimulus]) -> None:
    """Attach a one-span-per-character layer (whitespace included).

    Args:
        stimuli: Stimuli to annotate in place. One that already carries the
            layer is left untouched; one whose text is not whitespace-normalized
            raises ``ValueError``, since its character indices would not address
            the same positions as the focal-area geometry.
    """
    from replica.core import AnnotationLayer, Span

    for stimulus in stimuli:
        if CHARACTER_LAYER in stimulus.layers:
            continue
        text = stimulus.text
        if text != " ".join(text.strip().split()):
            raise ValueError(
                f"{stimulus.stimulus_id}: text is not whitespace-normalized, so "
                "character indices would not match the focal-area geometry"
            )
        stimulus.add_layer(
            AnnotationLayer(
                name=CHARACTER_LAYER,
                spans=[Span(i, i + 1) for i in range(len(text))],
                values=list(text),
            )
        )
