"""Stimulus builders shared across the test suite."""

from replica.core import AnnotationLayer, Span, Stimulus


def word_stimulus(
    text: str,
    stimulus_id: str = "t",
    *,
    metadata: dict | None = None,
) -> Stimulus:
    """Build a stimulus whose ``words`` layer follows whitespace tokens."""
    words = text.split()
    spans = []
    cursor = 0
    for word in words:
        start = text.index(word, cursor)
        spans.append(Span(start, start + len(word)))
        cursor = start + len(word)
    stimulus = Stimulus(
        text=text,
        stimulus_id=stimulus_id,
        metadata={} if metadata is None else dict(metadata),
    )
    stimulus.add_layer(AnnotationLayer(name="words", spans=spans, values=words))
    return stimulus


def s0_word_stimulus(text: str) -> Stimulus:
    """Build a word stimulus with the id ``s0``."""
    return word_stimulus(text, "s0")
