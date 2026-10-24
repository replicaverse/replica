"""Cloze probabilities require the annotation spans they were collected for."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import ClozeLM


def _stimulus():
    stimulus = Stimulus(text="aa bb cc dd ee", stimulus_id="cloze")
    spans = [Span(start, start + 2) for start in range(0, 15, 3)]
    stimulus.add_layer(AnnotationLayer("words", spans, [None] * 5))
    stimulus.add_layer(
        AnnotationLayer("human_cloze", list(spans), [0.1, None, np.nan, 0.0, 0.8])
    )
    return stimulus


def test_aligned_alias_keeps_unrated_values_and_probability_floor():
    stimulus = _stimulus()
    stimulus.add_layer(
        AnnotationLayer("alias", list(stimulus.layers["words"].spans), [None] * 5)
    )
    expected = [np.nan, np.nan, np.nan, np.log(1e-10), np.log(0.8)]
    for target_layer in ("words", "alias"):
        outputs = ClozeLM()(stimulus, target_layer=target_layer)
        np.testing.assert_equal(
            [output.values["log_p_observed"] for output in outputs], expected
        )


@pytest.mark.parametrize(
    "mismatch", ["fused", "shifted", "missing_norm", "value_count", "vector_value"]
)
def test_misaligned_or_nonscalar_cloze_values_are_refused(mismatch):
    stimulus = _stimulus()
    norms = stimulus.layers["human_cloze"]
    if mismatch == "fused":
        stimulus.layers["words"] = AnnotationLayer(
            "words", [Span(0, 5), *norms.spans[2:]], [None] * 4
        )
    elif mismatch == "shifted":
        stimulus.layers["words"].spans[1] = Span(3, 4)
    elif mismatch == "missing_norm":
        stimulus.layers["human_cloze"] = AnnotationLayer(
            "human_cloze", norms.spans[:-1], norms.values[:-1]
        )
    elif mismatch == "value_count":
        norms.values.pop()
    else:
        norms.values[1] = np.array([0.5])
    with pytest.raises(ValueError, match="human_cloze"):
        ClozeLM()(stimulus)


def test_absent_cloze_layer_remains_an_error():
    stimulus = _stimulus()
    del stimulus.layers["human_cloze"]
    with pytest.raises(KeyError, match="human_cloze"):
        ClozeLM()(stimulus)
