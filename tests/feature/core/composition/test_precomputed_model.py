"""Tests for PrecomputedFeatureModel, including target_layer validation."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span
from replica.feature import FeatureModelOutput, PrecomputedFeatureModel
from tests._stimuli import word_stimulus


def _make_stimulus(n_words=3, stimulus_id="s0"):
    return word_stimulus(" ".join(f"w{i}" for i in range(n_words)), stimulus_id)


def _make_outputs(n):
    return [
        FeatureModelOutput(values={"log_p_observed": np.nan if i == 0 else -1.0})
        for i in range(n)
    ]


class TestPrecomputedFeatureModel:
    def test_returns_stored_outputs(self):
        stim = _make_stimulus(3)
        outputs = _make_outputs(3)
        model = PrecomputedFeatureModel("test", {stim.stimulus_id: outputs})
        result = model(stim)
        assert result is outputs
        assert model.name == "test"
        assert model.vocab is None

    def test_missing_stimulus_raises(self):
        stim = _make_stimulus(3)
        model = PrecomputedFeatureModel("m", {})
        with pytest.raises(KeyError):
            model(stim)

    def test_length_mismatch_raises(self):
        """Stored outputs count != target layer count raises ValueError."""
        stim = _make_stimulus(3)  # 3 words
        outputs = _make_outputs(2)  # only 2 outputs
        model = PrecomputedFeatureModel("m", {stim.stimulus_id: outputs})
        with pytest.raises(ValueError, match="stored 2.*has 3 units"):
            model(stim)

    def test_length_mismatch_custom_layer(self):
        """Validation works with a non-default target_layer."""
        stim = _make_stimulus(3)
        # Add a 5-unit custom layer
        char_spans = [Span(i, i + 1) for i in range(5)]
        stim.add_layer(AnnotationLayer("chars", char_spans, list("abcde")))
        outputs = _make_outputs(3)  # 3 outputs, but chars layer has 5
        model = PrecomputedFeatureModel("m", {stim.stimulus_id: outputs})
        with pytest.raises(ValueError, match="stored 3.*chars.*has 5 units"):
            model(stim, target_layer="chars")

    def test_matching_length_with_custom_layer(self):
        """No error when stored count matches custom target_layer."""
        stim = _make_stimulus(3)
        char_spans = [Span(i, i + 1) for i in range(5)]
        stim.add_layer(AnnotationLayer("chars", char_spans, list("abcde")))
        outputs = _make_outputs(5)
        model = PrecomputedFeatureModel("m", {stim.stimulus_id: outputs})
        result = model(stim, target_layer="chars")
        assert result is outputs

    def test_a_stored_unit_string_must_name_the_layers_unit(self):
        """A record naming its unit is held against the stimulus, after NFC."""
        stim = _make_stimulus(3)
        outputs = _make_outputs(3)
        for record, word in zip(outputs, ["w0", "w1", "w2"], strict=True):
            record.metadata["text_unit"] = word
        model = PrecomputedFeatureModel("m", {stim.stimulus_id: outputs})
        assert model(stim) is outputs
        outputs[1].metadata["text_unit"] = "w9"
        with pytest.raises(
            ValueError, match="unit 1 .* is 'w9', but the 'words' layer has 'w1'"
        ):
            model(stim)

    def test_unit_strings_compare_after_nfc_and_absent_ones_are_not_checked(self):
        stim = word_stimulus("Café au lait", "s1")
        outputs = _make_outputs(3)
        outputs[0].metadata["text_unit"] = "Café"
        model = PrecomputedFeatureModel("m", {stim.stimulus_id: outputs})
        assert model(stim) is outputs
