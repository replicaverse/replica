"""Tests for text segmentation, refinement checking, and aggregation.

Covers:
- Built-in segmenters (WhitespaceSegmenter, CharacterSegmenter, from_predicate)
- Refinement computation (compute_refinement)
- Value and mask aggregation
- Stimulus segmentation (segment_stimulus)
- Integration with evaluate() for compatibility checking
"""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Datum, Span, Stimulus
from replica.feature import FeatureModelOutput
from replica.segmentation import (
    CharacterSegmenter,
    Segmenter,
    WhitespaceSegmenter,
    aggregate_mask,
    aggregate_values,
    compute_refinement,
    from_predicate,
    segment_stimulus,
)

from ._stimuli import word_stimulus
from .analysis._fixtures import make_toy_dataset

# ---------------------------------------------------------------------------
# WhitespaceSegmenter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_spans",
    [
        ("hello world", [Span(0, 5), Span(6, 11)]),
        ("hello  world", [Span(0, 5), Span(7, 12)]),  # multiple spaces
        (" hello ", [Span(1, 6)]),  # leading/trailing
        ("", []),  # empty
        ("   ", []),  # whitespace only
        ("a\tb\nc", [Span(0, 1), Span(2, 3), Span(4, 5)]),  # tabs/newlines
        ("hello", [Span(0, 5)]),  # single word
    ],
)
def test_whitespace_segmenter_spans(text, expected_spans):
    """Spans over tabs, newlines, and run-on whitespace match the expectation."""
    assert WhitespaceSegmenter()(text) == expected_spans


def test_whitespace_segmenter_extracts_correct_strings():
    """Spans returned by WhitespaceSegmenter must extract back to the
    original whitespace-split words from the source text."""
    text = "the cat sat"
    spans = WhitespaceSegmenter()(text)
    assert [text[s.start : s.end] for s in spans] == ["the", "cat", "sat"]


# ---------------------------------------------------------------------------
# CharacterSegmenter
# ---------------------------------------------------------------------------


class TestCharacterSegmenter:
    def test_skip_whitespace_default(self):
        spans = CharacterSegmenter()("hi there")
        assert len(spans) == 7  # 2 + 5, skipping the space

    def test_include_whitespace(self):
        spans = CharacterSegmenter(include_whitespace=True)("hi there")
        assert len(spans) == 8  # every character including space

    def test_each_span_is_one_char(self):
        spans = CharacterSegmenter()("abc")
        assert all(len(s) == 1 for s in spans)
        assert spans[0] == Span(0, 1)
        assert spans[1] == Span(1, 2)
        assert spans[2] == Span(2, 3)

    def test_empty(self):
        assert CharacterSegmenter()("") == []


@pytest.mark.parametrize(
    "segmenter, expected_name",
    [
        (WhitespaceSegmenter(), "words"),
        (CharacterSegmenter(), "characters"),
    ],
)
def test_segmenter_name(segmenter, expected_name):
    """Hardcoded `.name` attribute on stateless segmenter classes."""
    assert segmenter.name == expected_name


# ---------------------------------------------------------------------------
# from_predicate
# ---------------------------------------------------------------------------


class TestFromPredicate:
    def test_whitespace_predicate(self):
        seg = from_predicate(lambda t, i: t[i].isspace(), "words")
        spans = seg("hello world")
        assert len(spans) == 2
        assert spans[0] == Span(0, 5)
        assert spans[1] == Span(6, 11)

    def test_hyphen_predicate(self):
        seg = from_predicate(lambda t, i: t[i] == "-", "parts")
        spans = seg("well-known-fact")
        assert len(spans) == 3
        strings = ["well-known-fact"[s.start : s.end] for s in spans]
        assert strings == ["well", "known", "fact"]

    def test_name_set(self):
        seg = from_predicate(lambda t, i: False, "single")
        assert seg.name == "single"

    def test_no_separators(self):
        """Predicate that never fires -> single span."""
        seg = from_predicate(lambda t, i: False, "full")
        spans = seg("hello world")
        assert len(spans) == 1
        assert spans[0] == Span(0, 11)

    def test_all_separators(self):
        """Predicate that always fires -> no spans."""
        seg = from_predicate(lambda t, i: True, "none")
        spans = seg("hello")
        assert len(spans) == 0

    def test_conforms_to_protocol(self):
        from replica.segmentation import Segmenter

        seg = from_predicate(lambda t, i: t[i].isspace(), "words")
        assert isinstance(seg, Segmenter)


# ---------------------------------------------------------------------------
# compute_refinement
# ---------------------------------------------------------------------------


class TestComputeRefinement:
    def test_exact_match(self):
        """Same spans -> each target gets exactly one source."""
        source = [Span(0, 5), Span(6, 11)]
        target = [Span(0, 5), Span(6, 11)]
        mapping = compute_refinement(source, target)
        assert mapping == [[0], [1]]

    def test_words_refine_sentence(self):
        """Two word spans refine one sentence span."""
        source = [Span(0, 5), Span(6, 11)]
        target = [Span(0, 11)]
        mapping = compute_refinement(source, target)
        assert mapping == [[0, 1]]

    def test_words_refine_two_sentences(self):
        """Four words map into two sentences."""
        source = [Span(0, 3), Span(4, 7), Span(9, 12), Span(13, 16)]
        target = [Span(0, 7), Span(9, 16)]
        mapping = compute_refinement(source, target)
        assert mapping == [[0, 1], [2, 3]]

    def test_chars_refine_words(self):
        # "ab cd" -> chars a(0,1), b(1,2), c(3,4), d(4,5)
        # words: ab(0,2), cd(3,5)
        source = [Span(0, 1), Span(1, 2), Span(3, 4), Span(4, 5)]
        target = [Span(0, 2), Span(3, 5)]
        mapping = compute_refinement(source, target)
        assert mapping == [[0, 1], [2, 3]]

    def test_incompatible_span_crosses_boundary(self):
        """Source span straddles two target spans -> error."""
        source = [Span(0, 7)]  # crosses the boundary at position 5
        target = [Span(0, 5), Span(5, 10)]
        with pytest.raises(ValueError, match="not fully contained"):
            compute_refinement(source, target)

    def test_empty_target_group(self):
        """Target span with no source spans gets empty list."""
        source = [Span(0, 3)]
        target = [Span(0, 3), Span(5, 8)]
        mapping = compute_refinement(source, target)
        assert mapping == [[0], []]

    def test_empty_both(self):
        mapping = compute_refinement([], [])
        assert mapping == []


# ---------------------------------------------------------------------------
# aggregate_values / aggregate_mask
# ---------------------------------------------------------------------------


class TestAggregateValues:
    def test_identity(self):
        values = np.array([1.0, 2.0, 3.0])
        mapping = [[0], [1], [2]]
        result = aggregate_values(values, mapping)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_sum(self):
        values = np.array([100.0, 200.0, 150.0, 300.0])
        mapping = [[0, 1], [2, 3]]
        result = aggregate_values(values, mapping)
        np.testing.assert_array_equal(result, [300.0, 450.0])

    def test_nan_propagation(self):
        values = np.array([100.0, np.nan, 150.0])
        mapping = [[0, 1], [2]]
        result = aggregate_values(values, mapping)
        assert np.isnan(result[0])
        assert result[1] == 150.0

    def test_empty_group(self):
        values = np.array([1.0, 2.0])
        mapping = [[0, 1], []]
        result = aggregate_values(values, mapping)
        assert result[0] == 3.0
        assert np.isnan(result[1])


class TestAggregateMask:
    def test_all_true(self):
        mask = np.array([True, True, True, True])
        mapping = [[0, 1], [2, 3]]
        result = aggregate_mask(mask, mapping)
        np.testing.assert_array_equal(result, [True, True])

    def test_partial_false(self):
        """If any source is excluded, target is excluded."""
        mask = np.array([True, False, True, True])
        mapping = [[0, 1], [2, 3]]
        result = aggregate_mask(mask, mapping)
        np.testing.assert_array_equal(result, [False, True])

    def test_empty_group(self):
        mask = np.array([True])
        mapping = [[0], []]
        result = aggregate_mask(mask, mapping)
        assert result[0]
        assert not result[1]


# ---------------------------------------------------------------------------
# segment_stimulus
# ---------------------------------------------------------------------------


class TestSegmentStimulus:
    def test_creates_new_layer(self):
        stim = Stimulus(text="hello world", stimulus_id="s0")
        seg = CharacterSegmenter()
        layer_name = segment_stimulus(stim, seg)
        assert layer_name == "characters"
        assert "characters" in stim.layers
        # 10 non-whitespace characters
        assert len(stim.layers["characters"]) == 10

    def test_reuses_matching_layer(self):
        """If existing layer has identical spans, reuse it."""
        stim = Stimulus(text="hello world", stimulus_id="s0")
        spans = [Span(0, 5), Span(6, 11)]
        stim.add_layer(AnnotationLayer("words", spans, ["hello", "world"]))
        seg = WhitespaceSegmenter()
        layer_name = segment_stimulus(stim, seg)
        assert layer_name == "words"
        # Should not have added a new layer
        assert len(stim.layers) == 1

    def test_conflict_uses_different_name(self):
        """If existing layer has different spans, use _seg_ prefix."""
        stim = Stimulus(text="hello world", stimulus_id="s0")
        # Pre-add a units layer that treats the whole text as one unit
        stim.add_layer(AnnotationLayer("words", [Span(0, 11)], ["hello world"]))
        seg = WhitespaceSegmenter()
        layer_name = segment_stimulus(stim, seg)
        assert layer_name == "_seg_words"
        assert "_seg_words" in stim.layers
        assert len(stim.layers["_seg_words"]) == 2

    def test_extract_strings(self):
        stim = Stimulus(text="the cat sat", stimulus_id="s0")
        seg = WhitespaceSegmenter()
        layer_name = segment_stimulus(stim, seg)
        layer = stim.layers[layer_name]
        strings = layer.extract_strings(stim.text)
        assert strings == ["the", "cat", "sat"]

    def test_conflict_preserves_original_layer(self):
        """Conflicting segmenter creates _seg_ layer but leaves original intact."""
        stim = Stimulus(text="hello world", stimulus_id="s0")
        stim.add_layer(AnnotationLayer("words", [Span(0, 11)], ["hello world"]))
        seg = WhitespaceSegmenter()
        segment_stimulus(stim, seg)
        # Original layer unchanged
        assert len(stim.layers["words"]) == 1
        assert stim.layers["words"].spans == [Span(0, 11)]
        # New layer added
        assert len(stim.layers["_seg_words"]) == 2

    def test_multiple_segmenters_on_same_stimulus(self):
        """Applying two different segmenters creates two layers."""
        stim = Stimulus(text="hello world", stimulus_id="s0")
        ws_name = segment_stimulus(stim, WhitespaceSegmenter())
        ch_name = segment_stimulus(stim, CharacterSegmenter())
        assert ws_name == "words"
        assert ch_name == "characters"
        assert len(stim.layers) == 2

    def test_empty_text(self):
        """Segmenting empty text creates an empty layer."""
        stim = Stimulus(text="", stimulus_id="s0")
        layer_name = segment_stimulus(stim, WhitespaceSegmenter())
        assert layer_name == "words"
        assert len(stim.layers["words"]) == 0


# ---------------------------------------------------------------------------
# End-to-end: segmentation with evaluate()
# ---------------------------------------------------------------------------


class TestEvaluateWithSegmentation:
    """Test evaluate() integration with segmentation parameter."""

    def _make_word_dataset(self, n_stimuli=4, n_words=5):
        """Create a word-level SPR dataset."""
        return make_toy_dataset(
            n_stimuli=n_stimuli,
            n_words=n_words,
            dataset_name="toy_seg",
            seed=7,
        )

    @staticmethod
    def _make_const_lm():
        class ConstLM:
            name = "const"
            vocab = ["a"]

            def __call__(self, stimulus, extract=None, target_layer="words"):
                n = len(stimulus.layers[target_layer])
                return [
                    FeatureModelOutput(
                        values={"log_p_observed": np.nan if t == 0 else -1.0}
                    )
                    for t in range(n)
                ]

        return ConstLM()

    def test_no_segmentation_uses_native_layer(self):
        """segmentation=None uses the data's native 'words' layer."""
        from replica.analysis import OLS, evaluate
        from replica.analysis.specification.config import AnalysisConfig
        from replica.analysis.specification.evaluation import EvaluationSpec
        from replica.feature import ModelFeature, Surprisal

        ds = self._make_word_dataset()
        results = evaluate(
            ds,
            EvaluationSpec(
                predictors=[ModelFeature(self._make_const_lm(), Surprisal())],
                targets=[ds.feature("spr_rt")],
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), n_folds=2
            ),
        )
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)

    def test_matching_segmenter_works(self):
        """WhitespaceSegmenter on word-level data works (exact span match)."""
        from replica.analysis import OLS, evaluate
        from replica.analysis.specification.config import AnalysisConfig
        from replica.analysis.specification.evaluation import EvaluationSpec
        from replica.feature import ModelFeature, Surprisal

        ds = self._make_word_dataset()
        results = evaluate(
            ds,
            EvaluationSpec(
                predictors=[ModelFeature(self._make_const_lm(), Surprisal())],
                targets=[ds.feature("spr_rt")],
                segmentation=WhitespaceSegmenter(),
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), n_folds=2
            ),
        )
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)

    def test_coarser_segmentation_aggregates(self):
        """Half-sentence segmentation sums word-level SPR RT (additive)."""
        from replica.analysis import OLS, evaluate
        from replica.analysis.specification.config import AnalysisConfig
        from replica.analysis.specification.evaluation import EvaluationSpec

        # Build dataset with 6-word stimuli: "aa bb cc dd ee ff"
        # Segmented into 3 phrases of 2 words each -> enough units after
        # the first-unit exclusion.
        from replica.dataset import DatasetInfo, SPRDataset
        from replica.feature import ModelFeature, Surprisal

        class ToyDS(SPRDataset):
            info = DatasetInfo(name="toy_agg")

            def load(self, path):
                pass

        ds = ToyDS()
        rng = np.random.default_rng(99)
        for i in range(8):
            stim = word_stimulus("aa bb cc dd ee ff", f"s{i}")
            ds._stimuli.append(stim)
            ds._data.append(
                Datum(
                    stimulus=stim,
                    measure_name="spr_rt",
                    values=rng.uniform(200, 600, size=6),
                    participant_id="p0",
                )
            )

        # Phrase segmenter: split on spaces between word pairs
        # "aa bb | cc dd | ee ff" -> 3 phrases
        class PhraseSegmenter(Segmenter):
            def __init__(self):
                super().__init__("phrases")

            def segment(self, text: str) -> list[Span]:
                words = text.split()
                spans, pos = [], 0
                for j in range(0, len(words), 2):
                    chunk = words[j : j + 2]
                    start = text.index(chunk[0], pos)
                    end = text.index(chunk[-1], pos) + len(chunk[-1])
                    spans.append(Span(start, end))
                    pos = end
                return spans

        results = evaluate(
            ds,
            EvaluationSpec(
                predictors=[ModelFeature(self._make_const_lm(), Surprisal())],
                targets=[ds.feature("spr_rt")],
                segmentation=PhraseSegmenter(),
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), n_folds=2
            ),
        )
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)

    def test_multiple_data_layers_no_segmentation_raises(self):
        """evaluate() without segmentation on multi-layer data raises."""
        from replica.analysis import OLS, evaluate
        from replica.analysis.specification.config import AnalysisConfig
        from replica.analysis.specification.evaluation import EvaluationSpec
        from replica.dataset import DatasetInfo, SPRDataset
        from replica.feature import ModelFeature, Surprisal

        class ToyDS(SPRDataset):
            info = DatasetInfo(name="toy_multi")

            def load(self, path):
                pass

        ds = ToyDS()
        text = "ab cd"
        # Stimulus with both "words" and "characters" layers
        stim = Stimulus(text=text, stimulus_id="s0")
        stim.add_layer(
            AnnotationLayer(
                "words",
                [Span(0, 2), Span(3, 5)],
                ["ab", "cd"],
            )
        )
        stim.add_layer(
            AnnotationLayer(
                "characters",
                [Span(0, 1), Span(1, 2), Span(3, 4), Span(4, 5)],
                ["a", "b", "c", "d"],
            )
        )
        ds._stimuli.append(stim)
        # Data on "words" layer
        ds._data.append(
            Datum(
                stimulus=stim,
                measure_name="spr_rt",
                values=np.array([300.0, 400.0]),
                participant_id="p0",
            )
        )
        # Data on "characters" layer
        ds._data.append(
            Datum(
                stimulus=stim,
                measure_name="spr_rt",
                values=np.array([100.0, 200.0, 150.0, 250.0]),
                participant_id="p1",
                layer="characters",
            )
        )

        with pytest.raises(ValueError, match="multiple annotation layers"):
            evaluate(
                ds,
                EvaluationSpec(
                    predictors=[ModelFeature(self._make_const_lm(), Surprisal())],
                    targets=[ds.feature("spr_rt")],
                ),
                analysis=AnalysisConfig(
                    design=None, transform_dv=np.log2, model=OLS(), n_folds=2
                ),
            )

    def test_exclusion_mask_aggregated_correctly(self):
        """Exclusion mask at word level propagates through aggregation."""
        # If first word is excluded (as per SPR default), and the segmenter
        # aggregates 2 words per sentence, the first sentence should be excluded.
        from replica.dataset import DatasetInfo, SPRDataset

        class ToyDS(SPRDataset):
            info = DatasetInfo(name="toy_excl")

            def load(self, path):
                pass

        ds = ToyDS()
        stim = word_stimulus("aa bb cc dd", "s0")
        ds._stimuli.append(stim)
        ds._data.append(
            Datum(
                stimulus=stim,
                measure_name="spr_rt",
                values=np.array([300.0, 400.0, 500.0, 600.0]),
                participant_id="p0",
            )
        )

        # Word-level mask: first word excluded by SPR default
        mask = ds.exclusion_mask(ds._data[0])
        assert not mask[0]  # first word excluded

        # 2-word sentence segmentation: [0:5) = "aa bb", [6:11) = "cc dd"
        half_seg = from_predicate(lambda t, i: t[i] == " " and i == 5, "phrases")
        phrase_spans = half_seg(stim.text)
        assert len(phrase_spans) == 2

        mapping = compute_refinement(stim.layers["words"].spans, phrase_spans)
        agg_mask = aggregate_mask(mask, mapping)
        # First phrase contains excluded word -> excluded
        assert not agg_mask[0]
        # Second phrase is fully included
        assert agg_mask[1]


def test_repeated_segmenter_name_preserves_every_existing_layer():
    class Fixed(Segmenter):
        def __init__(self, spans):
            super().__init__("parts")
            self.spans = spans

        def segment(self, text):
            return self.spans

    stimulus = Stimulus("abc")
    layers = [
        segment_stimulus(stimulus, Fixed(spans))
        for spans in [[Span(0, 1)], [Span(0, 2)], [Span(0, 3)]]
    ]
    assert len(set(layers)) == 3
    assert stimulus.layers[layers[1]].spans == [Span(0, 2)]
    assert segment_stimulus(stimulus, Fixed([Span(0, 2)])) == layers[1]


class _WholeText(Segmenter):
    """One unit spanning the whole text, under a registered name."""

    def __init__(self, name="whole"):
        super().__init__(name)

    def segment(self, text):
        return [Span(0, len(text))]


@pytest.fixture
def registered(monkeypatch):
    """A segmenter registered as ``whole`` for one test."""
    from replica import segmentation

    monkeypatch.setattr(segmentation, "SEGMENTERS", dict(segmentation.SEGMENTERS))
    segmentation.register_segmenter("whole")(_WholeText)
    return segmentation


def test_registered_segmentations_attach_their_layer_once(registered):
    stimuli = [word_stimulus("ab cd", "s1"), word_stimulus("ef", "s2")]
    registered.attach_segmentation("whole", stimuli)
    registered.attach_segmentation("whole", stimuli)
    assert [s.unit_strings("whole") for s in stimuli] == [["ab cd"], ["ef"]]
    assert all(not any(n.startswith("_seg_") for n in s.layers) for s in stimuli)


def test_a_conflicting_layer_of_the_registered_name_is_refused(registered):
    stimulus = word_stimulus("ab cd", "s1")
    stimulus.add_layer(AnnotationLayer("whole", [Span(0, 2)], ["ab"]))
    with pytest.raises(ValueError, match="exists with other spans"):
        registered.attach_segmentation("whole", [stimulus])
    assert set(stimulus.layers) == {"words", "whole"}


def test_reserved_duplicate_and_malformed_names_are_refused(registered):
    for name in ("words", "characters", "tokens", "Whole", "a_b", "-x"):
        with pytest.raises(ValueError):
            registered.register_segmenter(name)
    with pytest.raises(ValueError, match="already registered"):
        registered.register_segmenter("whole")(_WholeText)
    with pytest.raises(KeyError):
        registered.attach_segmentation("absent", [])
