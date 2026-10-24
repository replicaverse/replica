"""Unit tests for ``dataset.reaggregation.aggregate_fixations_to_layer``.

These exercise the corner cases that the cross-corpus round-trip tests
don't cover (empty input, NaN duration, regressions, re-fixations
within a run, off-text fixations, cross-layer aggregation via
char_offset, run_id-vs-cursor disagreement).
"""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.dataset.contracts import Fixation
from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer
from tests._stimuli import word_stimulus


def _stimulus(words: list[str], stimulus_id: str = "s") -> Stimulus:
    """Build a minimal stimulus whose ``words`` layer holds *words*."""
    return word_stimulus(" ".join(words), stimulus_id)


def _fix(
    *,
    unit_index=None,
    char_offset=None,
    duration=200.0,
    start_time=0.0,
    end_time=200.0,
    fixation_index=1,
    metadata=None,
    char_offset_is_exact=False,
    unit_layer="words",
):
    return Fixation(
        participant_id="p",
        stimulus_id="s",
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        x=0.0,
        y=0.0,
        unit_index=unit_index,
        char_offset=char_offset,
        fixation_index=fixation_index,
        metadata=metadata or {},
        char_offset_is_exact=char_offset_is_exact,
        unit_layer=unit_layer,
    )


def test_empty_input_returns_empty_arrays():
    stim = _stimulus(["a", "b", "c"])
    out = aggregate_fixations_to_layer([], stim, "words")
    assert np.all(np.isnan(out["first_fixation"]))
    assert np.all(out["first_run"] == 0.0)
    assert np.all(out["first_pass"] == 0.0)
    assert np.all(out["total_time"] == 0.0)
    assert out["first_fixation"].shape == (3,)


def test_repeated_visits_do_not_interleave_their_trial_clocks():
    stimulus = _stimulus(["one", "two"])
    events = [
        _fix(
            unit_index=0,
            char_offset=0,
            duration=100,
            start_time=0,
            metadata={"trial_index": 1, "run_id": 1},
        ),
        _fix(
            unit_index=1,
            char_offset=4,
            duration=200,
            start_time=100,
            metadata={"trial_index": 1, "run_id": 1},
        ),
        _fix(
            unit_index=1,
            char_offset=4,
            duration=300,
            start_time=0,
            metadata={"trial_index": 2, "run_id": 1},
        ),
        _fix(
            unit_index=0,
            char_offset=0,
            duration=400,
            start_time=300,
            metadata={"trial_index": 2, "run_id": 1},
        ),
    ]
    result = aggregate_fixations_to_layer(events, stimulus)
    np.testing.assert_equal(result["first_fixation"], [100, 200])
    np.testing.assert_equal(result["first_run"], [100, 200])
    np.testing.assert_equal(result["total_time"], [500, 500])


def test_source_word_run_ids_do_not_control_a_larger_region():
    stimulus = _stimulus(["one", "two"])
    stimulus.add_layer(AnnotationLayer("sentence", [Span(0, 7)], ["one two"]))
    events = [
        _fix(unit_index=0, char_offset=0, duration=100, metadata={"run_id": 1}),
        _fix(unit_index=1, char_offset=4, duration=200, metadata={"run_id": 1}),
        _fix(unit_index=0, char_offset=0, duration=300, metadata={"run_id": 2}),
    ]
    result = aggregate_fixations_to_layer(events, stimulus, "sentence")
    np.testing.assert_equal(result["first_run"], [600])


def test_mixed_trial_identity_is_rejected():
    stimulus = _stimulus(["one"])
    events = [
        _fix(unit_index=0, char_offset=0),
        _fix(unit_index=0, char_offset=0, metadata={"trial_index": 2}),
    ]
    with pytest.raises(ValueError, match="with and without trial_index"):
        aggregate_fixations_to_layer(events, stimulus)


def test_zero_unit_layer():
    stim = Stimulus(text="", stimulus_id="empty", metadata={})
    stim.add_layer(AnnotationLayer("words", [], []))
    out = aggregate_fixations_to_layer([], stim, "words")
    for v in out.values():
        assert v.shape == (0,)


def test_missing_layer_raises():
    stim = _stimulus(["a"])
    with pytest.raises(KeyError):
        aggregate_fixations_to_layer([_fix(unit_index=0)], stim, "missing")


def test_single_first_pass_fixation():
    """One fixation on word 0 → first_fixation=200, first_run=200, total=200."""
    stim = _stimulus(["the", "cat", "sat"])
    out = aggregate_fixations_to_layer(
        [_fix(unit_index=0, char_offset=0, duration=200, start_time=0)],
        stim,
        "words",
    )
    assert out["first_fixation"][0] == 200
    assert out["first_run"][0] == 200
    assert out["first_pass"][0] == 1
    assert out["total_time"][0] == 200
    # Other words untouched
    assert np.isnan(out["first_fixation"][1])
    assert out["first_run"][1] == 0
    assert out["first_pass"][1] == 0
    assert out["total_time"][1] == 0


def test_re_fixation_within_first_run():
    """Two consecutive fixations on word 0, both in first run.

    first_fixation = duration of the first fixation only.
    first_run      = sum of both.
    total_time     = sum of both.
    """
    stim = _stimulus(["the", "cat"])
    fixes = [
        _fix(unit_index=0, char_offset=0, duration=180, start_time=0),
        _fix(unit_index=0, char_offset=0, duration=120, start_time=200),
    ]
    out = aggregate_fixations_to_layer(fixes, stim, "words")
    assert out["first_fixation"][0] == 180
    assert out["first_run"][0] == 300
    assert out["total_time"][0] == 300


def test_regression_excludes_from_first_run():
    """Fixation order: word 0, word 1, word 0 (regression).

    The regression fixation on word 0 contributes to total_time but
    not first_run, and the flag says word 0 was reached in first pass.
    """
    stim = _stimulus(["the", "cat", "sat"])
    fixes = [
        _fix(unit_index=0, char_offset=0, duration=200, start_time=0),
        _fix(unit_index=1, char_offset=4, duration=180, start_time=210),
        _fix(unit_index=0, char_offset=0, duration=150, start_time=400),
    ]
    out = aggregate_fixations_to_layer(fixes, stim, "words")
    assert out["first_fixation"][0] == 200
    assert out["first_run"][0] == 200  # regression not counted
    assert out["first_pass"][0] == 1
    assert out["total_time"][0] == 350  # regression counted in total
    assert out["first_run"][1] == 180
    assert out["first_pass"][1] == 1
    assert out["total_time"][1] == 180


def test_regression_entry_has_a_first_run_but_no_first_pass():
    """Fixation order: word 1, then word 0. Word 0 is first entered by a
    regression, so its first run is the 150 ms visit and its flag is 0."""
    stim = _stimulus(["the", "cat", "sat"])
    fixes = [
        _fix(unit_index=1, char_offset=4, duration=180, start_time=0),
        _fix(unit_index=0, char_offset=0, duration=150, start_time=200),
    ]
    out = aggregate_fixations_to_layer(fixes, stim, "words")
    assert out["first_run"][0] == 150
    assert out["first_pass"][0] == 0
    assert out["first_run"][1] == 180
    assert out["first_pass"][1] == 1


def test_explicit_run_id_overrides_cursor():
    """If metadata['run_id'] is set, it takes precedence over cursor inference.

    EyeLink's RUN_ID is the source of truth in PROVO/MECO_char; the
    cursor inference is only a fallback.
    """
    stim = _stimulus(["the", "cat"])
    fixes = [
        _fix(unit_index=0, char_offset=0, duration=100, metadata={"run_id": 1}),
        # Cursor inference would say "first-pass" (still on word 0,
        # never moved), but explicit run_id=2 marks this as a re-read.
        _fix(unit_index=0, char_offset=0, duration=200, metadata={"run_id": 2}),
    ]
    out = aggregate_fixations_to_layer(fixes, stim, "words")
    assert out["first_fixation"][0] == 100
    assert out["first_run"][0] == 100  # only run_id=1 counted
    assert out["total_time"][0] == 300


def test_off_text_fixation_is_skipped():
    """A fixation with char_offset between spans (whitespace) → unit_index=None
    via lookup → skipped from all aggregations, doesn't disrupt cursor."""
    stim = _stimulus(["the", "cat"])  # text = "the cat", spans (0,3) and (4,7)
    fixes = [
        _fix(unit_index=0, char_offset=0, duration=200),  # on "the"
        _fix(
            unit_index=None, char_offset=3, duration=80, char_offset_is_exact=True
        ),  # whitespace
        _fix(unit_index=1, char_offset=4, duration=180),  # on "cat"
    ]
    out = aggregate_fixations_to_layer(fixes, stim, "words")
    # The off-text fixation must not be counted on either word.
    assert out["total_time"][0] == 200
    assert out["total_time"][1] == 180
    # And it must not break first-run inference for word 1.
    assert out["first_run"][1] == 180


@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_duration_preserves_arrival_and_unknown_sums(missing):
    stim = _stimulus(["the", "cat", "sat"])
    fixes = [
        _fix(unit_index=1, char_offset=4, duration=missing),
        _fix(unit_index=1, char_offset=4, duration=80),
        _fix(unit_index=0, char_offset=0, duration=100),
    ]
    out = aggregate_fixations_to_layer(fixes, stim)
    np.testing.assert_equal(out["first_pass"], [0, 1, 0])
    np.testing.assert_equal(out["first_fixation"], [100, np.nan, np.nan])
    np.testing.assert_equal(out["first_run"], [100, np.nan, 0])
    np.testing.assert_equal(out["total_time"], [100, np.nan, 0])


@pytest.mark.parametrize("duration", [-1, np.inf])
def test_invalid_fixation_duration_raises(duration):
    with pytest.raises(ValueError, match="nonnegative or missing"):
        aggregate_fixations_to_layer(
            [_fix(unit_index=0, char_offset=0, duration=duration)], _stimulus(["word"])
        )


@pytest.mark.parametrize("layer", ["words", "region"])
def test_source_unit_without_offset_preserves_native_and_coarser_measures(layer):
    stimulus = _stimulus(["the", "cat", "sat"])
    stimulus.add_layer(
        AnnotationLayer("region", [Span(0, 7), Span(8, 11)], ["the cat", "sat"])
    )
    events = [
        _fix(unit_index=unit, duration=100, start_time=index * 100)
        for index, unit in enumerate([0, 1, 0, 2])
    ]
    result = aggregate_fixations_to_layer(events, stimulus, layer)
    if layer == "words":
        np.testing.assert_equal(result["first_fixation"], [100, 100, 100])
        np.testing.assert_equal(result["first_run"], [100, 100, 100])
        np.testing.assert_equal(result["go_past"], [100, 200, 100])
        np.testing.assert_equal(result["total_time"], [200, 100, 100])
    else:
        np.testing.assert_equal(result["first_fixation"], [100, 100])
        np.testing.assert_equal(result["first_run"], [300, 100])
        np.testing.assert_equal(result["go_past"], [300, 100])
        np.testing.assert_equal(result["total_time"], [300, 100])


@pytest.mark.parametrize("unit_index", [-1, 2])
def test_invalid_source_unit_without_offset_is_rejected(unit_index):
    with pytest.raises(ValueError, match="valid source unit_index"):
        aggregate_fixations_to_layer(
            [_fix(unit_index=unit_index)], _stimulus(["the", "cat"])
        )


def test_missing_source_layer_without_offset_is_rejected():
    with pytest.raises(KeyError, match="missing"):
        aggregate_fixations_to_layer(
            [_fix(unit_index=0, unit_layer="missing")], _stimulus(["the"])
        )


def test_exact_landing_requires_an_offset_for_assigned_units():
    with pytest.raises(ValueError, match="exact character landing requires"):
        aggregate_fixations_to_layer(
            [_fix(unit_index=0, char_offset_is_exact=True)], _stimulus(["the"])
        )


def test_cross_layer_via_char_offset():
    """A fixation with unit_index aligned to the words layer can still be
    aggregated to a characters layer, char_offset takes precedence over
    unit_index, so the lookup goes to the right span."""
    stim = _stimulus(["the", "cat"])  # text = "the cat"
    # Add a characters layer (each non-whitespace char is one span)
    char_spans = [Span(i, i + 1) for i, ch in enumerate(stim.text) if ch != " "]
    char_strings = [stim.text[s.start : s.end] for s in char_spans]
    stim.add_layer(AnnotationLayer("characters", char_spans, char_strings))

    fixes = [
        # unit_index points at WORD 0 ("the"), but char_offset says char 1 ('h')
        _fix(unit_index=0, char_offset=1, duration=180, char_offset_is_exact=True),
    ]
    out = aggregate_fixations_to_layer(fixes, stim, "characters")
    # 'h' (char index 1) should get the fixation, not 't' (char index 0).
    assert out["total_time"][0] == 0  # 't'
    assert out["total_time"][1] == 180  # 'h'
    assert out["total_time"][2] == 0  # 'e'


def test_unsorted_input_is_sorted_by_start_time():
    """The aggregator must reorder by start_time; cursor inference depends
    on temporal order."""
    stim = _stimulus(["the", "cat"])
    # Submit out of time order
    fixes = [
        _fix(unit_index=1, char_offset=4, duration=150, start_time=200),  # 2nd
        _fix(unit_index=0, char_offset=0, duration=180, start_time=0),  # 1st
    ]
    out = aggregate_fixations_to_layer(fixes, stim, "words")
    # After sorting, word 0 is first-pass (start_time=0), word 1 too.
    assert out["first_fixation"][0] == 180
    assert out["first_run"][0] == 180
    assert out["first_fixation"][1] == 150
    assert out["first_run"][1] == 150


def test_first_run_over_a_region_is_not_the_sum_of_word_first_runs():
    stimulus = _stimulus(["a", "b", "c"])
    stimulus.add_layer(
        AnnotationLayer("region", [Span(0, 3), Span(4, 5)], ["a b", "c"])
    )
    # Revisiting a starts a new word run but continues the run in region a b.
    fixations = [
        _fix(
            unit_index=offset // 2,
            char_offset=offset,
            duration=100,
            start_time=100 * index,
        )
        for index, offset in enumerate([0, 2, 0, 4])
    ]
    words = aggregate_fixations_to_layer(fixations, stimulus, "words")
    region = aggregate_fixations_to_layer(fixations, stimulus, "region")
    assert words["first_run"][:2].sum() == 200
    assert region["first_run"][0] == 300
    assert words["total_time"][:2].sum() == region["total_time"][0] == 300


@pytest.mark.parametrize("char_offset", [None, 0])
def test_word_anchor_cannot_establish_character_durations(char_offset):
    stimulus = _stimulus(["word"])
    stimulus.add_layer(
        AnnotationLayer("characters", [Span(i, i + 1) for i in range(4)], list("word"))
    )
    with pytest.raises(ValueError, match="splits source area"):
        aggregate_fixations_to_layer(
            [_fix(unit_index=0, char_offset=char_offset)], stimulus, "characters"
        )


@pytest.mark.parametrize(
    "spans",
    [[Span(1, 2), Span(0, 1)], [Span(0, 2), Span(1, 3)], [Span(0, 20)], [Span(0, 0)]],
)
def test_invalid_target_spans_are_rejected(spans):
    stimulus = _stimulus(["word"])
    stimulus.add_layer(AnnotationLayer("custom", spans, [""] * len(spans)))
    with pytest.raises(ValueError, match="Unit spans"):
        aggregate_fixations_to_layer([], stimulus, "custom")


def test_missing_duration_only_invalidates_go_past_windows_containing_it():
    stimulus = _stimulus(["a", "b", "c"])
    events = [
        _fix(unit_index=i, char_offset=2 * i, duration=d, start_time=i)
        for i, d in enumerate([100, np.nan, 200])
    ]
    result = aggregate_fixations_to_layer(events, stimulus)
    np.testing.assert_equal(result["go_past"], [100, np.nan, 200])


def test_same_numeric_indices_do_not_reuse_runs_after_merging_regions():
    stimulus = _stimulus(["a", "b", "c"])
    stimulus.add_layer(
        AnnotationLayer("region", [Span(0, 3), Span(4, 5)], ["a b", "c"])
    )
    events = [
        _fix(unit_index=0, char_offset=0, duration=100, metadata={"run_id": run})
        for run in [1, 2]
    ]
    assert (
        aggregate_fixations_to_layer(events, stimulus, "region")["first_run"][0] == 200
    )


def test_ambiguous_landings_resolve_inside_a_custom_unit():
    from dataclasses import replace

    stimulus = _stimulus(["one", "two", "three"])
    stimulus.add_layer(
        AnnotationLayer("regions", [Span(0, 7), Span(8, 13)], ["one two", "three"])
    )
    events = [
        replace(_fix(duration=100), candidate_unit_indices=(0, 1)),
        _fix(unit_index=1, duration=200, start_time=100),
        _fix(unit_index=2, duration=300, start_time=300),
    ]
    native = aggregate_fixations_to_layer(events, stimulus)
    assert np.isnan(native["first_fixation"][:2]).all()
    result = aggregate_fixations_to_layer(events, stimulus, "regions")
    for measure in ("first_run", "go_past", "right_bounded", "total_time"):
        np.testing.assert_equal(result[measure], [300, 300])
    np.testing.assert_equal(result["first_fixation"], [100, 300])
    assert events[0].candidate_unit_indices == (0, 1)
    assert events[0].unit_index is None


@pytest.mark.parametrize("off_text_ends_run", [False, True])
@pytest.mark.parametrize("trials", [(1, 1, 1, 1), (1, 1, 2, 2)])
def test_known_ambiguous_measures_agree_with_every_possible_landing(
    off_text_ends_run, trials
):
    from dataclasses import replace
    from itertools import permutations, product

    from replica.dataset.loading.fixation_utils import reading_measures_from_fixations

    # Exhaust all short paths around one uncertain event. Every reported value
    # must hold under both source interpretations; uncertain values may be NaN.
    for path in product((None, 0, 1, 2), repeat=3):
        for candidates in permutations((None, 0, 1, 2), 2):
            for position in range(4):
                events = [
                    _fix(unit_index=unit, duration=10 * (i + 1), fixation_index=i)
                    for i, unit in enumerate(path)
                ]
                events.insert(
                    position,
                    replace(_fix(duration=75), candidate_unit_indices=candidates),
                )
                events = [
                    replace(
                        event, fixation_index=i, metadata={"trial_index": trials[i]}
                    )
                    for i, event in enumerate(events)
                ]
                result = reading_measures_from_fixations(
                    events, 3, off_text_ends_run=off_text_ends_run
                )
                for candidate in candidates:
                    resolved = list(events)
                    resolved[position] = replace(
                        events[position],
                        unit_index=candidate,
                        candidate_unit_indices=(),
                    )
                    expected = reading_measures_from_fixations(
                        resolved, 3, off_text_ends_run=off_text_ends_run
                    )
                    for measure, values in result.items():
                        known = np.isfinite(values)
                        np.testing.assert_equal(
                            values[known],
                            expected[measure][known],
                            err_msg=f"{path=} {candidates=} {position=} {measure=}",
                        )


def test_uncertainty_in_a_later_visit_does_not_reopen_an_earlier_run():
    from dataclasses import replace

    stimulus = _stimulus(["one", "two"])
    events = [
        _fix(unit_index=0, duration=100, metadata={"trial_index": 1}),
        replace(
            _fix(duration=200, metadata={"trial_index": 2}),
            candidate_unit_indices=(0, 1),
        ),
    ]
    result = aggregate_fixations_to_layer(events, stimulus)
    for measure in ("first_fixation", "first_run", "go_past"):
        assert result[measure][0] == 100
    assert np.isnan(result["total_time"][0])


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_unit_indices": (0,)},
        {"candidate_unit_indices": (0, 0)},
        {"candidate_unit_indices": (0, 3)},
        {"candidate_unit_indices": (0, 1), "unit_index": 0},
        {"candidate_unit_indices": (0, 1), "char_offset": 0},
        {"candidate_unit_indices": (0, 1), "char_offset_is_exact": True},
    ],
)
def test_invalid_alternative_landings_are_rejected(changes):
    from dataclasses import replace

    with pytest.raises(ValueError, match="fixation|Fixation"):
        aggregate_fixations_to_layer(
            [replace(_fix(), **changes)], _stimulus(["one", "two"])
        )


@pytest.mark.parametrize("require_exit", [False, True])
def test_go_past_right_exit_policy_preserves_closed_paths_and_trial_boundaries(
    require_exit,
):
    stimulus = _stimulus(["one", "two", "three"])
    events = [
        _fix(
            unit_index=unit,
            duration=duration,
            fixation_index=i,
            metadata={"trial_index": trial},
        )
        for i, (unit, duration, trial) in enumerate(
            [(0, 100, 1), (1, 200, 1), (0, 300, 1), (2, 400, 2)], 1
        )
    ]
    arrays = aggregate_fixations_to_layer(
        events, stimulus, go_past_requires_right_exit=require_exit
    )
    np.testing.assert_equal(
        arrays["go_past"],
        [100, np.nan if require_exit else 500, np.nan if require_exit else 400],
    )
    np.testing.assert_equal(arrays["total_time"], [400, 200, 400])
    stimulus.add_layer(
        AnnotationLayer("text", [Span(0, len(stimulus.text))], [stimulus.text])
    )
    whole = aggregate_fixations_to_layer(
        events, stimulus, "text", go_past_requires_right_exit=require_exit
    )
    np.testing.assert_equal(whole["go_past"], [np.nan if require_exit else 600])
    np.testing.assert_equal(whole["total_time"], [1000])


def test_known_whitespace_landings_keep_order_and_regression_dwell():
    stimulus = _stimulus(["one", "two", "three"])
    stimulus.add_layer(
        AnnotationLayer(
            "areas",
            [Span(0, 3), Span(3, 4), Span(4, 7), Span(7, 8), Span(8, 13)],
            ["one", " ", "two", " ", "three"],
        )
    )
    events = [
        _fix(unit_index=unit, unit_layer="areas", duration=duration, fixation_index=i)
        for i, (unit, duration) in enumerate(
            [(2, 100), (1, 200), (0, 300), (2, 400), (4, 500)], 1
        )
    ]
    arrays = aggregate_fixations_to_layer(
        events, stimulus, include_inter_unit_fixations=True
    )
    np.testing.assert_equal(arrays["go_past"], [300, 1000, 500])
    np.testing.assert_equal(arrays["first_pass"], [0, 1, 1])
    np.testing.assert_equal(arrays["total_time"], [300, 500, 500])
    events = [
        _fix(unit_index=1, unit_layer="areas", duration=200, fixation_index=1),
        _fix(unit_index=0, unit_layer="areas", duration=300, fixation_index=2),
    ]
    arrays = aggregate_fixations_to_layer(
        events, stimulus, include_inter_unit_fixations=True
    )
    np.testing.assert_equal(arrays["first_pass"], [0, 0, 0])
    assert set(stimulus.layers) == {"words", "areas"}
