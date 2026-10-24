"""Text validation respects source measurement regions."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replica.core import AnnotationLayer, Datum, Span, Stimulus
from replica.dataset.validation import stim_content_issues


@pytest.mark.parametrize("same_regions", [False, True])
def test_identical_text_is_duplicate_only_with_the_same_measured_regions(same_regions):
    text = "a b"
    spans = [Span(0, 1), Span(2, 3)]
    stimuli = []
    datums = []
    for identifier, regions in [
        ("first", spans),
        ("second", spans if same_regions else [Span(0, 3)]),
    ]:
        stimulus = Stimulus(
            text=text,
            stimulus_id=identifier,
            layers={
                "words": AnnotationLayer(
                    "words", regions, [text[s.start : s.end] for s in regions]
                )
            },
        )
        stimuli.append(stimulus)
        datums.append(Datum(stimulus, "total_time", np.ones(len(regions))))
    dataset = SimpleNamespace(stimuli=lambda: stimuli, data=lambda: datums)
    report = stim_content_issues(dataset)
    assert report.counts == ({"duplicate_stim_text": 1} if same_regions else {})


@pytest.mark.parametrize("separator", [" ", "\u00a0", "\u202f"])
def test_horizontal_spaces_are_valid_inside_a_measured_region(separator):
    text = f"a{separator}b"
    stimulus = Stimulus(
        text,
        layers={"words": AnnotationLayer("words", [Span(0, len(text))], [text])},
    )
    datum = Datum(stimulus, "total_time", np.array([100.0]))
    dataset = SimpleNamespace(stimuli=lambda: [stimulus], data=lambda: [datum])
    assert stim_content_issues(dataset).counts == {}


@pytest.mark.parametrize(
    "text,regions,expected",
    [
        ('a "b"', [Span(0, 1), Span(3, 4)], {}),
        ("a b", [Span(0, 1)], {"text_not_covered_by_units": 1}),
    ],
)
def test_unmeasured_punctuation_does_not_hide_uncovered_letters(
    text, regions, expected
):
    stimulus = Stimulus(
        text,
        layers={
            "words": AnnotationLayer(
                "words", regions, [text[s.start : s.end] for s in regions]
            )
        },
    )
    datum = Datum(stimulus, "total_time", np.ones(len(regions)))
    dataset = SimpleNamespace(stimuli=lambda: [stimulus], data=lambda: [datum])
    assert stim_content_issues(dataset).counts == expected


@pytest.mark.needs_data
def test_zuco_repeated_text_retains_the_released_trial_identities(data_root):
    import h5py

    from replica.dataset import ZuCo2

    root = Path(data_root) / "eye-tracking/original/zuco2"
    source = root / "resultsYAC_NR.mat"
    if not source.exists():
        pytest.skip("ZuCo 2 NR release unavailable")
    source_trials = defaultdict(list)
    with h5py.File(source) as handle:
        for index, reference in enumerate(handle["sentenceData/content"][()].ravel()):
            text = (
                np.asarray(handle[reference][()], dtype="<u2")
                .tobytes()
                .decode("utf-16le")
            )
            source_trials[text].append(index)
    expected = sorted(ids for ids in source_trials.values() if len(ids) > 1)
    assert expected

    dataset = ZuCo2()
    dataset.load(str(root), max_subjects=1, load_eeg=False)
    loaded_trials = defaultdict(list)
    for stimulus in dataset.stimuli():
        index = int(stimulus.stimulus_id.rsplit("_", 1)[1])
        loaded_trials[stimulus.text].append(index)
    repeated = sorted(sorted(ids) for ids in loaded_trials.values() if len(ids) > 1)
    assert repeated == expected
