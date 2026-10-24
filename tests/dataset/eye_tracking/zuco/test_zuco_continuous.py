"""Continuous ZuCo recordings from both MATLAB layouts.

One synthetic trial covers the release's rules: a fixation too short to be
assigned, a fixation on the edge of two rectangles, a refixation, a fixation
that outlasts the trial, and a fixation off the text.
"""

import itertools
import logging

import numpy as np
import pytest
from scipy.io import savemat

from replica.dataset.eeg.zuco.zuco1 import ZuCo1EEG
from replica.dataset.eeg.zuco.zuco2 import ZuCo2EEG
from replica.dataset.eye_tracking.zuco.base import EEG_BANDS, N_ELECTRODES
from replica.dataset.eye_tracking.zuco.continuous import (
    SAMPLE_RATE_HZ,
    FragmentMismatch,
    WordFragments,
    locate_fragment,
    sentence_recordings,
    word_fixation_pairs,
)
from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1
from replica.dataset.eye_tracking.zuco.zuco2 import ZuCo2
from tests._stimuli import word_stimulus

CHANNELS = 3
WORDS = ("ab", "cd", "ef", "gh")
MEASURES = ("first_fixation", "first_run", "go_past", "total_time")
BOUNDS = np.array([[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10], [30, 0, 40, 10]])
#: x, y and duration (samples) of the trial's fixations, in trial order.
FIXATIONS = {
    "x": np.array([5.0, 10.0, 25.0, 15.0, 35.0, 50.0]),
    "y": np.array([5.0] * 6),
    "duration": np.array([30.0, 60.0, 80.0, 70.0, 90.0, 100.0]),
}
#: Onset sample of each fixation the release assigns to a word; the last one
#: outlasts the 400-sample trial.
ONSETS = {1: 100, 2: 180, 3: 270, 4: 360}
N_SAMPLES = 400


def _gaze(fixation, onset, samples):
    clock = 1000.0 + 2.0 * (onset + np.arange(samples))
    x = np.full(samples, FIXATIONS["x"][fixation])
    y = np.full(samples, FIXATIONS["y"][fixation])
    pupil = np.full(samples, 3000.0 + fixation)
    return np.column_stack([clock, x, y, pupil])


def _trial(seed=0):
    """Return the trial's EEG and the released fragments under each word."""
    rng = np.random.default_rng(seed)
    eeg = rng.standard_normal((N_SAMPLES, CHANNELS))
    overrun = rng.standard_normal((51, CHANNELS))
    fragments = {}
    for fixation, onset in ONSETS.items():
        samples = int(FIXATIONS["duration"][fixation]) + 1
        fragment = np.vstack([eeg[onset : onset + samples], overrun])[:samples]
        fragments[fixation] = (fragment, _gaze(fixation, onset, samples))
    # Pairs in release order: (1, 0), (1, 1), (2, 2), (3, 1), (4, 3).
    words = [
        WordFragments((fragments[1][0],), (fragments[1][1],), (1,)),
        WordFragments(
            (fragments[1][0], fragments[3][0]),
            (fragments[1][1], fragments[3][1]),
            (2, 4),
        ),
        WordFragments((fragments[2][0],), (fragments[2][1],), (3,)),
        WordFragments((fragments[4][0],), (fragments[4][1],), (5,)),
    ]
    return eeg, words


EXPECTED_TABLE = {
    "fixation_index": [2, 2, 3, 4, 5],
    "unit_index": [0, 1, 2, 1, 3],
    "onset_sample": [100, 100, 180, 270, 360],
    "samples": [61, 61, 81, 71, 91],
    "truncated": [False, False, False, False, True],
}
EXPECTED_EVENTS = np.array([100, 100, 180, 360]) / SAMPLE_RATE_HZ


def test_word_fixation_pairs_follow_the_release_rule():
    inside, fixations, words = word_fixation_pairs(
        FIXATIONS["x"], FIXATIONS["y"], FIXATIONS["duration"], BOUNDS.astype(float)
    )
    assert inside.shape == (6, 4)
    assert inside[1].tolist() == [True, True, False, False]
    assert fixations.tolist() == [1, 1, 2, 3, 4]
    assert words.tolist() == [0, 1, 2, 1, 3]


def test_locate_fragment_prefers_a_whole_match_and_reports_the_tail():
    eeg, _ = _trial()
    assert locate_fragment(eeg, eeg[180:261]) == (180, False)
    overrun = np.vstack([eeg[360:], np.ones((5, CHANNELS))])
    assert locate_fragment(eeg, overrun) == (360, True)


@pytest.mark.parametrize(
    "recording, fragment, message",
    [
        (
            np.tile(np.arange(3.0)[:, None], (2, 1)),
            np.arange(2.0)[:, None],
            "occurs 2 times",
        ),
        (np.arange(4.0)[:, None], np.array([[7.0], [8.0]]), "occurs nowhere"),
        (
            np.array([[1.0], [2.0], [1.0], [2.0], [1.0]]),
            np.array([[1.0], [2.0], [1.0], [3.0]]),
            "2 ways",
        ),
    ],
)
def test_locate_fragment_rejects_ambiguous_or_absent_fragments(
    recording, fragment, message
):
    with pytest.raises(ValueError, match=message):
        locate_fragment(recording, fragment)


def _recordings(eeg, words, **overrides):
    arguments = {
        "stimulus": word_stimulus(" ".join(WORDS), "s"),
        "participant_id": "p",
        "eeg": eeg,
        "words": words,
        "fixations": FIXATIONS,
        "bounds": BOUNDS.astype(float),
        "metadata": {"task": "NR"},
    }
    arguments.update(overrides)
    return sentence_recordings(**arguments)


def test_sentence_recordings_time_every_located_fixation():
    eeg, words = _trial()
    eeg_recording, gaze = _recordings(eeg, words)
    assert eeg_recording.values is eeg
    assert eeg_recording.sample_interval == gaze.sample_interval == 1 / SAMPLE_RATE_HZ
    assert (eeg_recording.modality, gaze.modality) == ("eeg", "eye-tracking")
    assert (eeg_recording.measure_name, gaze.measure_name) == ("eeg", "gaze")
    for recording in (eeg_recording, gaze):
        np.testing.assert_allclose(recording.event_times, EXPECTED_EVENTS)
        assert recording.metadata == {"task": "NR", "fixations": EXPECTED_TABLE}
    # The two recordings own their onsets and their tables.
    gaze.event_times[0] = -1.0
    gaze.metadata["fixations"]["samples"][0] = -1
    assert eeg_recording.event_times[0] == EXPECTED_EVENTS[0]
    assert eeg_recording.metadata["fixations"]["samples"][0] == 61
    assert gaze.dimension_labels == {"channel": ("x", "y", "pupil")}
    assert gaze.values.shape == (N_SAMPLES, 3)
    located = np.zeros(N_SAMPLES, dtype=bool)
    for onset, samples in ((100, 61), (180, 81), (270, 71), (360, 91)):
        located[onset : onset + samples] = True
    assert np.array_equal(np.isfinite(gaze.values[:, 0]), located)
    np.testing.assert_array_equal(gaze.values[100], [10.0, 5.0, 3001.0])
    np.testing.assert_array_equal(gaze.values[399], [35.0, 5.0, 3004.0])


def _with_position(words, unit, positions):
    words = list(words)
    words[unit] = WordFragments(words[unit].eeg, words[unit].gaze, positions)
    return words


def test_sentence_recordings_reject_fragments_that_disagree_with_the_release():
    eeg, words = _trial()
    with pytest.raises(FragmentMismatch, match="position 3 is not assigned"):
        _recordings(eeg, _with_position(words, 0, (3,)))
    assert issubclass(FragmentMismatch, ValueError)
    with pytest.raises(ValueError, match="differ in number"):
        _recordings(eeg, _with_position(words, 0, (1, 2)))
    with pytest.raises(
        ValueError, match=r"EEG fragment of fixation 3 has shape \(80, 3\)"
    ):
        shorter = WordFragments((words[2].eeg[0][:-1],), words[2].gaze, (3,))
        _recordings(eeg, [*words[:2], shorter, words[3]])
    with pytest.raises(ValueError, match="gaze fragment of fixation 3"):
        narrow = WordFragments(words[2].eeg, (words[2].gaze[0][:, :3],), (3,))
        _recordings(eeg, [*words[:2], narrow, words[3]])
    with pytest.raises(ValueError, match="word 0, fixation 2: fragment occurs nowhere"):
        _recordings(eeg[::-1].copy(), words)
    with pytest.raises(ValueError, match="one rectangle per word"):
        _recordings(eeg, words, bounds=BOUNDS[:3].astype(float))


def _file_trials():
    """Trials for the file writers: the full trial, then the two skipped kinds."""
    eeg, words = _trial()
    dwell = [61.0, 61.0, 81.0, 91.0]
    complete = {"words": WORDS, "eeg": eeg, "fragments": words, "dwell": dwell}
    without_eeg = dict(complete, eeg=None)
    without_fragments = dict(complete, fragment_fields=False)
    return [complete, None, without_eeg, without_fragments]


def _write_zuco2(directory, subject, trials, task="NR"):
    """Lay the trials out as the ZuCo 2.0 MATLAB v7.3 files do."""
    import h5py

    counter = itertools.count()
    with h5py.File(directory / f"results{subject}_{task}.mat", "w") as fh:
        store = fh.create_group("#refs#")

        def dataset(data, **attrs):
            field = store.create_dataset(f"d{next(counter)}", data=data)
            for key, value in attrs.items():
                field.attrs[key] = value
            return field.ref

        def refs(references):
            return dataset(np.array([[r] for r in references], dtype=h5py.ref_dtype))

        def empty():
            return dataset(np.zeros((1, 2), dtype=np.uint64), MATLAB_empty=1)

        def scalar(value):
            return dataset(np.array([[float(value)]]))

        columns = {"word": [], "rawData": [], "allFixations": [], "wordbounds": []}
        for trial in trials:
            if trial is None:  # a sentence this reader never saw
                for column in columns.values():
                    column.append(scalar(np.nan))
                continue
            group = store.create_group(f"w{next(counter)}")
            group.create_dataset(
                "content",
                data=np.array(
                    [
                        [dataset(np.array([[ord(c)] for c in w], dtype=np.uint16))]
                        for w in trial["words"]
                    ],
                    dtype=h5py.ref_dtype,
                ),
            )
            fragments = trial["fragments"]
            for field in ("FFD", "GD", "GPT", "TRT"):
                group.create_dataset(
                    field,
                    data=np.array(
                        [
                            [scalar(d) if f.positions else empty()]
                            for d, f in zip(trial["dwell"], fragments, strict=True)
                        ],
                        dtype=h5py.ref_dtype,
                    ),
                )
            group.create_dataset(
                "nFixations",
                data=np.array(
                    [[scalar(len(f.positions))] for f in fragments],
                    dtype=h5py.ref_dtype,
                ),
            )
            group.create_dataset(
                "fixPositions",
                data=np.array(
                    [
                        [
                            dataset(np.array(f.positions, dtype=float)[:, None])
                            if f.positions
                            else empty()
                        ]
                        for f in fragments
                    ],
                    dtype=h5py.ref_dtype,
                ),
            )
            for index, field in enumerate(EEG_BANDS, start=1):
                group.create_dataset(
                    field,
                    data=np.array(
                        [
                            [
                                dataset(
                                    np.full((N_ELECTRODES, 1), float(index * (w + 1)))
                                )
                                if f.positions
                                else empty()
                            ]
                            for w, f in enumerate(fragments)
                        ],
                        dtype=h5py.ref_dtype,
                    ),
                )
            if trial.get("fragment_fields", True):
                for field, attribute in (("rawEEG", "eeg"), ("rawET", "gaze")):
                    group.create_dataset(
                        field,
                        data=np.array(
                            [
                                [
                                    refs([dataset(a) for a in getattr(f, attribute)])
                                    if f.positions
                                    else empty()
                                ]
                                for f in fragments
                            ],
                            dtype=h5py.ref_dtype,
                        ),
                    )
            columns["word"].append(group.ref)
            columns["rawData"].append(
                scalar(np.nan) if trial["eeg"] is None else dataset(trial["eeg"])
            )
            fixations = store.create_group(f"f{next(counter)}")
            for key, values in FIXATIONS.items():
                fixations.create_dataset(key, data=values[:, None])
            columns["allFixations"].append(fixations.ref)
            columns["wordbounds"].append(dataset(BOUNDS.T.astype(float)))
        sentences = fh.create_group("sentenceData")
        for column, references in columns.items():
            sentences.create_dataset(
                column, data=np.array([[r] for r in references], dtype=h5py.ref_dtype)
            )


def _cell(arrays):
    cell = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        cell[index] = array
    return cell


def _write_zuco1(directory, subject, trials, task="NR"):
    """Lay the trials out as the ZuCo 1.0 MATLAB v5 files do: channels first."""
    sentences = []
    for trial in trials:
        if trial is None:
            sentences.append({"content": "unseen", "word": []})
            continue
        words = np.empty(len(trial["words"]), dtype=object)
        for index, (text, dwell, fragments) in enumerate(
            zip(trial["words"], trial["dwell"], trial["fragments"], strict=True)
        ):
            entry = {"content": text, "nFixations": float(len(fragments.positions))}
            for field in ("FFD", "GD", "GPT", "TRT"):
                entry[field] = float(dwell) if fragments.positions else np.empty((0, 0))
            entry["fixPositions"] = (
                np.array(fragments.positions, dtype=float)
                if fragments.positions
                else np.empty((0, 0))
            )
            for band, field in enumerate(EEG_BANDS, start=1):
                entry[field] = (
                    np.full(N_ELECTRODES, float(band * (index + 1)))
                    if fragments.positions
                    else np.empty((0, 0))
                )
            if trial.get("fragment_fields", True):
                entry["rawEEG"] = (
                    _cell([a.T for a in fragments.eeg])
                    if fragments.eeg
                    else np.empty((0, 0))
                )
                entry["rawET"] = (
                    _cell([a.T for a in fragments.gaze])
                    if fragments.gaze
                    else np.empty((0, 0))
                )
            words[index] = entry
        sentence = {
            "content": " ".join(trial["words"]),
            "word": words,
            "wordbounds": BOUNDS.astype(float),
            "allFixations": dict(FIXATIONS),
        }
        if trial["eeg"] is not None:
            sentence["rawData"] = trial["eeg"].T
        sentences.append(sentence)
    savemat(directory / f"results{subject}_{task}.mat", {"sentenceData": sentences})


@pytest.mark.parametrize("cls, write", [(ZuCo2, _write_zuco2), (ZuCo1, _write_zuco1)])
def test_loaders_build_the_recordings_from_the_released_layout(
    tmp_path, cls, write, caplog
):
    trials = _file_trials()
    write(tmp_path, "AAA", trials)
    dataset = cls(continuous=True)
    with caplog.at_level(logging.WARNING, logger="replica"):
        dataset.load(str(tmp_path))
    assert (
        "no recordings for 1 sentences without EEG, 1 without fixation fragments "
        "and 0 whose fragments contradict the sentence EEG" in caplog.text
    )
    assert len(dataset.stimuli()) == 3
    eeg, gaze = dataset.recordings()
    assert (eeg.measure_name, gaze.measure_name) == ("eeg", "gaze")
    assert eeg.stimulus is dataset.stimuli()[0]
    assert eeg.participant_id == "AAA"
    np.testing.assert_array_equal(eeg.values, trials[0]["eeg"])
    np.testing.assert_allclose(eeg.event_times, EXPECTED_EVENTS)
    assert eeg.metadata["fixations"] == EXPECTED_TABLE
    assert eeg.metadata["task"] == "NR" and eeg.metadata["sentence_idx"] == 0
    # The located fixations are the physical events of the fixation stream.
    events = {
        e.fixation_index: e for e in dataset.fixations("AAA", eeg.stimulus.stimulus_id)
    }
    for index, samples in zip(
        EXPECTED_TABLE["fixation_index"], EXPECTED_TABLE["samples"], strict=True
    ):
        assert events[index].metadata["source_duration_samples"] + 1 == samples
    assert set(dataset.modalities) == {"eye-tracking", "eeg"}
    (only_gaze,) = dataset.recordings(modality="eye-tracking")
    assert only_gaze is gaze
    # Band power rides along: the mean over electrodes per word, NaN unfixated.
    (theta,) = [
        d
        for d in dataset._data
        if d.measure_name == "eeg_theta_1" and d.stimulus is eeg.stimulus
    ]
    np.testing.assert_array_equal(theta.values, [1.0, 2.0, 3.0, 4.0])
    assert eeg.stimulus.layers["eeg_gamma_2_AAA"].values[2].shape == (N_ELECTRODES,)
    # Datum measures and the two recording measures share one vocabulary.
    assert dataset.measures == sorted(
        {"eeg", "gaze", *MEASURES, *(f"eeg_{b}" for b in EEG_BANDS.values())}
    )


@pytest.mark.parametrize("cls, write", [(ZuCo2, _write_zuco2), (ZuCo1, _write_zuco1)])
def test_without_continuous_the_loaders_build_no_recordings(tmp_path, cls, write):
    write(tmp_path, "AAA", _file_trials())
    dataset = cls()
    dataset.load(str(tmp_path), load_eeg=False)
    assert dataset.recordings() == []
    assert len(dataset._data) == 3 * 4
    assert not any(d.measure_name.startswith("eeg_") for d in dataset._data)


@pytest.mark.parametrize("cls, write", [(ZuCo2, _write_zuco2), (ZuCo1, _write_zuco1)])
def test_participants_select_the_task_files(tmp_path, cls, write):
    write(tmp_path, "AAA", _file_trials()[:1])
    write(tmp_path, "BBB", _file_trials()[:1])
    dataset = cls(continuous=True)
    dataset.load(str(tmp_path), participants=["BBB"])
    assert dataset.participants == ["BBB"]
    assert {r.participant_id for r in dataset.recordings()} == {"BBB"}
    with pytest.raises(ValueError, match=r"no task file for participants \['CCC'\]"):
        cls().load(str(tmp_path), participants=["CCC"])


@pytest.mark.parametrize(
    "cls, write", [(ZuCo2EEG, _write_zuco2), (ZuCo1EEG, _write_zuco1)]
)
def test_the_eeg_views_forward_the_recordings(tmp_path, cls, write):
    write(tmp_path, "AAA", _file_trials()[:1])
    dataset = cls(continuous=True)
    dataset.load(str(tmp_path), participants=["AAA"])
    assert [r.measure_name for r in dataset.recordings()] == ["eeg", "gaze"]
    assert dataset.recordings()[0].stimulus is dataset.stimuli()[0]
    assert sorted(d.measure_name for d in dataset._data) == sorted(
        f"eeg_{band}" for band in EEG_BANDS.values()
    )
    assert cls().load(str(tmp_path)) is None


def test_a_fragment_that_contradicts_the_release_skips_the_trial(tmp_path, caplog):
    trials = _file_trials()[:1]
    broken = trials[0]["fragments"]
    trials[0]["fragments"] = [
        *broken[:3],
        WordFragments(broken[3].eeg, broken[3].gaze, (3,)),
    ]
    _write_zuco2(tmp_path, "AAA", trials)
    dataset = ZuCo2(continuous=True)
    with caplog.at_level(logging.DEBUG, logger="replica"):
        dataset.load(str(tmp_path))
    assert dataset.recordings() == []
    assert (
        "ZuCo AAA zuco2_NR_0: word 3: fixation position 3 is not assigned"
        in caplog.text
    )
    assert (
        "0 without fixation fragments and 1 whose fragments contradict" in caplog.text
    )
    assert len(dataset._data) == 4 + len(EEG_BANDS)


@pytest.mark.needs_data
class TestReleases:
    """The located fixations of the released files, pinned per reader."""

    def _load(self, cls, name, participant, files="zuco2"):
        """Load one reader of a registered view whose files sit under ``files``."""
        from replica.dataset.catalog.registry import get

        if not get(files).default_path().exists():
            pytest.skip(f"{files} data not found at {get(files).default_path()}")
        dataset = cls(continuous=True)
        dataset.load(str(get(name).default_path()), participants=[participant])
        return dataset

    def test_zuco2_reader_yac_normal_reading(self):
        dataset = self._load(ZuCo2, "zuco2", "YAC")
        eeg = dataset.recordings(measure="eeg")
        assert len(eeg) == len(dataset.recordings(measure="gaze")) == 247
        first = next(r for r in eeg if r.stimulus.stimulus_id == "zuco2_NR_0")
        assert first.values.shape == (4760, 105)
        assert first.metadata["fixations"]["onset_sample"][:4] == [151, 230, 299, 382]
        assert first.metadata["fixations"]["unit_index"][:4] == [1, 0, 2, 5]
        self._check_fixated_words_are_timed(dataset, eeg)

    def test_zuco1_reader_zab_normal_reading(self):
        dataset = self._load(ZuCo1, "zuco1", "ZAB", files="zuco1")
        eeg = dataset.recordings(measure="eeg")
        assert len(eeg) == 295
        first = next(r for r in eeg if r.stimulus.stimulus_id == "zuco1_NR_0")
        assert first.values.shape == (1837, 105)
        assert first.metadata["fixations"]["onset_sample"][:4] == [182, 287, 395, 475]
        self._check_fixated_words_are_timed(dataset, eeg)

    @staticmethod
    def _check_fixated_words_are_timed(dataset, recordings):
        """A word has an onset exactly when the release gives it a first fixation."""
        released = {
            (d.participant_id, d.stimulus.stimulus_id): np.asarray(
                d.metadata["released_values"]
            )
            for d in dataset._data
            if d.measure_name == "first_fixation"
        }
        for recording in recordings:
            durations = released[
                (recording.participant_id, recording.stimulus.stimulus_id)
            ]
            np.testing.assert_array_equal(
                np.isfinite(recording.event_times), durations > 0
            )
            gaze = dataset.recordings(
                measure="gaze",
                participant=recording.participant_id,
                stimulus_id=recording.stimulus.stimulus_id,
            )
            assert len(gaze) == 1 and gaze[0].values.shape == (len(recording.values), 3)
