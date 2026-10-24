"""Frank et al. (2015) ERP loader: synthetic contracts and real-data invariants.

The synthetic tests write a release file with the documented array layout. The
real-data tests pin the paper's counts, the agreement with the participant
means that de Varda et al. (2023) published, and the stimulus identity shared
with the UCL loaders.
"""

import csv
import logging
import os

import numpy as np
import pytest
from scipy.io import savemat

from replica.dataset import Frank2015ERP, describe_dataset
from replica.dataset.catalog.registry import get
from replica.dataset.eeg.frank2015_erp import (
    BASELINES,
    COMPONENTS,
    N_PARTICIPANTS,
    ucl_sentence_numbers,
)
from replica.dataset.eye_tracking.devarda2023.covariates import (
    COVARIATE_COLUMNS,
    read_devarda_norms,
)

SENTENCES = ((9, "First second third."), (4, "Only two."))


def _write_ucl_stimuli(directory, sentences=SENTENCES):
    """Write a UCL ``stimuli.txt`` numbering *sentences*."""
    rows = ["sent_nr\tsentence\tquestion\tanswer"]
    rows += [f"{sent_nr}\t{text}\t-\t-" for sent_nr, text in sentences]
    (directory / "stimuli.txt").write_text("\n".join(rows) + "\n", encoding="cp1252")


def _write_release(directory, sentences=SENTENCES, *, nan_trial=None):
    """Write a release file for *sentences* and return its arrays by sentence.

    Trial flags: word 1 of participant 1 is an artefact, every first and last
    word is rejected, and word 1 of participant 3 is rejected without being an
    artefact. ``nan_trial`` names a (sentence, word, participant) whose
    amplitudes are NaN, as the package's unrecorded trials are.
    """
    n = len(sentences)
    cells = {
        key: np.empty((n, 1), dtype=object)
        for key in ("sentences", "ERP", "ERPbase", "artefact", "reject", "logwordfreq")
    }
    arrays = []
    for index, (_, text) in enumerate(sentences):
        words = text.split()
        n_words = len(words)
        amplitude = (
            np.arange(n_words * N_PARTICIPANTS * len(COMPONENTS), dtype=float).reshape(
                n_words, N_PARTICIPANTS, len(COMPONENTS)
            )
            + 1000 * index
        )
        if nan_trial is not None and nan_trial[0] == index:
            amplitude[nan_trial[1], nan_trial[2], :] = np.nan
        artefact = np.zeros((n_words, N_PARTICIPANTS), dtype=np.uint8)
        artefact[1, 0] = 1
        reject = artefact.copy()
        reject[0, :] = 1
        reject[-1, :] = 1
        reject[1, 2] = 1
        cells["sentences"][index, 0] = np.array(words, dtype=object)
        cells["ERP"][index, 0] = amplitude
        cells["ERPbase"][index, 0] = amplitude / 10
        cells["artefact"][index, 0] = artefact
        cells["reject"][index, 0] = reject
        cells["logwordfreq"][index, 0] = (
            np.arange(n_words, dtype=float).reshape(-1, 1) - 5
        )
        arrays.append(
            {
                "amplitude": amplitude,
                "baseline": amplitude / 10,
                "artefact": artefact,
                "reject": reject,
            }
        )
    cells["sentence_position"] = np.tile(
        np.arange(1, n + 1, dtype=np.uint8), (N_PARTICIPANTS, 1)
    )
    savemat(os.path.join(directory, "stimuli_erp.mat"), cells)
    return arrays


def _load(directory, sentences=SENTENCES, **kwargs):
    _write_ucl_stimuli(directory, sentences)
    dataset = Frank2015ERP(**kwargs)
    dataset.load(str(directory), ucl_path=str(directory))
    return dataset


def _values(dataset, measure, participant, stimulus_id):
    (datum,) = [
        d
        for d in dataset.data(measure, participant=participant)
        if d.stimulus.stimulus_id == stimulus_id
    ]
    return datum


def test_loader_attributes():
    dataset = Frank2015ERP()
    assert dataset.info.name == "frank2015_erp"
    assert dataset.default_measure == "n400"
    assert dataset.presentation == ["rapid serial visual presentation"]
    assert set(dataset.measures) == set(COMPONENTS) | set(BASELINES.values())


def test_catalog_describes_the_release_without_data():
    described = describe_dataset("frank2015_erp")
    assert described.paradigm == "eeg"
    assert set(described.measures) == set(COMPONENTS) | set(BASELINES.values())
    assert described.download_keys == ("frank2015_erp", "ucl_et")


@pytest.mark.parametrize("default_ucl_path", [False, True])
def test_stimuli_carry_the_ucl_sentence_numbers(
    tmp_path, monkeypatch, default_ucl_path
):
    _write_release(tmp_path)
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(tmp_path / "data"))
    if default_ucl_path:
        ucl_path = get("ucl_et").default_path()
        ucl_path.mkdir(parents=True)
        _write_ucl_stimuli(ucl_path)
        dataset = Frank2015ERP()
        dataset.load(str(tmp_path))
    else:
        dataset = _load(tmp_path)
    assert [s.stimulus_id for s in dataset.stimuli()] == ["ucl_9", "ucl_4"]
    assert dataset.stimuli()[0].unit_strings("words") == ["First", "second", "third."]
    assert dataset.stimuli()[1].metadata == {
        "sent_nr": 4,
        "release_index": 2,
        "release_log_frequency": [-5.0, -4.0],
    }
    assert dataset.participants == sorted(
        (str(p) for p in range(1, N_PARTICIPANTS + 1)), key=str
    )
    assert len(dataset._data) == 2 * N_PARTICIPANTS * 2 * len(COMPONENTS)


def test_reject_policy_removes_every_flagged_trial(tmp_path):
    arrays = _write_release(tmp_path, nan_trial=(0, 1, 5))
    dataset = _load(tmp_path)
    for participant in range(N_PARTICIPANTS):
        datum = _values(dataset, "n400", str(participant + 1), "ucl_9")
        expected = arrays[0]["amplitude"][
            :, participant, COMPONENTS.index("n400")
        ].copy()
        expected[arrays[0]["reject"][:, participant] == 1] = np.nan
        np.testing.assert_equal(datum.values, expected)
        assert datum.metadata["exclusions"] == "reject"
        assert datum.metadata["reject"] == arrays[0]["reject"][:, participant].tolist()
        assert (
            datum.metadata["artefact"] == arrays[0]["artefact"][:, participant].tolist()
        )
        assert datum.metadata["session_position"] == 1
    # The unrecorded trial stays missing whatever the policy.
    assert np.isnan(_values(dataset, "n400", "6", "ucl_9").values[1])
    assert np.isnan(
        _values(_load(tmp_path, exclusions="none"), "n400", "6", "ucl_9").values[1]
    )


@pytest.mark.parametrize("exclusions", ["artefact", "none"])
def test_looser_policies_keep_rejected_but_clean_trials(tmp_path, exclusions):
    arrays = _write_release(tmp_path)
    dataset = _load(tmp_path, exclusions=exclusions)
    source = arrays[0]["amplitude"][:, :, COMPONENTS.index("lan")]
    first = _values(dataset, "lan", "1", "ucl_9").values
    third = _values(dataset, "lan", "3", "ucl_9").values
    # Participant 1's second word is an artefact; participant 3's is rejected
    # for another reason; first and last words are rejected for everyone.
    if exclusions == "artefact":
        np.testing.assert_equal(first, [source[0, 0], np.nan, source[2, 0]])
    else:
        np.testing.assert_equal(first, source[:, 0])
    np.testing.assert_equal(third, source[:, 2])
    assert all(d.metadata["exclusions"] == exclusions for d in dataset._data)


def test_baselines_follow_the_amplitudes(tmp_path):
    arrays = _write_release(tmp_path)
    dataset = _load(tmp_path)
    for name in COMPONENTS:
        amplitude = _values(dataset, name, "2", "ucl_4").values
        baseline = _values(dataset, BASELINES[name], "2", "ucl_4").values
        np.testing.assert_equal(baseline, amplitude / 10)
        np.testing.assert_equal(np.isnan(baseline), arrays[1]["reject"][:, 1] == 1)


def test_unknown_exclusions_are_rejected():
    with pytest.raises(ValueError, match="exclusions"):
        Frank2015ERP(exclusions="all")


def test_a_release_sentence_outside_ucl_is_rejected(tmp_path):
    _write_release(tmp_path)
    with pytest.raises(ValueError, match="not a UCL sentence: 'Only two.'"):
        _load(tmp_path, sentences=SENTENCES[:1])


def test_a_missing_ucl_release_is_reported(tmp_path):
    _write_release(tmp_path)
    with pytest.raises(FileNotFoundError, match="stimuli.txt"):
        Frank2015ERP().load(str(tmp_path), ucl_path=str(tmp_path))


def test_ucl_sentences_sharing_a_text_are_rejected(tmp_path):
    _write_ucl_stimuli(tmp_path, ((1, "Only two."), (2, "Only two.")))
    with pytest.raises(ValueError, match="share a text"):
        ucl_sentence_numbers(tmp_path)


def test_release_sentences_sharing_a_ucl_number_are_rejected(tmp_path):
    _write_release(tmp_path, sentences=((9, "Only two."), (4, "Only two.")))
    with pytest.raises(ValueError, match="both UCL sentence 4"):
        _load(tmp_path, sentences=((4, "Only two."),))


def _rewrite_release(directory, edit):
    """Load the synthetic release, apply *edit* to its arrays, and save it."""
    from scipy.io import loadmat

    cells = {
        key: value
        for key, value in loadmat(directory / "stimuli_erp.mat").items()
        if not key.startswith("__")
    }
    edit(cells)
    savemat(directory / "stimuli_erp.mat", cells)


def test_array_shape_disagreement_is_rejected(tmp_path):
    _write_release(tmp_path)

    def truncate(cells):
        cells["artefact"][0, 0] = cells["artefact"][0, 0][:, :5]

    _rewrite_release(tmp_path, truncate)
    with pytest.raises(ValueError, match="array shapes"):
        _load(tmp_path)


def test_session_position_shape_disagreement_is_rejected(tmp_path):
    _write_release(tmp_path)

    def truncate(cells):
        cells["sentence_position"] = cells["sentence_position"][:, :1]

    _rewrite_release(tmp_path, truncate)
    with pytest.raises(ValueError, match="sentence_position"):
        _load(tmp_path)


def test_an_artefact_outside_the_rejections_is_rejected(tmp_path):
    _write_release(tmp_path)

    def flag(cells):
        cells["artefact"][0, 0][1, 4] = 1

    _rewrite_release(tmp_path, flag)
    with pytest.raises(ValueError, match="not rejected"):
        _load(tmp_path)


def test_each_datum_owns_its_metadata(tmp_path):
    _write_release(tmp_path)
    dataset = _load(tmp_path)
    trial = [
        d for d in dataset.data(participant="1") if d.stimulus.stimulus_id == "ucl_9"
    ]
    assert len(trial) == 2 * len(COMPONENTS)
    trial[0].metadata["probe"] = True
    trial[0].metadata["reject"][0] = 7
    assert all("probe" not in d.metadata for d in trial[1:])
    assert all(d.metadata["reject"][0] == 1 for d in trial[1:])


def test_covariates_carry_the_release_frequency(tmp_path):
    _write_release(tmp_path)
    dataset = _load(tmp_path)
    covs = dataset.covariates(dataset.stimuli()[0])
    np.testing.assert_equal(covs["release_log_frequency"], [-5.0, -4.0, -3.0])
    np.testing.assert_equal(covs["unit_length"], [5.0, 6.0, 6.0])
    assert "cloze" not in covs


def _write_norms(path, rows):
    columns = ["sentence", "word", "context_length", *COVARIATE_COLUMNS.values()]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _norm_row(sentence, position, value):
    return {
        "sentence": sentence,
        "word": sentence.split()[position],
        "context_length": position,
        **{column: value for column in COVARIATE_COLUMNS.values()},
    }


@pytest.mark.parametrize("default_norms_path", [False, True])
def test_norms_attach_by_sentence_text(tmp_path, monkeypatch, default_norms_path):
    _write_release(tmp_path)
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(tmp_path / "data"))
    table_dir = get("devarda2023").default_path() if default_norms_path else tmp_path
    table_dir.mkdir(parents=True, exist_ok=True)
    table = table_dir / "all_measures.csv"
    _write_norms(
        table,
        [
            _norm_row("First second third.", 1, 0.25),
            _norm_row("First second third.", 2, 0.5),
            _norm_row("Elsewhere entirely.", 1, 0.75),
        ],
    )
    dataset = _load(tmp_path)
    assert not set(COVARIATE_COLUMNS) & set(dataset.covariates(dataset.stimuli()[0]))
    assert (
        dataset.attach_predictability_norms(None if default_norms_path else table) == 1
    )
    assert dataset.devarda_unmatched == ["Elsewhere entirely."]
    covs = dataset.covariates(dataset.stimuli()[0])
    for name in COVARIATE_COLUMNS:
        np.testing.assert_equal(covs[name], [np.nan, 0.25, 0.5])
    # An unmatched stimulus keeps the schema, with NaN throughout.
    unmatched = dataset.covariates(dataset.stimuli()[1])
    for name in COVARIATE_COLUMNS:
        np.testing.assert_equal(unmatched[name], [np.nan, np.nan])
    # The returned arrays are copies.
    covs["cloze"][1] = -1
    assert dataset.covariates(dataset.stimuli()[0])["cloze"][1] == 0.25


def test_norms_are_forgotten_on_reload_and_absent_before_load(tmp_path):
    _write_release(tmp_path)
    table = tmp_path / "all_measures.csv"
    _write_norms(table, [_norm_row("First second third.", 1, 0.25)])
    _write_ucl_stimuli(tmp_path)
    dataset = Frank2015ERP()
    assert dataset.attach_predictability_norms(table) == 0
    assert dataset.devarda_unmatched == ["First second third."]
    dataset.load(str(tmp_path), ucl_path=str(tmp_path))
    assert "cloze" not in dataset.covariates(dataset.stimuli()[0])
    assert dataset.attach_predictability_norms(table) == 1
    dataset.load(str(tmp_path), ucl_path=str(tmp_path))
    assert "cloze" not in dataset.covariates(dataset.stimuli()[0])


def test_norms_read_na_as_missing(tmp_path):
    table = tmp_path / "all_measures.csv"
    row = _norm_row("First second third.", 1, 0.25)
    row["cloze_p_smoothed"] = "NA"
    row["rating_sd"] = ""
    _write_norms(table, [row])
    norms = read_devarda_norms(table)
    values = norms["First second third."][1]
    assert np.isnan(values["cloze"]) and np.isnan(values["predictability_rating_sd"])
    assert values["cloze_surprisal"] == 0.25


def test_norms_with_a_misplaced_word_are_rejected(tmp_path):
    table = tmp_path / "all_measures.csv"
    row = _norm_row("First second third.", 1, 0.25)
    row["word"] = "third."
    _write_norms(table, [row])
    with pytest.raises(ValueError, match="not at position 1"):
        read_devarda_norms(table)


def test_norms_repeating_a_position_are_rejected(tmp_path):
    table = tmp_path / "all_measures.csv"
    row = _norm_row("First second third.", 1, 0.25)
    _write_norms(table, [row, row])
    with pytest.raises(ValueError, match="repeated position"):
        read_devarda_norms(table)


_UCL_STIMULI = "sent_nr\tsentence\tquestion\tanswer\n1\tFirst second third.\t-\t-\n2\tOnly two.\t-\t-\n"


def _write_ucl(directory):
    """Write a two-sentence UCL release: eye-tracking and self-paced reading."""
    (directory / "stimuli.txt").write_text(_UCL_STIMULI, encoding="cp1252")
    et = [
        "subj_nr\tsent_nr\tsent_pos\tcorrect\tanswer_time\tword_pos\tword\t"
        "RTfirstfix\tRTfirstpass\tRTrightbound\tRTgopast"
    ]
    spr = ["subj_nr\tsent_nr\tsent_pos\tcorrect\tanswer_time\tword_pos\tword\tRT"]
    displayed = {1: ["First", "second", "third."], 2: ["Only", "twoo."]}
    for sent_nr, words in displayed.items():
        for position, word in enumerate(words, start=1):
            et.append(
                f"1\t{sent_nr}\t1\t-\tNaN\t{position}\t{word}\t200\t200\t200\t200"
            )
            spr.append(f"1\t{sent_nr}\t1\t-\tNaN\t{position}\t{word}\t300")
    (directory / "eyetracking.RT.txt").write_text("\n".join(et) + "\n")
    (directory / "selfpacedreading.RT.txt").write_text("\n".join(spr) + "\n")


def test_ucl_loaders_attach_the_norms_by_displayed_text(tmp_path):
    from replica.dataset import UCL_ET, UCL_SPR

    _write_ucl(tmp_path)
    table = tmp_path / "all_measures.csv"
    _write_norms(
        table,
        [_norm_row("First second third.", 2, 0.5), _norm_row("Only two.", 1, 0.75)],
    )
    eye_tracking = UCL_ET()
    eye_tracking.load(str(tmp_path))
    assert eye_tracking.attach_predictability_norms(table) == 2
    by_id = {s.stimulus_id: s for s in eye_tracking.stimuli()}
    np.testing.assert_equal(
        eye_tracking.covariates(by_id["ucl_1"])["cloze"], [np.nan, np.nan, 0.5]
    )
    assert "prev_word_fixated" in eye_tracking.covariates(by_id["ucl_1"])
    # The self-paced release displayed sentence 2 with a typo, so only
    # sentence 1 matches and sentence 2 carries NaN.
    spr = UCL_SPR()
    spr.load(str(tmp_path))
    assert spr.attach_predictability_norms(table) == 1
    assert spr.devarda_unmatched == ["Only two."]
    by_id = {s.stimulus_id: s for s in spr.stimuli()}
    np.testing.assert_equal(spr.covariates(by_id["ucl_2"])["cloze"], [np.nan, np.nan])
    np.testing.assert_equal(
        spr.covariates(by_id["ucl_1"])["cloze_surprisal"], [np.nan, np.nan, 0.5]
    )


def _write_devarda_release(directory, rows):
    from replica.dataset.eye_tracking.devarda2023 import MEASURE_COLUMNS

    columns = [
        "sent_id",
        "sentence",
        "list",
        "context_length",
        "word",
        *MEASURE_COLUMNS,
        *COVARIATE_COLUMNS.values(),
    ]
    with open(directory / "all_measures.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**dict.fromkeys(MEASURE_COLUMNS, 100), **row})


def test_devarda_attaches_its_own_norms_on_load(tmp_path):
    from replica.dataset import DeVarda2023

    _write_devarda_release(
        tmp_path,
        [
            {
                "sent_id": 1,
                "sentence": "One two three.",
                "list": 1,
                "context_length": 1,
                "word": "two",
                **{c: 0.5 for c in COVARIATE_COLUMNS.values()},
            },
            {
                "sent_id": 1,
                "sentence": "One two three.",
                "list": 1,
                "context_length": 2,
                "word": "three.",
                **{c: 0.25 for c in COVARIATE_COLUMNS.values()},
            },
        ],
    )
    dataset = DeVarda2023()
    dataset.load(str(tmp_path))
    covs = dataset.covariates(dataset.stimuli()[0])
    np.testing.assert_equal(covs["cloze"], [np.nan, 0.5, 0.25])
    np.testing.assert_equal(covs["cloze_entropy"], [np.nan, 0.5, 0.25])
    assert dataset.devarda_unmatched == []


def test_devarda_rejects_a_normed_row_whose_word_is_misplaced(tmp_path):
    from replica.dataset import DeVarda2023

    _write_devarda_release(
        tmp_path,
        [
            {
                "sent_id": 1,
                "sentence": "One two three.",
                "list": 1,
                "context_length": 1,
                "word": "three.",
                **{c: 0.5 for c in COVARIATE_COLUMNS.values()},
            },
        ],
    )
    with pytest.raises(ValueError, match="not at position 1"):
        DeVarda2023().load(str(tmp_path))


# One sentence's recorded block: 100 ms (25 samples) before its first word to
# 924 ms (231 samples) after its last, words 175 samples (700 ms) apart.
_BEFORE, _AFTER, _SOA = 25, 231, 175


def _session(blocks, n_channels=3, gap=20, soa=_SOA):
    """Lay sentences out in one EEGLAB-like session with NaN between them.

    Args:
        blocks: ``(release_index, n_words)`` per sentence, in session order.
        n_channels: Channels of the session.
        gap: NaN samples before every sentence and after the last.
        soa: Samples between word onsets; beyond ``_BEFORE + _AFTER`` the
            words' windows are separate and NaN lies between them.

    Returns:
        The ``(n_channels, n_samples)`` data, the ``(type, latency)`` events,
        and per sentence its block ``[start, end)`` and its word onsets.
    """
    data, events, expected = [], [], []
    end = 0
    for release_index, n_words in blocks:
        start = end + gap
        onsets = [start + _BEFORE + soa * w for w in range(n_words)]
        end = onsets[-1] + _AFTER
        block = (
            np.arange(n_channels * (end - start), dtype=np.float32).reshape(
                n_channels, end - start
            )
            + 1000 * release_index
        )
        for onset in onsets[1:]:
            block[:, onset + _AFTER - soa - start : onset - _BEFORE - start] = np.nan
        data.append(np.full((n_channels, gap), np.nan, dtype=np.float32))
        data.append(block)
        events.append((50 + release_index, onsets[0] + 1))
        events += [(w + 2, onset + 1) for w, onset in enumerate(onsets[1:])]
        expected.append((release_index, start, end, onsets))
    data.append(np.full((n_channels, gap), np.nan, dtype=np.float32))
    return np.hstack(data), events, expected


def _write_eeg(
    directory, participant, data, events, *, srate=250, labels=None, part2=None
):
    """Write ``EEG<participant>.mat`` as EEGLAB structures."""

    def structure(data, events):
        names = labels or [f"c{i}" for i in range(data.shape[0])]
        return {
            "data": data,
            "srate": srate,
            "chanlocs": np.array([(name,) for name in names], dtype=[("labels", "O")]),
            "event": np.array(events, dtype=[("type", "f8"), ("latency", "f8")]),
        }

    payload = {"EEG": structure(data, events)}
    if part2 is not None:
        payload["EEG_part2"] = structure(*part2)
    savemat(os.path.join(directory, f"EEG{participant:02d}.mat"), payload)


def _load_continuous(directory, participants):
    _write_ucl_stimuli(directory)
    dataset = Frank2015ERP(continuous=True)
    dataset.load(str(directory), ucl_path=str(directory), participants=participants)
    return dataset


def test_continuous_recordings_follow_the_word_events(tmp_path):
    _write_release(tmp_path)
    data, events, expected = _session([(2, 2), (1, 3)])
    _write_eeg(tmp_path, 1, data, events, labels=["VEOG", "HEOG", "50"])
    dataset = _load_continuous(tmp_path, ["1"])
    recordings = dataset.recordings(modality="eeg")
    assert [r.participant_id for r in recordings] == ["1", "1"]
    for recording, (release_index, start, end, onsets) in zip(
        recordings, expected, strict=True
    ):
        assert recording.stimulus is dataset.stimuli()[release_index - 1]
        assert recording.values.dtype == np.float32
        np.testing.assert_array_equal(recording.values, data[:, start:end].T)
        assert recording.sample_interval == 1 / 250
        assert recording.start_time == start / 250
        np.testing.assert_allclose(recording.event_times, np.array(onsets) / 250)
        assert recording.event_times[0] - recording.start_time == pytest.approx(0.1)
        assert recording.dimension_labels == {"channel": ("VEOG", "HEOG", "50")}
        assert recording.metadata == {
            "session_position": release_index,
            "release_index": release_index,
            "part": 1,
            "reference": "average mastoids",
            "unrecorded_samples": 0,
        }
    # The amplitudes are loaded for everyone whatever the continuous selection.
    assert len(dataset._data) == len(_load(tmp_path)._data)
    assert _load(tmp_path).recordings() == []


def test_continuous_reads_the_requested_participants_and_parts(tmp_path, caplog):
    _write_release(tmp_path)
    data, events, _ = _session([(1, 3)])
    _write_eeg(tmp_path, 9, data, events)
    resumed, resumed_events, (resumed_block,) = _session([(2, 2)])
    # The second part resumes inside a sentence: its first events belong to no
    # sentence, and the next block starts after those samples and a gap.
    lead = np.hstack(
        [
            np.full((3, 5), 7.0, dtype=np.float32),
            np.full((3, 20), np.nan, dtype=np.float32),
        ]
    )
    part2 = (
        np.hstack([lead, resumed]),
        [(3, 2), (4, 4)] + [(kind, latency + 25) for kind, latency in resumed_events],
    )
    _write_eeg(tmp_path, 6, data, events, part2=part2)
    with caplog.at_level(logging.WARNING, logger="replica"):
        dataset = _load_continuous(tmp_path, ["6", "9"])
    assert "EEG06.mat: 2 word events before the first sentence event" in caplog.text
    six = dataset.recordings(participant="6")
    assert [(r.metadata["part"], r.metadata["release_index"]) for r in six] == [
        (1, 1),
        (2, 2),
    ]
    assert six[1].start_time == (resumed_block[1] + 25) / 250
    assert (
        dataset.recordings(participant="9")[0].metadata["reference"] == "left mastoid"
    )
    assert len(dataset.recordings()) == 3
    with pytest.raises(
        ValueError, match=r"no continuous EEG for participants \['25'\]"
    ):
        _load_continuous(tmp_path, ["25"])
    assert len(_load_continuous(tmp_path, [9]).recordings()) == 1


def test_a_long_soa_leaves_an_unrecorded_gap_inside_the_sentence(tmp_path):
    """The words' windows are separate, and the recording spans the gap."""
    _write_release(tmp_path)
    data, events, ((_, start, end, onsets),) = _session([(1, 3)], soa=300)
    _write_eeg(tmp_path, 1, data, events)
    (recording,) = _load_continuous(tmp_path, ["1"]).recordings()
    gap = 300 - _BEFORE - _AFTER
    assert recording.values.shape == (end - start, 3)
    assert recording.metadata["unrecorded_samples"] == 2 * gap
    missing = np.isnan(recording.values).all(axis=1)
    assert np.array_equal(
        np.flatnonzero(missing),
        np.r_[
            np.arange(onsets[0] + _AFTER, onsets[1] - _BEFORE) - start,
            np.arange(onsets[1] + _AFTER, onsets[2] - _BEFORE) - start,
        ],
    )
    np.testing.assert_allclose(recording.event_times, np.array(onsets) / 250)


def test_a_part_without_a_sentence_event_only_warns(tmp_path, caplog):
    _write_release(tmp_path)
    data, events, _ = _session([(1, 3)])
    orphans = np.full((3, 30), 1.0, dtype=np.float32)
    _write_eeg(tmp_path, 1, data, events, part2=(orphans, [(2, 3), (3, 9)]))
    with caplog.at_level(logging.WARNING, logger="replica"):
        dataset = _load_continuous(tmp_path, ["1"])
    assert "EEG01.mat: 2 word events before the first sentence event" in caplog.text
    assert [r.metadata["part"] for r in dataset.recordings()] == [1]


def _broken_session(kind):
    data, events, expected = _session([(1, 3), (2, 2)])
    (_, start, _, onsets), (_, second_start, _, _) = expected
    options: dict = {}
    if kind == "sample rate":
        options["srate"] = 500
        message = "sampled at 500"
    elif kind == "unknown sentence":
        events[0] = (53, events[0][1])
        message = "sentence 3, which the release does not have"
    elif kind == "positions":
        events[1], events[2] = (3, events[1][1]), (2, events[2][1])
        message = "word positions 2 to 3"
    elif kind == "latency order":
        events[1], events[2] = (2, events[2][1]), (3, events[1][1])
        message = "increasing samples"
    elif kind == "partial sample":
        data[0, start + 5] = np.nan
        message = "some channels only"
    elif kind == "event on a gap":
        data[:, onsets[1] - 1 : onsets[1] + 2] = np.nan
        message = "a word event falls on an unrecorded sample"
    elif kind == "merged blocks":  # no gap between the two sentences
        data[:, onsets[-1] + _AFTER : second_start] = 0.0
        message = "sentence 2 shares a recorded block with another sentence"
    else:  # "repeated sentence"
        events[3] = (51, events[3][1])
        message = "sentence 1 is presented twice"
    return data, events, options, message


@pytest.mark.parametrize(
    "kind",
    [
        "sample rate",
        "unknown sentence",
        "positions",
        "latency order",
        "partial sample",
        "event on a gap",
        "merged blocks",
        "repeated sentence",
    ],
)
def test_continuous_rejects_files_whose_events_do_not_describe_sentences(
    tmp_path, kind
):
    _write_release(tmp_path)
    data, events, options, message = _broken_session(kind)
    _write_eeg(tmp_path, 1, data, events, **options)
    with pytest.raises(ValueError, match=message):
        _load_continuous(tmp_path, ["1"])


@pytest.mark.needs_data
class TestFrank2015EEGRealData:
    """The continuous EEG of four participants against the README and the SOAs."""

    PARTICIPANTS = ("1", "6", "8", "9", "10", "14", "17")

    @pytest.fixture(autouse=True)
    def load_dataset(self):
        spec = get("frank2015_erp")
        if not (spec.default_path() / "EEG01.mat").exists():
            pytest.skip(f"Frank et al. (2015) EEG not found at {spec.default_path()}")
        self.path = spec.default_path()
        self.ds = Frank2015ERP(continuous=True)
        self.ds.load(str(self.path), participants=self.PARTICIPANTS)

    def test_recordings_per_participant_follow_the_readme(self):
        """Whole sentences per participant, and the README's trial counts less
        the word events of a recording that starts or resumes mid-sentence."""
        counts = {p: len(self.ds.recordings(participant=p)) for p in self.PARTICIPANTS}
        assert counts == {
            "1": 205,
            "6": 194,
            "8": 205,
            "9": 204,
            "10": 205,
            "14": 202,
            "17": 205,
        }
        words = {
            p: sum(
                int(np.isfinite(r.event_times).sum())
                for r in self.ds.recordings(participant=p)
            )
            for p in self.PARTICIPANTS
        }
        assert words == {
            "1": 1931,
            "6": 1832 - 5,
            "8": 1931,
            "9": 1916,
            "10": 1931,
            "14": 1901 - 3,
            "17": 1931,
        }
        parts = {r.metadata["part"] for r in self.ds.recordings(participant="6")}
        assert parts == {1, 2}
        assert self.ds.recordings(participant="9")[0].metadata["reference"] == (
            "left mastoid"
        )
        positions = {
            d.stimulus.stimulus_id: d.metadata["session_position"]
            for d in self.ds.data("n400", participant="1")
        }
        for recording in self.ds.recordings(participant="1"):
            assert (
                recording.metadata["session_position"]
                == positions[recording.stimulus.stimulus_id]
            )

    def test_every_recording_starts_100_ms_before_its_first_word(self):
        for recording in self.ds.recordings():
            assert recording.values.shape[1] == 34
            assert recording.values.dtype == np.float32
            missing = np.isnan(recording.values).all(axis=1)
            assert missing.sum() == recording.metadata["unrecorded_samples"]
            assert np.isfinite(recording.values[~missing]).all()
            assert recording.event_times[0] - recording.start_time == pytest.approx(0.1)
            assert recording.dimension_labels["channel"][:2] == ("VEOG", "HEOG")

    def test_the_three_long_soas_of_the_readme_leave_gaps(self):
        """An SOA beyond the 1,024 ms word window (100 ms before, 924 ms after)
        leaves NaN between the words; the README lists exactly three."""
        gaps = {
            (r.participant_id, r.metadata["release_index"]): r.metadata[
                "unrecorded_samples"
            ]
            for r in self.ds.recordings()
            if r.metadata["unrecorded_samples"]
        }
        assert gaps == {
            ("8", 47): (1868 - 1024) // 4,
            ("10", 1): (3112 - 1024) // 4,
            ("17", 126): (1784 - 1024) // 4,
        }

    def test_event_onsets_reproduce_the_released_soas(self):
        """The events sit on the 4 ms sample grid; the released SOAs are in
        milliseconds, so they agree to a sample apart from a few trials."""
        from scipy.io import loadmat

        soas = loadmat(self.path / "stimuli_erp.mat")["SOAs"][:, 0]
        deviations = []
        for recording in self.ds.recordings(participant="1"):
            released = soas[recording.metadata["release_index"] - 1][:, 0]
            measured = np.diff(recording.event_times) * 1000
            deviations.extend(np.abs(measured - released[: len(measured)]))
        assert len(deviations) == 1931 - 205
        assert np.median(deviations) == pytest.approx(1.0)
        assert max(deviations) <= 10.0


@pytest.mark.needs_data
class TestFrank2015ERPRealData:
    """Counts and agreements pinned against the release and its siblings."""

    @pytest.fixture(autouse=True)
    def load_dataset(self):
        spec = get("frank2015_erp")
        if not (spec.default_path() / "stimuli_erp.mat").exists():
            pytest.skip(f"Frank et al. (2015) data not found at {spec.default_path()}")
        self.path = spec.default_path()
        self.ds = spec.factory()

    def _sibling(self, name):
        spec = get(name)
        if not spec.default_path().exists():
            pytest.skip(f"{name} data not found at {spec.default_path()}")
        return spec.factory()

    def test_paper_counts(self):
        """205 sentences, 1,931 word tokens, 24 participants (Frank et al., 2015)."""
        assert len(self.ds.stimuli()) == 205
        assert sum(len(s.layers["words"]) for s in self.ds.stimuli()) == 1931
        assert len(self.ds.participants) == 24
        assert len(self.ds._data) == 205 * 24 * 2 * len(COMPONENTS)

    def test_first_and_last_words_are_rejected_for_everyone(self):
        """The paper analyses neither sentence-initial nor sentence-final words."""
        for datum in self.ds.data("n400"):
            assert datum.metadata["reject"][0] == 1
            assert datum.metadata["reject"][-1] == 1
            assert np.isnan(datum.values[0]) and np.isnan(datum.values[-1])

    def test_unrecorded_trials_match_the_readme(self):
        """The README lists 1931 minus 1832, 1916, 1917, 1901 and 1880 trials
        missing for participants 6, 9, 13, 14 and 20."""
        dataset = Frank2015ERP(exclusions="none")
        dataset.load(str(self.path))
        missing = {}
        for datum in dataset.data("n400"):
            missing[datum.participant_id] = missing.get(datum.participant_id, 0) + int(
                np.isnan(datum.values).sum()
            )
        assert {k: v for k, v in missing.items() if v} == {
            "6": 99,
            "9": 15,
            "13": 14,
            "14": 30,
            "20": 51,
        }

    def test_artefact_means_reproduce_de_varda(self):
        """de Varda et al. (2023) average every component over the trials
        that are not artefacts; their 1,726 by 6 values must come back."""
        devarda = get("devarda2023").default_path()
        table = next(
            (
                devarda / n
                for n in ("all_measures.csv", "aligned.csv")
                if (devarda / n).exists()
            ),
            None,
        )
        if table is None:
            pytest.skip(f"de Varda et al. (2023) data not found at {devarda}")
        dataset = Frank2015ERP(exclusions="artefact")
        dataset.load(str(self.path))
        stacked = {}
        for datum in dataset._data:
            stacked.setdefault((datum.stimulus.text, datum.measure_name), []).append(
                datum.values
            )
        checked = 0
        with open(table, newline="") as stream:
            for row in csv.DictReader(stream):
                position = int(row["context_length"])
                for name in COMPONENTS:
                    column = np.stack(stacked[(row["sentence"], name)])[:, position]
                    assert np.isfinite(column).any()
                    assert abs(np.nanmean(column) - float(row[name.upper()])) < 1e-9
                    checked += 1
        assert checked == 1726 * len(COMPONENTS)

    def test_stimulus_identity_is_shared_with_the_ucl_loaders(self):
        eye_tracking = self._sibling("ucl_et")
        erp = {s.stimulus_id: s.text for s in self.ds.stimuli()}
        assert erp == {s.stimulus_id: s.text for s in eye_tracking.stimuli()}
        spr = {s.stimulus_id for s in self._sibling("ucl_spr").stimuli()}
        assert set(erp) <= spr

    def test_norms_attach_to_the_participant_level_loaders(self):
        devarda = get("devarda2023").default_path()
        if not devarda.exists():
            pytest.skip(f"de Varda et al. (2023) data not found at {devarda}")
        assert self.ds.attach_predictability_norms() == 205
        assert self.ds.devarda_unmatched == []
        norms = read_devarda_norms(devarda)
        for stimulus in self.ds.stimuli():
            cloze = self.ds.covariates(stimulus)["cloze"]
            assert np.isnan(cloze[0])
            for position, values in norms[stimulus.text].items():
                assert cloze[position] == values["cloze"]
        eye_tracking = self._sibling("ucl_et")
        assert eye_tracking.attach_predictability_norms() == 205
        # The self-paced reading release displayed one sentence with a typo,
        # so that sentence's norms stay unmatched.
        spr = self._sibling("ucl_spr")
        assert spr.attach_predictability_norms() == 204
        assert spr.devarda_unmatched == ["Scott got up and washed his plate."]
