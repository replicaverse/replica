"""Reject inputs that would silently change stimulus or observation identity."""

import csv
from pathlib import Path

import numpy as np
import pytest

from replica.dataset.catalog.facts import counts_for
from replica.dataset.eye_tracking.devarda2023 import MEASURE_COLUMNS, DeVarda2023
from replica.dataset.eye_tracking.geco import GECO
from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1
from replica.dataset.eye_tracking.zuco.zuco2 import ZuCo2
from replica.dataset.fmri.narratives import Narratives
from replica.dataset.spr.natural_stories import NaturalStories


def test_aggregate_means_do_not_imply_individual_flags(tmp_path):
    source = tmp_path / "all_measures.csv"
    rows = [
        {
            "sent_id": 1,
            "sentence": "one two three",
            "context_length": i,
            "list": i + 1,
            **dict.fromkeys(MEASURE_COLUMNS, value),
        }
        for i, value in enumerate([75.8095238095, 0.0])
    ]
    with source.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    dataset = DeVarda2023()
    dataset.load(str(tmp_path))
    assert counts_for(dataset)["participants_averaged"]
    assert dataset.participants == ["averaged"]
    assert not {"first_pass", "first_run"} & set(dataset.measures)
    assert dataset.stimuli()[0].metadata["cloze_lists"] == {0: 1, 1: 2}
    dataset.drop_skips = False
    np.testing.assert_equal(
        dataset.data("first_run_in_first_pass")[0].values, [75.8095238095, 0.0, np.nan]
    )
    dataset.drop_skips = True
    np.testing.assert_equal(
        dataset.data("first_run_in_first_pass")[0].values,
        [75.8095238095, np.nan, np.nan],
    )


def test_geco_requires_the_declared_duration_columns(tmp_path):
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "DATA"
    sheet.append(["PP_NR", "PART", "TRIAL", "WORD_ID_WITHIN_TRIAL", "WORD"])
    sheet.append([1, 1, 1, 1, "word"])
    source = tmp_path / "input.xlsx"
    workbook.save(source)
    with pytest.raises(ValueError, match="missing columns"):
        GECO()._load_xlsx(str(source), openpyxl)


@pytest.mark.parametrize("tokens", [["one", "two"], ["one", "two", "three"]])
def test_zuco1_checks_text_identity_across_participants(tmp_path, tokens):
    from scipy.io import savemat

    for subject, words in [("A", ["red", "fox"]), ("B", tokens)]:
        fields = np.array(
            [(word, 100, 200, 300, 400) for word in words],
            dtype=[(name, object) for name in ("content", "FFD", "GD", "GPT", "TRT")],
        )
        savemat(
            tmp_path / f"results{subject}_NR.mat", {"sentenceData": {"word": fields}}
        )
    with pytest.raises(ValueError, match="conflicting text"):
        ZuCo1().load(str(tmp_path))


def _zuco2_file(path, sentences):
    import h5py

    with h5py.File(path, "w") as stream:
        refs = []
        for i, words in enumerate(sentences):
            group = stream.create_group(f"s{i}")
            tokens = [
                stream.create_dataset(
                    f"s{i}w{j}", data=np.array([ord(c) for c in word])
                )
                for j, word in enumerate(words)
            ]
            group.create_dataset(
                "content",
                data=np.array([[t.ref] for t in tokens], dtype=h5py.ref_dtype),
            )
            for field in ("FFD", "GD", "GPT", "TRT"):
                value = stream.create_dataset(f"s{i}{field}", data=[[100]])
                group.create_dataset(
                    field,
                    data=np.array([[value.ref]] * len(words), dtype=h5py.ref_dtype),
                )
            refs.append([group.ref])
        stream.create_group("sentenceData").create_dataset(
            "word", data=np.array(refs, dtype=h5py.ref_dtype)
        )


def test_zuco2_checks_same_length_text_identity(tmp_path):
    _zuco2_file(tmp_path / "resultsA_NR.mat", [["one", "two"]])
    _zuco2_file(tmp_path / "resultsB_NR.mat", [["red", "fox"]])
    with pytest.raises(ValueError, match="conflicting text"):
        ZuCo2().load(str(tmp_path), load_eeg=False)


def test_zuco2_retains_sentences_absent_from_the_first_file(tmp_path):
    _zuco2_file(tmp_path / "resultsA_NR.mat", [["one", "two"]])
    _zuco2_file(tmp_path / "resultsB_NR.mat", [["one", "two"], ["red", "fox"]])
    dataset = ZuCo2()
    dataset.load(str(tmp_path), load_eeg=False)
    assert [s.text for s in dataset.stimuli()] == ["one two", "red fox"]
    assert len(dataset.data("first_run", participant="B")) == 2
    assert not any(m.startswith("eeg_") for m in dataset.measures)


def test_zuco2_does_not_silently_remove_an_unreadable_participant(tmp_path):
    _zuco2_file(tmp_path / "resultsA_NR.mat", [["one", "two"]])
    (tmp_path / "resultsB_NR.mat").write_text("truncated")
    with pytest.raises(OSError):
        ZuCo2().load(str(tmp_path), load_eeg=False)


@pytest.mark.parametrize(
    "rows",
    [
        "one\t1\t1\n\t2\t1\nthree\t3\t1\n",
        "one\t1\t1\nthree\t3\t1\n",
        "one\t1\t1\ntwo\t1\t1\n",
    ],
)
def test_natural_stories_rejects_positions_that_would_shift_responses(tmp_path, rows):
    (tmp_path / "all_stories.tok").write_text("word\tzone\titem\n" + rows)
    with pytest.raises(
        ValueError, match="missing token identity|duplicate or missing zones"
    ):
        NaturalStories().load(str(tmp_path))


@pytest.mark.parametrize("load_arrays", [False, True])
def test_narratives_numpy_honors_selection_and_metadata_only(
    tmp_path, monkeypatch, load_arrays
):
    (tmp_path / "story").mkdir()
    (tmp_path / "story_align.csv").write_text("one,one,0,0.5\n")
    for subject in ("sub_1", "sub_2"):
        np.save(tmp_path / "story" / f"{subject}.npy", np.ones((3, 4)))
    original_load = np.load
    calls = []

    def checked_load(path, *args, **kwargs):
        calls.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked_load)
    dataset = Narratives(story="story", trim_trs=(0, 0))
    dataset.load(
        str(tmp_path),
        align_path=str(tmp_path),
        participants=["sub_1"],
        load_arrays=load_arrays,
    )
    assert dataset.participants == ["sub_1", "sub_2"]
    assert [r.participant_id for r in dataset.recordings()] == (
        ["sub_1"] if load_arrays else []
    )
    assert len(calls) == int(load_arrays)
    assert all(path.endswith("sub_1.npy") for path in calls)


def test_narratives_defaults_to_the_afni_derivatives(tmp_path):
    (tmp_path / "story_align.csv").write_text("one,one,0,0.5\n")
    func = tmp_path / "afni-nosmooth" / "sub-01" / "func"
    func.mkdir(parents=True)
    for hemi in ("L", "R"):
        name = f"sub-01_task-story_space-fsaverage6_hemi-{hemi}_desc-clean.func.gii"
        (func / name).touch()
    dataset = Narratives(story="story", trim_trs=(0, 0))
    dataset.load(str(tmp_path), align_path=str(tmp_path), load_arrays=False)
    assert dataset.participants == ["sub-01"]


def test_narratives_without_a_bold_source_raises(tmp_path):
    (tmp_path / "story_align.csv").write_text("one,one,0,0.5\n")
    dataset = Narratives(story="story", trim_trs=(0, 0))
    with pytest.raises(FileNotFoundError, match="No BOLD run for story 'story'"):
        dataset.load(str(tmp_path), align_path=str(tmp_path))


def test_narratives_subject_cap_applies_after_selection(tmp_path):
    (tmp_path / "story").mkdir()
    (tmp_path / "story_align.csv").write_text("one,one,0,0.5\n")
    for subject in ("sub_1", "sub_2"):
        np.save(tmp_path / "story" / f"{subject}.npy", np.ones((3, 4)))
    dataset = Narratives(story="story", trim_trs=(0, 0))
    dataset.load(
        str(tmp_path), align_path=str(tmp_path), participants=["sub_2"], max_subjects=1
    )
    assert dataset.participants == ["sub_1", "sub_2"]
    assert [r.participant_id for r in dataset.recordings()] == ["sub_2"]


def test_devarda_keeps_both_reading_presentations_after_load(tmp_path):
    path = tmp_path / "eye-tracking" / "processed" / "devarda2023"
    _write_rows(
        path / "aligned.csv",
        [
            {
                "sent_id": 1,
                "sentence": "one two",
                "context_length": 1,
                "list": 1,
                **dict.fromkeys(MEASURE_COLUMNS, 100.0),
            }
        ],
    )
    dataset = DeVarda2023()
    dataset.load(str(path))
    assert dataset.info.name == "de Varda et al. (2023)"
    assert dataset.presentation == ["natural reading", "self-paced reading"]


def test_devarda_records_each_measure_under_its_modality(tmp_path):
    _write_rows(
        tmp_path / "all_measures.csv",
        [
            {
                "sent_id": 1,
                "sentence": "one two",
                "context_length": 1,
                "list": 1,
                **dict.fromkeys(MEASURE_COLUMNS, 100.0),
            }
        ],
    )
    dataset = DeVarda2023()
    dataset.load(str(tmp_path))
    assert dataset.modalities == ["behavioral", "eeg", "eye-tracking"]
    by_modality = {
        modality: {d.measure_name for d in dataset.responses(modality=modality)}
        for modality in dataset.modalities
    }
    assert by_modality == {
        "behavioral": {"spr_rt"},
        "eeg": {"elan", "lan", "n400", "epnp", "p600", "pnp"},
        "eye-tracking": {
            "first_fixation_in_first_pass",
            "first_run_in_first_pass",
            "right_bounded",
            "go_past_in_first_pass",
        },
    }


def test_devarda_verifies_the_selected_alias(tmp_path, monkeypatch):
    from replica.dataset.contracts import DatasetInfo, FileEntry
    from replica.dataset.metadata import io as metadata
    from replica.dataset.metadata.io import ChecksumMismatchError, compute_sha256

    declared = tmp_path / "aligned.csv"
    declared.write_text("verified bytes")
    (tmp_path / "all_measures.csv").write_text("different bytes")
    info = DatasetInfo(
        name="devarda2023",
        files=[FileEntry(path="aligned.csv", checksum=compute_sha256(declared))],
    )
    monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
    monkeypatch.setattr(
        metadata, "find_dataset_yaml", lambda _: tmp_path / "dataset.yaml"
    )
    monkeypatch.setattr(metadata, "load_yaml", lambda _: info)
    with pytest.raises(ChecksumMismatchError, match="all_measures.csv"):
        DeVarda2023().load(str(tmp_path))


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _eyelink_rows():
    from replica.dataset.measures import EYELINK_IA_MEASURES

    return [
        {
            "IA_ID": i,
            "IA_LABEL": word,
            "Text_ID": 1,
            "Word_Number": i + 1,
            "Word_Cleaned": word,
            "Participant_ID": "p1",
            "participant_id": "p1",
            "TRIAL_INDEX": 1,
            "article_id": 1,
            "article_batch": 1,
            "paragraph_id": 1,
            "difficulty_level": "Adv",
            "question_preview": "False",
            "repeated_reading_trial": "False",
            "practice_trial": "False",
            **dict.fromkeys(EYELINK_IA_MEASURES, 200),
            "IA_SKIP": 0,
        }
        for i, word in enumerate(["one", "two"], 1)
    ]


def _copco_rows():
    from replica.dataset.eye_tracking.copco import MEASURE_COLUMNS

    return [
        {
            "speechId": 1,
            "paragraphId": 0,
            "sentenceId": 1,
            "wordId": i,
            "trialId": 1,
            "word": word,
            **dict.fromkeys(MEASURE_COLUMNS, 200),
        }
        for i, word in enumerate(["one", "two"], 1)
    ]


@pytest.mark.parametrize(
    ("key", "required", "optional", "rows"),
    [
        (
            "provo",
            "Provo_Corpus-Eyetracking_Data.csv",
            [
                "Provo_Corpus-Predictability_Norms.csv",
                "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv",
            ],
            _eyelink_rows,
        ),
        ("onestop", "ia_Paragraph.csv", ["fixations_Paragraph.csv"], _eyelink_rows),
        (
            "copco",
            "ExtractedFeatures/P*.csv",
            ["FixationReports/FIX_report_P*.txt"],
            _copco_rows,
        ),
    ],
)
def test_verification_skips_optional_files_that_are_absent(
    tmp_path, monkeypatch, key, required, optional, rows
):
    from replica.dataset import load_dataset
    from replica.dataset.catalog.registry import REGISTRY
    from replica.dataset.contracts import DatasetInfo, FileEntry
    from replica.dataset.metadata import io as metadata
    from replica.dataset.metadata.io import compute_aggregate_sha256, compute_sha256

    shipped = metadata.load_yaml(
        metadata.find_dataset_yaml(Path(REGISTRY[key].default_subdir))
    )
    assert {entry.path for entry in shipped.files} == {required, *optional}
    glob = "*" in required
    source = tmp_path / required.replace("*", "01")
    _write_rows(source, rows())
    checksum = (
        compute_aggregate_sha256(tmp_path, [str(source.relative_to(tmp_path))])
        if glob
        else compute_sha256(source)
    )
    info = DatasetInfo(
        name=key,
        files=[
            FileEntry(path=required, checksum=checksum, n_files=int(glob)),
            *(
                FileEntry(path=name, checksum="sha256:" + "0" * 64, n_files=int(glob))
                for name in optional
            ),
        ],
    )
    monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
    monkeypatch.setattr(
        metadata, "find_dataset_yaml", lambda _, **kwargs: tmp_path / "dataset.yaml"
    )
    monkeypatch.setattr(metadata, "load_yaml", lambda _: info)
    dataset = load_dataset(key, path=tmp_path)
    assert dataset.stimuli()
    source.write_text(source.read_text() + "\n")
    with pytest.raises(metadata.ChecksumMismatchError):
        load_dataset(key, path=tmp_path)


def test_zuco_verifies_only_selected_task_files(tmp_path, monkeypatch):
    from replica.dataset.contracts import DatasetInfo, FileEntry
    from replica.dataset.metadata import io as metadata
    from replica.dataset.metadata.io import ChecksumMismatchError

    (tmp_path / "resultsA_NR.mat").write_text("wrong bytes")
    info = DatasetInfo(
        name="zuco2",
        files=[
            FileEntry(path="resultsA_NR.mat", checksum="sha256:" + "0" * 64),
            FileEntry(path="resultsA_TSR.mat", checksum="sha256:" + "1" * 64),
        ],
    )
    monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
    monkeypatch.setattr(
        metadata, "find_dataset_yaml", lambda _: tmp_path / "dataset.yaml"
    )
    monkeypatch.setattr(metadata, "load_yaml", lambda _: info)
    with pytest.raises(ChecksumMismatchError, match="resultsA_NR.mat"):
        ZuCo2().load(str(tmp_path))


@pytest.mark.parametrize("word", ["", " ", float("nan")])
def test_meco_nonwords_cannot_become_model_context(word):
    from replica.dataset.eye_tracking.meco.word_utils import clean_word

    with pytest.raises(ValueError, match="nonword handling"):
        clean_word(word)
