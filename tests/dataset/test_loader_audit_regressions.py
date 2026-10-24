"""Release defects must not shift responses or corrupt predictor context."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from replica.dataset import PROVO, SBSAT, CopCo, OneStop, PoTeC
from replica.dataset.catalog.sources import DATASETS
from replica.dataset.eye_tracking.geco import _word_regions
from replica.dataset.eye_tracking.meco.waves import MECOL2W2
from replica.dataset.eye_tracking.onestop import _NEEDED_COLS
from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1
from replica.dataset.eye_tracking.zuco.zuco2 import ZuCo2
from replica.dataset.fetch import _fetch_dataset_source
from replica.dataset.measures import EYELINK_IA_MEASURES
from tests.dataset.eye_tracking.meco.test_meco_waves import _frame


@pytest.mark.parametrize("dataset_class", [ZuCo1, ZuCo2])
@pytest.mark.parametrize("measure", ["first_run", "total_time", "eeg_theta_1"])
def test_zuco_does_not_apply_an_implicit_duration_window(dataset_class, measure):
    from replica.core import Datum

    dataset = dataset_class()
    stimulus = dataset._make_stimulus(["one", "two", "three", "four", "five"], "test")
    datum = Datum(
        stimulus=stimulus,
        measure_name=measure,
        values=np.array([200, 50, 5000, np.nan, 200]),
    )
    np.testing.assert_equal(
        dataset.exclusion_mask(datum), [False, True, True, False, True]
    )


def _csv(path, rows, fields=None, delimiter=","):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields or list(rows[0]), delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("use_passage", [True, False])
@pytest.mark.parametrize("first_pass_filter", [True, False])
@pytest.mark.parametrize("text_id", [18, 42])
def test_provo_duplicate_word_number_preserves_both_words(
    tmp_path, use_passage, first_pass_filter, text_id
):
    words = ["After", "the", *[f"word{i}" for i in range(47)], "evolution."]
    rows = []
    for i, word in enumerate(words):
        row = {
            "Text_ID": text_id,
            "Word_Number": i + 2,
            "IA_ID": i + 2,
            "Word_Cleaned": word.rstrip("."),
            "Participant_ID": "p1",
            "IA_FIRST_FIXATION_TIME": i * 100 + 1,
            **{col: 100 + i for col in EYELINK_IA_MEASURES},
        }
        if i == 49:
            row["Word_Number"] = 3
        rows.append(row)
    rows[2]["IA_DWELL_TIME"] = 0
    path = tmp_path / "ia.csv"
    _csv(path, rows)
    ds = PROVO(first_pass_filter=first_pass_filter)
    ds._load_et(str(path), passages={text_id: words} if use_passage else None)
    [datum] = ds.data("total_time")
    assert datum.values[1] == 101
    assert datum.values[49] == 149
    assert np.isnan(datum.values[2])
    ds.drop_skips = False
    assert ds.data("total_time")[0].values[2] == 0
    ds.drop_skips = True
    assert np.isnan(ds.data("total_time")[0].values[2])
    assert datum.stimulus.layers["words"].values[-1].rstrip(".") == "evolution"


def test_provo_initial_area_has_durations_but_no_cloze_target(tmp_path):
    norms = tmp_path / "norms.csv"
    _csv(
        norms,
        [
            {
                "Text_ID": 1,
                "Text": "There are now rumblings.",
                "Word_Number": 2,
                "Word": "are",
                "Response": "are",
                "Response_Proportion": 0.5,
            }
        ],
    )
    source = tmp_path / "ia.csv"
    _csv(
        source,
        [
            {
                "Text_ID": 1,
                "IA_ID": i,
                "IA_LABEL": word + " ",
                "Word_Number": "NA" if i == 1 else i,
                "Word_Cleaned": "NA" if i == 1 else word,
                "Participant_ID": "p1",
                **dict.fromkeys(EYELINK_IA_MEASURES, 169 if i == 1 else 0),
                "IA_SKIP": 0 if i == 1 else 1,
            }
            for i, word in enumerate(["There", "are", "now", "rumblings."], 1)
        ],
    )
    ds = PROVO()
    ds._load_et(str(source), passages=ds._load_passages(str(norms)))
    ds._load_cloze(str(norms))
    [stimulus] = ds.stimuli()
    assert stimulus.text == "There are now rumblings."
    assert stimulus.metadata["source_ia_to_unit"] == {1: 0, 2: 1, 3: 2, 4: 3}
    np.testing.assert_equal(
        stimulus.layers["human_cloze"].values, [np.nan, 0.5, np.nan, np.nan]
    )
    for drop in [True, False, True]:
        ds.drop_skips = drop
        for datum in ds.data():
            if datum.measure_name == "first_pass":
                np.testing.assert_equal(datum.values, [1, 0, 0, 0])
                continue
            np.testing.assert_equal(
                datum.values, [169, *([np.nan] * 3 if drop else [0] * 3)]
            )


def test_provo_official_regions_and_shifted_cloze_targets(tmp_path):
    source = json.loads(
        (Path(__file__).parent / "fixtures/provo_source_alignment.json").read_text()
    )
    ia, norms = tmp_path / "ia.csv", tmp_path / "norms.csv"
    _csv(ia, source["areas"])
    rows = [dict(r, Text=source["passages"][r["Text_ID"]]) for r in source["norms"]]
    with norms.open("w", newline="", encoding="mac_roman") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ds = PROVO()
    ds._load_et(str(ia), passages=ds._load_passages(str(norms)))
    ds._load_cloze(str(norms))
    stimuli = {s.metadata["text_id"]: s for s in ds.stimuli()}
    for row in source["areas"]:
        t, area = int(row["Text_ID"]), int(row["IA_ID"])
        assert stimuli[t].metadata["source_ia_to_unit"][area] == area - 1
    assert stimuli[55].unit_strings()[8] == "livres--a"
    assert stimuli[55].unit_strings()[-1] == "profession--writing."
    assert len(stimuli[55].layers["words"]) == 60
    for text, source_number, position, word in [
        (3, 46, 44, "While"),
        (3, 59, 57, "nationwide."),
        (13, 20, 18, '"I'),
        (13, 54, 52, "carts."),
        (18, 3, 2, "the"),
        (36, 24, 24, "betting"),
        (36, 39, 39, "world."),
        (55, 11, 9, "large"),
    ]:
        s = stimuli[text]
        assert s.unit_strings()[position] == word
        assert s.metadata["source_cloze_to_unit"][source_number] == position
        expected = sum(
            float(r["Response_Proportion"])
            for r in rows
            if int(r["Text_ID"]) == text
            and int(r["Word_Number"]) == source_number
            and r["Word"].lower() == r["Response"].lower()
        )
        assert s.layers["human_cloze"].values[position] == expected
    assert np.isnan(np.asarray(stimuli[55].layers["human_cloze"].values)[[8, 59]]).all()
    assert len(stimuli[18].metadata["source_unaligned_cloze"]) == 17


def test_provo_unverified_area_cannot_receive_a_neighbours_duration():
    with pytest.raises(ValueError, match="Unaligned Provo area"):
        PROVO()._align_wn_to_passage([(1, "wrong")], ["right"])


def test_onestop_excludes_practice_and_aligns_each_readers_variant(tmp_path):
    rows = []
    sequences = {"p1": ["deep-fried", "food"], "p2": ["deep-", "fried", "food"]}
    for participant, words in sequences.items():
        for i, word in enumerate(words, start=1):
            rows.append(
                {
                    "participant_id": participant,
                    "TRIAL_INDEX": 1,
                    "IA_ID": i,
                    "IA_LABEL": word,
                    "article_id": 1,
                    "article_batch": 1,
                    "paragraph_id": 1,
                    "difficulty_level": "Adv",
                    "question_preview": "False",
                    "repeated_reading_trial": "False",
                    "practice_trial": "False",
                    **{col: 100 * i for col in EYELINK_IA_MEASURES},
                }
            )
    rows[0]["IA_DWELL_TIME"] = 0
    rows.append({**rows[0], "participant_id": "practice", "practice_trial": "True"})
    path = tmp_path / "ia.csv"
    _csv(path, rows, fields=sorted(_NEEDED_COLS))
    ds = OneStop()
    ds._load_ia_report(str(path))
    assert len(ds.stimuli()) == 2
    assert set(ds.participants) == {"p1", "p2"}
    for datum in ds.data("total_time"):
        words = sequences[datum.participant_id]
        assert datum.stimulus.layers["words"].values == words
        assert datum.values[-1] == 100 * len(words)
    assert np.isnan(
        next(d for d in ds.data("total_time") if d.participant_id == "p1").values[0]
    )
    ds.drop_skips = False
    assert ds.data("total_time", participant="p1")[0].values[0] == 0.0
    ds.drop_skips = True
    assert np.isnan(ds.data("total_time", participant="p1")[0].values[0])

    # Both readers' last IA refers to "food", despite having different IDs.
    fix_rows = []
    for participant, words in sequences.items():
        fix_rows.append(
            {
                **dict.fromkeys(ds._FIX_COLS, "1"),
                "participant_id": participant,
                "difficulty_level": "Adv",
                "question_preview": "False",
                "repeated_reading_trial": "False",
                "practice_trial": "False",
                "CURRENT_FIX_INTEREST_AREA_INDEX": len(words),
                "CURRENT_FIX_START": 0,
                "CURRENT_FIX_END": 100,
                "CURRENT_FIX_DURATION": 100,
            }
        )
    fix_rows.append({**fix_rows[0], "practice_trial": "True"})
    fix_path = tmp_path / "fix.csv"
    _csv(fix_path, fix_rows)
    ds._fixations = []
    ds._load_fixations(str(fix_path))
    assert len(ds.fixations()) == 2
    stimuli = {s.stimulus_id: s for s in ds.stimuli()}
    for fixation in ds.fixations():
        assert (
            stimuli[fixation.stimulus_id].layers["words"].values[fixation.unit_index]
            == "food"
        )


def _sbsat_row(book, page, ia, label, duration=100, trial=1):
    return {
        "type": "reading",
        "book_name": book,
        "page": page,
        "RECORDING_SESSION_LABEL": "p1",
        "TRIAL_INDEX": trial,
        "CURRENT_FIX_INTEREST_AREA_ID": ia,
        "CURRENT_FIX_INTEREST_AREA_LABEL": label,
        "CURRENT_FIX_DURATION": duration,
        "CURRENT_FIX_INTEREST_AREA_DWELL_TIME": duration,
    }


def test_sbsat_off_text_events_advance_time_without_adding_word_dwell(tmp_path):
    from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer

    rows = [
        _sbsat_row("example", 1, 1, "one", 100),
        _sbsat_row("example", 1, "", "", 70),
        _sbsat_row("example", 1, 1, "one", 90),
        _sbsat_row("example", 1, 2, "two", 80),
    ]
    path = tmp_path / "fix.csv"
    _csv(path, rows)
    dataset = SBSAT()
    dataset._load_fixation_data(str(path))
    events = dataset.fixations()
    assert [f.unit_index for f in events] == [0, None, 0, 1]
    assert [f.start_time for f in events] == [0, 100, 170, 260]
    np.testing.assert_equal(dataset.data("total_time")[0].values, [190, 80])
    result = aggregate_fixations_to_layer(events, dataset.stimuli()[0])
    np.testing.assert_equal(dataset.data("first_run")[0].values, [100, 80])
    assert [event.metadata.get("run_id") for event in events] == [1, None, 2, 1]
    np.testing.assert_equal(result["first_run"], [100, 80])


def test_sbsat_total_includes_page_revisits_and_merges_line_wrapped_word(tmp_path):
    rows = [
        _sbsat_row("genome", 3, 44, "was"),
        _sbsat_row("genome", 3, 45, "gee-whiz,", 110),
        _sbsat_row("genome", 3, 46, "gee-whiz,", 120),
        _sbsat_row("genome", 3, 47, "fun"),
        _sbsat_row("genome", 3, 45, "gee-whiz,", 130, trial=2),
    ]
    path = tmp_path / "fix.csv"
    _csv(path, rows)
    ds = SBSAT()
    ds._load_fixation_data(str(path))
    assert ds.stimuli()[0].layers["words"].values == ["was", "gee-whiz,", "fun"]
    assert ds.data("first_fixation")[0].values[1] == 110
    assert ds.data("first_run")[0].values[1] == 230
    assert ds.data("first_pass")[0].values[1] == 1
    assert ds.data("total_time")[0].values[1] == 360
    assert sum(f.duration for f in ds.fixations() if f.unit_index == 1) == 360


def test_sbsat_recovers_ampersand_and_punctuation_without_inventing_durations(tmp_path):
    path = tmp_path / "fix.csv"
    _csv(
        path,
        [
            _sbsat_row("dickens", 3, 45, "Sempere"),
            _sbsat_row("dickens", 3, 46, "Sons"),
            _sbsat_row("dickens", 3, 47, "didn\x92t"),
        ],
    )
    ds = SBSAT()
    ds._load_fixation_data(str(path))
    assert ds.stimuli()[0].layers["words"].values == ["Sempere &", "Sons", "didn’t"]
    np.testing.assert_equal(ds.data("total_time")[0].values, [100, 100, 100])
    ds.drop_skips = False
    np.testing.assert_equal(ds.data("total_time")[0].values, [100, 100, 100])
    assert ds.fixations()[0].unit_index == 0
    assert ds.fixations()[1].unit_index == 1


@pytest.mark.parametrize("book,page,area", [("dickens", 3, 45), ("genome", 3, 46)])
def test_sbsat_rejects_changed_repair_targets(tmp_path, book, page, area):
    path = tmp_path / "fix.csv"
    _csv(path, [_sbsat_row(book, page, area, "unexpected")])
    with pytest.raises(ValueError, match="repair no longer matches"):
        SBSAT()._load_fixation_data(str(path))


@pytest.mark.parametrize("defect", ["schema", "identity", "position", "text"])
def test_rsc_source_errors_cannot_silently_drop_rows(tmp_path, defect):
    from replica.dataset.eye_tracking.rsc import RSC

    row = {
        "item.id": 1,
        "word.serial.no": 1,
        "DATA_FILE": "1.edf",
        "item.text": "One sentence.",
        **dict.fromkeys(EYELINK_IA_MEASURES, 200),
        "IA_SKIP": 0,
    }
    rows = [row.copy(), row.copy()]
    if defect == "schema":
        for entry in rows:
            del entry["item.id"]
    elif defect == "identity":
        rows[1]["item.id"] = "invalid"
    elif defect == "position":
        rows[1]["word.serial.no"] = 3
    else:
        rows[1]["item.text"] = "Different sentence."
    path = tmp_path / "data.csv"
    _csv(path, rows, delimiter="\t")
    with pytest.raises(ValueError, match="RSC"):
        RSC._read_measures(path)


def test_meco_multiword_areas_preserve_responses_and_event_alignment(tmp_path):
    frame = _frame()
    frame.loc[(frame.itemid == "1") & (frame.wordnum == 2), "word"] = "5.\xa0Juni"
    ds = MECOL2W2()
    ds._load_frame(frame)
    ds._configure_fixation_store(tmp_path)
    [stim] = [s for s in ds.stimuli() if s.metadata["text_id"] == 1]
    assert stim.text == "text1 5. Juni"
    assert stim.layers["words"].values == ["text1", "5. Juni"]
    assert stim.metadata["source_word_numbers"] == [1, 2]
    for datum in ds._data:
        if datum.stimulus is stim:
            np.testing.assert_array_equal(np.isnan(datum.values), [False, False])
    assert ds._fixation_lookup[("site_1", 1)].word_indices == {1: 0, 2: 1}


def test_meco_l1_blank_space_areas_do_not_become_underscores(tmp_path, monkeypatch):
    import pandas as pd

    from replica.dataset.eye_tracking.meco.l1 import MECO_L1

    frame = _frame()
    frame["lang"] = "ee"
    frame["subid"] = frame.uniform_id
    blank = frame[frame.itemid == "8"].copy()
    blank = blank[blank.wordnum == 2]
    blank["wordnum"] = 9
    blank["word"] = ""
    blank["dur"] = 355.0
    frame.loc[(frame.itemid == "8") & (frame.wordnum == 2), "wordnum"] = 10
    frame = pd.concat([frame, blank], ignore_index=True)
    monkeypatch.setattr(
        "replica.dataset.eye_tracking.meco.l1.read_r_dataframe", lambda _: frame
    )
    ds = MECO_L1(language="ee")
    ds._load_rda("fixture.rda")
    ds._configure_fixation_store(tmp_path)
    stimulus = next(s for s in ds.stimuli() if s.metadata["text_id"] == 8)
    assert stimulus.text == "text8 alpha-beta"
    assert stimulus.metadata["source_word_numbers"] == [1, 10]
    assert stimulus.metadata["source_nonword_numbers"] == [9]
    datum = next(d for d in ds._data if d.stimulus is stimulus)
    assert datum.metadata["source_nonword_measures"][0]["total_time"] == 355.0
    mapping = ds._fixation_lookup[("site_1", 8)]
    assert stimulus.layers[mapping.unit_layer].values[mapping.word_indices[9]] == " "


@pytest.mark.parametrize(("reported", "actual"), [(3, 5), (6, 10), (7, 11)])
def test_meco_l2_corrects_text_identity_and_retains_reported_event_key(
    tmp_path, reported, actual
):
    import pandas as pd

    frame = _frame()
    mislabeled = frame[
        (frame.uniform_id == "site_1") & (frame.itemid == str(actual))
    ].copy()
    mislabeled["uniform_id"] = "ch_s_1"
    mislabeled["itemid"] = str(reported)
    ds = MECOL2W2()
    ds._load_frame(pd.concat([frame, mislabeled], ignore_index=True))
    ds._configure_fixation_store(tmp_path)
    assert len(ds.stimuli()) == 12
    [datum] = [d for d in ds.data("total_time") if d.participant_id == "ch_s_1"]
    assert datum.stimulus.metadata["text_id"] == actual
    assert datum.metadata["source_item_id"] == reported
    assert ds._fixation_lookup[("ch_s_1", reported)].stimulus is datum.stimulus


def test_geco_only_reconciles_identified_export_damage():
    from replica.dataset.eye_tracking.geco import _reconcile_dutch_label

    assert _reconcile_dutch_label("Wel", '"Wel', "3-5-23") == '"Wel'
    assert _reconcile_dutch_label("intu´tie", "intuïtie", "3-19-34") == "intuïtie"
    assert _reconcile_dutch_label("klap", "#NAME?", "4-65-102") == "klap"
    with pytest.raises(ValueError, match="Conflicting"):
        _reconcile_dutch_label("cat", "dog", "1-1-1")


@pytest.mark.parametrize("label", ["null", "0", "nan"])
def test_potec_official_layout_preserves_literal_null_and_sentences(tmp_path, label):
    from replica.dataset.catalog.sentences import attach_sentences

    rows = [
        dict(
            FD=0,
            FRT=0,
            FFD=0,
            FPRT=0,
            TFT=0,
            RPD_inc=0,
            word_index_in_text=140,
            sent_index_in_text=3,
            word_with_punct=label,
        )
    ]
    _csv(tmp_path / "reader1_p3_merged.tsv", rows, delimiter="\t")
    ds = PoTeC(measurement_source="released")
    ds.load(str(tmp_path))
    [stimulus] = ds.stimuli()
    assert stimulus.text == "null"
    assert stimulus.metadata["source_sentence_numbers"] == [3]
    attach_sentences("potec", [stimulus])
    assert stimulus.layers["sentences"].extract_strings(stimulus.text) == ["null"]
    assert np.isnan(ds.data("first_run")[0].values[0])
    assert ds.data("first_pass")[0].values[0] == 0
    ds.drop_skips = False
    assert ds.data("first_run")[0].values[0] == 0


def test_copco_recovers_only_label_validated_missing_character_events(tmp_path):
    from replica.dataset.eye_tracking.copco import MEASURE_COLUMNS

    features = tmp_path / "ExtractedFeatures"
    features.mkdir()
    _csv(
        features / "P01.csv",
        [
            {
                "speechId": 1,
                "paragraphId": 0,
                "sentenceId": 1,
                "wordId": 1,
                "trialId": 1,
                "word": "Røn",
                "char_IA_ids": "[1, 3]",
                **dict.fromkeys(MEASURE_COLUMNS, 100),
                "word_total_fix_dur": 200,
            }
        ],
    )
    fix_dir = tmp_path / "FixationReports"
    fix_dir.mkdir()
    rows = [
        {
            "RECORDING_SESSION_LABEL": "P01",
            "paragraphid": 0,
            "speechid": 1,
            "TRIAL_INDEX": 1,
            "CURRENT_FIX_INDEX": i,
            "CURRENT_FIX_DURATION": duration,
            "CURRENT_FIX_INTEREST_AREA_INDEX": area,
            "CURRENT_FIX_INTEREST_AREA_LABEL": label,
        }
        for i, (area, label, duration) in enumerate(
            [(1, "R", 100), (2, "Ã¸", 30), (3, "n", 100), (2, "g", 40)], start=1
        )
    ]
    _csv(fix_dir / "FIX_report_P01.txt", rows, delimiter="\t")
    ds = CopCo()
    ds.load(str(tmp_path))
    assert [f.duration for f in ds.fixations()] == [100, 30, 100, 40]
    assert [f.metadata["inferred_character_mapping"] for f in ds.fixations()] == [
        False,
        True,
        False,
        False,
    ]
    assert [f.unit_index for f in ds.fixations()] == [0, 0, 0, None]
    # Published per-word summaries retain their original character inventory.
    assert ds.data("total_time")[0].values[0] == 200


def test_geco_regions_preserve_context_and_only_identifiable_responses():
    words, ids, mapping = _word_regions(
        {1: "well", 2: 'enough..." blow', 4: "them"}, (1, 107), english=True
    )
    assert words == ["well", 'enough..." blow', "them"]
    assert ids == [1, 2, 4]
    assert mapping == {1: 0, 2: 1, 4: 2}
    words, _, mapping = _word_regions(
        {43: 'freely..." miserable', 44: 'freely..." miserable', 45: "animal"},
        (2, 57),
        english=True,
    )
    assert words == ['freely..."', "miserable", "animal"]
    assert mapping == {43: 0, 44: 1, 45: 2}
    words, _, mapping = _word_regions(
        {18: "gravely.", 19: "Unknown 19", 20: "'To"}, (3, 132), english=True
    )
    assert words == ["gravely.", "'To"]
    assert mapping == {18: 0, 20: 1}


def test_potec_uses_corrected_words_and_explicit_positions(tmp_path):
    directory = tmp_path / "reading_measures"
    directory.mkdir()
    common = {
        "FD": 100,
        "FRT": 200,
        "FFD": 100,
        "FPRT": 200,
        "TFT": 300,
        "RPD_inc": 250,
    }
    rows = [
        {
            **common,
            "word_with_punct": "weiter.",
            "word_index_in_text": 2,
            "sent_index_in_text": 2,
        },
        {
            **common,
            "word_with_punct": '"Agarose-Sieb"',
            "word_index_in_text": 1,
            "sent_index_in_text": 1,
        },
    ]
    _csv(directory / "reader1_b0_rm.tsv", rows, delimiter="\t")
    ds = PoTeC(measurement_source="released")
    ds.load(str(tmp_path))
    assert ds.stimuli()[0].layers["words"].values == ['"Agarose-Sieb"', "weiter."]
    assert ds.stimuli()[0].metadata["sentence_boundaries"] == [0, 1]
    bad_root = tmp_path / "inconsistent"
    bad_directory = bad_root / "reading_measures"
    bad_directory.mkdir(parents=True)
    _csv(bad_directory / "reader1_b0_rm.tsv", rows, delimiter="\t")
    rows[0]["word_with_punct"] = "wrong"
    _csv(bad_directory / "reader2_b0_rm.tsv", rows, delimiter="\t")
    with pytest.raises(ValueError, match="Word alignment mismatch"):
        PoTeC(measurement_source="released").load(str(bad_root))


def test_copco_fetch_covers_the_loader_inputs_without_raw_gaze(tmp_path, monkeypatch):
    from replica.dataset import fetch

    downloaded = []

    def download(file_id, out):
        assert out.parent.is_dir()
        downloaded.append((file_id, out.relative_to(tmp_path).as_posix()))
        out.touch()

    monkeypatch.setattr(fetch, "download_osf", download)
    _fetch_dataset_source(DATASETS["copco"], tmp_path, force=False)
    names = {name for _, name in downloaded}
    assert len([name for name in names if name.startswith("ExtractedFeatures/")]) == 57
    assert "ExtractedFeatures/P39.csv" in names
    assert "FixationReports/FIX_report_P39.txt" in names
    assert not list(tmp_path.glob("P*.csv"))
    assert not list(tmp_path.glob("*.zip"))


def test_sbsat_page_departure_ends_gaze_even_when_the_next_visit_starts_on_same_word(
    tmp_path,
):
    path = tmp_path / "fix.csv"
    _csv(
        path,
        [
            _sbsat_row("test", 1, 4, "word", 90, trial=1),
            _sbsat_row("test", 1, 4, "word", 100, trial=2),
        ],
    )
    ds = SBSAT()
    ds._load_fixation_data(str(path))
    assert ds.data("first_run")[0].values[0] == 90
    assert ds.data("first_pass")[0].values[0] == 1
    assert ds.data("total_time")[0].values[0] == 190


def test_invalid_stimulus_identity_is_not_silently_dropped(tmp_path):
    directory = tmp_path / "ExtractedFeatures"
    directory.mkdir()
    (directory / "P01.csv").write_text("speechId,paragraphId\nbroken,1\n")
    with pytest.raises(ValueError, match="Invalid CopCo speech or paragraph ID"):
        CopCo().load(str(tmp_path))


def test_fixation_report_preserves_literal_quote_labels():
    from io import StringIO

    from replica.dataset.eye_tracking.copco import _fixation_rows

    header = [
        '"RECORDING_SESSION_LABEL"',
        '"paragraphid"',
        '"speechid"',
        '"TRIAL_INDEX"',
        '"CURRENT_FIX_DURATION"',
        '"CURRENT_FIX_INDEX"',
        '"CURRENT_FIX_INTEREST_AREA_LABEL"',
    ]
    rows = [
        ['"P01"', "0", "202201", "11", "193", "114", '"""'],
        ['"P01"', "0", "202201", "11", "133", "115", '"a"'],
    ]
    source = StringIO("\n".join("\t".join(row) for row in [header, *rows]))
    parsed = list(_fixation_rows(source))
    assert [row["CURRENT_FIX_INTEREST_AREA_LABEL"] for row in parsed] == ['"', "a"]
    assert [row["CURRENT_FIX_DURATION"] for row in parsed] == ["193", "133"]
    assert [row["speechid"] for row in parsed] == ["202201", "202201"]


def test_wrong_fixation_report_schema_raises():
    from io import StringIO

    from replica.dataset.eye_tracking.copco import _fixation_rows

    with pytest.raises(ValueError, match="missing columns"):
        list(_fixation_rows(StringIO("participant\tduration\np1\t100\n")))


def test_maze_identity_and_source_answers_survive_selection(tmp_path, monkeypatch):
    import pandas as pd

    from replica.dataset import NaturalStoriesMaze

    frame = pd.DataFrame(
        [
            {
                "sentence": sentence,
                "subject": "p",
                "word_num": i,
                "word": word,
                "type": "story",
                "correct": "no" if i == 2 else "yes",
                "rt": rt,
            }
            for sentence, words in [
                ("One two three.", ["One", "two", "three."]),
                ("Four five six.", ["Four", "five", "six."]),
            ]
            for i, (word, rt) in enumerate(zip(words, [100, 0, 300], strict=True))
        ]
    )
    identities = []
    for rows in [frame, frame.iloc[::-1]]:
        monkeypatch.setattr(
            NaturalStoriesMaze, "_read", staticmethod(lambda _, rows=rows: rows.copy())
        )
        dataset = NaturalStoriesMaze(analysed_sample=False, post_error="keep")
        dataset.load(str(tmp_path))
        identities.append({s.text: s.stimulus_id for s in dataset.stimuli()})
        for datum in dataset.data("maze_rt"):
            np.testing.assert_equal(datum.values, [100, 0, np.nan])
            assert not dataset.exclusion_mask(datum).any()
            assert datum.metadata["source_correct"] == ["yes", "yes", "no"]
            assert datum.metadata["source_rt"] == [100, 0, 300]
    assert identities[0] == identities[1]


def two_study_n400(tmp_path):
    """Two studies' norms and baselines of one sentence, built from a small TSV.

    Args:
        tmp_path: Directory that receives the TSV the loader reads.
    """
    import pandas as pd

    from replica.dataset.eeg.szewczyk_n400 import SzewczykN400

    frame = pd.DataFrame(
        [
            {
                "FullSentence": "A bird.",
                "dataset": study,
                "Subject": subject,
                "cloze_p": 0.1,
                "logfreq": frequency,
                "concr": 4.0,
                "old20": 2.0,
                "n400": amplitude,
                "bline": baseline,
            }
            for study, subject, frequency, amplitude, baseline in [
                ("study1", "p1", 1.0, -3.0, 0.2),
                ("study2", "p2", 2.0, -4.0, 0.7),
            ]
        ]
    )
    frame.to_csv(tmp_path / "szewczyk_2022.tsv", sep="\t", index=False)
    ann = pd.DataFrame([{"article": "1", "sent_id": 0, "tokenN_in_sent": 1}])
    dataset = SzewczykN400()
    dataset._build_per_subject(tmp_path / "annotation", ann, {"1": [["A", "bird."]]})
    return dataset


def test_n400_keeps_study_norms_separate_and_participant_baselines(tmp_path):
    dataset = two_study_n400(tmp_path)
    assert len(dataset.stimuli()) == 2
    for datum in dataset.data("n400"):
        expected = 1.0 if datum.participant_id == "p1" else 2.0
        assert dataset.covariates(datum.stimulus)["log_frequency"][1] == expected
    baseline = {
        datum.participant_id: datum.values[1] for datum in dataset.data("n400_baseline")
    }
    assert baseline == {"p1": 0.2, "p2": 0.7}


def test_bsc_fixation_indices_are_one_based_within_each_trial():
    from replica.dataset.eye_tracking.bsc import BSC

    dataset = BSC()
    stimuli, indices = dataset._build_stimuli({1: [(1, "一"), (2, "二")]})
    rows = [("p1", 1, 1, 0, 200.0), ("p1", 1, 2, 1, 150.0), ("p2", 1, 2, 0, 90.0)]
    dataset._append_raw_fixations(rows, stimuli, indices)
    indices_by_reader = {}
    for event in dataset.fixations():
        indices_by_reader.setdefault(event.participant_id, []).append(
            event.fixation_index
        )
    assert indices_by_reader == {"p1": [1, 2], "p2": [1]}
