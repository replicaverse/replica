"""Source parsers must distinguish zero duration from unavailable measurements."""

import csv

import numpy as np
import pandas as pd
import pytest

from replica.dataset import load_dataset
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.measures import (
    EYELINK_IA_MEASURES,
    FIRST_PASS_GO_PAST,
    FIRST_PASS_RUN,
    PYMOVEMENTS_FIRST_PASS_COLUMN,
    PYMOVEMENTS_RM_MEASURES,
)

WORDS = ["start", "skipped", "unavailable"]


def _write(path, rows, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _check(
    dataset,
    expected=(200.0, 0.0, np.nan),
    measure=FIRST_PASS_RUN,
    participant=None,
    source="default",
):
    """A duration reads its zero as a skip under the default view and as 0 ms otherwise."""
    stored = [d.values.copy() for d in dataset._data]
    expected = np.asarray(expected)
    for exclude in (True, False, True):
        dataset.drop_skips = exclude
        want = np.where(expected == 0, np.nan, expected) if exclude else expected
        for datum in dataset.data(measure, participant=participant, source=source):
            np.testing.assert_equal(datum.values, want)
            assert not dataset.exclusion_mask(datum)[np.isnan(expected)].any()
        assert dataset.data(measure, participant=participant, source=source)
    for datum, original in zip(dataset._data, stored, strict=True):
        np.testing.assert_equal(datum.values, original)


def _check_flag(dataset, expected=(1.0, 0.0, np.nan), participant=None):
    """The first-pass flag is never a skip: both views read it as stored."""
    expected = np.asarray(expected)
    for exclude in (True, False):
        dataset.drop_skips = exclude
        datums = dataset.data("first_pass", participant=participant)
        assert datums
        for datum in datums:
            np.testing.assert_equal(datum.values, expected)
            assert not dataset.exclusion_mask(datum)[np.isnan(expected)].any()


@pytest.mark.parametrize("key", ["provo", "celer", "onestop", "rsc", "rsc_extended"])
def test_eyelink_summary_skip_and_unknown(tmp_path, key):
    rows = []
    for i, word in enumerate(WORDS, 1):
        rows.append(
            {
                "IA_ID": i,
                "IA_LABEL": word,
                "Text_ID": 1,
                "Word_Number": i + 1,
                "Word_Cleaned": word,
                "Participant_ID": "p1",
                "RECORDING_SESSION_LABEL": "p1",
                "sentenceid": "s1",
                "sentence": " ".join(WORDS),
                "DATA_FILE": "p1",
                "participant_id": "p1",
                "TRIAL_INDEX": 1,
                "article_id": 1,
                "article_batch": 1,
                "paragraph_id": 1,
                "difficulty_level": "Adv",
                "question_preview": "False",
                "repeated_reading_trial": "False",
                "practice_trial": "False",
                "item.id": 1,
                "word.serial.no": i,
                "item.text": " ".join(WORDS),
                **dict.fromkeys(EYELINK_IA_MEASURES, 200 if i == 1 else "NA"),
                "IA_DWELL_TIME": [200, 0, "NA"][i - 1],
                "IA_SKIP": [0, 1, "NA"][i - 1],
            }
        )
    ds = REGISTRY[key].loader_cls()(**REGISTRY[key].init_kwargs)
    path = tmp_path / "rows.csv"
    _write(path, rows, "\t" if key in {"celer", "rsc", "rsc_extended"} else ",")
    if key == "provo":
        ds._load_et(str(path), passages={1: WORDS})
    elif key == "celer":
        ds._load_tsv(path)
    elif key == "onestop":
        ds._load_ia_report(str(path))
    else:
        ds._build(*ds._read_measures(path))
    _check(ds)
    _check(ds, measure="first_run")
    _check_flag(ds)


@pytest.mark.parametrize("key", ["emtec", "potec"])
def test_pymovements_summary_skip_and_unknown(tmp_path, key):
    rows = [
        {
            "item_id": "1",
            "model": "test",
            "decoding_strategy": "test",
            "subject_id": "p1",
            "word_id": i,
            "word": word,
            "word_with_punct": word,
            "word_index_in_text": i,
            "sent_index_in_text": 1,
            **dict.fromkeys(PYMOVEMENTS_RM_MEASURES, 200 if i == 1 else "NA"),
            PYMOVEMENTS_FIRST_PASS_COLUMN: [200, 0, "NA"][i - 1],
            "TFT": [200, 0, "NA"][i - 1],
        }
        for i, word in enumerate(WORDS, 1)
    ]
    if key == "emtec":
        path = tmp_path / "rows.tsv"
        _write(path, rows, "\t")
        ds = REGISTRY[key].loader_cls()()
        ds._load_data(path)
    else:
        _write(tmp_path / "reading_measures/reader1_b0_rm.tsv", rows, "\t")
        ds = load_dataset(key, path=tmp_path, measurement_source="released")
    _check(ds)
    _check(ds, measure="first_run")
    _check_flag(ds)


@pytest.mark.parametrize("key", ["emtec", "potec"])
def test_pymovements_regression_entry_leaves_the_first_run_unknown(tmp_path, key):
    """FPRT establishes the first run on first-pass words only."""
    rows = [
        {
            "item_id": "1",
            "model": "test",
            "decoding_strategy": "test",
            "subject_id": "p1",
            "word_id": i,
            "word": word,
            "word_with_punct": word,
            "word_index_in_text": i,
            "sent_index_in_text": 1,
            **dict.fromkeys(PYMOVEMENTS_RM_MEASURES, 200),
            PYMOVEMENTS_FIRST_PASS_COLUMN: [200, 0, "NA"][i - 1],
            "TFT": 200,
        }
        for i, word in enumerate(WORDS, 1)
    ]
    if key == "emtec":
        path = tmp_path / "rows.tsv"
        _write(path, rows, "\t")
        ds = REGISTRY[key].loader_cls()()
        ds._load_data(path)
    else:
        _write(tmp_path / "reading_measures/reader1_b0_rm.tsv", rows, "\t")
        ds = load_dataset(key, path=tmp_path, measurement_source="released")
    _check(ds, (200.0, np.nan, np.nan), measure="first_run")
    _check(ds, (200.0, 0.0, np.nan))
    _check_flag(ds)


def test_geco_summary_skip_and_unknown(tmp_path):
    import openpyxl

    from replica.dataset.eye_tracking.geco import GECO, MEASURE_COLUMNS

    rows = [
        {
            "PP_NR": 1,
            "PART": 1,
            "TRIAL": 1,
            "WORD_ID_WITHIN_TRIAL": i,
            "WORD": word,
            **dict.fromkeys(MEASURE_COLUMNS, 200 if i == 1 else "."),
            "WORD_TOTAL_READING_TIME": [200, 0, "."][i - 1],
            "WORD_SKIP": [0, 1, "."][i - 1],
        }
        for i, word in enumerate(WORDS, 1)
    ]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DATA"
    ws.append(list(rows[0]))
    for row in rows:
        ws.append(list(row.values()))
    path = tmp_path / "geco.xlsx"
    wb.save(path)
    ds = GECO()
    ds._load_xlsx(str(path), openpyxl)
    _check(ds)
    _check(ds, measure="first_run")
    _check_flag(ds)


def test_copco_summary_skip_and_unknown(tmp_path):
    from replica.dataset.eye_tracking.copco import MEASURE_COLUMNS

    rows = [
        {
            "speechId": 1,
            "paragraphId": 0,
            "sentenceId": 1,
            "wordId": i,
            "trialId": 1,
            "word": word,
            **dict.fromkeys(MEASURE_COLUMNS, 200 if i == 1 else "NA"),
            "word_total_fix_dur": [200, 0, "NA"][i - 1],
        }
        for i, word in enumerate(WORDS, 1)
    ]
    _write(tmp_path / "ExtractedFeatures/P01.csv", rows)
    ds = load_dataset("copco", path=tmp_path)
    _check(ds, measure="first_run")
    # Without the fixation reports the release gives no first-pass flag.
    with pytest.raises(ValueError, match="first_pass"):
        ds.data(FIRST_PASS_RUN)


@pytest.mark.parametrize(
    "key", ["meco_du", "meco_l1_w2_ru_mo", "meco_l2_w1", "meco_l2_w2"]
)
def test_meco_frames_preserve_skip_and_unknown(monkeypatch, key):
    from replica.dataset.eye_tracking.meco import l1
    from replica.dataset.eye_tracking.meco.word_utils import MEASURE_COLUMNS

    rows = [
        {
            "uniform_id": "p1",
            "subid": "p1",
            "lang": "ru_mo" if "l1_w2" in key else "du",
            "itemid": str(item),
            "trialid": item,
            "wordnum": i,
            "word": word,
            **dict.fromkeys(MEASURE_COLUMNS, 200 if i == 1 else np.nan),
            "dur": [200, 0, np.nan][i - 1],
            "firstrun.skip": [0, 1, np.nan][i - 1],
        }
        for item in range(1, 13)
        for i, word in enumerate(WORDS, 1)
    ]
    frame = pd.DataFrame(rows)
    spec = REGISTRY[key]
    ds = spec.loader_cls()(**spec.init_kwargs)
    if key == "meco_du":
        monkeypatch.setattr(l1, "read_r_dataframe", lambda path: frame)
        ds._load_rda("synthetic.rda")
    else:
        ds._load_frame(frame)
    _check(ds, measure="first_run_in_first_pass", source="released")
    _check(ds, measure="first_run", source="released")
    _check_flag(ds)


def test_ucl_summary_preserves_skip_and_unknown(tmp_path):
    from replica.dataset.eye_tracking.ucl import (
        FIRST_PASS_COLUMN,
        MEASURE_COLUMNS,
        UCL_ET,
    )

    ds = UCL_ET()
    ds._sent_map = {1: ds._make_stimulus(WORDS, "s1")}
    path = tmp_path / "et.tsv"
    _write(
        path,
        [
            {
                "subj_nr": "1",
                "sent_nr": 1,
                "word_pos": i,
                **dict.fromkeys(MEASURE_COLUMNS, [200, 0, "NA"][i - 1]),
                FIRST_PASS_COLUMN: [200, 0, "NA"][i - 1],
            }
            for i in range(1, 4)
        ],
        "\t",
    )
    ds._load_et(path)
    _check(ds)
    # The table alone cannot say whether a word outside first pass was fixated.
    _check(ds, (200.0, np.nan, np.nan), measure="first_run")
    _check_flag(ds)


def test_devarda_unmeasured_positions_stay_missing(tmp_path):
    from replica.dataset.eye_tracking.devarda2023 import (
        FIRST_PASS_COLUMN,
        MEASURE_COLUMNS,
        DeVarda2023,
    )

    path = tmp_path / "all_measures.csv"
    _write(
        path,
        [
            {
                "sent_id": 1,
                "sentence": " ".join(WORDS),
                "list": 1,
                "context_length": i,
                **dict.fromkeys(MEASURE_COLUMNS, [200, 0][i]),
                FIRST_PASS_COLUMN: [200, 0][i],
            }
            for i in range(2)
        ],
    )
    ds = DeVarda2023()
    ds._load_csv(path)
    _check(ds)
    assert "first_run" not in ds.measures
    assert "first_pass" not in ds.measures


def test_ucl_fixation_totals_preserve_unavailable_duration():
    from types import SimpleNamespace

    from replica.core import Datum
    from replica.dataset.eye_tracking.ucl import UCL_ET

    ds = UCL_ET()
    stim = ds._make_stimulus(WORDS, "s1")
    ds._data = [Datum(stim, "first_pass", np.array([1.0, 0.0, np.nan]), "p1")]
    ds._fixations = [
        SimpleNamespace(
            participant_id="p1", stimulus_id="s1", unit_index=i, duration=duration
        )
        for i, duration in [(0, 200.0), (2, np.nan)]
    ]
    ds._load_total_time("synthetic.fix.txt")
    _check(ds, measure="total_time")


def test_rastros_response_alignment_preserves_skip_and_unknown():
    from replica.dataset.eye_tracking.rastros import RastrOS

    ds = RastrOS()
    ds._words = pd.DataFrame(
        {
            "Word_Unique_ID": [1, 2, 3],
            "context_id": [1] * 3,
            "word_index": [0, 1, 2],
            "word": WORDS,
            "separator": ["", " ", " "],
            "hits": [1] * 3,
            "n_responses": [2] * 3,
        }
    )
    ds._responses = pd.DataFrame(
        {
            "Word_Unique_ID": [1, 2, 3],
            "RECORDING_SESSION_LABEL": ["p1"] * 3,
            **{col: [200, np.nan, np.nan] for col in EYELINK_IA_MEASURES},
            "IA_DWELL_TIME": [200, 0, np.nan],
            "IA_SKIP": [0, 1, np.nan],
        }
    )
    ds._build()
    _check(ds)
    _check(ds, measure="first_run")
    _check_flag(ds)


def test_bsc_stream_distinguishes_absence_from_missing_duration():
    from replica.dataset.eye_tracking.bsc import BSC

    ds = BSC()
    stimuli, indices = ds._build_stimuli({1: list(enumerate(WORDS, 1))})
    ds._append_data({("p1", 1): [(0, 1, 200.0), (1, 3, np.nan)]}, stimuli, indices)
    _check(ds, measure="total_time")


def test_dundee_known_skips_and_missing_aggregates_stay_distinct():
    from replica.dataset.eye_tracking.dundee.shain2024 import Dundee

    ds = Dundee()
    stim = ds._make_stimulus(WORDS, "s1")
    rows = pd.DataFrame(
        {
            "_region": [0, 2],
            "_aligned": [True, True],
            "time": [0, 0.2],
            "fdurGP": [200.0, np.nan],
        }
    )
    ds._append_measures(stim, "p1", rows, 3)
    _check(ds, measure=FIRST_PASS_GO_PAST)


def test_dundee_first_pass_column_yields_run_and_flag():
    """An empty fdurFP on a visited word leaves both unknown; an unvisited word
    is a skip; a positive value is a first run reached in first pass."""
    from replica.dataset.eye_tracking.dundee.shain2024 import Dundee

    ds = Dundee()
    stim = ds._make_stimulus(WORDS, "s1")
    rows = pd.DataFrame(
        {
            "_region": [0, 2],
            "_aligned": [True, True],
            "time": [0, 0.2],
            "fdurFP": [200.0, np.nan],
        }
    )
    ds._append_measures(stim, "p1", rows, 3)
    _check(ds, measure="first_run")
    _check(ds)
    _check_flag(ds)


def test_dundee_official_first_pass_skips_include_later_regression_visits(tmp_path):
    from tests.dataset.eye_tracking.dundee.test_dundee_official import _load

    ds = _load(tmp_path, [(1, 200), (3, 100), (2, 150)])
    # Words 2 and 4 have observed zero first-pass time; word 2 has later total time
    # and a first run of its own.
    _check(ds, [200, 0, 100, 0])
    _check(ds, [200, 150, 100, 0], measure="first_run")
    _check_flag(ds, [1, 0, 1, 0])


def test_sbsat_stream_skips_are_zero(tmp_path):
    from replica.dataset import SBSAT
    from tests.dataset.test_loader_audit_regressions import _sbsat_row

    rows = [
        _sbsat_row("test", 1, 1, "start", 200),
        _sbsat_row("test", 1, 3, "end", 300),
    ]
    rows.append(
        {**_sbsat_row("test", 1, 2, "skipped"), "RECORDING_SESSION_LABEL": "p2"}
    )
    path = tmp_path / "sbsat.csv"
    _write(path, rows)
    ds = SBSAT()
    ds._load_fixation_data(str(path))
    _check(ds, [200, 0, 300], participant="p1")
    _check(ds, [200, 0, 300], measure="first_run", participant="p1")
    _check_flag(ds, [1, 0, 1], participant="p1")


def test_meco_character_stream_skip_and_missing_duration(tmp_path):
    _write(
        tmp_path / "stimulus_en.csv",
        [
            {"trial_id": 1, "letternum": i, "character": char}
            for i, char in enumerate("abc", 1)
        ],
    )
    _write(
        tmp_path / "en.csv",
        [
            {
                "trial_id": 1,
                "letternum": i,
                "reader": "p1",
                "dur": dur,
                "right_bounded_time": dur,
                "start": i,
            }
            for i, dur in [(1, 200), (3, "NA")]
        ],
    )
    _check(load_dataset("meco_char_en", path=tmp_path), measure="total_time")


def test_zuco1_matlab_empty_is_zero_but_nan_stays_missing(tmp_path):
    from scipy.io import savemat

    from replica.dataset.eye_tracking.zuco.zuco1 import MEASURE_COLUMNS, ZuCo1

    words = [
        {"content": word, **dict.fromkeys(MEASURE_COLUMNS, value)}
        for word, value in zip(WORDS, [200.0, np.array([]), np.nan], strict=True)
    ]
    savemat(tmp_path / "resultsTEST_NR.mat", {"sentenceData": {"word": words}})
    ds = ZuCo1()
    ds.load(str(tmp_path))
    for measure in MEASURE_COLUMNS.values():
        _check(ds, expected=(400.0, 0.0, np.nan), measure=measure)
        _check(ds, expected=(400.0, 0.0, np.nan), measure=measure, source="released")
    for datum in ds._data:
        assert datum.metadata["source_unit"] == "samples"
        np.testing.assert_equal(datum.metadata["released_values"], [200, 0, np.nan])


def test_zuco2_matlab_empty_is_zero_but_nan_stays_missing(tmp_path):
    h5py = pytest.importorskip("h5py")
    from replica.dataset.eye_tracking.zuco.zuco2 import MEASURE_COLUMNS, ZuCo2

    ds = ZuCo2()
    stim = ds._make_stimulus(WORDS, "s1")
    ds._sent_idx_map = {0: 0}
    with h5py.File(tmp_path / "zuco.mat", "w") as fh:
        words = fh.create_group("words")
        refs = []
        for i, value in enumerate([[[200.0]], [0, 0], [[np.nan]]]):
            field = fh.create_dataset(f"value{i}", data=value)
            if i == 1:
                field.attrs["MATLAB_empty"] = 1
            refs.append([field.ref])
        words.create_dataset("content", data=refs, dtype=h5py.ref_dtype)
        for column in MEASURE_COLUMNS:
            words.create_dataset(column, data=refs, dtype=h5py.ref_dtype)
        sd = fh.create_group("sentenceData")
        sd.create_dataset("word", data=[[words.ref]], dtype=h5py.ref_dtype)
        ds._load_subject(fh, "p1", [stim], load_eeg=False)
    for measure in MEASURE_COLUMNS.values():
        _check(ds, expected=(400.0, 0.0, np.nan), measure=measure)
        _check(ds, expected=(400.0, 0.0, np.nan), measure=measure, source="released")
    for datum in ds._data:
        assert datum.metadata["source_unit"] == "samples"
        np.testing.assert_equal(datum.metadata["released_values"], [200, 0, np.nan])
