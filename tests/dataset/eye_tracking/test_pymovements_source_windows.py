"""Validate released arrival and first-pass windows against their source fields."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replica.dataset.eye_tracking.emtec import EMTeC
from replica.dataset.eye_tracking.potec import PoTeC

COLUMNS = {
    "FD": "first_fixation",
    "FFD": "first_fixation_in_first_pass",
    "FPRT": "first_run_in_first_pass",
}


def _views(ds, expected):
    stored = {d.measure_name: d.values.copy() for d in ds._data}
    for drop in (False, True):
        ds.drop_skips = drop
        for column, measure in COLUMNS.items():
            values = expected[column]
            [datum] = ds.data(measure)
            np.testing.assert_equal(
                datum.values, np.where(values == 0, np.nan, values) if drop else values
            )
    for datum in ds._data:
        np.testing.assert_equal(datum.values, stored[datum.measure_name])


@pytest.mark.parametrize("loader", [EMTeC, PoTeC])
def test_later_entry_and_missing_first_pass_preserve_released_windows(tmp_path, loader):
    expected = {
        "FD": np.array([100, 150, 0, np.nan, 180]),
        "FFD": np.array([100, 0, 0, np.nan, np.nan]),
        "FRT": np.array([200, 250, 0, np.nan, 300]),
        "FPRT": np.array([200, 0, 0, np.nan, np.nan]),
    }
    rows = []
    for i, word in enumerate(["forward", "late", "skipped", "missing", "unknownpass"]):
        rows.append(
            {
                "subject_id": "p1",
                "item_id": "1",
                "model": "test",
                "decoding_strategy": "test",
                "word_id": i,
                "word_index_in_text": i + 1,
                "sent_index_in_text": 1,
                "word": word,
                "word_with_punct": word,
                **{col: values[i] for col, values in expected.items()},
                "TFT": [250, 350, 0, np.nan, 450][i],
                "RPD_inc": [250, 0, 0, np.nan, np.nan][i],
            }
        )
    filename = "reading_measures.csv" if loader is EMTeC else "reader1_b0_merged.tsv"
    pd.DataFrame(rows).to_csv(tmp_path / filename, sep="\t", index=False, na_rep="NaN")
    ds = loader(measurement_source="released")
    ds.load(str(tmp_path))
    _views(ds, expected)


@pytest.mark.parametrize("loader", [EMTeC, PoTeC])
@pytest.mark.needs_data
def test_real_release_windows(data_root, tmp_path, loader):
    base = (
        Path(data_root)
        / "eye-tracking/original"
        / ("emtec" if loader is EMTeC else "potec")
    )
    if loader is EMTeC:
        source = base / "reading_measures.csv"
        if not source.exists():
            pytest.skip(f"Missing {source}")
        frame = pd.read_csv(source, sep="\t", keep_default_na=False)
        keys = ["subject_id", "item_id", "model", "decoding_strategy"]
        frame = frame[(frame[keys] == frame.iloc[0][keys]).all(axis=1)].sort_values(
            "word_id"
        )
    else:
        sources = sorted(base.glob("reader*_merged.tsv"))
        if not sources:
            pytest.skip(f"Missing {base}")
        source = sources[0]
        frame = pd.read_csv(source, sep="\t", keep_default_na=False).sort_values(
            "word_index_in_text"
        )
    frame.to_csv(tmp_path / source.name, sep="\t", index=False)
    ds = loader(measurement_source="released")
    ds.load(str(tmp_path))
    expected = {col: frame[col].to_numpy(dtype=float) for col in COLUMNS}
    assert ((expected["FD"] > 0) & (expected["FFD"] == 0)).any()
    _views(ds, expected)


@pytest.mark.parametrize("loader", [EMTeC, PoTeC])
@pytest.mark.parametrize("missing_column", COLUMNS)
def test_missing_released_window_is_an_error(tmp_path, loader, missing_column):
    row = {
        "subject_id": "p1",
        "item_id": "1",
        "model": "test",
        "decoding_strategy": "test",
        "word_id": 0,
        "word_index_in_text": 1,
        "sent_index_in_text": 1,
        "word": "word",
        "word_with_punct": "word",
        "FD": 100,
        "FFD": 100,
        "FRT": 200,
        "FPRT": 200,
        "TFT": 250,
        "RPD_inc": 250,
    }
    del row[missing_column]
    filename = "reading_measures.csv" if loader is EMTeC else "reader1_b0_merged.tsv"
    pd.DataFrame([row]).to_csv(tmp_path / filename, sep="\t", index=False)
    with pytest.raises(KeyError, match=missing_column):
        loader(measurement_source="released").load(str(tmp_path))


@pytest.mark.parametrize("loader", [EMTeC, PoTeC])
def test_invalid_released_first_run_does_not_turn_a_fixation_into_a_skip(
    tmp_path, loader
):
    rows = [
        {
            "subject_id": "p1",
            "item_id": "1",
            "model": "test",
            "decoding_strategy": "test",
            "word_id": index,
            "word_index_in_text": index + 1,
            "sent_index_in_text": 1,
            "word": word,
            "word_with_punct": word,
            "FD": 100,
            "FFD": first_pass,
            "FRT": 0,
            "FPRT": first_pass,
            "TFT": 100,
            "RPD_inc": first_pass,
        }
        for index, (word, first_pass) in enumerate([("forward", 100), ("later", 0)])
    ]
    filename = "reading_measures.csv" if loader is EMTeC else "reader1_b0_merged.tsv"
    pd.DataFrame(rows).to_csv(tmp_path / filename, sep="\t", index=False)
    ds = loader(measurement_source="released")
    ds.load(str(tmp_path))
    for drop in (False, True):
        ds.drop_skips = drop
        [first_run] = ds.data("first_run")
        np.testing.assert_equal(first_run.values, [100, np.nan])
        [first_fixation] = ds.data("first_fixation")
        np.testing.assert_equal(first_fixation.values, [100, 100])
