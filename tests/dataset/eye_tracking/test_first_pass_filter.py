"""The optional filter selects observations without changing measured durations."""

import csv

import numpy as np
import openpyxl
import pytest

from replica.dataset import GECO, PROVO, describe_dataset, load_dataset
from replica.dataset.eye_tracking.geco import MEASURE_COLUMNS as GECO_MEASURES
from replica.dataset.measures import EYELINK_IA_MEASURES as PROVO_MEASURES


@pytest.mark.parametrize("key,loader", [("geco", GECO), ("provo", PROVO)])
def test_first_pass_filter_is_discoverable_and_loadable(
    monkeypatch, tmp_path, key, loader
):
    options = {p.name: p for p in describe_dataset(key).parameters}
    assert options["first_pass_filter"].default is False
    assert set(options["first_pass_filter"].choices) == {True, False}
    monkeypatch.setattr(loader, "load", lambda self, path: None)
    assert load_dataset(key, path=tmp_path).first_pass_filter is False
    assert load_dataset(key, path=tmp_path, first_pass_filter=True).first_pass_filter


@pytest.mark.parametrize("key", ["geco", "provo"])
def test_first_pass_filter_preserves_text_and_retained_durations(tmp_path, key):
    # First-fixation order: 1 -> 3 -> 2 (regression) -> 7 (jump 4)
    # -> 12 (jump 5) -> 13. Word 13's predecessor remains 12 even though
    # the separate jump limit subsequently excludes the observation on 12.
    times = {1: 100, 3: 200, 2: 300, 7: 400, 12: 500, 13: 600}
    words = [f"word{i}" for i in range(1, 15)]
    columns = GECO_MEASURES if key == "geco" else PROVO_MEASURES
    rows = []
    for position, word in enumerate(words, start=1):
        row = {
            column: 100 * (j + 1) + position if position in times else 0
            for j, column in enumerate(columns)
        }
        if key == "geco":
            row.update(
                PP_NR="reader",
                PART=1,
                TRIAL=1,
                WORD_ID_WITHIN_TRIAL=position,
                WORD=word,
                WORD_FIRST_FIXATION_TIME=times.get(position, "."),
                WORD_SKIP=int(position not in times),
            )
        else:
            row.update(
                Text_ID=1,
                Participant_ID="reader",
                IA_ID=position + 1,
                Word_Number=position + 1,
                Word_Cleaned=word,
                IA_FIRST_FIXATION_TIME=times.get(position, "."),
                IA_SKIP=int(position not in times),
            )
        rows.append(row)

    if key == "geco":
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "DATA"
        sheet.append(list(rows[0]))
        for row in rows:
            sheet.append(list(row.values()))
        path = tmp_path / "MonolingualReadingData.xlsx"
        workbook.save(path)
        workbook.close()
    else:
        path = tmp_path / "ia.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    datasets = []
    for enabled in (False, True):
        dataset = (GECO if key == "geco" else PROVO)(first_pass_filter=enabled)
        if key == "geco":
            dataset.load(str(tmp_path))
        else:
            dataset._load_et(str(path), passages={1: words})
        dataset.drop_skips = False
        datasets.append(dataset)

    unfiltered, filtered = datasets
    assert unfiltered.stimuli()[0].text == filtered.stimuli()[0].text
    assert filtered.stimuli()[0].unit_strings("words") == words
    keep = np.zeros(14, dtype=bool)
    keep[[0, 2, 6, 12]] = True
    for measure in ("first_fixation", "first_run", "go_past", "total_time"):
        original = unfiltered.data(measure)[0].values
        expected = np.where(keep, original, np.nan)
        np.testing.assert_equal(filtered.data(measure)[0].values, expected)
        assert original[1] > 0  # a real observation first reached by regression
        assert original[11] > 0  # a real observation excluded only by jump length
        assert original[13] == 0  # include-skips cannot undo the first-pass filter

    # The flag is filtered like the durations; the never-fixated word stays 0.
    def flag(dataset):
        return next(d for d in dataset.data() if d.measure_name == "first_pass").values

    np.testing.assert_equal(flag(filtered), np.where(keep, flag(unfiltered), np.nan))
    assert flag(unfiltered)[13] == 0

    expected_delta = np.full(14, np.nan)
    expected_delta[[2, 6, 12]] = [2, 4, 1]
    expected_previous = np.full(14, np.nan)
    expected_previous[[2, 6, 12]] = [0, 0, 1]
    np.testing.assert_equal(filtered.data("wdelta")[0].values, expected_delta)
    np.testing.assert_equal(filtered.data("prevwasfix")[0].values, expected_previous)
