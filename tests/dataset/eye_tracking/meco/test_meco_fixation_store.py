"""Synthetic tests for MECO's lazy Arrow-backed fixation store."""

from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from replica.dataset.eye_tracking.meco.prepare import (
    PARQUET_SCHEMA,
    _csv_to_parquet,
    _parquet_schema,
    _rda_to_parquet_with_pyreadr,
    validate_parquet,
)
from replica.dataset.eye_tracking.meco.releases import (
    FIXATION_RELEASES as RELEASES,
)
from replica.dataset.eye_tracking.meco.waves import MECOL1W2, MECOL2W2
from tests.dataset.eye_tracking.meco.test_meco_waves import _frame


def _release_for_test(monkeypatch, tmp_path, *, rows: int = 3):
    release = replace(
        RELEASES["meco_l2_w2"],
        expected_rows=rows,
        parquet_filename="fixations_test.parquet",
    )
    monkeypatch.setitem(RELEASES, "meco_l2_w2", release)
    return release, tmp_path / release.parquet_filename


def _fixation_rows():
    common = {
        "participant_id": "site_1",
        "site": "du",
        "source_item_id": 1,
        "source_trial_id": 12,
        "start_time": 10.0,
        "end_time": 30.0,
        "duration": 20.0,
        "x": 100.0,
        "y": 200.0,
        "pupil_size": 700.0,
        "blink": 0.0,
        "source_ia_number": 1,
        "source_ia": "text1",
        "run_id": 1,
    }
    return [
        {
            **common,
            "fixation_index": 1,
            "source_word_number": 1,
            "source_word": "text1",
        },
        {
            **common,
            "fixation_index": 2,
            "start_time": 40.0,
            "end_time": 65.0,
            "duration": 25.0,
            "source_word_number": 2,
            "source_word": "alpha-beta",
            "run_id": 7,
        },
        {
            **common,
            "participant_id": "site_2",
            "fixation_index": 1,
            "source_word_number": None,
            "source_word": None,
            "source_ia_number": None,
            "source_ia": None,
        },
    ]


def test_lazy_table_streaming_and_filtered_object_materialization(
    monkeypatch, tmp_path
):
    release, parquet_path = _release_for_test(monkeypatch, tmp_path)
    table = pa.Table.from_pylist(_fixation_rows(), schema=PARQUET_SCHEMA)
    pq.write_table(
        table.replace_schema_metadata(_parquet_schema(release).metadata), parquet_path
    )

    ds = MECOL2W2()
    ds._load_frame(_frame())
    ds._configure_fixation_store(tmp_path)

    assert ds.fixation_table(participant="site_1").num_rows == 2
    aligned = ds.fixation_table(participant="site_1", aligned=True)
    assert aligned.column_names[-4:] == [
        "stimulus_id",
        "unit_index",
        "char_offset",
        "alignment_method",
    ]
    assert aligned["unit_index"].to_pylist() == [0, 1]
    with pytest.raises(ValueError, match="more than one million"):
        ds.fixations()

    fixations = ds.fixations(participant="site_1")
    assert len(fixations) == 2
    assert fixations[0].stimulus_id.startswith("mecol2w2_all_1")
    assert fixations[0].unit_index == 0
    assert fixations[0].char_offset == 0
    assert fixations[1].unit_index == 1
    assert fixations[1].metadata["source_trial_id"] == 12
    assert fixations[1].metadata["pupil_size"] == 700.0
    assert fixations[1].metadata["source_word_run_id"] == 7
    assert "run_id" not in fixations[1].metadata
    from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer

    runs = aggregate_fixations_to_layer(fixations, ds.stimuli()[0])["first_run"]
    assert runs[:2].tolist() == [20.0, 25.0]
    stimulus_id = fixations[0].stimulus_id
    assert ds.fixation_table(stimulus_id=stimulus_id).num_rows == 3
    assert all(
        fixation.stimulus_id == stimulus_id
        for fixation in ds.fixations(stimulus_id=stimulus_id)
    )
    validate_parquet(parquet_path, release)


def test_off_text_fixation_has_no_linguistic_alignment(monkeypatch, tmp_path):
    _release, parquet_path = _release_for_test(monkeypatch, tmp_path)
    table = pa.Table.from_pylist(_fixation_rows(), schema=PARQUET_SCHEMA)
    pq.write_table(
        table.replace_schema_metadata(_parquet_schema(_release).metadata), parquet_path
    )
    ds = MECOL2W2()
    ds._load_frame(_frame())
    ds._configure_fixation_store(tmp_path)

    [fixation] = ds.fixations(participant="site_2")
    assert fixation.unit_index is None
    assert fixation.char_offset is None


def test_csv_conversion_uses_canonical_schema(monkeypatch, tmp_path):
    release, parquet_path = _release_for_test(monkeypatch, tmp_path, rows=1)
    csv_path = tmp_path / "selected.csv"
    csv_path.write_text(
        "uniform_id,lang,itemid,trialid,fixid,start,stop,dur,xn,yn,ps,blink,"
        "wordnum,word,ianum,ia,word.runid\n"
        "p1,en,1,1,1,0,100,100,12,34,700,0,1,Hello,1,Hello,1\n",
        encoding="utf-8",
    )
    _csv_to_parquet(csv_path, parquet_path, release)

    validate_parquet(parquet_path, release)
    assert pq.read_table(parquet_path).schema.remove_metadata() == PARQUET_SCHEMA


def test_pyreadr_fallback_uses_same_canonical_schema(monkeypatch, tmp_path):
    release, parquet_path = _release_for_test(monkeypatch, tmp_path, rows=2)
    source_row = {
        "uniform_id": "p1",
        "lang": "en",
        "itemid": "1",
        "trialid": 1,
        "fixid": 2,
        "start": 0.0,
        "stop": 100.0,
        "dur": 100.0,
        "xn": 12.0,
        "yn": 34.0,
        "ps": 700.0,
        "blink": 0.0,
        "wordnum": 1,
        "word": "Hello",
        "ianum": 1,
        "ia": "Hello",
        "word.runid": 1,
    }
    monkeypatch.setattr(
        "pyreadr.read_r",
        lambda _path: {
            release.r_key: pd.DataFrame(
                [source_row, {**source_row, "fixid": 1, "start": -10.0}]
            )
        },
    )
    _rda_to_parquet_with_pyreadr(tmp_path / "unused.rda", parquet_path, release)
    validate_parquet(parquet_path, release)
    assert pq.read_table(parquet_path)["fixation_index"].to_pylist() == [1, 2]


def test_missing_prepared_store_has_actionable_commands(monkeypatch, tmp_path):
    _release_for_test(monkeypatch, tmp_path)
    ds = MECOL2W2()
    ds._load_frame(_frame())
    ds._configure_fixation_store(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch meco_l2_w2_fixations"):
        ds.fixation_table(participant="site_1")
    assert ds.data("first_run", source="released")[0].values.tolist() == [201.0, 202.0]
    with pytest.raises(FileNotFoundError, match="fetch meco_l2_w2_fixations"):
        ds.data("first_run")


@pytest.mark.parametrize("site", ["ch_s", "ch_t"])
def test_l1w2_chinese_sites_expose_matching_fixations(monkeypatch, tmp_path, site):
    release = replace(RELEASES["meco_l1_w2"], expected_rows=3)
    monkeypatch.setitem(RELEASES, "meco_l1_w2", release)
    rows = [{**row, "site": site} for row in _fixation_rows()]
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(
        table.replace_schema_metadata(_parquet_schema(release).metadata),
        tmp_path / release.parquet_filename,
    )
    frame = _frame()
    frame["lang"] = site
    ds = MECOL1W2(site=site)
    ds._load_frame(frame)
    ds._configure_fixation_store(tmp_path)
    assert ds.fixation_table().num_rows == 3
    assert [event.unit_index for event in ds.fixations(participant="site_1")] == [0, 1]


def test_reaggregation_counts_duplicate_source_events_once(monkeypatch, tmp_path):
    from replica.core import AnnotationLayer, Span

    rows = _fixation_rows()[:2]
    release, parquet_path = _release_for_test(monkeypatch, tmp_path, rows=4)
    table = pa.Table.from_pylist(
        [rows[0], rows[0], rows[1], rows[1]], schema=PARQUET_SCHEMA
    )
    pq.write_table(
        table.replace_schema_metadata(_parquet_schema(release).metadata), parquet_path
    )
    frame = _frame()
    frame = frame[frame.uniform_id == "site_1"].copy()
    frame["dur"] = frame.wordnum.map({1: 20.0, 2: 25.0})
    dataset = MECOL2W2()
    dataset._load_frame(frame)
    dataset._configure_fixation_store(tmp_path)
    dataset.drop_skips = False
    datum = dataset.data("total_time")[0]
    stimulus = datum.stimulus
    stimulus.add_layer(
        AnnotationLayer("text", [Span(0, len(stimulus.text))], [stimulus.text])
    )
    raw = dataset.fixations(participant="site_1")
    assert len(raw) == 4
    assert dataset.fixation_table().num_rows == 4
    assert dataset.reaggregate(datum, datum.layer, fixations=raw).values.tolist() == [
        20.0,
        25.0,
    ]
    assert dataset.reaggregate(datum, "text", fixations=raw).values.tolist() == [45.0]
    assert len(raw) == 4
    changed = [raw[0], replace(raw[0], duration=21.0), raw[2]]
    with pytest.raises(ValueError, match="Conflicting MECO fixation rows"):
        dataset.reaggregate(datum, "text", fixations=changed)


def test_blank_source_area_retains_its_whitespace_span():
    from replica.dataset.eye_tracking.meco.l1 import MECO_L1

    dataset = MECO_L1(language="ee")
    stimulus = dataset._make_stimulus(
        ["one", "two"],
        "s",
        {"source_word_numbers": [1, 3], "source_nonword_numbers": [2]},
    )
    mapping = dataset._source_mapping(stimulus)
    layer = stimulus.layers[mapping.unit_layer]
    assert layer.values == ["one", " ", "two"]
    assert mapping.word_indices == {1: 0, 2: 1, 3: 2}


@pytest.mark.parametrize(
    "scanpath, released, standard, totals, flags, merged_run",
    [
        (
            [(2, 50, 1), (1, 100, 2), (0, 200, 2), (1, 300, 2)],
            [200, 400, 50, float("nan")],
            [200, 100, 50, 0],
            [200, 400, 50, float("nan")],
            [0, 0, 1, 0],
            650,
        ),
        (
            [(0, 100, 1), (None, 60, None), (0, 200, 1), (1, 300, 1)],
            [300, 300, float("nan"), float("nan")],
            [100, 300, 0, 0],
            [300, 300, float("nan"), float("nan")],
            [1, 1, 0, 0],
            100,
        ),
    ],
    ids=["regression-episode", "off-text-interruption"],
)
def test_standard_and_released_runs_have_distinct_windows(
    monkeypatch, tmp_path, scanpath, released, standard, totals, flags, merged_run
):
    import numpy as np

    from replica.core import AnnotationLayer, Datum, Span

    ds = MECOL2W2()
    stimulus = ds._make_stimulus(
        ["a", "b", "c", "d"],
        "test",
        {
            "source_item_id": 1,
            "source_word_numbers": [1, 2, 3, 4],
            "reader_group": "L2",
            "wave": 2,
            "site": "du",
        },
    )
    first = [float("nan")] * 4
    rows = []
    start = 0
    for index, (unit, duration, episode) in enumerate(scanpath, 1):
        if unit is not None and np.isnan(first[unit]):
            first[unit] = duration
        word = None if unit is None else stimulus.layers["words"].values[unit]
        number = None if unit is None else unit + 1
        rows.append(
            {
                **_fixation_rows()[0],
                "participant_id": "p",
                "fixation_index": index,
                "start_time": start,
                "end_time": start + duration,
                "duration": duration,
                "source_word_number": number,
                "source_word": word,
                "source_ia_number": number,
                "source_ia": word,
                "run_id": episode,
            }
        )
        start += duration
    for name, values in {
        "first_fixation": first,
        "first_run": released,
        "first_pass": flags,
        "total_time": totals,
    }.items():
        ds._data.append(
            Datum(stimulus, name, np.array(values, float), participant_id="p")
        )
    release, path = _release_for_test(monkeypatch, tmp_path, rows=len(rows))
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(
        table.replace_schema_metadata(_parquet_schema(release).metadata), path
    )
    ds._configure_fixation_store(tmp_path)
    np.testing.assert_equal(ds.data("first_run", source="released")[0].values, released)
    assert not any(d.measure_name == "first_run" for d in ds._data)
    stimulus.add_layer(
        AnnotationLayer("whole", [Span(0, len(stimulus.text))], [stimulus.text])
    )
    for drop_skips in (False, True):
        ds.drop_skips = drop_skips
        [datum] = ds.data("first_run")
        expected = np.array(standard, float)
        if drop_skips:
            expected[expected == 0] = np.nan
        np.testing.assert_equal(datum.values, expected)
        expected_pass = np.where(np.array(flags) == 1, standard, 0.0)
        if drop_skips:
            expected_pass[expected_pass == 0] = np.nan
        np.testing.assert_equal(
            ds.data("first_run_in_first_pass")[0].values, expected_pass
        )
        assert datum.metadata["measurement_source"] == "fixations"
        np.testing.assert_equal(ds.reaggregate(datum, "words").values, expected)
        assert ds.reaggregate(datum, "whole").values.tolist() == [merged_run]
        [source] = ds.data("first_run", source="released")
        np.testing.assert_equal(ds.reaggregate(source, "words").values, source.values)
        [restricted] = ds.data("first_run_in_first_pass", source="released")
        np.testing.assert_equal(
            ds.reaggregate(restricted, "words").values, restricted.values
        )
        with pytest.raises(ValueError, match="native units"):
            ds.reaggregate(source, "whole")
        from replica.dataset import FixationCalculation

        result = ds.reaggregate(
            source,
            "whole",
            measure="first_run",
            calculation=FixationCalculation(first_run="popeye_episode@1"),
        )
        assert result.values.tolist() == [
            sum(d for u, d, episode in scanpath if u is not None)
        ]

    assert sum(d.measure_name == "first_run" for d in ds._data) == 1
    ds._data = [d for d in ds._data if d.measure_name != "first_run"]
    next(d for d in ds._data if d.measure_name == "total_time").values[0] += 1
    with pytest.raises(ValueError, match="do not reproduce total_time"):
        ds.data("first_run")
    assert not any(d.measure_name == "first_run" for d in ds._data)
