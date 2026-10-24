"""CELER events must preserve report identities and reproduce native measures."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span
from replica.dataset import CELER, FixationCalculation, describe_dataset


def _write(path, rows):
    fields = list(rows[0])
    path.write_text(
        "\t".join(fields)
        + "\n"
        + "".join("\t".join(str(row[k]) for k in fields) + "\n" for row in rows)
    )


@pytest.fixture
def reports(tmp_path):
    words = ['"aa', "bb", "cc", 'dd"']
    identity = dict(
        DATA_FILE="reader.edf",
        sentenceid="s",
        trial="2",
        list="1",
        dataset_version="2",
        EYE_USED="RIGHT",
    )
    durations = [100, 120, 70, 80, 50, 60, 90]
    areas = [10, 30, 20, 10, None, 10, 30]
    summary = [
        dict(
            **identity,
            IA_ID=area,
            IA_LABEL=word,
            sentence=" ".join(words),
            shared_text=1,
            IA_LEFT=area * 10,
            IA_RIGHT=area * 10 + 20,
            IA_FIRST_FIXATION_DURATION=first,
            IA_FIRST_RUN_DWELL_TIME=first,
            IA_DWELL_TIME=total,
            IA_REGRESSION_PATH_DURATION=path,
            IA_SKIP=skip,
            TRIAL_FIXATION_COUNT=7,
            TRIAL_DWELL_TIME=sum(durations),
        )
        for area, word, first, total, path, skip in zip(
            [10, 20, 30, 40],
            words,
            [100, 70, 120, 0],
            [240, 70, 210, 0],
            [100, 210, 420, 0],
            [0, 1, 0, 1],
            strict=True,
        )
    ]
    rows = [
        dict(
            **identity,
            CURRENT_FIX_INDEX=i,
            CURRENT_FIX_DURATION=duration,
            CURRENT_FIX_X=0 if area is None else area * 10 + 10,
            CURRENT_FIX_Y=100,
            CURRENT_FIX_INTEREST_AREA_ID="." if area is None else area,
            CURRENT_FIX_INTEREST_AREA_LABEL="."
            if area is None
            else words[area // 10 - 1],
            CURRENT_FIX_INTEREST_AREA_LEFT="." if area is None else area * 10,
            CURRENT_FIX_INTEREST_AREA_RIGHT="." if area is None else area * 10 + 20,
            CURRENT_FIX_INTEREST_AREA_RUN_ID=run,
            CURRENT_FIX_NEAREST_INTEREST_AREA=20,
        )
        for i, (area, duration, run) in enumerate(
            zip(areas, durations, [1, 1, 1, 2, ".", 3, 2], strict=True), 1
        )
    ]
    _write(tmp_path / "sent_ia.tsv", summary)
    _write(tmp_path / "sent_fix.tsv", rows)
    return tmp_path, summary, rows


@pytest.mark.parametrize("drop_skips", [False, True])
def test_native_roundtrip_and_custom_windows(reports, drop_skips):
    path, _, _ = reports
    ds = CELER()
    ds.load(str(path))
    ds.drop_skips = drop_skips
    events = ds.fixations()
    assert len(events) == 7
    assert events[4].unit_index is None
    assert events[4].duration == 50
    assert all(e.metadata["synthetic_start_time"] for e in events)
    assert [e.metadata["source_run_id"] for e in events] == [1, 1, 1, 2, None, 3, 2]
    assert ds.fixations(participant="absent") == []
    stimulus = ds.stimuli()[0]
    stimulus.add_layer(
        AnnotationLayer("sentence", [Span(0, len(stimulus.text))], [stimulus.text])
    )
    for measure in ds.measures:
        datum = ds.data(measure)[0]
        np.testing.assert_equal(ds.reaggregate(datum, "words").values, datum.values)
    for measure, value in dict(
        first_fixation=100, first_run=370, go_past=520, total_time=520
    ).items():
        datum = ds.data(measure)[0]
        np.testing.assert_equal(ds.reaggregate(datum, "sentence").values, [value])
        np.testing.assert_equal(
            ds.reaggregate(datum, "sentence", calculation=FixationCalculation()).values,
            [value],
        )
    # An explicit off-text path policy counts the otherwise excluded 50 ms.
    datum = ds.data("go_past")[0]
    alternative = ds.reaggregate(
        datum,
        "sentence",
        calculation=FixationCalculation(go_past_includes_off_text=True),
    )
    np.testing.assert_equal(alternative.values, [570])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("trial", "3", "presentation"),
        ("EYE_USED", "LEFT", "presentation"),
        ("CURRENT_FIX_INTEREST_AREA_ID", "11", "unknown area"),
        ("CURRENT_FIX_INTEREST_AREA_LABEL", "wrong", "label or bounds"),
        ("CURRENT_FIX_INTEREST_AREA_LEFT", "101", "label or bounds"),
        ("CURRENT_FIX_DURATION", "99", "count or total dwell"),
        ("CURRENT_FIX_INDEX", "0", "indices or durations"),
    ],
)
def test_inconsistent_source_event_is_rejected(reports, field, value, match):
    path, _, rows = reports
    rows[0][field] = value
    _write(path / "sent_fix.tsv", rows)
    ds = CELER()
    ds.load(str(path))
    with pytest.raises(ValueError, match=match):
        ds.fixations()


def test_missing_report_and_missing_events_are_distinct(reports):
    path, _, rows = reports
    ds = CELER()
    ds.load(str(path))
    _write(path / "sent_fix.tsv", rows[:-1])
    with pytest.raises(ValueError, match="count or total dwell"):
        ds.fixations()
    (path / "sent_fix.tsv").unlink()
    with pytest.raises(NotImplementedError, match="sent_fix.tsv"):
        ds.fixations()
    assert len(ds.data("total_time")) == 1


def test_catalog_ties_event_access_to_the_fixation_report(reports):
    path, _, _ = reports
    ds = CELER()
    ds.load(str(path))
    assert ds.fixations()
    status = describe_dataset("celer").fixations
    assert status == "available when sent_fix.tsv is present"


@pytest.mark.parametrize("malformed", ["header", "row"])
def test_malformed_report_is_rejected(reports, malformed):
    path, _, _ = reports
    report = path / "sent_fix.tsv"
    lines = report.read_text().splitlines()
    if malformed == "header":
        lines[0] = lines[0].replace("CURRENT_FIX_DURATION", "unknown_column")
    else:
        lines[1] = "\t".join(lines[1].split("\t")[:-1])
    report.write_text("\n".join(lines) + "\n")
    ds = CELER()
    ds.load(str(path))
    with pytest.raises(ValueError, match="missing columns|TSV header"):
        ds.fixations()


@pytest.mark.parametrize(
    "trial,area,accepted", [(45, ".", True), (44, ".", False), (45, 10, False)]
)
def test_unmatched_trials_require_documented_unread_status(
    reports, trial, area, accepted
):
    path, _, rows = reports
    extra = dict(
        rows[4],
        DATA_FILE="unread.edf",
        sentenceid="unread",
        list="37",
        trial=str(trial),
        CURRENT_FIX_INTEREST_AREA_ID=area,
        CURRENT_FIX_DURATION="1",
    )
    _write(path / "sent_fix.tsv", [*rows, extra])
    ds = CELER()
    ds.load(str(path))
    if accepted:
        assert len(ds.fixations()) == 7
        assert len(ds.data("total_time")) == 1
        np.testing.assert_equal(
            ds.data("total_time", calculation=FixationCalculation())[0].values,
            ds.data("total_time")[0].values,
        )
    else:
        with pytest.raises(ValueError, match="no interest-area summary"):
            ds.fixations()


@pytest.mark.needs_data
@pytest.mark.slow
def test_complete_release_roundtrips_all_measures(data_root):
    path = Path(data_root) / "eye-tracking/original/celer"
    if not (path / "sent_ia.tsv").is_file() or not (path / "sent_fix.tsv").is_file():
        pytest.skip("CELER interest-area and fixation reports are required")
    ds = CELER()
    ds.load(str(path))
    references = defaultdict(list)
    for measure in ds.measures:
        for datum in ds._native_responses(measure).values():
            references[datum.participant_id, datum.stimulus.stimulus_id].append(datum)
    compared = 0
    for key, events in ds.fixation_trials():
        cache = {}
        for reference in references.pop(key, []):
            for drop_skips in (False, True):
                ds.drop_skips = drop_skips
                datum = ds._skips_as_missing(reference) if drop_skips else reference
                result = ds.reaggregate(
                    datum,
                    datum.layer,
                    fixations=events,
                    _reference=reference,
                    _cache=cache,
                )
                np.testing.assert_allclose(
                    result.values, datum.values, rtol=0, atol=1e-6, equal_nan=True
                )
            compared += len(reference.values)
    assert not references, "Some summary trials lack source events"
    assert compared > 1_000_000
