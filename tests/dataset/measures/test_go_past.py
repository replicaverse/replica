"""Distinguish first arrival from first-pass entry using independent examples."""

import numpy as np
import pandas as pd
import pytest

from replica import AnalysisConfig, prepare_observations
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import Datum
from replica.dataset.eye_tracking.emtec import EMTeC
from replica.dataset.loading.fixation_utils import first_pass_flags, go_past_duration
from replica.dataset.measures import FIRST_PASS_GO_PAST
from replica.dataset.modelblocks import ModelBlocksDataset
from replica.feature import UnitLength


@pytest.mark.parametrize("include_off_text", [False, True])
def test_off_text_policy_only_changes_paths_containing_the_event(include_off_text):
    units = [None, 0, 1, None, 0, 2]
    durations = [90, 100, 200, 50, 70, 300]
    actual = go_past_duration(units, durations, 3, include_off_text=include_off_text)
    np.testing.assert_equal(actual, [100, 320 if include_off_text else 270, 300])


def test_missing_off_text_duration_invalidates_only_the_open_path():
    actual = go_past_duration(
        [None, 0, None, 1, 2],
        [np.nan, 100, np.nan, 200, 300],
        3,
        include_off_text=True,
    )
    np.testing.assert_equal(actual, [np.nan, 200, 300])


def _dataset(values, flags=None, total=None, *, conditional=False):
    ds = EMTeC()
    stimulus = ds._make_stimulus(
        ["one", "two", "three", "four", "five"], "trial", {"item_id": "7"}
    )
    ds._stimuli = [stimulus]
    ds._data = [
        Datum(
            stimulus,
            FIRST_PASS_GO_PAST if conditional else "go_past",
            np.array(values, dtype=float),
            "reader",
        )
    ]
    for name, column in [("first_pass", flags), ("total_time", total)]:
        if column is not None:
            ds._data.append(
                Datum(stimulus, name, np.array(column, dtype=float), "reader")
            )
    return ds


@pytest.mark.parametrize("drop_skips", [False, True])
def test_a_later_first_visit_is_measured_only_by_the_arrival_window(drop_skips):
    # Visit word 2, regress to word 1 which the eye skipped, revisit 2, then 3.
    # Word 2's go-past includes the regression, while word 1 has no first pass.
    units, durations = [1, 0, 1, 2], [180, 200, 70, 150]
    ds = _dataset(go_past_duration(units, durations, 5), first_pass_flags(units, 5))
    ds.drop_skips = drop_skips
    np.testing.assert_array_equal(ds.data("go_past")[0].values[:3], [200, 450, 150])
    expected = [np.nan if drop_skips else 0, 450, 150]
    np.testing.assert_array_equal(ds.data(FIRST_PASS_GO_PAST)[0].values[:3], expected)


@pytest.mark.parametrize("drop_skips", [False, True])
def test_conditional_release_keeps_missing_distinct_from_zero(drop_skips):
    # A first-pass visit, a late visit, a never-visited word, a missing source
    # measurement despite a known skip, and an unavailable observation.
    source = [300, 0, 0, np.nan, np.nan]
    ds = _dataset(
        source, [1, 0, 0, 0, np.nan], [350, 200, 0, 100, np.nan], conditional=True
    )
    stored = ds._data[0].values
    ds._data[0].values = stored + 2j
    with pytest.raises(ValueError, match="real-valued"):
        ds._add_arrival_go_past()
    assert len(ds._data) == 3
    ds._data[0].values = stored
    ds._add_arrival_go_past()
    ds.drop_skips = drop_skips
    zero = np.nan if drop_skips else 0
    np.testing.assert_array_equal(
        ds.data(FIRST_PASS_GO_PAST)[0].values, [300, zero, zero, np.nan, np.nan]
    )
    np.testing.assert_array_equal(
        ds.data("go_past")[0].values, [300, np.nan, zero, np.nan, np.nan]
    )
    np.testing.assert_array_equal(ds._data[0].values, source)


def test_a_conditional_release_needs_no_inferred_flag():
    ds = _dataset([300, 0, np.nan, 250, 200], conditional=True)
    ds.drop_skips = False
    ds._add_arrival_go_past()
    np.testing.assert_array_equal(
        ds.data("go_past")[0].values, [300, np.nan, np.nan, 250, 200]
    )
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[ds.feature(FIRST_PASS_GO_PAST)],
    )
    table = prepare_observations(ds, spec, AnalysisConfig(transform_dv=lambda x: x))
    # First word structurally excluded; source missing stays excluded, zero kept.
    np.testing.assert_array_equal(table.y, [0, 250, 200])


def test_modelblocks_keeps_its_conditional_column_verbatim():
    df = pd.DataFrame(
        {
            "word": ["one", "two", "three"],
            "subject": ["r"] * 3,
            "docid": [1] * 3,
            "sentid": [1] * 3,
            "sentpos": [1, 2, 3],
            "fdurGP": [300, 0, np.nan],
        }
    )
    ds = ModelBlocksDataset(dataframe=df, dv_column="fdurGP")
    assert ds.measures == [FIRST_PASS_GO_PAST]
    np.testing.assert_array_equal(
        ds.data(FIRST_PASS_GO_PAST)[0].values, [300, 0, np.nan]
    )
    assert not ds.data("go_past")
    assert not ds.is_additive(FIRST_PASS_GO_PAST)


@pytest.mark.parametrize("column", ["RPD_inc", "RTgopast", "firstrun.gopast", "fdurGP"])
def test_source_spellings_name_the_conditional_measure(column):
    from replica.dataset.measures import canonical_measure

    assert canonical_measure(column) == FIRST_PASS_GO_PAST


def test_eyelink_spelling_keeps_the_arrival_measure():
    from replica.dataset.measures import canonical_measure

    assert canonical_measure("IA_REGRESSION_PATH_DURATION") == "go_past"


def test_no_registered_loader_sums_non_additive_reading_windows():
    from replica.dataset.catalog.registry import REGISTRY

    for spec in REGISTRY.values():
        loader = spec.loader_cls()
        non_additive = {
            "first_fixation",
            "first_run",
            "right_bounded",
            "go_past",
            "first_fixation_in_first_pass",
            "first_run_in_first_pass",
            FIRST_PASS_GO_PAST,
        }
        assert not (non_additive & loader.additive_measures), spec.loader_class


def test_conditional_go_past_declarations_name_both_quantities():
    from replica.dataset.measures.sources import MEASURE_SOURCES

    for name in [
        "MECO_L1",
        "MECOL1W2",
        "MECOL2W1",
        "MECOL2W2",
        "UCL_ET",
        "Dundee",
    ]:
        declared = MEASURE_SOURCES[name]
        released = declared[FIRST_PASS_GO_PAST]
        assert released.quantity == "regression_path_in_first_pass"
        assert released.rule == "released"
        assert declared["go_past"].quantity == "regression_path"
        assert declared["go_past"].rule == "conditional_column"
        assert declared["go_past"].source == released.source


def test_complete_fixation_go_past_declarations():
    from replica.dataset.measures.sources import MEASURE_SOURCES

    for name in ("EMTeC", "PoTeC"):
        declared = MEASURE_SOURCES[name]
        assert declared["go_past"].quantity == "regression_path"
        assert declared[FIRST_PASS_GO_PAST].quantity == "regression_path_in_first_pass"
        assert declared["go_past"].rule == "fixation_stream"
        assert declared[FIRST_PASS_GO_PAST].rule == "fixation_stream"
