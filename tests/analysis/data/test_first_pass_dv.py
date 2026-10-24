"""A dependent variable restricted to first pass reaches the regression as the
first run where the flag holds and as a skip where it does not."""

import numpy as np
import pytest

from replica import AnalysisConfig, prepare_observations
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import DERIVED_DVS, AnnotationLayer, Datum, Span, Stimulus, resolve_dv
from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.measures import FIRST_PASS_RUN
from replica.feature import UnitLength

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def _dataset(*, with_flag: bool) -> EyeTrackingDataset:
    class _DS(EyeTrackingDataset):
        info = DatasetInfo(name="toy_et")

        def load(self, path):
            pass

    ds = _DS()
    spans, pos = [], 0
    for word in WORDS:
        spans.append(Span(pos, pos + len(word)))
        pos += len(word) + 1
    stimulus = Stimulus(text=" ".join(WORDS), stimulus_id="s1")
    stimulus.add_layer(AnnotationLayer("words", spans, WORDS))
    ds._stimuli.append(stimulus)
    for reader, offset in (("r1", 0.0), ("r2", 10.0)):
        first_run = np.array([210.0, 180.0, 250.0, 0.0, 300.0, 190.0])
        first_run[first_run > 0] += offset
        ds._data.append(Datum(stimulus, "first_run", first_run, reader))
        if with_flag:
            # Word 4 was first entered by a regression; word 3 was never fixated.
            flag = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            ds._data.append(Datum(stimulus, "first_pass", flag, reader))
    return ds


def _rows(ds: EyeTrackingDataset, measure: str) -> int:
    """Rows the analysis layer keeps for one DV, under the dataset's skip policy.

    The DV is left untransformed, so a kept zero is not lost to the log.
    """
    spec = EvaluationSpec(predictors=[UnitLength()], targets=[ds.feature(measure)])
    config = AnalysisConfig(transform_dv=lambda values: values)
    return len(prepare_observations(ds, spec, config, target_name=measure).y)


def test_restriction_turns_a_regression_entry_into_a_skip():
    ds = _dataset(with_flag=True)
    # Six words less the first unit, one released skip, two readers.
    assert _rows(ds, "first_run") == 8
    # The regression entry joins the skip and is masked with it.
    assert _rows(ds, "first_run_in_first_pass") == 6
    ds.drop_skips = False
    assert _rows(ds, FIRST_PASS_RUN) == 10
    conditioned = ds.data(measure=FIRST_PASS_RUN, participant="r1")[0]
    np.testing.assert_array_equal(conditioned.values, [210, 180, 250, 0, 0, 190])


def test_the_stored_values_are_untouched_by_the_restriction():
    ds = _dataset(with_flag=True)
    before = [d.values.copy() for d in ds._data]
    ds.data(measure=FIRST_PASS_RUN)
    for datum, values in zip(ds._data, before, strict=True):
        np.testing.assert_array_equal(datum.values, values)


def test_a_corpus_without_the_flag_refuses_the_restriction():
    ds = _dataset(with_flag=False)
    assert _rows(ds, "first_run") == 8
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        targets=[ds.feature("first_run_in_first_pass")],
    )
    with pytest.raises(ValueError, match="first_pass"):
        prepare_observations(ds, spec)
    with pytest.raises(ValueError, match="first_pass"):
        ds.data(measure=FIRST_PASS_RUN)


def test_a_missing_flag_for_one_reader_drops_only_that_reader():
    ds = _dataset(with_flag=True)
    ds._data = [
        d
        for d in ds._data
        if not (d.measure_name == "first_pass" and d.participant_id == "r2")
    ]
    assert _rows(ds, "first_run_in_first_pass") == 3


def test_the_derived_variables_name_the_supported_first_pass_windows():
    assert set(DERIVED_DVS) == {
        "first_fixation_in_first_pass",
        FIRST_PASS_RUN,
        "go_past_in_first_pass",
    }
    assert resolve_dv(FIRST_PASS_RUN) == ("first_run", "first_pass")
    assert resolve_dv("total_time") == ("total_time", None)
