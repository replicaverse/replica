"""Participant aggregation preserves missing values and skip policies."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Datum, Span, Stimulus
from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.measures import FIRST_PASS_RUN
from replica.multiverse.run import aggregation


def _participant_dataset(rows: dict[str, list[list[float]]]):
    """An eye-tracking dataset with one first_run datum per participant and a
    first-pass flag of ones, so the derived variable reads the same values."""

    class _DS(EyeTrackingDataset):
        info = DatasetInfo(name="toy_et")

        def load(self, path):
            pass

    ds = _DS()
    for sid, participant_rows in rows.items():
        words = [f"w{i}" for i in range(len(participant_rows[0]))]
        spans, pos = [], 0
        for word in words:
            spans.append(Span(pos, pos + len(word)))
            pos += len(word) + 1
        stimulus = Stimulus(text=" ".join(words), stimulus_id=sid)
        stimulus.add_layer(AnnotationLayer("words", spans, words))
        ds._stimuli.append(stimulus)
        for i, values in enumerate(participant_rows):
            ds._data.append(
                Datum(stimulus, "first_run", np.asarray(values, dtype=float), f"p{i}")
            )
            ds._data.append(
                Datum(stimulus, "first_pass", np.ones(len(values)), f"p{i}")
            )
    return ds


def test_aggregate_over_participants_excl_drops_non_fixations():
    ds = _participant_dataset({"s1": [[200.0, 0.0], [100.0, 300.0]]})

    n_stim, dropped = aggregation.aggregate_over_participants(
        ds, FIRST_PASS_RUN, "excl"
    )

    assert (n_stim, dropped) == (1, 1)
    (datum,) = [d for d in ds._data if d.measure_name == FIRST_PASS_RUN]
    assert datum.participant_id == "aggregated"
    # Zeros are non-fixations and are excluded from the mean.
    np.testing.assert_allclose(datum.values, [150.0, 300.0])


def test_aggregate_over_participants_zero_keeps_skips_as_zero():
    ds = _participant_dataset({"s1": [[200.0, 0.0], [100.0, 300.0]]})

    aggregation.aggregate_over_participants(ds, FIRST_PASS_RUN, "zero")

    (datum,) = [d for d in ds._data if d.measure_name == FIRST_PASS_RUN]
    np.testing.assert_allclose(datum.values, [150.0, 150.0])


def test_aggregate_over_participants_all_skipped_is_zero_and_rejects_unknown():
    ds = _participant_dataset({"s1": [[0.0], [0.0]]})
    aggregation.aggregate_over_participants(ds, FIRST_PASS_RUN, "zero")
    (datum,) = [d for d in ds._data if d.measure_name == FIRST_PASS_RUN]
    assert datum.values[0] == 0.0

    with pytest.raises(ValueError, match="unknown aggregation strategy"):
        aggregation.aggregate_over_participants(
            _participant_dataset({"s1": [[1.0]]}), FIRST_PASS_RUN, "sideways"
        )


def test_aggregate_excl_all_skipped_column_is_nan():
    import numpy as np

    ds = _participant_dataset({"s1": [[0.0], [0.0]]})

    aggregation.aggregate_over_participants(ds, FIRST_PASS_RUN, "excl")

    (datum,) = [d for d in ds._data if d.measure_name == FIRST_PASS_RUN]
    assert np.isnan(datum.values[0])


@pytest.mark.parametrize(
    "strategy, expected",
    [("zero", [150.0, 300.0, np.nan, 0.0]), ("excl", [300.0, 300.0, np.nan, np.nan])],
)
def test_aggregate_distinguishes_skips_from_unavailable_measurements(
    strategy, expected
):
    ds = _participant_dataset(
        {"s1": [[0.0, np.nan, np.nan, 0.0], [300.0, 300.0, np.nan, np.nan]]}
    )
    aggregation.aggregate_over_participants(ds, FIRST_PASS_RUN, strategy)
    (datum,) = [d for d in ds._data if d.measure_name == FIRST_PASS_RUN]
    np.testing.assert_equal(datum.values, expected)
