"""Shared structural assertions for dataset-loader tests."""

from replica.dataset.measures import with_derived

#: Loaders whose source carries a first-pass skip flag, a first-pass
#: conditional column or a fixation stream expose the first run and the flag.
FLAGGED_MEASURES = [
    "first_fixation",
    "first_pass",
    "first_run",
    "go_past",
    "total_time",
]


def assert_eye_tracking_measures(dataset, expected=FLAGGED_MEASURES) -> None:
    """Assert the loaded data stores exactly the expected canonical measures."""
    stored = {datum.measure_name for datum in dataset.data()}
    assert stored, "the dataset holds no loaded data"
    assert sorted(with_derived(stored)) == sorted(with_derived(expected))
    assert sorted(dataset.measures) == sorted(with_derived(expected))


def measure_values(dataset, measure):
    """Return the first datum array for a named measure."""
    for datum in dataset.data():
        if datum.measure_name == measure:
            return datum.values
    raise AssertionError(f"no {measure} datum")
