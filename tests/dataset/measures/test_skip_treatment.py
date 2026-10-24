"""Skipped words in eye-tracking durations: missing by default, 0 ms on request."""

import numpy as np
import pytest

from replica.core import Datum
from replica.dataset import describe_dataset, load_dataset
from replica.dataset.base import EyeTrackingDataset
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.measures import CANONICAL_RT_MEASURES
from replica.multiverse.run.aggregation import aggregate_over_participants
from tests._stimuli import word_stimulus


class _Toy(EyeTrackingDataset):
    """Two readers on one three-word sentence; the second reader skips word 1."""

    def load(self, path: str) -> None:
        stimulus = word_stimulus("one two three", "s1")
        self._stimuli = [stimulus]
        self._data = [
            Datum(stimulus, "first_run", np.array([200.0, 300.0, 250.0]), "r1"),
            Datum(stimulus, "first_run", np.array([180.0, 0.0, 210.0]), "r2"),
            Datum(stimulus, "total_time", np.array([260.0, 0.0, 270.0]), "r2"),
        ]


@pytest.fixture
def toy() -> _Toy:
    dataset = _Toy()
    dataset.load("")
    return dataset


def test_skipped_words_are_missing_by_default(toy):
    assert toy.drop_skips is True
    r2 = toy.data(measure="first_run", participant="r2")[0]
    assert np.isnan(r2.values[1])
    stored = toy._data[1]
    stored.values = stored.values + 2j
    with pytest.raises(ValueError, match="real-valued"):
        toy.data(measure="first_run", participant="r2")
    np.testing.assert_equal(stored.values, [180 + 2j, 2j, 210 + 2j])


def test_skipped_words_read_as_zero_when_skips_are_kept(toy):
    toy.drop_skips = False
    for measure in ("first_run", "total_time"):
        r2 = toy.data(measure=measure, participant="r2")[0]
        assert r2.values[1] == 0.0
        assert not np.isnan(r2.values).any()
    # The stored datum keeps its zero, so the flag can be switched back.
    toy.drop_skips = True
    assert np.isnan(toy.data(measure="first_run", participant="r2")[0].values[1])


def test_the_two_treatments_give_the_paper_s_two_averages(toy):
    """Per-word means over readers: skips excluded, then skips as 0 ms."""
    aggregate_over_participants(toy, measure="first_run", strategy="excl")
    excluded = toy.data(measure="first_run")[0].values
    np.testing.assert_allclose(excluded, [190.0, 300.0, 230.0])

    fresh = _Toy()
    fresh.load("")
    aggregate_over_participants(fresh, measure="first_run", strategy="zero")
    zeroed = fresh.data(measure="first_run")[0].values
    np.testing.assert_allclose(zeroed, [190.0, 150.0, 230.0])


EYE_TRACKING_KEYS = [
    key
    for key, spec in REGISTRY.items()
    if issubclass(spec.loader_cls(), EyeTrackingDataset)
]


@pytest.mark.parametrize("key", [*EYE_TRACKING_KEYS, "meco"])
@pytest.mark.parametrize(
    "drop_skips", [None, True, False], ids=["default", "exclude", "include"]
)
def test_each_loader_exposes_reversible_skip_policy(
    monkeypatch, tmp_path, key, drop_skips
):
    """Exercise the public factory for every registered eye-tracking variant.

    File I/O is replaced by release-style datums containing zero skips and unavailable
    NaN measurements. Constructors, option routing, data access and masks are real. Raw
    file parsing is covered by each loader's source-specific tests.
    """

    def load(self, path):
        stimulus = word_stimulus("one two three four", "s1")
        self._stimuli = [stimulus]
        self._data = [
            Datum(stimulus, measure, np.array([200.0, 0.0, np.nan, 300.0]), "r1")
            for measure in sorted(CANONICAL_RT_MEASURES)
        ]
        self._data.append(
            Datum(stimulus, "eeg_theta_1", np.array([1.0, 0.0, np.nan, -1.0]), "r1")
        )

    spec = REGISTRY["meco_en" if key == "meco" else key]
    monkeypatch.setattr(spec.loader_cls(), "load", load)
    options = {} if drop_skips is None else {"drop_skips": drop_skips}
    dataset = load_dataset(key, path=tmp_path, **options)
    initial_policy = True if drop_skips is None else drop_skips
    assert dataset.drop_skips is initial_policy
    stored = [datum.values.copy() for datum in dataset._data]

    for policy in (initial_policy, not initial_policy, initial_policy):
        dataset.drop_skips = policy
        expected = (
            [200.0, np.nan, np.nan, 300.0] if policy else [200.0, 0.0, np.nan, 300.0]
        )
        for measure in CANONICAL_RT_MEASURES:
            [datum] = dataset.data(measure=measure, participant="r1")
            np.testing.assert_equal(datum.values, expected)
            np.testing.assert_equal(
                dataset.exclusion_mask(datum), [False, not policy, False, True]
            )
        np.testing.assert_equal(dataset.data("eeg_theta_1")[0].values, stored[-1])
        assert dataset.exclusion_mask(dataset.data("eeg_theta_1")[0])[1]
        for datum, original in zip(dataset._data, stored, strict=True):
            np.testing.assert_equal(datum.values, original)
        assert not dataset.data(participant="unknown")


@pytest.mark.parametrize("key", [*EYE_TRACKING_KEYS, "meco"])
def test_each_loader_documents_both_skip_options(key):
    [option] = [p for p in describe_dataset(key).parameters if p.name == "drop_skips"]
    assert option.default is True
    assert option.choices == (True, False)


def test_default_mask_excludes_a_raw_zero_duration(toy):
    datum = toy._data[0]
    datum.values[1] = 0.0
    assert not toy.exclusion_mask(datum)[1]
    toy.drop_skips = False
    assert toy.exclusion_mask(datum)[1]
