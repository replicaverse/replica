"""Shared source checks and independent round-trip assertions for fixation loaders.

Corpus tests inherit FixationsTestBase and implement its load fixture. Add
SyntheticTimingChecks when starts are constructed, or RecordedTimingChecks
when the source provides both endpoints.
"""

import abc
from collections import defaultdict

import numpy as np
import pytest

from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer


def group_fixations(fixations) -> dict:
    """Group fixation records by the datum identity used for aggregation."""
    grouped: dict = defaultdict(list)
    for fixation in fixations:
        grouped[(fixation.participant_id, fixation.stimulus_id)].append(fixation)
    return grouped


def assert_fixation_round_trip(
    dataset,
    *,
    source_measure: str,
    aggregate_measure: str | None = None,
    fixations_by_key: dict | None = None,
    unit_layer: str = "words",
    tolerance: float = 0.0,
    exclusive_tolerance: bool = False,
    missing_reconstruction: str = "zero",
    min_compared: int,
    via_dataset: bool = False,
    aggregation_cache: dict | None = None,
) -> None:
    """Assert that fixation aggregation reconstructs a dataset measure.

    Every observed cell must match within the stated numeric tolerance.
    Missing source cells must reconstruct as zero or NaN according to the
    selected measure policy; dataset reaggregation must preserve NaNs directly.
    """
    if missing_reconstruction not in {"zero", "nan"}:
        raise ValueError("missing_reconstruction must be 'zero' or 'nan'")
    if fixations_by_key is None:
        fixations_by_key = group_fixations(dataset.fixations())
    aggregate_measure = aggregate_measure or source_measure

    references = dataset._native_responses(source_measure) if via_dataset else {}
    n_compared = 0
    n_mismatches = 0
    missing_mismatches = []
    examples = []
    for datum in dataset.data(measure=source_measure):
        stimulus = datum.stimulus
        key = (datum.participant_id, stimulus.stimulus_id)
        events = fixations_by_key.get(key, [])
        if via_dataset:
            aggregate = dataset.reaggregate(
                datum,
                unit_layer,
                fixations=events,
                _cache=aggregation_cache,
                _reference=references[
                    (
                        stimulus.stimulus_id,
                        datum.participant_id,
                        datum.layer,
                        source_measure,
                    )
                ],
            ).values
        else:
            aggregate = aggregate_fixations_to_layer(events, stimulus, unit_layer)[
                aggregate_measure
            ]
        assert aggregate.shape == datum.values.shape

        for unit_index, (upstream, reconstructed) in enumerate(
            zip(datum.values, aggregate, strict=True)
        ):
            if np.isnan(upstream):
                matches = (
                    np.isnan(reconstructed)
                    if via_dataset or missing_reconstruction == "nan"
                    else reconstructed == 0.0
                )
                if not matches:
                    missing_mismatches.append(
                        (key, unit_index, upstream, reconstructed)
                    )
                continue
            n_compared += 1
            if not np.isfinite(reconstructed):
                missing_mismatches.append((key, unit_index, upstream, reconstructed))
                continue

            difference = abs(reconstructed - upstream)
            matches = np.isfinite(reconstructed) and (
                difference < tolerance
                if exclusive_tolerance
                else difference <= tolerance
            )
            if not matches:
                n_mismatches += 1
                if len(examples) < 5:
                    examples.append(
                        (
                            datum.participant_id,
                            stimulus.stimulus_id,
                            unit_index,
                            upstream,
                            reconstructed,
                        )
                    )

    assert not missing_mismatches, (
        f"{len(missing_mismatches)} missingness mismatches for {source_measure}: "
        f"{missing_mismatches[:5]}"
    )
    assert n_compared > min_compared, f"too few cells compared: {n_compared}"
    assert n_mismatches == 0, (
        f"{n_mismatches}/{n_compared} cells diverge {source_measure} "
        f"(tolerance {tolerance}). Examples: {examples}"
    )


def assert_trial_dwell_times(
    dataset,
    *,
    tolerance: float,
    min_compared: int,
    positive_only: bool = False,
) -> None:
    """Check source dwell metadata against per-trial fixation-duration sums."""
    sums: dict = defaultdict(float)
    upstream: dict = {}
    for fixation in dataset.fixations():
        if fixation.unit_index is None:
            continue
        trial_index = fixation.metadata.get("trial_index")
        if trial_index is None:
            continue
        key = (
            fixation.participant_id,
            fixation.stimulus_id,
            trial_index,
            fixation.unit_index,
        )
        if np.isfinite(fixation.duration):
            sums[key] += fixation.duration
        dwell_time = fixation.metadata.get("dwell_time")
        if (
            dwell_time is not None
            and np.isfinite(dwell_time)
            and (not positive_only or dwell_time > 0)
        ):
            upstream[key] = dwell_time

    mismatches = [
        (key, expected, sums.get(key, 0.0))
        for key, expected in upstream.items()
        if abs(sums.get(key, 0.0) - expected) > tolerance
    ]
    assert len(upstream) > min_compared, f"too few cells compared: {len(upstream)}"
    assert not mismatches, (
        f"{len(mismatches)}/{len(upstream)} trial dwell cells diverge. "
        f"Examples: {mismatches[:5]}"
    )


class FixationsTestBase(abc.ABC):
    """Smoke + invariant tests every fixation-aware corpus must pass."""

    min_fixations: int = 1000
    unit_layer: str = "words"

    @pytest.fixture(autouse=True)
    @abc.abstractmethod
    def load(self, data_root):
        """Subclass must populate ``self.ds`` via the data root fixture."""

    def test_fixations_loaded(self):
        assert len(self.ds.fixations()) >= self.min_fixations

    def test_filter_by_participant(self):
        assert self.ds.participants, "Fixation records must identify participants"
        a = next(iter(self.ds.participants))
        sub = self.ds.fixations(participant=a)
        all_fix = self.ds.fixations()
        assert 0 < len(sub) <= len(all_fix)
        assert all(f.participant_id == a for f in sub)

    def test_filter_by_stimulus(self):
        a = next(iter(self.ds.stimuli())).stimulus_id
        sub = self.ds.fixations(stimulus_id=a)
        assert 0 < len(sub) <= len(self.ds.fixations())
        assert all(f.stimulus_id == a for f in sub)

    def test_unit_indices_in_range(self):
        n_units = {
            s.stimulus_id: len(s.layers[self.unit_layer]) for s in self.ds.stimuli()
        }
        for f in self.ds.fixations():
            if f.unit_index is None:
                continue
            assert 0 <= f.unit_index < n_units[f.stimulus_id], (
                f"unit_index {f.unit_index} out of range for "
                f"{f.stimulus_id} (n_units={n_units[f.stimulus_id]})"
            )


class SyntheticTimingChecks:
    """Constructed start times must be identified in event provenance."""

    def test_synthetic_timing_marked(self):
        # Sample first 100, any unflagged fixation indicates a regression.
        for f in self.ds.fixations()[:100]:
            assert f.metadata.get("synthetic_start_time") is True


class RecordedTimingChecks:
    """Independent source endpoints must agree with recorded durations."""

    def test_durations_self_consistent(self):
        # EyeLink's convention: duration = end - start + 1 (inclusive
        # endpoints). 1.5 ms tolerance covers integer rounding.
        for f in self.ds.fixations()[:5000]:
            diff = abs(f.duration - (f.end_time - f.start_time))
            assert diff <= 1.5, (
                f"duration={f.duration} but end-start={f.end_time - f.start_time}"
            )
