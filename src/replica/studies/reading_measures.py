"""Response selections shared by the new-corpus reading studies."""

from dataclasses import replace

import numpy as np

from replica.dataset import FixationCalculation
from replica.dataset.eye_tracking.rsc import RSC
from replica.dataset.eye_tracking.zuco.base import ZuCoTaskDataset
from replica.dataset.measures import FIRST_PASS_RUN

_ZUCO_DERIVED_MEASURES = frozenset({FIRST_PASS_RUN, "first_pass", "right_bounded"})
_ZUCO_CALCULATION = FixationCalculation()


def reading_response_policy(dataset, measure: str) -> dict:
    """Describe the selected response without changing the loader's defaults.

    Args:
        dataset: Loaded reading dataset.
        measure: Exact response requested by the study.
    """
    if isinstance(dataset, ZuCoTaskDataset) and measure in _ZUCO_DERIVED_MEASURES:
        return {
            "source": "fixation_events",
            "native_reference_measure": "first_run",
            "calculation": _ZUCO_CALCULATION.to_dict(),
            "event_filter": "on_word_and_at_least_100_ms",
            "assignment": "physical_event_candidates",
        }
    if isinstance(dataset, RSC) and measure in {"go_past", "go_past_in_first_pass"}:
        exclusions = []
        for datum in dataset.data(measure=measure):
            indices = np.flatnonzero(~dataset.go_past_consistency_mask(datum)).tolist()
            if indices:
                exclusions.append(
                    {
                        "stimulus_id": datum.stimulus.stimulus_id,
                        "participant_id": datum.participant_id,
                        "unit_indices": indices,
                    }
                )
        return {
            "source": "dataset_default",
            "release": dataset.release,
            "consistency_screen": "go_past_not_shorter_than_first_run",
            "exclude_inconsistent_go_past": dataset.exclude_inconsistent_go_past,
            "screen_stage": "source_contradictions_before_skip_and_analysis_filters",
            "excluded_observations": sum(
                len(row["unit_indices"]) for row in exclusions
            ),
            "exclusions": exclusions,
        }
    return {"source": "dataset_default"}


def reading_responses(dataset, measures, *, drop_skips=None) -> dict:
    """Select study responses with explicit source calculations and exclusions.

    ZuCo's first-pass measures are computed from its fixation events, with its
    released first run as the validation reference; the shared mapping and
    calculation keep ambiguous landings unknown. RSC's optional consistency
    screen masks contradictory go-past observations in copied responses,
    leaving the released datums intact.

    Args:
        dataset: Loaded reading dataset.
        measures: Exact response names requested by the study.
        drop_skips: Optional override, restored before returning or raising.
    """
    measures = tuple(measures)
    derived = (
        set(measures) & _ZUCO_DERIVED_MEASURES
        if isinstance(dataset, ZuCoTaskDataset)
        else set()
    )
    previous = getattr(dataset, "drop_skips", True)
    if drop_skips is not None:
        dataset.drop_skips = drop_skips
    try:
        result = {
            measure: list(dataset.data(measure=measure))
            for measure in measures
            if measure not in derived
        }
        if isinstance(dataset, RSC):
            result = {
                measure: [
                    replace(
                        datum,
                        values=np.where(
                            dataset.go_past_consistency_mask(datum),
                            datum.values,
                            np.nan,
                        ),
                    )
                    for datum in datums
                ]
                for measure, datums in result.items()
            }
        if not derived:
            return result
        references = dataset._native_responses("first_run")
        pending = {
            (d.participant_id, d.stimulus.stimulus_id): d for d in references.values()
        }
        if not pending or len(pending) != len(references):
            raise ValueError("ZuCo first-pass calculation needs unique native trials")
        result.update({measure: [] for measure in derived})
        seen = set()
        for pair, events in dataset.fixation_trials():
            if pair in seen or pair not in pending:
                raise ValueError("ZuCo events do not match unique native trials")
            seen.add(pair)
            reference = pending.pop(pair)
            datum = (
                dataset._skips_as_missing(reference)
                if dataset.drop_skips
                else reference
            )
            cache = {}
            for measure in sorted(derived):
                result[measure].append(
                    dataset.reaggregate(
                        datum,
                        datum.layer,
                        measure=measure,
                        calculation=_ZUCO_CALCULATION,
                        fixations=events,
                        _reference=reference,
                        _cache=cache,
                    )
                )
        if pending:
            raise ValueError("ZuCo first-pass calculation requires every source trial")
        return result
    finally:
        if drop_skips is not None:
            dataset.drop_skips = previous
