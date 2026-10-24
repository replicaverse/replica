"""Internal helpers for raw-fixation loaders.

Centralizes algorithms shared across several eye-tracking loaders:

* :func:`append_fixation`, canonical event construction.

* :func:`first_run_durations`, first-run aggregation from a fixation stream,
  and :func:`first_pass_flags`, which units that stream reached in first pass.

* :func:`go_past_duration`, regression-path aggregation from the same stream.

* :class:`RunIdInferrer`, EyeLink-style ``run_id`` inference from a
  temporally-ordered fixation stream when the source CSV doesn't carry
  ``CURRENT_FIX_INTEREST_AREA_RUN_ID``.

* :class:`SyntheticTimer`, trial-relative ``start_time`` from cumulative
  fixation durations when the source CSV doesn't carry per-fixation
  start/end timestamps.

``RunIdInferrer`` and ``SyntheticTimer`` keep state internal and require
fixations in source-file (temporal) order.
"""

import heapq
from collections import defaultdict
from collections.abc import Hashable, Iterable, MutableSequence, Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np

from replica.core import Datum
from replica.dataset.contracts import Fixation


def mark_unfixated(
    measures: Sequence[np.ndarray],
    measured_indices: Iterable[int],
    visited_indices: Iterable[int],
) -> None:
    """Store zero for measured units absent from a trial's fixation stream.

    Args:
        measures: Duration arrays to update in place.
        measured_indices: Units with a valid mapping into the source stream.
            Unmapped or ambiguous units are not eligible for zero filling.
        visited_indices: Units with any fixation event, including events with
            unavailable durations. Missing measures there remain unavailable.
    """
    unfixated = list(set(measured_indices) - set(visited_indices))
    for values in measures:
        values[unfixated] = 0.0


def append_fixation(
    output: MutableSequence[Fixation],
    *,
    participant_id: str,
    stimulus_id: str,
    start_time: float,
    end_time: float,
    duration: float,
    x: float,
    y: float,
    unit_index: int | None,
    char_offset: int | None,
    fixation_index: int,
    metadata: dict[str, Any] | None = None,
    unit_layer: str = "words",
    char_offset_is_exact: bool = False,
    candidate_unit_indices: tuple[int | None, ...] = (),
) -> None:
    """Append one event using REPLICA's canonical fixation schema.

    Args:
        output: List the new :class:`~replica.dataset.contracts.Fixation` is added to.
        participant_id: Reader the event belongs to.
        stimulus_id: Stimulus the reader was looking at.
        start_time: Trial-relative onset, milliseconds.
        end_time: Trial-relative offset, milliseconds.
        duration: Dwell time of this fixation, milliseconds.
        x: Horizontal gaze position in the release's pixel space.
        y: Vertical gaze position in the release's pixel space.
        unit_index: 0-based position in the stimulus's unit layer, ``None`` when
            the fixation fell off the text.
        char_offset: Character offset of that unit's span into the stimulus text,
            or an exact landing when established by the source. May be omitted
            when ``unit_index`` identifies the source area.
        fixation_index: 1-based ordinal of this fixation within its trial.
        metadata: Corpus-specific extras (pupil size, run id); the event stores a
            copy.
        unit_layer: Source annotation layer indexed by ``unit_index``.
        char_offset_is_exact: Whether the offset is the exact character the eye
            landed on.
        candidate_unit_indices: Conflicting source areas, reported position first;
            None denotes a possible off-text landing.
    """
    output.append(
        Fixation(
            participant_id=participant_id,
            stimulus_id=stimulus_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            x=x,
            y=y,
            unit_index=unit_index,
            char_offset=char_offset,
            fixation_index=fixation_index,
            metadata=dict(metadata or {}),
            unit_layer=unit_layer,
            char_offset_is_exact=char_offset_is_exact,
            candidate_unit_indices=candidate_unit_indices,
        )
    )


def first_run_durations(
    ia_ids: list,
    durations: list,
    ia_to_idx: dict,
    n_words: int,
    *,
    off_text_ends_run: bool = True,
    run_ids: Sequence[int | None] | None = None,
) -> np.ndarray:
    """First-run duration per word from an ordered fixation stream.

    The first run is the sum of fixations from the eye's first arrival on a
    word, even by a regression, until its first departure. Fixations must be
    supplied in temporal order.

    Args:
        ia_ids: Interest-area identifier each fixation landed on, in temporal
            order.
        durations: Fixation durations in milliseconds, parallel to *ia_ids*.
        ia_to_idx: Interest-area identifier to 0-based word position.
        n_words: Length of the output array, one entry per word.
        off_text_ends_run: Whether an identifier absent from ``ia_to_idx``
            closes the open run, as the UCL release counts it. Otherwise such
            fixations are skipped without disturbing the accumulation state.
        run_ids: Optional source run IDs for these same units; missing IDs
            use the observed contiguous visit.

    Returns:
        A ``(n_words,)`` array, NaN for never-fixated words.
    """
    first_run = np.full(n_words, np.nan)
    seen: set[int] = set()
    current: int | None = None
    in_first_run = False
    ids = run_ids if run_ids is not None else [None] * len(ia_ids)
    for ia_id, duration, source_run in zip(ia_ids, durations, ids, strict=True):
        unit = ia_to_idx.get(ia_id)
        if unit is None:
            if off_text_ends_run:
                current = None
                in_first_run = False
            continue
        if unit != current:
            current = unit
            in_first_run = unit not in seen
        if unit not in seen:
            first_run[unit] = 0.0
            seen.add(unit)
        qualifies = source_run == 1 if source_run is not None else in_first_run
        if qualifies:
            first_run[unit] += np.nan if duration is None else duration
    return first_run


def first_pass_units(unit_indices: Iterable[int | None]) -> set[int]:
    """Units the eye reached before it had ever been further right.

    A unit first fixated after the reader has already fixated a later unit is
    entered by a regression, so it has no first pass. Off-text fixations
    (``None``) neither count nor advance the frontier.

    Args:
        unit_indices: 0-based unit of each fixation, in temporal order.
    """
    eligible: set[int] = set()
    seen: set[int] = set()
    furthest = -1
    for unit in unit_indices:
        if unit is None:
            continue
        if unit not in seen:
            seen.add(unit)
            if unit > furthest:
                eligible.add(unit)
        furthest = max(furthest, unit)
    return eligible


def first_pass_flags(unit_indices: Iterable[int | None], n_units: int) -> np.ndarray:
    """The first-pass flag per unit from the fixation order.

    Args:
        unit_indices: 0-based unit of each fixation, in temporal order;
            ``None`` or out-of-range entries are off-text.
        n_units: Length of the output array, one entry per unit.

    Returns:
        1.0 for units the eye reached before it had ever been further right,
        0.0 for every other unit, whether skipped or first entered by a
        regression.
    """
    flags = np.zeros(n_units)
    for unit in first_pass_units(unit_indices):
        if 0 <= unit < n_units:
            flags[unit] = 1.0
    return flags


def go_past_duration(
    unit_indices: list[int | None],
    durations: list[float],
    n_units: int,
    *,
    include_off_text: bool = False,
    require_right_exit: bool = False,
) -> np.ndarray:
    """Go-past (regression-path) duration from an ordered fixation stream.

    The time from the eye's first arrival on a unit until it first moves beyond
    it to the right, regressions to earlier units included. Off-text fixations
    never end a region and count only when ``include_off_text`` is true.
    Fixations must be in temporal order.

    Args:
        unit_indices: 0-based unit each fixation landed on, in temporal order;
            ``None`` or out-of-range entries are off-text.
        durations: Fixation durations in milliseconds, parallel to
            *unit_indices*.
        n_units: Length of the output array, one entry per unit.
        include_off_text: Count off-text dwell within an open regression path.
        require_right_exit: Leave an unfinished path unavailable at trial end.

    Returns:
        An ``(n_units,)`` array, NaN for never-fixated units.
    """
    go_past = np.full(n_units, np.nan)
    started_at: dict[int, tuple[float, int]] = {}
    open_units: list[int] = []
    elapsed = 0.0
    missing = 0

    def elapsed_since(unit: int) -> float:
        start, missing_before = started_at[unit]
        return elapsed - start if missing == missing_before else np.nan

    for unit, duration in zip(unit_indices, durations, strict=True):
        if unit is None or not 0 <= unit < n_units:
            if include_off_text:
                if duration is None or np.isnan(duration):
                    missing += 1
                else:
                    elapsed += duration
            continue
        while open_units and open_units[0] < unit:
            passed = heapq.heappop(open_units)
            go_past[passed] = elapsed_since(passed)
        if unit not in started_at:
            started_at[unit] = (elapsed, missing)
            heapq.heappush(open_units, unit)
        if duration is None or np.isnan(duration):
            missing += 1
        else:
            elapsed += duration
    while open_units:
        passed = heapq.heappop(open_units)
        if not require_right_exit:
            go_past[passed] = elapsed_since(passed)
    return go_past


def right_bounded_durations(
    indices: Sequence[int | None], durations: Sequence[float], n_units: int
) -> np.ndarray:
    """Sum each unit's dwell before the eye first reaches a later unit.

    Args:
        indices: Landing units in event order, with None for off-text events.
        durations: Event durations; a missing duration makes its unit unknown.
        n_units: Number of units in reading order.
    """
    values = np.zeros(n_units)
    furthest = -1
    for unit, duration in zip(indices, durations, strict=True):
        if unit is None:
            continue
        if not 0 <= unit < n_units:
            raise ValueError("Fixation unit is outside the stimulus")
        if unit >= furthest:
            values[unit] += duration
        furthest = max(furthest, unit)
    return values


class RunIdInferrer:
    """Infer EyeLink-style ``run_id`` per (trial, unit) without source data.

    EyeLink's RUN_ID counts runs per interest area:

    * The very first fixation on a given unit gets ``run_id = 1``
      regardless of whether the reader has already progressed past it
      (a non-linear scan path can land on word 5 before word 0).
    * Consecutive fixations on the same unit (no intervening other unit)
      stay in the same run, they share the previous ``run_id``.
    * Returning to that unit after fixating something else starts a new
      run on it: ``run_id = N+1`` where ``N`` is the count of completed
      prior runs on the same unit.

    State is keyed by an opaque ``trial_key`` (anything hashable
    ``(participant_id, trial_index)`` is the typical choice). The caller
    is responsible for passing fixations in temporal order.
    """

    def __init__(self) -> None:
        """Start with no trial seen and no run opened on any unit."""
        self._prev_unit: dict[Hashable, int] = {}
        self._last_run_id: dict[Hashable, int] = {}
        # Per-(trial_key, unit_index): how many runs have STARTED on this
        # unit so far (i.e. the run_id of the current/last run).
        self._run_count: dict[tuple, int] = defaultdict(int)

    def next(self, trial_key: Hashable, unit_index: int | None) -> int | None:
        """Run id for one fixation, advancing the trial's state.

        Must be called in temporal order within each trial.

        Args:
            trial_key: Opaque hashable identifying the trial, typically
                ``(participant_id, trial_index)``.
            unit_index: 0-based landing unit, or None for an off-text fixation.

        Returns:
            1 on the unit's first run, and one more per later return to it;
            consecutive fixations on the same unit share a run id. Off-text
            fixations end the current run and return None.
        """
        if unit_index is None:
            self._prev_unit.pop(trial_key, None)
            self._last_run_id.pop(trial_key, None)
            return None
        prev = self._prev_unit.get(trial_key)
        unit_key = (trial_key, unit_index)
        if prev == unit_index:
            # Same unit as previous fixation → continuation of current run.
            run_id = self._last_run_id.get(trial_key, 1)
        else:
            # Different unit (or first fixation) → start a new run.
            self._run_count[unit_key] += 1
            run_id = self._run_count[unit_key]

        self._prev_unit[trial_key] = unit_index
        self._last_run_id[trial_key] = run_id
        return run_id


class SyntheticTimer:
    """Trial-relative ``start_time`` from cumulative fixation durations.

    Used when the source CSV ships only ``fixation_duration`` (no per-row
    start/end timestamps). The synthesized timeline is monotonic within
    each trial and starts at 0; absolute session timing is unrecoverable.

    Loaders setting timing this way also stamp
    ``Fixation.metadata["synthetic_start_time"] = True`` to flag that
    timestamps are not comparable across trials.
    """

    SENTINEL_KEY = "synthetic_start_time"

    def __init__(self) -> None:
        """Start every trial's cursor at 0 milliseconds."""
        self._cum: dict[Hashable, float] = defaultdict(float)

    def next(self, trial_key: Hashable, duration: float) -> tuple[float, float]:
        """Return ``(start_time, end_time)`` and advance the trial cursor.

        Non-finite durations (NaN, inf) are treated as 0-length for
        cursor purposes, so a corrupt row doesn't cascade.

        Args:
            trial_key: Opaque hashable identifying the trial, typically
                ``(participant_id, trial_index)``.
            duration: This fixation's dwell time in milliseconds.
        """
        start = self._cum[trial_key]
        if duration is None or not np.isfinite(duration):
            end = start
        else:
            end = start + duration
        self._cum[trial_key] = end
        return start, end


def _sort_fixation_visit(visit: list[Fixation], order: str) -> list[Fixation]:
    """Order a source visit without mixing its clock with other visits."""
    if order == "fixation_index":
        if len({e.fixation_index for e in visit}) != len(visit):
            raise ValueError("Duplicate fixation indices within a trial")
        ordered = sorted(visit, key=lambda e: e.fixation_index)
    elif all(e.start_time is not None and np.isfinite(e.start_time) for e in visit):
        ordered = sorted(visit, key=lambda e: e.start_time)
    else:
        ordered = sorted(visit, key=lambda e: e.fixation_index)
    return ordered


def _uncertain_positions(events):
    """Find units whose dwell or first-pass status depends on a source conflict."""
    uncertain_pass, uncertain_total = set(), set()
    frontier = -1
    for event in events:
        if event.candidate_unit_indices:
            candidates = {u for u in event.candidate_unit_indices if u is not None}
            uncertain_total.update(candidates)
            uncertain_pass.update(candidates)
            uncertain_pass.update(range(frontier + 1, max(candidates) + 1))
        elif event.unit_index is not None:
            frontier = max(frontier, event.unit_index)
    return uncertain_pass, uncertain_total


def _uncertain_window_end(indices, unit, start, stop, measure, off_text_ends_run):
    """Find the boundary whose uncertain landing could change a reading window."""
    if measure in {"first_fixation", "first_pass"}:
        return start + 1
    for i in range(start + 1, stop):
        landing = indices[i]
        if measure == "first_run":
            if landing != unit and (landing is not None or off_text_ends_run):
                return i
        elif landing is not None and landing > unit:
            return i
    return stop


def uncertain_fixation_masks(visits, n_units, *, off_text_ends_run=True):
    """Conservatively mark measures affected by alternative source positions.

    The first candidate supplies only a provisional scanpath. Windows
    containing a conflict, including their terminating landing, remain
    unknown. A trial boundary is not a landing.

    Args:
        visits: Event sequences in presentation order, each already sorted.
        n_units: Number of ordered units in the target layer.
        off_text_ends_run: Whether off-text landings terminate the first run.

    Returns:
        Per-measure masks and the implicated first-pass and total-time unit sets.
    """
    events = [event for visit in visits for event in visit]
    bad = {i for i, event in enumerate(events) if event.candidate_unit_indices}
    measures = (
        "first_fixation",
        "first_run",
        "first_pass",
        "go_past",
        "total_time",
        "right_bounded",
    )
    if not bad:
        return {name: np.zeros(n_units, dtype=bool) for name in measures}, set(), set()
    indices = [
        e.candidate_unit_indices[0] if e.candidate_unit_indices else e.unit_index
        for e in events
    ]
    ends = []
    offset = 0
    for visit in visits:
        offset += len(visit)
        ends.extend([offset] * len(visit))
    uncertain_pass, uncertain_total = _uncertain_positions(events)
    first_events = {}
    for i, unit in enumerate(indices):
        if unit is not None:
            first_events.setdefault(unit, i)
    masks = {}
    for measure in measures:
        invalid = np.zeros(n_units, dtype=bool)
        for unit in range(n_units):
            start = first_events.get(unit, len(events))
            invalid[unit] = start in bad
            if measure == "total_time":
                invalid[unit] |= unit in uncertain_total
            else:
                stop = (
                    ends[start]
                    if start < len(events) and measure != "right_bounded"
                    else len(events)
                )
                end = _uncertain_window_end(
                    indices, unit, start, stop, measure, off_text_ends_run
                )
                if measure in {"first_run", "go_past", "right_bounded"}:
                    invalid[unit] |= any(start <= i <= end and i < stop for i in bad)
                if unit in uncertain_pass:
                    invalid[unit] |= any(i < end for i in bad)
        masks[measure] = invalid
    return masks, uncertain_pass, uncertain_total


def _mask_uncertain_landings(arrays, visits, off_text_ends_run):
    """Apply source-position uncertainty to durations and their restrictions."""
    if not any(e.candidate_unit_indices for visit in visits for e in visit):
        return
    masks, _, _ = uncertain_fixation_masks(
        visits, len(arrays["first_pass"]), off_text_ends_run=off_text_ends_run
    )
    flag = arrays["first_pass"]
    for measure in ("first_fixation", "first_run", "go_past"):
        invalid = masks["first_pass"] | ((flag == 1) & masks[measure])
        if measure == "go_past":
            # Preserve Dundee's conservative availability rule for an
            # uncertain regression-path window, including conditional paths.
            invalid |= masks[measure]
        arrays[measure + "_in_first_pass"][invalid] = np.nan
    for measure, invalid in masks.items():
        arrays[measure][invalid] = np.nan


def copco_extractor_go_past(events: list[Fixation], n_units: int) -> np.ndarray:
    """Reproduce CopCo's extractor using source event-index adjacency.

    The extractor extends leftward only after the first fixation, and only
    if the unit is revisited. Subsequent on-unit fixations must immediately
    follow the last selected event. This is not the full regression path.

    Args:
        events: Mapped events from one participant, stimulus and source trial.
        n_units: Number of ordered target units.

    Returns:
        Extractor durations, with NaN for units never fixated.
    """
    if (
        len(
            {
                (e.participant_id, e.stimulus_id, e.metadata.get("trial_index"))
                for e in events
            }
        )
        > 1
    ):
        raise ValueError("CopCo's released calculation requires one source trial")
    if any(e.candidate_unit_indices for e in events):
        raise ValueError("CopCo's released calculation requires resolved landings")
    positions = defaultdict(list)
    durations = {}
    for event in sorted(events, key=lambda e: e.fixation_index):
        if event.fixation_index in durations:
            raise ValueError("Duplicate CopCo fixation indices")
        durations[event.fixation_index] = event.duration
        if event.unit_index is not None:
            if not 0 <= event.unit_index < n_units:
                raise ValueError("CopCo fixation is outside its unit layer")
            positions[event.unit_index].append(event.fixation_index)
    values = np.full(n_units, np.nan)
    left = set()
    for unit in sorted(positions):
        first, *returns = positions[unit]
        selected = [first]
        if returns:
            index = first + 1
            while index in left:
                selected.append(index)
                index += 1
        for index in returns:
            if index == selected[-1] + 1:
                selected.append(index)
        values[unit] = sum(durations[index] for index in selected)
        left.update(positions[unit])
    return values


def popeye_episode_ranks(indices: Sequence[int | None]) -> list[int]:
    """Rank each unit's regression episodes in one ordered presentation.

    An episode starts when a backward step follows a non-backward step.
    Off-text events are excluded from the episode sequence, as in popEye's
    in-text event selection. Ranks are rebuilt from the target units.

    Args:
        indices: Ordered target-unit landings; ``None`` denotes off-text.
    """
    episode = 1
    previous = None
    previous_regression = False
    episodes: dict[int, dict[int, int]] = {}
    ranks = []
    for unit in indices:
        if unit is None:
            ranks.append(0)
            continue
        regression = previous is not None and unit < previous
        if regression and not previous_regression:
            episode += 1
        unit_episodes = episodes.setdefault(unit, {})
        ranks.append(unit_episodes.setdefault(episode, len(unit_episodes) + 1))
        previous, previous_regression = unit, regression
    return ranks


def _validate_window_algorithms(
    events, first_run_algorithm, go_past_algorithm, use_source_runs
):
    """Reject unknown algorithms and unsupported combinations before calculation."""
    if first_run_algorithm not in {"standard@1", "popeye_episode@1"}:
        raise ValueError(f"Unknown first-run algorithm: {first_run_algorithm!r}")
    if go_past_algorithm not in {"standard@1", "copco_extractor@1"}:
        raise ValueError(f"Unknown go-past algorithm: {go_past_algorithm!r}")
    if use_source_runs and first_run_algorithm != "standard@1":
        raise ValueError("Episode reconstruction cannot use source run IDs")
    alternative = (first_run_algorithm, go_past_algorithm) != (
        "standard@1",
        "standard@1",
    )
    if alternative and any(e.candidate_unit_indices for e in events):
        raise ValueError("Alternative calculations require resolved target landings")


def _first_run_ids(ordered, indices, use_source_runs, algorithm):
    """Select native run evidence or reconstruct target-unit episode ranks."""
    if algorithm == "popeye_episode@1":
        return popeye_episode_ranks(indices)
    if use_source_runs:
        return [e.metadata.get("run_id") for e in ordered]
    return None


def _visit_go_past(
    ordered,
    indices,
    durations,
    n_units,
    algorithm,
    include_off_text,
    require_right_exit,
):
    """Calculate one presentation's regression paths with the selected window."""
    if algorithm == "copco_extractor@1":
        return copco_extractor_go_past(
            [replace(e, duration=d) for e, d in zip(ordered, durations, strict=True)],
            n_units,
        )
    return go_past_duration(
        indices,
        durations,
        n_units,
        include_off_text=include_off_text,
        require_right_exit=require_right_exit,
    )


def _modelblocks_reading_measures(events, n_units, floor_durations, order):
    """Replay the Dundee extractor's pointers on mapped units and interruptions.

    ModelBlocks process_dundee.py, revision 3ae5120327005cde1c02576eda24876997e20550.
    Interruptions are the intervals that follow physical events.
    First-run and go-past are available only on first-pass arrivals.
    """
    if len({e.metadata.get("trial_index") for e in events}) > 1:
        raise ValueError("ModelBlocks calculations require one presentation")
    if any(e.candidate_unit_indices or e.unit_index is None for e in events):
        raise ValueError("ModelBlocks calculations require resolved on-word landings")
    run = np.zeros(n_units)
    path = np.zeros(n_units)
    total = np.zeros(n_units)
    seen = set()
    frontier = -1
    fp = gp = None
    for event in _sort_fixation_visit(list(events), order):
        unit = event.unit_index
        intervals = event.metadata.get("following_interruptions")
        if intervals is None:
            raise ValueError(
                "ModelBlocks calculations require recorded interruption intervals"
            )
        duration = float(
            np.floor(event.duration) if floor_durations else event.duration
        )
        if unit not in seen:
            run[unit] = path[unit] = np.nan
            seen.add(unit)
        if unit > frontier:
            fp = gp = unit
            run[unit] = path[unit] = 0.0
        elif unit < frontier:
            fp = None
        frontier = max(frontier, unit)
        if fp is not None:
            run[fp] += duration
        if gp is not None:
            path[gp] += duration
        total[unit] += duration
        for interval in intervals:
            elapsed = float(interval["duration"])
            if not np.isfinite(elapsed) or elapsed < 0:
                raise ValueError(
                    "Interruption durations must be finite and nonnegative"
                )
            if floor_durations:
                elapsed = float(np.floor(elapsed))
            if interval["closes_windows"]:
                fp = gp = None
            if fp is not None:
                run[fp] += elapsed
            if gp is not None:
                path[gp] += elapsed
            total[unit] += elapsed
    return {"first_run": run, "go_past": path, "total_time": total}


def _calculation_windows(events, selected, use_source_runs):
    """Validate selections and separate the interval overrides from dwell windows."""
    first_run_algorithm, go_past_algorithm, total_time_algorithm = (
        selected[name] for name in ("first_run", "go_past", "total_time")
    )
    if total_time_algorithm not in {"standard@1", "modelblocks@1"}:
        raise ValueError(f"Unknown total-time algorithm: {total_time_algorithm!r}")
    if "modelblocks@1" in selected.values() and use_source_runs:
        raise ValueError("ModelBlocks reconstruction cannot use source run IDs")
    first_run_algorithm = (
        "standard@1" if first_run_algorithm == "modelblocks@1" else first_run_algorithm
    )
    go_past_algorithm = (
        "standard@1" if go_past_algorithm == "modelblocks@1" else go_past_algorithm
    )
    _validate_window_algorithms(
        events, first_run_algorithm, go_past_algorithm, use_source_runs
    )
    return first_run_algorithm, go_past_algorithm


def _apply_interval_calculations(
    arrays, events, n_units, selected, floor_durations, order
):
    """Replace only measures explicitly selecting interruption attribution."""
    flag = arrays["first_pass"]
    if "modelblocks@1" in selected.values():
        source = _modelblocks_reading_measures(events, n_units, floor_durations, order)
        for measure, algorithm in selected.items():
            if algorithm == "modelblocks@1":
                arrays[measure] = source[measure]
                if measure != "total_time":
                    arrays[measure + "_in_first_pass"] = np.where(
                        flag == 1, source[measure], 0.0
                    )


def reading_measures_from_fixations(
    events: Sequence[Fixation],
    n_units: int,
    *,
    floor_durations: bool = False,
    off_text_ends_run: bool = True,
    go_past_includes_off_text: bool = False,
    go_past_requires_right_exit: bool = False,
    use_source_runs: bool = False,
    first_run_algorithm: str = "standard@1",
    go_past_algorithm: str = "standard@1",
    total_time_algorithm: str = "standard@1",
    allow_missing: bool = False,
    order: Literal["fixation_index", "start_time"] = "fixation_index",
    unfixated_first_fixation: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute reading measures from events assigned to ordered units.

    Visits keep input order and events are sorted within each visit. Total time
    covers all visits; first fixation, first run and go-past start at a unit's
    first visit, and go-past ends at that visit's boundary. Off-text events add
    no word dwell; ``go_past_includes_off_text`` sets their regression-path share.

    Args:
        events: Events from one participant and stimulus, with target unit indices.
        n_units: Number of target units.
        floor_durations: Floor each duration to integer milliseconds before sums.
        off_text_ends_run: End the current run on an off-text event.
        go_past_includes_off_text: Count off-text dwell in regression paths.
        go_past_requires_right_exit: Keep paths without a rightward exit unavailable.
        use_source_runs: Use supplied run IDs when they refer to the target units.
        first_run_algorithm: Versioned first-run window.
        go_past_algorithm: Versioned regression-path window.
        total_time_algorithm: Fixation dwell or ModelBlocks interruption attribution.
        allow_missing: Preserve missing durations as unknown affected measures.
        order: Source field defining event order within a visit.
        unfixated_first_fixation: Value for a unit with no observed fixation.
    """
    selected = {
        "first_run": first_run_algorithm,
        "go_past": go_past_algorithm,
        "total_time": total_time_algorithm,
    }
    first_run_algorithm, go_past_algorithm = _calculation_windows(
        events, selected, use_source_runs
    )
    if order not in {"fixation_index", "start_time"}:
        raise ValueError(f"Unknown fixation order: {order!r}")
    if len({(e.participant_id, e.stimulus_id) for e in events}) > 1:
        raise ValueError("Fixation aggregation requires one participant and stimulus")
    visits: dict[object, list[Fixation]] = {}
    for event in events:
        visits.setdefault(event.metadata.get("trial_index"), []).append(event)
    if None in visits and len(visits) > 1:
        raise ValueError("Cannot mix fixations with and without trial_index")

    first = np.full(n_units, unfixated_first_fixation)
    run = np.zeros(n_units)
    total = np.zeros(n_units)
    path = np.zeros(n_units)
    seen: set[int] = set()
    all_indices: list[int | None] = []
    all_durations: list[float] = []
    ordered_visits = []
    for visit in visits.values():
        ordered = _sort_fixation_visit(visit, order)
        ordered_visits.append(ordered)
        for event in ordered:
            if event.candidate_unit_indices and (
                event.unit_index is not None
                or event.char_offset is not None
                or event.char_offset_is_exact
                or len(set(event.candidate_unit_indices)) < 2
                or any(
                    u is not None and not 0 <= u < n_units
                    for u in event.candidate_unit_indices
                )
            ):
                raise ValueError("Invalid alternative fixation assignments")
        indices = [
            e.candidate_unit_indices[0] if e.candidate_unit_indices else e.unit_index
            for e in ordered
        ]
        if any(u is not None and not 0 <= u < n_units for u in indices):
            raise ValueError("Fixation unit is outside the stimulus")
        durations = [
            np.nan if e.duration is None else float(e.duration) for e in ordered
        ]
        if any(
            np.isinf(d) or d < 0 or (not allow_missing and np.isnan(d))
            for d in durations
        ):
            raise ValueError(
                "Fixation durations must be nonnegative or missing"
                if allow_missing
                else "Complete fixation measures require finite nonnegative durations"
            )
        if floor_durations:
            durations = [float(np.floor(d)) for d in durations]
        new_units = set(u for u in indices if u is not None) - seen
        visit_path = _visit_go_past(
            ordered,
            indices,
            durations,
            n_units,
            go_past_algorithm,
            go_past_includes_off_text,
            go_past_requires_right_exit,
        )
        for unit in new_units:
            path[unit] = visit_path[unit]
        run_ids = _first_run_ids(ordered, indices, use_source_runs, first_run_algorithm)
        visit_run = first_run_durations(
            indices,
            durations,
            {i: i for i in range(n_units)},
            n_units,
            off_text_ends_run=off_text_ends_run,
            run_ids=run_ids,
        )
        for unit in new_units:
            run[unit] = visit_run[unit]
        for unit, duration in zip(indices, durations, strict=True):
            if unit is None:
                continue
            if unit not in seen:
                first[unit] = duration
                seen.add(unit)
            total[unit] += duration
        all_indices.extend(indices)
        all_durations.extend(durations)
    flag = first_pass_flags(all_indices, n_units)
    arrays = {
        "first_fixation": first,
        "first_run": run,
        "first_pass": flag,
        "total_time": total,
        "go_past": path,
        "right_bounded": right_bounded_durations(all_indices, all_durations, n_units),
        "first_fixation_in_first_pass": np.where(flag == 1, first, 0.0),
        "first_run_in_first_pass": np.where(flag == 1, run, 0.0),
        "go_past_in_first_pass": np.where(flag == 1, path, 0.0),
    }
    _mask_uncertain_landings(arrays, ordered_visits, off_text_ends_run)
    _apply_interval_calculations(
        arrays, events, n_units, selected, floor_durations, order
    )
    return arrays


def replace_with_fixation_measures(
    data: list[Datum],
    fixations: Sequence[Fixation],
    *,
    floor_durations: bool = False,
    off_text_ends_run: bool = True,
) -> None:
    """Replace summary measures in place, retaining their released values.

    Every participant/stimulus pair must have exactly one complete source trial.
    Missing trials are errors rather than observations of zero reading time.

    Args:
        data: Summary datums to update, preserving their previous values in metadata.
        fixations: Complete source events for the same participant/stimulus pairs.
        floor_durations: Convert each duration to integer milliseconds.
        off_text_ends_run: End the current run on an off-word event.
    """
    groups: dict[tuple[str, str], list[Fixation]] = defaultdict(list)
    for event in fixations:
        groups[(event.participant_id, event.stimulus_id)].append(event)
    expected = {(d.participant_id, d.stimulus.stimulus_id) for d in data}
    if expected != groups.keys():
        raise ValueError(
            "Fixation trials do not match the released participant/stimulus pairs"
        )
    arrays = {}
    for datum in data:
        key = (datum.participant_id, datum.stimulus.stimulus_id)
        if key not in arrays:
            events = groups[key]
            if len({e.metadata.get("trial_index") for e in events}) != 1:
                raise ValueError(f"Multiple fixation trials for {key}")
            arrays[key] = reading_measures_from_fixations(
                events,
                len(datum.values),
                floor_durations=floor_durations,
                off_text_ends_run=off_text_ends_run,
            )
        datum.metadata = {
            **datum.metadata,
            "measurement_source": "fixations",
            "duration_rounding": "floor_ms" if floor_durations else "none",
            "off_text_ends_run": off_text_ends_run,
            "released_values": datum.values.tolist(),
            "released_metadata": dict(datum.metadata),
        }
        datum.metadata.pop("source_measure", None)
        datum.metadata.pop("unavailable_outside_first_pass", None)
        datum.values = arrays[key][datum.measure_name]
