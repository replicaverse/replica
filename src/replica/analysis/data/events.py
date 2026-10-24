"""Join predictor arrays and response rows onto dataset event timelines."""

import numpy as np

from replica.analysis.contracts.events import (
    EventCovariateDataset,
    EventIndex,
    EventStream,
    ResponseEvents,
    TimedDataset,
)

__all__ = [
    "EventCovariateDataset",
    "EventIndex",
    "EventStream",
    "ResponseEvents",
    "TimedDataset",
    "build_response_events",
    "join_feature_values",
    "series_codes",
]


def series_codes(
    participant_id: np.ndarray, stimulus_id: np.ndarray
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """Dense codes for (participant, stimulus) series, in first-appearance order.

    Args:
        participant_id: Participant label per event, in event order.
        stimulus_id: Stimulus label per event, aligned with *participant_id*.

    Returns:
        The per-event integer codes, and the ``(participant, stimulus)`` to
        code table that produced them.
    """
    codes = np.empty(len(participant_id), dtype=np.int64)
    table: dict[tuple[str, str], int] = {}
    for i, key in enumerate(zip(participant_id, stimulus_id, strict=True)):
        key = (str(key[0]), str(key[1]))
        if key not in table:
            table[key] = len(table)
        codes[i] = table[key]
    return codes, table


def join_feature_values(
    index: EventIndex,
    feature_arrays: dict[str, dict[str, np.ndarray]],
    names: list[str],
) -> dict[str, np.ndarray]:
    """Per-event predictor columns, read from word-level feature arrays.

    Values are looked up by stimulus and unit. Non-finite values become zero,
    as the reference's impulse transform does; an event with no entry raises,
    since a zero impulse there would read as a weaker effect.

    Args:
        index: Event rows to fill, supplying the stimulus and unit to look up.
        feature_arrays: Feature name to stimulus id to that stimulus's
            per-word values, the same table the row-aligned pipeline reads.
        names: Features to turn into impulse columns.

    Returns:
        One column per name, each with one entry per event.

    Raises:
        ValueError: When any event has no value in a requested feature.
    """
    out: dict[str, np.ndarray] = {}
    for name in names:
        per_stimulus = feature_arrays.get(name, {})
        column = np.zeros(len(index), dtype=np.float64)
        missed = 0
        for i, (sid, unit) in enumerate(
            zip(index.stimulus_id, index.unit_index, strict=True)
        ):
            values = per_stimulus.get(str(sid))
            unit = int(unit)
            if values is None or not 0 <= unit < len(values):
                missed += 1
                continue
            value = values[unit]
            column[i] = 0.0 if not np.isfinite(value) else float(value)
        if missed:
            raise ValueError(
                f"feature {name!r} has no value for {missed} of {len(index)} "
                "events; the impulse stream and the feature arrays disagree"
            )
        out[name] = column
    return out


def build_response_events(
    *,
    dataset,
    dv_name: str,
    rows: dict,
    feature_arrays: dict[str, dict[str, np.ndarray]],
    control_names: tuple[str, ...],
    focal_names: tuple[str, ...],
    y: np.ndarray | None = None,
    group: np.ndarray | None = None,
    layer: str = "words",
) -> ResponseEvents:
    """Place the analyzed rows on the dataset's event timeline.

    Each response takes the timestamp of the single event matching its
    participant, stimulus and unit, the word it responds to.

    Args:
        dataset: Corpus under analysis, which has to implement
            :class:`TimedDataset` to supply the impulse index and any impulse
            it owns itself.
        dv_name: Reading measure being analyzed, passed on to
            :meth:`TimedDataset.event_index`.
        rows: ``participant_id``, ``stimulus_id`` and ``unit_index`` arrays for
            the analyzed rows, in flat-array order.
        feature_arrays: Feature name to stimulus id to per-word values, the
            columns the row-aligned pipeline uses, joined onto the events by
            stimulus and unit.
        control_names: Baseline impulses, which any dataset-owned covariate
            joins.
        focal_names: Impulses an ablation removes to form the baseline model.
        y: Dependent variable per analyzed row, kept so a model can build a
            design for rows outside the fold's train and test split.
        group: Grouping code per analyzed row, usually the participant, kept
            for the same reason as *y*.
        layer: Stimulus annotation layer the rows' ``unit_index`` counts in,
            passed on to :meth:`TimedDataset.event_index`.

    Returns:
        The response side, row-aligned to the flat arrays, carrying the shared
        impulse table.

    Raises:
        TypeError: When *dataset* carries no event timeline.
        ValueError: When the dataset supplies an unrequested impulse, its
            index counts units in another layer, or an analyzed row has no
            event or more than one event in the timeline.
    """
    if not isinstance(dataset, TimedDataset):
        raise TypeError(
            f"{type(dataset).__name__} carries no event timeline: a "
            "continuous-time analysis needs a dataset implementing "
            "event_index(measure)"
        )
    index = dataset.event_index(dv_name, layer=layer)
    if index.layer != layer:
        raise ValueError(
            f"{type(dataset).__name__}.event_index counts units in layer "
            f"{index.layer!r}, but the analyzed rows count them in {layer!r}"
        )
    owned: dict[str, np.ndarray] = {}
    if isinstance(dataset, EventCovariateDataset):
        control_names = tuple(
            dict.fromkeys(control_names + tuple(dataset.event_impulses()))
        )
    names = list(dict.fromkeys(control_names + focal_names))
    if isinstance(dataset, EventCovariateDataset):
        owned = dataset.event_values(names, index)
        unknown = set(owned) - set(names)
        if unknown:
            raise ValueError(
                f"{type(dataset).__name__} supplied impulses this analysis did "
                f"not ask for: {sorted(unknown)}"
            )
    joined = join_feature_values(
        index, feature_arrays, [n for n in names if n not in owned]
    )
    impulses = EventStream(index=index, values={**joined, **owned})

    lookup: dict[tuple[str, str, int], int] = {}
    repeated: set[tuple[str, str, int]] = set()
    for i, (p, s, u) in enumerate(
        zip(index.participant_id, index.stimulus_id, index.unit_index, strict=True)
    ):
        key = (str(p), str(s), int(u))
        if key in lookup:
            repeated.add(key)
        lookup[key] = i
    participant = rows["participant_id"]
    stimulus = rows["stimulus_id"]
    unit = rows["unit_index"]
    time = np.zeros(len(unit), dtype=float)
    series = np.zeros(len(unit), dtype=np.int64)
    event_rows = np.empty(len(unit), dtype=np.int64)
    missing = ambiguous = 0
    for i in range(len(unit)):
        key = (str(participant[i]), str(stimulus[i]), int(unit[i]))
        event_row = lookup.get(key)
        if event_row is None:
            missing += 1
            continue
        if key in repeated:
            ambiguous += 1
            continue
        time[i] = index.time[event_row]
        series[i] = index.series[event_row]
        event_rows[i] = event_row
    if missing:
        raise ValueError(
            f"{missing} of {len(unit)} analyzed rows have no event in the "
            "dataset's timeline; the event index must cover every observation"
        )
    if ambiguous:
        raise ValueError(
            f"{ambiguous} of {len(unit)} analyzed rows match more than one event "
            "with the same participant, stimulus and unit, so their timestamps "
            "are ambiguous; give each presentation its own unit index"
        )
    item_codes: dict[object, int] = {}
    item = np.empty(len(unit), dtype=np.int64)
    for i in range(len(unit)):
        key = (
            index.item_id[event_rows[i]]
            if index.item_id is not None
            else (str(stimulus[i]), int(unit[i]))
        )
        if key not in item_codes:
            item_codes[key] = len(item_codes)
        item[i] = item_codes[key]
    return ResponseEvents(
        time=time,
        series=series,
        impulses=impulses,
        control_names=tuple(control_names),
        focal_names=tuple(focal_names),
        item=item,
        y=None if y is None else np.asarray(y, dtype=float),
        group=None if group is None else np.asarray(group, dtype=np.int64),
    )
