"""Map fixation events to text units and recompute reading measures."""

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from replica.core import AnnotationLayer, Span, Stimulus
from replica.segmentation import _validate_spans

if TYPE_CHECKING:
    from replica.dataset.contracts import FixationCalculation


def _source_unit_mappings(events, stimulus, layer_name, starts):
    """Validate the resolution of source areas and map them to target spans."""
    layer = stimulus.layers[layer_name]
    source_mappings: dict[str, dict[int, int | None]] = {}
    for event in events:
        if (event.char_offset_is_exact and not event.candidate_unit_indices) or (
            event.unit_index is None
            and event.char_offset is None
            and not event.candidate_unit_indices
        ):
            continue
        source_name = event.unit_layer
        if source_name in source_mappings:
            continue
        source = stimulus.layers[source_name]
        _validate_spans(source.spans, len(stimulus.text))
        mapping: dict[int, int | None] = {}
        for index, span in enumerate(source.spans):
            candidate = int(np.searchsorted(starts, span.start, side="right")) - 1
            if candidate >= 0 and layer.spans[candidate].contains(span):
                mapping[index] = candidate
            elif any(target.overlaps(span) for target in layer.spans):
                raise ValueError(
                    f"Target layer {layer_name!r} splits source area {source_name}[{index}]; "
                    "character-level landing evidence is required"
                )
            elif span.extract(stimulus.text).isspace():
                mapping[index] = None
            else:
                raise ValueError("Target units must cover every source area")
        source_mappings[source_name] = mapping
    return source_mappings


def _map_candidate_units(event, source_mappings):
    """Resolve alternative source positions when they share a target unit."""
    if (
        event.unit_index is not None
        or event.char_offset is not None
        or event.char_offset_is_exact
        or len(set(event.candidate_unit_indices)) < 2
    ):
        raise ValueError("Invalid alternative fixation assignments")
    mapping = source_mappings[event.unit_layer]
    if any(i is not None and i not in mapping for i in event.candidate_unit_indices):
        raise ValueError("Candidate fixation area is outside its source layer")
    candidates = tuple(
        dict.fromkeys(
            mapping[i] if i is not None else None for i in event.candidate_unit_indices
        )
    )
    return (candidates[0], ()) if len(candidates) == 1 else (None, candidates)


def map_fixations_to_layer(
    fixations,
    stimulus: Stimulus,
    layer_name: str = "words",
    *,
    use_source_runs: bool = True,
):
    """Map source landings to target spans without changing event order or duration.

    Args:
        fixations: Events belonging to the stimulus.
        stimulus: Text with the source and target annotation layers.
        layer_name: Target layer; its spans must cover complete source areas
            unless exact character landings permit finer boundaries.
        use_source_runs: Retain run IDs only when source and target spans match.

    Returns:
        Copied events with mapped assignments and unresolved alternatives.
    """
    layer = stimulus.layers[layer_name]
    _validate_spans(layer.spans, len(stimulus.text))
    events = list(fixations)
    if any(event.stimulus_id != stimulus.stimulus_id for event in events):
        raise ValueError("Fixation aggregation requires one participant and stimulus")
    starts = np.array([s.start for s in layer.spans])
    source_mappings = _source_unit_mappings(events, stimulus, layer_name, starts)
    matching_layers = {
        name: stimulus.layers[name].spans == layer.spans
        for name in {event.unit_layer for event in events}
        if name in stimulus.layers
    }
    mapped = []
    for event in events:
        unit = None
        candidates = event.candidate_unit_indices
        if candidates:
            unit, candidates = _map_candidate_units(event, source_mappings)
        elif event.unit_index is not None or event.char_offset is not None:
            if event.char_offset is not None and not (
                0 <= event.char_offset < len(stimulus.text)
            ):
                raise ValueError("Fixation character offset is outside the stimulus")
            if event.char_offset_is_exact:
                if event.char_offset is None:
                    raise ValueError("An exact character landing requires char_offset")
                if event.unit_index is not None:
                    source = stimulus.layers[event.unit_layer]
                    if not 0 <= event.unit_index < len(source) or not (
                        source.spans[event.unit_index].start
                        <= event.char_offset
                        < source.spans[event.unit_index].end
                    ):
                        raise ValueError(
                            "Fixation offset disagrees with its source area"
                        )
                candidate = (
                    int(np.searchsorted(starts, event.char_offset, side="right")) - 1
                )
                if candidate >= 0 and event.char_offset < layer.spans[candidate].end:
                    unit = candidate
                elif not stimulus.text[event.char_offset].isspace():
                    raise ValueError(
                        "Target units do not cover an observed character landing"
                    )
            else:
                source = stimulus.layers[event.unit_layer]
                if event.unit_index is None or not 0 <= event.unit_index < len(source):
                    raise ValueError(
                        "Fixation area anchor requires a valid source unit_index"
                    )
                span = source.spans[event.unit_index]
                if event.char_offset is not None and not (
                    span.start <= event.char_offset < span.end
                ):
                    raise ValueError("Fixation offset disagrees with its source area")
                unit = source_mappings[event.unit_layer][event.unit_index]
        same_spans = matching_layers.get(event.unit_layer, False)
        metadata = event.metadata
        if not use_source_runs or not same_spans or event.unit_index != unit:
            metadata = {k: v for k, v in metadata.items() if k != "run_id"}
        mapped.append(
            replace(
                event,
                unit_index=unit,
                unit_layer=layer_name,
                metadata=metadata,
                candidate_unit_indices=candidates,
            )
        )
    return mapped


def aggregate_fixations_to_layer(
    fixations,
    stimulus: Stimulus,
    layer_name: str = "words",
    *,
    floor_durations: bool = False,
    off_text_ends_run: bool = True,
    go_past_includes_off_text: bool = False,
    go_past_requires_right_exit: bool = False,
    include_inter_unit_fixations: bool = False,
    use_source_runs: bool = True,
    calculation: "FixationCalculation | None" = None,
) -> dict[str, np.ndarray]:
    """Map fixation landings to spans and compute reading measures.

    A target boundary may split an event's source area (``unit_layer``,
    ``unit_index``) only when ``char_offset_is_exact`` marks a character
    landing; target units must cover the source text areas, and whitespace may
    stay outside them. Measures come from ``reading_measures_from_fixations``.

    Args:
        fixations: Events from one participant and stimulus.
        stimulus: Shared text carrying source and target annotation layers.
        layer_name: Target annotation layer, default ``words``.
        floor_durations: Floor event durations before calculating measures.
        off_text_ends_run: Whether off-text events close the current run, the
            source policy for off-text run boundaries.
        go_past_includes_off_text: Count off-text dwell in regression paths.
        go_past_requires_right_exit: Keep paths without a rightward exit unavailable.
        include_inter_unit_fixations: Count known whitespace landings between units
            in reading windows, without assigning their dwell to a neighboring unit.
        use_source_runs: Honor source run IDs at identical native spans; other
            segmentations recompute runs from the event sequence.
        calculation: Explicit algorithms and policies, overriding the source
            options above and rebuilding all runs from target landings.

    Returns:
        Arrays for first fixation, first run, first pass, total time,
        go-past and right-bounded time, plus first-pass restrictions, computed
        from the events with no duration summary summed. Never-fixated units
        have NaN first fixation and zero other durations. Missing event
        durations leave affected measurements unknown.

    Raises:
        ValueError: Invalid spans, mixed identities, or insufficient landing precision.
        KeyError: A source or target layer is absent.
    """
    from replica.dataset.loading.fixation_utils import reading_measures_from_fixations

    if calculation is not None:
        floor_durations = calculation.floor_durations
        off_text_ends_run = calculation.off_text_ends_run
        go_past_includes_off_text = calculation.go_past_includes_off_text
        go_past_requires_right_exit = calculation.go_past_requires_right_exit
        include_inter_unit_fixations = calculation.include_inter_unit_fixations
        use_source_runs = False
    if include_inter_unit_fixations:
        fixations = list(fixations)
        include_inter_unit_fixations = any(
            span.extract(stimulus.text).isspace()
            for name in {
                event.unit_layer
                for event in fixations
                if event.unit_layer in stimulus.layers
            }
            for span in stimulus.layers[name].spans
        ) or any(
            event.char_offset_is_exact
            and event.char_offset is not None
            and 0 <= event.char_offset < len(stimulus.text)
            and stimulus.text[event.char_offset].isspace()
            for event in fixations
        )
    if include_inter_unit_fixations:
        target = stimulus.layers[layer_name]
        spans = []
        retained = []
        end = 0
        for span in target.spans:
            if end < span.start and stimulus.text[end : span.start].isspace():
                spans.append(Span(end, span.start))
            retained.append(len(spans))
            spans.append(span)
            end = span.end
        if end < len(stimulus.text) and stimulus.text[end:].isspace():
            spans.append(Span(end, len(stimulus.text)))
        if len(spans) != len(target):
            name = "_fixation_regions"
            while name in stimulus.layers:
                name = "_" + name
            expanded = replace(stimulus, layers=dict(stimulus.layers))
            expanded.add_layer(
                AnnotationLayer(
                    name, spans, [span.extract(stimulus.text) for span in spans]
                )
            )
            arrays = aggregate_fixations_to_layer(
                fixations,
                expanded,
                name,
                floor_durations=floor_durations,
                off_text_ends_run=off_text_ends_run,
                go_past_includes_off_text=go_past_includes_off_text,
                go_past_requires_right_exit=go_past_requires_right_exit,
                use_source_runs=use_source_runs,
                calculation=calculation,
            )
            return {measure: values[retained] for measure, values in arrays.items()}
    mapped = map_fixations_to_layer(
        fixations, stimulus, layer_name, use_source_runs=use_source_runs
    )
    return reading_measures_from_fixations(
        mapped,
        len(stimulus.layers[layer_name]),
        floor_durations=floor_durations,
        off_text_ends_run=off_text_ends_run,
        go_past_includes_off_text=go_past_includes_off_text,
        go_past_requires_right_exit=go_past_requires_right_exit,
        use_source_runs=use_source_runs,
        first_run_algorithm=calculation.first_run if calculation else "standard@1",
        go_past_algorithm=calculation.go_past if calculation else "standard@1",
        total_time_algorithm=calculation.total_time if calculation else "standard@1",
        allow_missing=True,
        order="start_time",
        unfixated_first_fixation=np.nan,
    )
