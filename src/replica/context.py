"""Preceding stimulus context with annotation layers aligned by character offsets.

Loaders declare ``context_sequence`` metadata with a group identity and a
one-based position. Context must be assembled before filtering or sharding.
Stimuli are joined with one space; their annotation spans retain source units.
"""

import hashlib
import json
from collections import defaultdict
from copy import deepcopy

import numpy as np

from .core import AnnotationLayer, Span, Stimulus

SENTENCE_LAYER = "sentences"
CONTEXT_SEQUENCE = "context_sequence"
PRECEDING_CONTEXT = "preceding_context"


def attach_preceding_context(stimuli: list[Stimulus]) -> None:
    """Attach preceding text and layers using loader-declared sequence membership.

    Args:
        stimuli: Complete sequences in any order. Each stimulus must declare
            ``context_sequence={"group": str, "position": int}`` in metadata.

    Raises:
        ValueError: Membership is missing, positions or IDs are duplicated,
            preceding positions are absent, or a stimulus has empty text.
    """
    groups: dict[str, dict[int, Stimulus]] = defaultdict(dict)
    ids: set[str] = set()
    for stimulus in stimuli:
        membership = stimulus.metadata.get(CONTEXT_SEQUENCE, {})
        group = membership.get("group")
        position = membership.get("position")
        if not isinstance(group, str) or not group or type(position) is not int:
            raise ValueError(
                f"Missing context sequence membership for {stimulus.stimulus_id}"
            )
        if position < 1:
            raise ValueError(f"Invalid context position for {stimulus.stimulus_id}")
        if stimulus.stimulus_id in ids:
            raise ValueError(f"Duplicate stimulus id: {stimulus.stimulus_id}")
        ids.add(stimulus.stimulus_id)
        if position in groups[group]:
            raise ValueError(f"Duplicate context position {position} in {group!r}")
        groups[group][position] = stimulus
        if not stimulus.text.strip():
            raise ValueError(f"Empty context stimulus: {stimulus.stimulus_id}")
    prepared: list[tuple[Stimulus, dict]] = []
    for group, positions in groups.items():
        if sorted(positions) != list(range(1, max(positions) + 1)):
            raise ValueError(f"Missing preceding context in {group!r}")
        text = ""
        layers: dict[str, dict] | None = None
        preceding_ids: list[str] = []
        for _, stimulus in sorted(positions.items()):
            prepared.append(
                (
                    stimulus,
                    {
                        "group": group,
                        "text": text,
                        "layers": deepcopy(layers or {}),
                        "preceding_stimulus_ids": list(preceding_ids),
                    },
                )
            )
            offset = len(text) + bool(text)
            if layers is None:
                layers = {name: {"spans": [], "values": []} for name in stimulus.layers}
            # A layer is available as context only if every preceding stimulus has it.
            layers = {
                name: layer for name, layer in layers.items() if name in stimulus.layers
            }
            for name, layer in layers.items():
                original = stimulus.layers[name]
                layer["spans"].extend(
                    (s.start + offset, s.end + offset) for s in original.spans
                )
                layer["values"].extend(deepcopy(original.values))
            text += (" " if text else "") + stimulus.text
            preceding_ids.append(stimulus.stimulus_id)
    for stimulus, context in prepared:
        stimulus.metadata[PRECEDING_CONTEXT] = context


def context_input(
    stimulus: Stimulus, target_layer: str = "words"
) -> tuple[Stimulus, int]:
    """Combine preceding text with the target stimulus and return its unit offset.

    Args:
        stimulus: Target carrying metadata from :func:`attach_preceding_context`.
        target_layer: Annotation layer to align outputs to. It must exist on
            the target and every preceding stimulus.

    Returns:
        Combined stimulus and the index of its first target unit.
    """
    context = stimulus.metadata.get(PRECEDING_CONTEXT)
    if context is None:
        raise ValueError(f"Preceding context missing for {stimulus.stimulus_id}")
    if target_layer not in stimulus.layers:
        raise ValueError(
            f"Missing target layer {target_layer!r} on {stimulus.stimulus_id}"
        )
    prefix = context["text"]
    prefix_layers = context["layers"]
    if prefix and target_layer not in prefix_layers:
        raise ValueError(
            f"Missing preceding layer {target_layer!r} for {stimulus.stimulus_id}"
        )
    offset = len(prefix) + bool(prefix)
    combined = Stimulus(
        text=prefix + (" " if prefix else "") + stimulus.text,
        stimulus_id=f"{stimulus.stimulus_id}#context",
        metadata={CONTEXT_SEQUENCE: deepcopy(stimulus.metadata[CONTEXT_SEQUENCE])},
    )
    for name, layer in stimulus.layers.items():
        if prefix and name not in prefix_layers:
            continue
        previous = prefix_layers.get(name, {"spans": [], "values": []})
        combined.add_layer(
            AnnotationLayer(
                name=name,
                spans=[Span(start, end) for start, end in previous["spans"]]
                + [Span(s.start + offset, s.end + offset) for s in layer.spans],
                values=deepcopy(previous["values"]) + deepcopy(layer.values),
            )
        )
    return combined, len(prefix_layers.get(target_layer, {}).get("spans", []))


def _annotation_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Cannot fingerprint annotation value of type {type(value).__name__}"
    )


def corpus_digest(stimuli: list[Stimulus], unit: str, context_scope: str) -> str:
    """SHA-256 of the ordered corpus units a cell is computed on.

    The ``tokens`` layer comes from the tokenizer, part of the model identity,
    so only the corpus's own layers enter the digest: ``words``, the unit layer
    when it is another, the sentence layer under sentence scope, and the
    preceding context under story scope.

    Args:
        stimuli: Ordered stimuli after segmentation and sharding.
        unit: Unit the cell is scored at.
        context_scope: ``stimulus``, ``sentence`` or ``story``.
    """
    corpus = hashlib.sha256()
    ids = set()
    input_layers = {"words", unit} - {"tokens"}
    if context_scope == "sentence":
        input_layers.add(SENTENCE_LAYER)
    for stimulus in stimuli:
        if stimulus.stimulus_id in ids:
            raise ValueError(f"Duplicate stimulus id: {stimulus.stimulus_id}")
        ids.add(stimulus.stimulus_id)
        record = [
            stimulus.stimulus_id,
            stimulus.text,
            {
                name: {
                    "spans": [
                        (span.start, span.end) for span in stimulus.layers[name].spans
                    ],
                    "values": stimulus.layers[name].values,
                }
                for name in sorted(input_layers)
            },
        ]
        if context_scope == "story":
            record.append(stimulus.metadata[PRECEDING_CONTEXT])
        corpus.update(
            json.dumps(record, ensure_ascii=False, default=_annotation_value).encode()
        )
        corpus.update(b"\n")
    return corpus.hexdigest()
