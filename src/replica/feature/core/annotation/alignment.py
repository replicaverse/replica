"""Aligns tokens between annotation layers for mapping and aggregation.

:func:`align_by_spans` assigns each token to the unit containing its midpoint;
:func:`nested_members` and :func:`aggregate_nested` are the guarded form that
only accepts tokens lying inside one unit and raises on every other
configuration.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.core.protocols import FeatureModelOutput
from replica.utils.arrays import real_array_with_missing

#: Metadata key of a stimulus cut from a longer text: the character offset of
#: its text within the full stimulus, so per-token spans can be mapped back.
TEXT_OFFSET = "text_offset"


class TokenBoundaryError(ValueError):
    """A token crosses a unit boundary or lies outside every unit, or a unit has no token."""


@dataclass
class TokenAlignment:
    """Maps source layer indices to target layer index lists.

    Primary use case: ``mapping[word_i]`` = list of subword token indices
    that compose word_i.
    """

    source_layer: str
    target_layer: str
    mapping: list[list[int]]

    def aggregate(self, token_values: np.ndarray, method: str = "sum") -> np.ndarray:
        """Aggregate per-token values to source (word) level.

        Args:
            token_values: Array of per-token values (length = target layer size).
            method: Aggregation method, "sum", "mean", "first", "last".

        Returns:
            Array of per-word values (length = number of source spans).
        """
        result = np.full(len(self.mapping), np.nan)
        for i, token_indices in enumerate(self.mapping):
            if not token_indices:
                continue
            vals = token_values[token_indices]
            if np.all(np.isnan(vals)):
                continue
            if method == "sum":
                result[i] = np.nansum(vals)
            elif method == "mean":
                result[i] = np.nanmean(vals)
            elif method == "first":
                result[i] = vals[0]
            elif method == "last":
                result[i] = vals[-1]
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        return result


def align_by_spans(stimulus: Stimulus, source: str, target: str) -> TokenAlignment:
    """Build alignment by matching spans via midpoint containment.

    A target span is assigned to the source span whose interval contains
    the target span's midpoint. Target spans whose midpoint falls outside
    all source spans are dropped.

    Args:
        stimulus: Stimulus carrying both layers.
        source: Name of the coarser layer, one mapping row per span.
        target: Name of the finer layer, whose indices fill those rows.
    """
    source_layer = stimulus.layers[source]
    target_layer = stimulus.layers[target]

    mapping: list[list[int]] = [[] for _ in range(len(source_layer))]

    for t_idx, (t_span, _) in enumerate(target_layer):
        mid = t_span.midpoint
        for s_idx, (s_span, _) in enumerate(source_layer):
            if s_span.start <= mid < s_span.end:
                mapping[s_idx].append(t_idx)
                break

    return TokenAlignment(source_layer=source, target_layer=target, mapping=mapping)


def align_from_tokenizer(
    stimulus: Stimulus,
    tokenizer,
    layer_name: str | None = None,
    source_layer: str = "words",
) -> TokenAlignment:
    """Tokenize text with a HuggingFace tokenizer, create a token layer,
    and build the source→token alignment.

    Args:
        stimulus: Must already have the ``source_layer`` annotation layer.
        tokenizer: A HuggingFace tokenizer with ``return_offsets_mapping=True``.
        layer_name: Name for the new token layer. Defaults to the tokenizer's
            ``name_or_path`` attribute, or "tokens" if unavailable.
        source_layer: Annotation layer to align tokens against (default "words").

    Returns:
        TokenAlignment from ``source_layer`` to the new token layer.
    """
    if layer_name is None:
        layer_name = getattr(tokenizer, "name_or_path", "tokens") or "tokens"

    token_values, token_spans = tokenize_spans(stimulus.text, tokenizer)

    token_layer = AnnotationLayer(
        name=layer_name,
        spans=token_spans,
        values=token_values,
    )
    stimulus.add_layer(token_layer)

    return align_by_spans(stimulus, source_layer, layer_name)


def aligned_token_ids(
    stimulus: Stimulus,
    tokenizer,
    target_layer: str = "words",
    add_bos: bool = True,
    start_ids: Sequence[int] | None = None,
) -> tuple[list[int], TokenAlignment, int, int]:
    """Token-align ``target_layer`` and prepend BOS when requested.

    Args:
        stimulus: Stimulus to tokenize; it gains the token layer in place.
        tokenizer: Hugging Face tokenizer supporting ``return_offsets_mapping``.
        target_layer: Annotation layer whose units the token indices group under.
        add_bos: Prepend a start prefix so the first word has a defined
            context.
        start_ids: The prefix to prepend with *add_bos*; ``None`` prepends the
            tokenizer's ``bos_token_id`` when it has one, the generic
            convention, and a caller that resolves another prefix (see
            :mod:`replica.feature.backends.start_state`) passes it here.

    Returns:
        ``(token_ids, alignment, bos_offset, n_units)``. ``token_ids``
        includes the BOS, so the raw token stream is empty exactly when
        ``len(token_ids) == bos_offset``; ``alignment.mapping`` positions are
        un-BOS-prefixed and must be offset by ``bos_offset``.
    """
    alignment = align_from_tokenizer(stimulus, tokenizer, source_layer=target_layer)
    token_ids = [v for _, v in stimulus.layers[alignment.target_layer]]
    n_units = len(stimulus.layers[target_layer])
    prefix: list[int] = []
    if add_bos:
        if start_ids is not None:
            prefix = list(start_ids)
        elif getattr(tokenizer, "bos_token_id", None) is not None:
            prefix = [tokenizer.bos_token_id]
    token_ids = [*prefix, *token_ids]
    bos_offset = len(prefix)
    return token_ids, alignment, bos_offset, n_units


def tokenize_spans(text: str, tokenizer) -> tuple[list[int], list[Span]]:
    """Token ids and character spans of *text* under a Hugging Face tokenizer.

    No special tokens are added, and a token whose offsets are empty is left
    out. The spans are the tokenizer's own offsets: a byte-level tokenizer
    keeps a leading space inside the token's span and gives every piece of a
    byte-split character the same span.

    Args:
        text: Text to tokenize.
        tokenizer: Tokenizer supporting ``return_offsets_mapping``.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids: list[int] = []
    spans: list[Span] = []
    for token_id, (start, end) in zip(
        encoding["input_ids"], encoding["offset_mapping"], strict=True
    ):
        if start == end:
            continue
        ids.append(token_id)
        spans.append(Span(start, end))
    return ids, spans


def content_span(text: str, span: Span) -> Span | None:
    """The part of *span* that is not whitespace, or ``None`` for a blank span.

    Args:
        text: Text the span indexes.
        span: Character span of one token.
    """
    piece = text[span.start : span.end]
    stripped = piece.strip()
    if not stripped:
        return None
    start = span.start + len(piece) - len(piece.lstrip())
    return Span(start, start + len(stripped))


def nested_members(
    stimulus: Stimulus,
    layer: str,
    spans: Sequence[Span],
    labels: Sequence[str] | None = None,
) -> list[list[int]]:
    """Group token indices by the unit of *layer* that contains each token.

    A blank record belongs to the unit containing its whole span, any other
    record to the unit containing its non-blank content, since a byte-level
    token carries the space before a word inside its own span. Blanks outside
    every unit span are left out; nothing is split or interpolated.

    Args:
        stimulus: Text carrying the target layer.
        layer: Name of the target layer, such as ``words``.
        spans: Character span of each token in the stimulus text.
        labels: Token strings for error messages; the span text otherwise.

    Returns:
        The token indices of every unit, in layer order and text order.

    Raises:
        TokenBoundaryError: For a record that crosses a unit boundary, a
            non-blank record outside every unit, or a unit without a record,
            naming the stimulus, the record and the units concerned.
        ValueError: When the layer's units overlap or are out of order.
    """
    text = stimulus.text
    units = stimulus.layers[layer].spans
    if any(
        unit.start < previous.end
        for previous, unit in zip(units, units[1:], strict=False)
    ):
        raise ValueError(
            f"{stimulus.stimulus_id}: {layer} units overlap or are out of order"
        )
    starts = [unit.start for unit in units]
    ends = [unit.end for unit in units]

    def describe(index: int) -> str:
        unit = units[index]
        return f"unit {index} {text[unit.start : unit.end]!r} at [{unit.start}, {unit.end})"

    members: list[list[int]] = [[] for _ in units]
    for index, span in enumerate(spans):
        content = content_span(text, span)
        placed = span if content is None else content
        label = labels[index] if labels is not None else text[span.start : span.end]
        token = f"token {index} {label!r} at [{span.start}, {span.end})"
        first = bisect_right(ends, placed.start)
        last = bisect_left(starts, placed.end) - 1
        if first > last:
            if content is None:
                continue
            raise TokenBoundaryError(
                f"{stimulus.stimulus_id}: {token} lies outside every {layer} unit"
            )
        if first == last and units[first].contains(placed):
            members[first].append(index)
            continue
        touched = " and ".join(describe(j) for j in range(first, last + 1))
        raise TokenBoundaryError(
            f"{stimulus.stimulus_id}: {token} crosses a {layer} boundary; "
            f"it touches {touched}"
        )
    for index, indices in enumerate(members):
        if not indices:
            raise TokenBoundaryError(
                f"{stimulus.stimulus_id}: {layer} {describe(index)} has no token"
            )
    return members


def aggregate_nested(
    stimulus: Stimulus,
    layer: str,
    spans: Sequence[Span],
    values: np.ndarray,
    rule: str,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Aggregate per-token values to the units of one layer, refusing any split.

    Tokens are grouped by :func:`nested_members`. ``sum`` adds a unit's values
    in text order and ``first`` takes its first value; a NaN reaches its unit
    unchanged, so a unit whose first token is scored without a
    beginning-of-sequence token comes out NaN.

    Args:
        stimulus: Text carrying the target layer.
        layer: Name of the target layer, such as ``words``.
        spans: Character span of each token in the stimulus text.
        values: One real value per token; masked values are missing.
        rule: ``sum`` or ``first``.
        labels: Token strings for error messages; the span text otherwise.

    Returns:
        One value per unit of the layer, in layer order.
    """
    if rule not in ("sum", "first"):
        raise ValueError(f"unknown aggregation rule {rule!r}")
    values = np.asarray(real_array_with_missing(values), dtype=float)
    if len(spans) != len(values):
        raise ValueError(f"{len(spans)} token spans for {len(values)} values")
    members = nested_members(stimulus, layer, spans, labels)
    result = np.full(len(members), np.nan)
    for index, indices in enumerate(members):
        picked = values[indices]
        result[index] = float(np.sum(picked)) if rule == "sum" else float(picked[0])
    return result


#: Grain of records that score one whitespace word each.
WHITESPACE_UNIT = "whitespace_words"


def whitespace_spans(text: str) -> list[Span]:
    """Spans of the whitespace-delimited words of *text*, in order.

    Args:
        text: The stimulus text.
    """
    spans: list[Span] = []
    start = None
    for index, character in enumerate(text):
        if character.isspace():
            if start is not None:
                spans.append(Span(start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append(Span(start, len(text)))
    return spans


def aggregate_whitespace_words(
    stimulus: Stimulus,
    layer: str,
    rows: list[FeatureModelOutput],
    rule: str,
) -> list[FeatureModelOutput]:
    """Map one record per whitespace word onto the units of *layer*.

    A unit of several words takes the sum of their values.

    Args:
        stimulus: Text carrying *layer*.
        layer: Target layer, such as ``words``.
        rows: One record per whitespace word of the text, in text order.
        rule: ``sum``.

    Raises:
        ValueError: Record count or *rule* off.
        TokenBoundaryError: A whitespace word crosses a unit boundary.
    """
    spans = whitespace_spans(stimulus.text)
    if len(spans) != len(rows):
        raise ValueError(
            f"{stimulus.stimulus_id}: {len(rows)} whitespace-word records for "
            f"{len(spans)} whitespace words"
        )
    if rule != "sum":
        raise ValueError(f"unknown multi-word rule {rule!r}")
    labels = [stimulus.text[s.start : s.end] for s in spans]
    outputs = []
    for indices in nested_members(stimulus, layer, spans, labels):
        first = rows[indices[0]]
        if len(indices) == 1:
            outputs.append(first)
            continue
        values = {
            key: float(sum(float(rows[i].values[key]) for i in indices))
            for key in first.values
        }
        outputs.append(FeatureModelOutput(values=values, metadata=dict(first.metadata)))
    return outputs
