"""Sentence coordinates and fixed target transforms for scalar preparation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from replica.utils.arrays import array_with_missing


def _sentence_ids_for_indices(
    stimulus,
    sid_item_id: int,
    indices: np.ndarray,
    target_layer: str | None = None,
) -> np.ndarray:
    """Sentence ids aligned to retained indices for one datum.

    Supplied IDs must name each source word with an exact int64 value.
    Missing metadata retains the stimulus-level fallback.
    """
    stim_sentence_ids = stimulus.metadata.get("per_word_sentence_ids")
    if stim_sentence_ids is not None:
        words = stimulus.layers.get("words")
        message = (
            f"Stimulus {stimulus.stimulus_id!r}: per_word_sentence_ids must "
            "contain one integer ID per source word, within int64 range"
        )
        if (
            np.ma.isMaskedArray(stim_sentence_ids)
            and np.ma.getmaskarray(stim_sentence_ids).any()
        ):
            raise ValueError(message)
        ids = np.asarray(stim_sentence_ids, dtype=object)
        if words is None or ids.shape != (len(words),):
            raise ValueError(message)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in ids
        ):
            raise ValueError(message)
        try:
            with np.errstate(invalid="raise"):
                converted = ids.astype(np.int64)
        except (ValueError, OverflowError, FloatingPointError) as error:
            raise ValueError(message) from error
        if not np.equal(ids, converted.astype(object)).all():
            raise ValueError(message)
        spans = stimulus.layers[target_layer].spans if target_layer else words.spans
        if spans == words.spans:
            return converted[indices]
        sentence_ids = []
        for span in spans:
            overlapping = {
                sid
                for word, sid in zip(words.spans, converted, strict=True)
                if word.overlaps(span)
            }
            if len(overlapping) != 1:
                raise ValueError(
                    "Custom units require an unambiguous source sentence assignment when sentence IDs are present"
                )
            sentence_ids.append(overlapping.pop())
        return np.asarray(sentence_ids, dtype=int)[indices]
    return np.full(len(indices), sid_item_id, dtype=int)


def _transform_dv(
    transform: Callable[[np.ndarray], np.ndarray], values: np.ndarray
) -> np.ndarray:
    """Transform real responses, retaining masked results as missing values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        transformed = array_with_missing(transform(values))
    if np.iscomplexobj(transformed):
        raise ValueError("DV transform must return real values")
    if transformed.shape != values.shape:
        raise ValueError(
            f"DV transform must preserve shape {values.shape}; got {transformed.shape}"
        )
    return transformed
