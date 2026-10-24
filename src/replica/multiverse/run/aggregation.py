"""Participant aggregation for repeated-measures datasets."""

import numpy as np


def aggregate_over_participants(
    dataset,
    measure: str,
    strategy: str,
) -> tuple[int, int]:
    """Collapse per-participant measure datums into one datum per stimulus.

    Datums for other measures pass through untouched, so predictor pickles and
    stimulus ordering are unaffected.

    Args:
        dataset: Dataset whose datum list is rewritten in place.
        measure: Reading measure to collapse, resolved through the dataset's
            measure aliases.
        strategy: ``excl`` excludes zero durations; ``zero`` includes them.
            Both ignore unavailable measurements (NaN). An all-zero column
            averages to zero in include mode; an all-missing column stays NaN.

    Returns:
        The number of stimuli collapsed, and the number of per-participant
        datums the collapse replaced.

    Raises:
        ValueError: *strategy* is neither ``excl`` nor ``zero``.
    """
    from replica.core import Datum, resolve_dv

    # Read the stored values, restricted by the flag for a derived variable
    # but with released zeros intact, so the strategy decides about them.
    previous_policy = getattr(dataset, "drop_skips", None)
    if previous_policy is not None:
        dataset.drop_skips = False
    try:
        datums = dataset.data(measure=measure)
    finally:
        if previous_policy is not None:
            dataset.drop_skips = previous_policy
    by_stim: dict[str, list[Datum]] = {}
    for d in datums:
        by_stim.setdefault(d.stimulus.stimulus_id, []).append(d)
    replaced = {measure, resolve_dv(measure)[0]}
    other = [d for d in dataset._data if d.measure_name not in replaced]

    new_data: list[Datum] = list(other)
    dropped = 0
    for _sid, group in by_stim.items():
        matrix = np.stack([d.values for d in group], axis=0)
        if strategy == "excl":
            matrix = np.where(matrix == 0.0, np.nan, matrix)
        elif strategy == "zero":
            pass
        else:
            raise ValueError(f"unknown aggregation strategy {strategy!r}")
        counts = np.count_nonzero(~np.isnan(matrix), axis=0)
        agg = np.divide(
            np.nansum(matrix, axis=0),
            counts,
            out=np.full(matrix.shape[1], np.nan),
            where=counts > 0,
        )
        dropped += matrix.shape[0] - 1
        new_data.append(
            Datum(
                stimulus=group[0].stimulus,
                measure_name=measure,
                values=agg.astype(np.float64),
                participant_id="aggregated",
                layer=group[0].layer,
            )
        )
    dataset._data = new_data
    return len(by_stim), dropped
