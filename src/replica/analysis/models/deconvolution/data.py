"""Build fixed-depth CDR arrays from impulse and response streams."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_F32_EPS = np.float32(np.finfo(np.float32).eps)


def compute_history_intervals(
    X: pd.DataFrame, y: pd.DataFrame, series_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-response impulse-window row range ``[first_obs, last_obs)`` in X.

    ``first_obs`` is the start of the response's series in X; ``last_obs``
    is one past the last X row with ``time <= y.time + float32 eps``. Both
    streams must already be sorted by series then time.

    Args:
        X: Impulse table, carrying a ``time`` column and the ``series_ids``.
        y: Response table, carrying the same two.
        series_ids: Columns whose tuple names one time series, so a window
            never reaches across into another.
    """
    n_y = len(y)
    first_obs = np.zeros(n_y, dtype=np.int64)
    last_obs = np.zeros(n_y, dtype=np.int64)

    x_keys = [tuple(row) for row in X[series_ids].to_numpy()]
    y_keys = [tuple(row) for row in y[series_ids].to_numpy()]
    x_time = X["time"].to_numpy()
    y_time = y["time"].to_numpy()

    series_start: dict[tuple, int] = {}
    series_end: dict[tuple, int] = {}
    for j, key in enumerate(x_keys):
        if key not in series_start:
            series_start[key] = j
        series_end[key] = j + 1

    cursor: dict[tuple, int] = {}
    for i in range(n_y):
        key = y_keys[i]
        if key not in series_start:
            first_obs[i] = last_obs[i] = 0
            continue
        start = series_start[key]
        end = series_end[key]
        j = max(cursor.get(key, start), start)
        limit = y_time[i] + _F32_EPS
        while j < end and x_time[j] <= limit:
            j += 1
        cursor[key] = j
        first_obs[i] = start
        last_obs[i] = j
    return first_obs, last_obs


def clamp_response_times(
    y: pd.DataFrame, X: pd.DataFrame, first_obs: np.ndarray, last_obs: np.ndarray
) -> pd.DataFrame:
    """Clamp y.time up to its newest in-window impulse timestamp.

    Guarantees ``t_delta >= 0`` at the final window slot, which the
    DiracDelta gate's exact float32 zero test relies on. Performed in
    float64 pandas, as in the reference; casting to float32 happens in
    :func:`expand_history`.

    Args:
        y: Response table whose ``time`` column is raised; a copy is returned.
        X: Impulse table supplying the in-window timestamps.
        first_obs: Per-response first impulse row of the window.
        last_obs: Per-response row one past the newest impulse in the window.

    Returns:
        A copy of *y* whose ``time`` is at least its newest in-window impulse
        timestamp. A response with an empty window keeps its own time.
    """
    y = y.copy()
    has_window = last_obs > first_obs
    newest = np.full(len(y), -np.inf)
    newest[has_window] = X["time"].to_numpy()[last_obs[has_window] - 1]
    y["time"] = np.maximum(y["time"].to_numpy(), newest)
    return y


def expand_history(
    X_vals: np.ndarray,
    X_time: np.ndarray,
    first_obs: np.ndarray,
    last_obs: np.ndarray,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-aligned fixed-depth impulse windows.

    Returns ``(X_2d, time_X_2d, time_mask)`` with shapes
    ``(n_y, H, n_impulses)``; leading slots are zero padding (value 0,
    time 0, mask 0), so padded slots contribute exactly 0 to the
    convolution without masking.

    Args:
        X_vals: Impulse values, one row per impulse and one column per impulse
            channel.
        X_time: Timestamp of each row of *X_vals*.
        first_obs: Per-response first impulse row of the window.
        last_obs: Per-response row one past the newest impulse in the window.
        history_length: Slots ``H`` kept per response; a longer window is cut
            back to its ``H`` newest impulses.
    """
    n_y = len(first_obs)
    h = history_length
    k = X_vals.shape[1]
    X_2d = np.zeros((n_y, h, k), dtype=np.float32)
    time_X_2d = np.zeros((n_y, h, k), dtype=np.float32)
    time_mask = np.zeros((n_y, h, k), dtype=np.float32)
    for i in range(n_y):
        first = max(first_obs[i], last_obs[i] - h)
        n = last_obs[i] - first
        if n <= 0:
            continue
        X_2d[i, h - n :] = X_vals[first : last_obs[i]]
        time_X_2d[i, h - n :] = X_time[first : last_obs[i], None]
        time_mask[i, h - n :] = 1.0
    return X_2d, time_X_2d, time_mask


def apply_reference_cutoff(
    *,
    first_obs: np.ndarray,
    last_obs: np.ndarray,
    response_time: np.ndarray,
    response_series: np.ndarray,
    impulse_time: np.ndarray,
    series_starts: dict[int, int],
    t_delta_cutoff: float,
) -> None:
    """Move each series' oldest-impulse pointer as the reference does.

    The reference carries one pointer per series and tests the cutoff only
    while a response consumes new impulses; when the test fires it moves the
    pointer one past the newest consumed impulse rather than to the oldest one
    inside the window. A gap therefore empties the window and the responses
    after it regrow it, so the reach is a function of arrival order and not
    of the cutoff alone. ``first_obs`` is updated in place.

    Args:
        first_obs: Per-response first impulse row, rewritten in place.
        last_obs: Per-response row one past the newest impulse, read only.
        response_time: Timestamp of each response.
        response_series: Series code of each response.
        impulse_time: Timestamp of each impulse row.
        series_starts: First impulse row of each series code.
        t_delta_cutoff: Lag beyond which the pointer advances, in the event
            streams' own time unit.
    """
    order = np.lexsort((response_time, response_series))
    pointers: dict[int, tuple[int, int]] = {}
    for i in order:
        series = int(response_series[i])
        base = series_starts.get(series)
        if base is None:
            continue
        oldest, frontier = pointers.get(series, (base, base))
        target = int(last_obs[i])
        time = float(response_time[i])
        while frontier < target:
            frontier += 1
            if abs(float(impulse_time[oldest]) - time) > t_delta_cutoff:
                oldest = frontier
        pointers[series] = (oldest, frontier)
        first_obs[i] = oldest


def impulse_op(values: np.ndarray, op: str, ddof: int = 1) -> np.ndarray:
    """One named impulse transform over a whole column.

    ``z`` = (x-mean)/sd, ``c`` = x-mean, ``s`` = x/sd, ``powN`` = x**N,
    plus ``log``/``log1p``/``exp``. Pipe-separated operations compose from
    left to right. NaN and +inf become 0 first. ``ddof`` follows the reference
    variant: the parametric one divides by ``n-1``, the neural one by ``n``.

    Args:
        values: One impulse column, whole rather than per partition.
        op: The transform name, or several joined by ``|``.
        ddof: Delta degrees of freedom of the standard deviation ``z`` and
            ``s`` divide by.
    """
    values = np.where(np.isnan(values) | np.isposinf(values), 0.0, values)
    if "|" in op:
        for step in op.split("|"):
            values = impulse_op(values, step, ddof)
        return values
    if op == "z":
        return (values - values.mean()) / np.std(values, ddof=ddof)
    if op == "c":
        return values - values.mean()
    if op == "s":
        return values / np.std(values, ddof=ddof)
    if op == "log":
        return np.log(values)
    if op == "log1p":
        return np.log1p(values)
    if op == "exp":
        return np.exp(values)
    if op.startswith("pow"):
        return values ** float(op[3:])
    raise ValueError(f"unsupported impulse op: {op!r}")


def apply_impulse_ops(X: pd.DataFrame, ops: dict[str, str]) -> pd.DataFrame:
    """Column transforms ``{new_name: "op(col)"}`` over the whole X table.

    Transforms run over the full table, so scaling is identical across
    partitions, which is what the reference does.

    Args:
        X: Impulse table; a copy carrying the new columns is returned.
        ops: New column name to a ``"op(source_col)"`` expression, where the
            source column is read from *X*.
    """
    X = X.copy()
    for new_name, expr in ops.items():
        op, col = expr.rstrip(")").split("(", 1)
        X[new_name] = impulse_op(X[col].to_numpy(dtype=float), op)
    return X


def transform_impulses(
    matrix: np.ndarray, names: list[str], ops: dict[str, str], ddof: int = 1
) -> np.ndarray:
    """Apply per-impulse transforms to an event stream's columns.

    Impulses are shared by every partition, so a transform over the whole
    column is the same statistic the reference computes. Transforms are
    written for the full term set, so those naming an impulse an ablation
    removed have nothing to apply to.

    Args:
        matrix: Impulse values, one row per impulse and one column per name.
        names: Column names of *matrix*, in its column order.
        ops: Impulse name to the transform applied to that column; a name
            absent from *names* is skipped.
        ddof: Delta degrees of freedom passed to :func:`impulse_op`.
    """
    if not ops:
        return matrix
    out = matrix.astype(np.float64, copy=True)
    for i, name in enumerate(names):
        if name in ops:
            out[:, i] = impulse_op(out[:, i], ops[name], ddof)
    return out.astype(np.float32)


@dataclass
class CDRDesign:
    """Model-ready arrays for one response partition."""

    X_2d: np.ndarray
    time_X_2d: np.ndarray
    time_mask: np.ndarray
    time_y: np.ndarray
    y: np.ndarray
    gf_y: np.ndarray
    impulse_names: list[str]
    rangf_levels: dict[str, list[str]] = field(default_factory=dict)
    # Scalar for ordinary models, one value per response for a joint-response
    # model.  The latter mirrors CDRNN configurations whose left-hand side is
    # e.g. ``fdurFP + fdurGP + fdurSPsummed``.
    y_mean: float | np.ndarray = 0.0
    y_sd: float | np.ndarray = 1.0
    #: Optional second grouping code per response, for a model that gives the
    #: word an effect of its own alongside the participant's.
    item: np.ndarray | None = None


def build_design(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    impulse_names: list[str],
    dv: str,
    series_ids: list[str],
    history_length: int,
    rangf: list[str] | None = None,
    rangf_levels: dict[str, list[str]] | None = None,
    y_mean: float | np.ndarray | None = None,
    y_sd: float | np.ndarray | None = None,
) -> CDRDesign:
    """Assemble the CDR design for one (X, y) pair.

    ``rangf_levels`` / ``y_mean`` / ``y_sd`` are taken from the training
    partition when given (predict-time reuse); level codes are the sorted
    unique training values with one extra unknown-level index at the end.

    Args:
        X: Impulse table, sorted by series then time, carrying ``time`` and
            every column ``impulse_names`` asks for.
        y: Response table, carrying ``time``, *dv*, the ``series_ids``, and
            the ``rangf`` columns. Rows whose *dv* is not finite are dropped.
        impulse_names: Impulse columns to window, in the order the model's
            terms read them.
        dv: Response column of *y*.
        series_ids: Columns whose tuple names one time series.
        history_length: Impulse slots kept per response.
        rangf: Columns of *y* coded as random-effect grouping factors.
        rangf_levels: Level list per grouping factor; a factor absent here
            takes its levels from these rows.
        y_mean: Centre for the response, reused from the training partition.
            ``None`` takes the mean of these rows.
        y_sd: Scale for the response, reused from the training partition.
            ``None`` takes the sample standard deviation of these rows.
    """
    y = y[np.isfinite(y[dv].to_numpy(dtype=float))]
    first_obs, last_obs = compute_history_intervals(X, y, series_ids)
    y = clamp_response_times(y, X, first_obs, last_obs)

    X_vals = X[impulse_names].to_numpy(dtype=np.float32)
    X_time = X["time"].to_numpy(dtype=np.float32)
    X_2d, time_X_2d, time_mask = expand_history(
        X_vals, X_time, first_obs, last_obs, history_length
    )

    rangf = rangf or []
    levels = dict(rangf_levels or {})
    codes = np.zeros((len(y), len(rangf)), dtype=np.int64)
    for i, gf in enumerate(rangf):
        vals = y[gf].astype(str).to_numpy()
        if gf not in levels:
            levels[gf] = sorted(set(vals))
        table = {lev: j for j, lev in enumerate(levels[gf])}
        unk = len(levels[gf])
        codes[:, i] = np.array([table.get(v, unk) for v in vals], dtype=np.int64)

    y_vals = y[dv].to_numpy(dtype=np.float64)
    mean = float(y_vals.mean()) if y_mean is None else y_mean
    sd = float(y_vals.std(ddof=1)) if y_sd is None else y_sd

    return CDRDesign(
        X_2d=X_2d,
        time_X_2d=time_X_2d,
        time_mask=time_mask,
        time_y=y["time"].to_numpy(dtype=np.float32),
        y=y_vals,
        gf_y=codes,
        impulse_names=list(impulse_names),
        rangf_levels=levels,
        y_mean=mean,
        y_sd=sd,
    )


def build_multistream_design(
    X: list[pd.DataFrame],
    y: pd.DataFrame,
    *,
    impulse_sources: dict[str, int | None],
    dv: str,
    series_ids: list[str],
    history_length: int,
    rangf: list[str] | None = None,
    rangf_levels: dict[str, list[str]] | None = None,
    y_mean: float | None = None,
    y_sd: float | None = None,
) -> CDRDesign:
    """Assemble one design from asynchronous impulse streams.

    A response-aligned impulse sits in the last history slot at the response
    timestamp, as in the reference data builder. History intervals and
    response-time clamps are computed stream by stream, in source order.

    Args:
        X: Impulse tables, each sorted by series then time and already carrying
            any formula transform (for example ``s(tr)``) under its mapped name.
        y: Response table, carrying ``time``, *dv*, the ``series_ids``, the
            ``rangf`` columns, and any response-aligned impulse.
        impulse_sources: Impulse name to the index of its frame in *X*, or
            ``None`` for one stored in *y* and placed at the response's own
            timestamp. Its key order fixes the design's column order.
        dv: Response column of *y*.
        series_ids: Columns whose tuple names one time series.
        history_length: Impulse slots kept per response.
        rangf: Columns of *y* coded as random-effect grouping factors.
        rangf_levels: Level list per grouping factor; a factor absent here
            takes its levels from these rows.
        y_mean: Centre for the response, reused from the training partition.
            ``None`` takes the mean of these rows.
        y_sd: Scale for the response, reused from the training partition.
            ``None`` takes the sample standard deviation of these rows.
    """
    if not X:
        raise ValueError("at least one impulse stream is required")
    invalid_source = {
        name: source
        for name, source in impulse_sources.items()
        if source is not None and not 0 <= source < len(X)
    }
    if invalid_source:
        raise ValueError(f"invalid impulse source indices: {invalid_source!r}")

    impulse_names = list(impulse_sources)
    y = y[np.isfinite(y[dv].to_numpy(dtype=float))].copy()
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for source_idx, stream in enumerate(X):
        names = [
            name for name, source in impulse_sources.items() if source == source_idx
        ]
        if not names:
            continue
        first_obs, last_obs = compute_history_intervals(stream, y, series_ids)
        y = clamp_response_times(y, stream, first_obs, last_obs)
        values, times, mask = expand_history(
            stream[names].to_numpy(dtype=np.float32),
            stream["time"].to_numpy(dtype=np.float32),
            first_obs,
            last_obs,
            history_length,
        )
        for j, name in enumerate(names):
            arrays[name] = (values[:, :, j], times[:, :, j], mask[:, :, j])

    for name, source in impulse_sources.items():
        if source is not None:
            continue
        shape = (len(y), history_length)
        values = np.zeros(shape, dtype=np.float32)
        times = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        values[:, -1] = y[name].to_numpy(dtype=np.float32)
        times[:, -1] = y["time"].to_numpy(dtype=np.float32)
        mask[:, -1] = 1.0
        arrays[name] = (values, times, mask)

    missing = [name for name in impulse_names if name not in arrays]
    if missing:
        raise ValueError(f"impulses were not assembled: {missing!r}")

    X_2d = np.stack([arrays[name][0] for name in impulse_names], axis=2)
    time_X_2d = np.stack([arrays[name][1] for name in impulse_names], axis=2)
    time_mask = np.stack([arrays[name][2] for name in impulse_names], axis=2)

    rangf = rangf or []
    levels = dict(rangf_levels or {})
    codes = np.zeros((len(y), len(rangf)), dtype=np.int64)
    for i, gf in enumerate(rangf):
        values = y[gf].astype(str).to_numpy()
        if gf not in levels:
            levels[gf] = sorted(set(values))
        table = {level: j for j, level in enumerate(levels[gf])}
        unknown = len(levels[gf])
        codes[:, i] = np.asarray(
            [table.get(value, unknown) for value in values], dtype=np.int64
        )

    y_values = y[dv].to_numpy(dtype=np.float64)
    mean = float(y_values.mean()) if y_mean is None else y_mean
    sd = float(y_values.std(ddof=1)) if y_sd is None else y_sd
    return CDRDesign(
        X_2d=X_2d,
        time_X_2d=time_X_2d,
        time_mask=time_mask,
        time_y=y["time"].to_numpy(dtype=np.float32),
        y=y_values,
        gf_y=codes,
        impulse_names=impulse_names,
        rangf_levels=levels,
        y_mean=mean,
        y_sd=sd,
    )


def build_design_from_events(
    *,
    response_time: np.ndarray,
    response_series: np.ndarray,
    response_y: np.ndarray,
    response_group: np.ndarray | None,
    impulse_time: np.ndarray,
    impulse_series: np.ndarray,
    impulse_values: np.ndarray,
    impulse_names: list[str],
    history_length: int,
    t_delta_cutoff: float | None = None,
    cutoff_mode: str = "reference",
    y_mean: float | None = None,
    y_sd: float | None = None,
    item: np.ndarray | None = None,
) -> CDRDesign:
    """Assemble a design from an impulse stream and a response stream.

    Each response takes the impulses of its own series up to and including its
    timestamp, the half-open window the frame-based path computes, and its
    timestamp is clamped up to its newest in-window impulse so a simultaneous
    impulse stays simultaneous in float32.

    Args:
        response_time: Timestamp of each response.
        response_series: Series code of each response.
        response_y: Dependent variable per response; a second axis holds one
            column per response of a joint-response model.
        response_group: Integer grouping code per response, or ``None`` for a
            single all-zero column.
        impulse_time: Timestamp of each impulse; the stream is sorted by series
            then time.
        impulse_series: Series code of each impulse.
        impulse_values: Impulse values, one row per impulse and one column per
            name.
        impulse_names: Column names of *impulse_values*, in its column order.
        history_length: Impulse slots kept per response.
        t_delta_cutoff: Lag bound on how far back a window reaches, which the
            neural configurations set; ``None`` lets it reach the start of the
            series.
        cutoff_mode: ``reference`` keeps the released pointer behavior, for
            repetition; ``sliding`` applies the lag limit to each response
            independently, for corrected experiments. Recorded by fitted
            models.
        y_mean: Centre for the response, reused from the training partition.
            ``None`` takes the mean of these rows, per response column.
        y_sd: Scale for the response, reused from the training partition.
            ``None`` takes the sample standard deviation of these rows.
        item: Second grouping code per response, for a model that gives the
            word an effect alongside the participant's.
    """
    if cutoff_mode not in {"reference", "sliding"}:
        raise ValueError(f"unknown history cutoff mode: {cutoff_mode!r}")
    if t_delta_cutoff is not None and t_delta_cutoff <= 0:
        raise ValueError("history cutoff must be positive or None")
    order = np.lexsort((impulse_time, impulse_series))
    if not np.array_equal(order, np.arange(len(order))):
        impulse_time = impulse_time[order]
        impulse_series = impulse_series[order]
        impulse_values = impulse_values[order]

    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    for j, series in enumerate(impulse_series):
        series = int(series)
        starts.setdefault(series, j)
        ends[series] = j + 1

    n = len(response_time)
    first_obs = np.zeros(n, dtype=np.int64)
    last_obs = np.zeros(n, dtype=np.int64)
    times = np.asarray(response_time, dtype=float).copy()
    for i in range(n):
        series = int(response_series[i])
        start = starts.get(series)
        if start is None:
            continue
        end = ends[series]
        limit = times[i] + float(_F32_EPS)
        offset = int(np.searchsorted(impulse_time[start:end], limit, side="right"))
        first_obs[i] = start
        last_obs[i] = start + offset
        if t_delta_cutoff is not None and cutoff_mode == "sliding":
            first_obs[i] = start + int(
                np.searchsorted(
                    impulse_time[start : start + offset],
                    times[i] - t_delta_cutoff,
                    side="left",
                )
            )
        if offset:
            times[i] = max(times[i], float(impulse_time[start + offset - 1]))
    if t_delta_cutoff is not None and cutoff_mode == "reference":
        apply_reference_cutoff(
            first_obs=first_obs,
            last_obs=last_obs,
            response_time=np.asarray(response_time, dtype=float),
            response_series=np.asarray(response_series),
            impulse_time=impulse_time,
            series_starts=starts,
            t_delta_cutoff=float(t_delta_cutoff),
        )

    X_2d, time_X_2d, time_mask = expand_history(
        np.asarray(impulse_values, dtype=np.float32),
        np.asarray(impulse_time, dtype=np.float32),
        first_obs,
        last_obs,
        history_length,
    )
    y_values = np.asarray(response_y, dtype=np.float64)
    codes = (
        np.zeros((n, 1), dtype=np.int64)
        if response_group is None
        else np.asarray(response_group, dtype=np.int64).reshape(n, 1)
    )
    return CDRDesign(
        X_2d=X_2d,
        time_X_2d=time_X_2d,
        time_mask=time_mask,
        time_y=times.astype(np.float32),
        y=y_values,
        gf_y=codes,
        impulse_names=list(impulse_names),
        y_mean=(
            float(y_values.mean())
            if y_mean is None and y_values.ndim == 1
            else np.nanmean(y_values, axis=0)
            if y_mean is None
            else y_mean
        ),
        y_sd=(
            float(y_values.std(ddof=1))
            if y_sd is None and y_values.ndim == 1
            else np.nanstd(y_values, axis=0, ddof=1)
            if y_sd is None
            else y_sd
        ),
        item=item,
    )
