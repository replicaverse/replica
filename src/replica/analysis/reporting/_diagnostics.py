"""Generic fit-check tables and bounded per-fold prediction-error SVG plots."""

from html import escape
from typing import Any

from ._format import _details, _finite, _number, _text


def _status(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown / not provided"


def _fold_attempts(attempts: Any) -> str:
    if not isinstance(attempts, list) or not attempts:
        return "<p>Fold-attempt records were not provided.</p>"
    rows = []
    for attempt in attempts:
        details = {
            name: attempt[name]
            for name in ("reason", "warnings", "error")
            if name in attempt
        }
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{_text(value)}</td>"
                for value in (
                    attempt.get("fold_index"),
                    attempt.get("status"),
                    attempt.get("stage"),
                    attempt.get("n_train"),
                    attempt.get("n_test"),
                    _number(attempt.get("elapsed_seconds")),
                    details,
                )
            )
            + "</tr>"
        )
    return (
        "<p>Execution status is separate from the backend's fit checks.</p>"
        '<div class="scroll"><table><thead><tr><th>Fold</th><th>Status</th><th>Stage</th>'
        "<th>Train rows</th><th>Scored rows</th><th>Seconds</th><th>Recorded details</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _fit_diagnostics(metadata: dict[str, Any]) -> str:
    diagnostics = metadata.get("fit_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        return "<p>Fit diagnostics were not provided.</p>"
    rows = []
    for fold in diagnostics:
        for model in ("baseline", "full"):
            diagnostic = fold.get(model)
            if not isinstance(diagnostic, dict):
                values = ["Not provided"] * 6
            else:
                convergence = (
                    "Not applicable"
                    if diagnostic.get("convergence_applicable") is False
                    else _status(diagnostic.get("converged"))
                )
                rank, columns = diagnostic.get("rank"), diagnostic.get("n_columns")
                values = [
                    convergence,
                    _status(diagnostic.get("singular")),
                    "Unknown / not provided"
                    if rank is None
                    else f"{rank} / {columns if columns is not None else '?'}",
                    _status(diagnostic.get("diagnostic_checks_passed")),
                    diagnostic.get("selected_parameters", "Not provided"),
                    diagnostic.get("warnings", "Not provided"),
                ]
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{_text(value)}</td>"
                    for value in [fold.get("fold_index"), model.capitalize(), *values]
                )
                + "</tr>"
            )
    return (
        "<p>These are backend-reported fit checks, not an overall model-validity verdict. "
        "A non-iterative fit can mark convergence not applicable. Unknown values remain "
        "distinct from failed checks. Sampler diagnostic checks are not convergence.</p>"
        '<div class="scroll"><table><thead><tr><th>Fold</th><th>Model</th>'
        "<th>Converged</th><th>Random-effect singularity</th><th>Rank / columns</th>"
        "<th>Diagnostic checks passed</th><th>Selected parameters</th><th>Warnings</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + _details(
            "All fit diagnostics, selected parameters and backend details", diagnostics
        )
    )


def _axis(values: list[float]) -> tuple[float, float, float]:
    magnitude = max(abs(value) for value in values) or 1.0
    lo, hi = min(values) / magnitude, max(values) / magnitude
    if lo == hi:
        lo, hi = max(-1.0, lo - 0.1), min(1.0, hi + 0.1)
    return magnitude, lo, hi


def _tick(value: float, magnitude: float) -> str:
    return "0" if abs(value / magnitude) < 1e-12 else f"{value:.3g}"


def _scatter(
    points: list[tuple[float, float, float | None, str]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    identity: bool = False,
) -> str:
    """Draw bounded summary points; intervals are descriptive error SDs."""
    if not points:
        return "<p>No finite points are available.</p>"
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    for _, value, spread, _ in points:
        if spread is not None:
            ys.extend(end for end in (value - spread, value + spread) if _finite(end))
    if not identity:
        ys.append(0.0)
    xm, xl, xh = _axis(xs)
    ym, yl, yh = _axis(ys)

    def x(value: float) -> float:
        return 76 + 410 * ((value / xm - xl) / (xh - xl))

    def y(value: float) -> float:
        return 258 - 218 * ((value / ym - yl) / (yh - yl))

    marks = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        xv, yv = (xl + fraction * (xh - xl)) * xm, (yl + fraction * (yh - yl)) * ym
        xp, yp = x(xv), y(yv)
        marks.append(
            f'<path d="M{xp:.2f},258v5 M71,{yp:.2f}h5" stroke="#63717b"/>'
            f'<text x="{xp:.2f}" y="280" text-anchor="middle">{_tick(xv, xm)}</text>'
            f'<text x="67" y="{yp + 4:.2f}" text-anchor="end">{_tick(yv, ym)}</text>'
        )
    if identity:
        start, end = max(min(xs), min(ys)), min(max(xs), max(ys))
        if start <= end:
            marks.append(
                f'<path d="M{x(start):.2f},{y(start):.2f}L{x(end):.2f},{y(end):.2f}" '
                'stroke="#63717b" stroke-dasharray="5 4"/>'
            )
    else:
        marks.append(
            f'<path d="M76,{y(0):.2f}H486" stroke="#63717b" stroke-dasharray="5 4"/>'
        )
    for fitted, error, spread, label in points:
        xp, yp = x(fitted), y(error)
        if spread is not None and _finite(error - spread) and _finite(error + spread):
            marks.append(
                f'<path d="M{xp:.2f},{y(error - spread):.2f}V{y(error + spread):.2f}" '
                'stroke="#82b6b7" stroke-width="2"/>'
            )
        marks.append(
            f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="3" fill="#176e72">'
            f"<title>{escape(label)}</title></circle>"
        )
    return (
        f'<svg class="diagnostic-plot" viewBox="0 0 520 320" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>"
        f'<text x="260" y="20" text-anchor="middle" class="plot-title">{escape(title)}</text>'
        '<path d="M76,40V258H486" fill="none" stroke="#63717b"/>'
        + "".join(marks)
        + f'<text x="280" y="309" text-anchor="middle">{escape(x_label)}</text>'
        + f'<text transform="translate(15 150) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>'
        + "</svg>"
    )


def _prediction_summary(summary: dict[str, Any], model: str) -> str:
    bins = summary.get("fitted_bins") or []
    points = [
        (
            item["fitted_mean"],
            item["error_mean"],
            item.get("error_sd") if _finite(item.get("error_sd")) else None,
            f"{item.get('count')} rows; fitted mean {item['fitted_mean']}; "
            f"error mean {item['error_mean']}; SD {item.get('error_sd')}",
        )
        for item in bins
        if _finite(item.get("fitted_mean")) and _finite(item.get("error_mean"))
    ]
    residual = _scatter(
        points,
        title="Prediction errors versus fitted values",
        x_label="Mean fitted value in bin",
        y_label="Mean error ± SD",
    )
    qq = summary.get("normal_qq") or {}
    if qq.get("available"):
        qq_plot = _scatter(
            [
                (
                    point["theoretical"],
                    point["standardized_error"],
                    None,
                    f"Normal quantile {point['theoretical']}; "
                    f"standardized error {point['standardized_error']}",
                )
                for point in qq.get("points", [])
                if _finite(point.get("theoretical"))
                and _finite(point.get("standardized_error"))
            ],
            title="Normal Q–Q reference",
            x_label="Standard normal quantile",
            y_label="Ordered standardized error",
            identity=True,
        )
        scale = (
            "the scoring sigma"
            if qq.get("scale_source") == "scoring_sigma"
            else "finite-error RMS estimated on these scored rows (excluding nonfinite errors, including subtraction overflow)"
        )
        qq_note = (
            f"Errors are divided by {scale}. The dashed line is y = x. "
            f"{_number(qq.get('n_finite'))} finite standardized errors contribute; "
            f"{_number(qq.get('n_nonfinite'))} scored rows are excluded from the Q–Q "
            "reference, including any nonfinite raw errors. "
            "At most 101 selected order statistics include both extremes; intermediate "
            "points are omitted deterministically, not randomly sampled."
        )
    else:
        reasons = {
            "no_declared_gaussian_reference": "this backend declares no Gaussian reference",
            "no_finite_errors": "there are no finite scored errors",
            "no_finite_positive_standardization_scale": "no finite errors have a positive finite reference scale",
        }
        qq_plot = ""
        qq_note = f"Normal Q–Q reference unavailable: {reasons.get(str(qq.get('reason')), 'reference information was not recorded')}. "
    return (
        f"<h5>{escape(model.capitalize())}</h5>"
        f"<p>{_number(summary.get('n_finite_errors'))} finite errors; "
        f"{_number(summary.get('n_nonfinite_errors'))} nonfinite rows excluded from these summaries. "
        f"Error mean {_number(summary.get('error_mean'))}; "
        f"SD {_number(summary.get('error_sd'))}; RMS {_number(summary.get('error_rms'))}.</p>"
        + residual
        + f"<p>{len(points)} fitted-value bins include every finite row, with tied fitted "
        "values kept together. Points are bin means; bars show within-bin error SD, "
        "not standard errors or confidence intervals. Binning can conceal local patterns.</p>"
        + qq_plot
        + f"<p>{qq_note}</p>"
    )


def _prediction_diagnostics(record: dict[str, Any]) -> str:
    folds = record.get("folds") or []
    if not any(
        fold["metadata"].get("prediction_diagnostics") is not None for fold in folds
    ):
        return (
            "<p>Prediction-error summaries were not retained. Evaluate with "
            "<code>AnalysisConfig(per_fold=True, prediction_diagnostics=True)</code> "
            "to include bounded residual and Q–Q plots.</p>"
        )
    content = []
    for fold in folds:
        diagnostic = fold["metadata"].get("prediction_diagnostics")
        if not isinstance(diagnostic, dict):
            body = "<p>Prediction-error summaries were not retained for this fold.</p>"
        else:
            body = (
                f"<p>{_number(diagnostic.get('n_rows'))} scored rows; "
                f"{_number(diagnostic.get('n_train_overlap'))} also occur in this fold’s training data. "
                "Errors are observed minus predicted, on the model response scale after "
                "the response transform and training-fold preprocessing, even when scores "
                'are reported on the original scale.</p><div class="prediction-models">'
                + "".join(
                    "<div>" + _prediction_summary(diagnostic[model], model) + "</div>"
                    for model in ("baseline", "full")
                    if isinstance(diagnostic.get(model), dict)
                )
                + "</div>"
            )
        content.append(
            f"<details><summary>Fold {_text(fold['fold_index'])}</summary>{body}</details>"
        )
    return (
        "<p>Descriptive prediction-error checks, shown separately for each fold and model. "
        "These are not tests of model validity or residual independence. Cross-validation "
        "and hierarchical predictions can produce dependent errors; a normal reference "
        "alone does not establish a model’s assumptions.</p>" + "".join(content)
    )
