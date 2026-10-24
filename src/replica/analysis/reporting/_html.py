"""Dependency-free HTML rendering of analysis summaries and fold plots."""

from __future__ import annotations

import base64
import json
from html import escape
from typing import Any
from urllib.parse import quote

from ._diagnostics import _fit_diagnostics, _fold_attempts, _prediction_diagnostics
from ._format import _details, _finite, _number, _table, _text


def _fold_chart(folds: list[dict[str, Any]]) -> str:
    points = [fold for fold in folds if _finite(fold["delta_ll_per_observation"])]
    if not points:
        return "<p>No finite fold scores were retained.</p>"
    values = [fold["delta_ll_per_observation"] for fold in points]
    magnitude = max(abs(value) for value in values) or 1
    normalized = [value / magnitude for value in values]
    lo, hi = min(0, *normalized), max(0, *normalized)
    span = hi - lo or 1
    zero = 100 * (-lo / span)
    bars = []
    for fold in points:
        value = fold["delta_ll_per_observation"]
        end = 100 * ((value / magnitude - lo) / span)
        bars.append(
            f'<div class="fold-bar" title="{_text(fold["n_test"])} scored rows">'
            f'<span>Fold {_text(fold["fold_index"])}</span><div class="track">'
            f'<i class="zero" style="left:{zero:.3f}%"></i>'
            f'<i class="bar" style="left:{min(zero, end):.3f}%;width:{abs(end - zero):.3f}%;'
            f'background:{"#176e72" if value >= 0 else "#a54832"}"></i></div>'
            f"<span>{_number(value)}</span></div>"
        )
    return (
        f"<p>{len(points)} of {len(folds)} folds have finite gains. The vertical line marks zero.</p>"
        '<div class="fold-chart" tabindex="0" role="img" aria-label="Scored fold log-likelihood gains; values in the fold table">'
        + "".join(bars)
        + "</div>"
    )


def _folds(record: dict[str, Any]) -> str:
    folds = record["folds"]
    if not folds:
        return "<p>Fold results were not retained. Set <code>per_fold=True</code> before evaluation to include them.</p>"
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{value}</td>"
            for value in (
                _text(fold["fold_index"]),
                _text(fold["metadata"].get("n_train")),
                _text(fold["n_test"]),
                _number(fold["baseline_ll"]),
                _number(fold["full_ll"]),
                _number(fold["delta_ll_per_observation"]),
            )
        )
        + "</tr>"
        for fold in folds
    )
    return (
        "<p>Full minus baseline log likelihood, in nats per scored observation. "
        "Fold variation is descriptive; these bars are not confidence intervals.</p>"
        + _fold_chart(folds)
        + '<details><summary>Fold scores and row counts</summary><div class="scroll"><table><thead><tr>'
        "<th>Fold</th><th>Train rows</th><th>Scored rows</th><th>Baseline LL</th><th>Full LL</th><th>Gain / observation</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table></div></details>"
        "<p>Totals above use the result’s recorded aggregation, including repeated-CV normalization. "
        "Summed fold counts can include repeated observations. The configured splitter determines "
        "whether scoring is out of sample. Download the JSON for full numeric precision.</p>"
    )


def _observation_fields(record: dict[str, Any]) -> str:
    retained = [
        fold
        for fold in record.get("folds") or []
        if fold.get("observations") is not None
    ]
    if not retained:
        return (
            "<p>Observation vectors were not retained. Set "
            "<code>AnalysisConfig(per_fold=True, retain_predictions=True)</code> "
            "before evaluation to enable a separate CSV export.</p>"
        )
    rows = []
    for fold in retained:
        inventory = fold["observations"]
        fields = ", ".join(
            f"{name} ({value.get('dtype', 'unknown type')})"
            for name, value in inventory["fields"].items()
        )
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{_text(value)}</td>"
                for value in (
                    fold["fold_index"],
                    inventory["n_rows"],
                    fields,
                    (
                        "Key recorded"
                        if inventory["row_keys"] is not None
                        else "Key absent"
                    )
                    + (
                        "; labels recorded"
                        if inventory["row_ids"] is not None
                        else "; labels absent"
                    ),
                    ", ".join(inventory["factors"]) or "None retained",
                    "Available"
                    if inventory["delta_loglik"] is not None
                    else "Not retained",
                )
            )
            + "</tr>"
        )
    return (
        f"<p>{len(retained)} of {len(record['folds'])} folds retain aligned observation fields. "
        "Values stay outside the HTML and its embedded JSON. Export explicitly with "
        '<code>observations_path="observations.csv"</code>; errors are derived from '
        "the retained observed and predicted values. Counts can repeat rows across folds.</p>"
        '<div class="scroll"><table><thead><tr><th>Fold</th><th>Rows</th><th>Numeric fields</th>'
        "<th>Canonical key / readable labels</th><th>Factors</th><th>Per-row LL difference</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + "<p>Response and prediction fields use the model scale after response transformation "
        "and fold preprocessing. Original target units, transforms and scoring scale remain "
        "in the recorded metadata; transformed values are not labelled with original units.</p>"
    )


def _observation_links(report: dict[str, Any]) -> str:
    export = report.get("observation_export")
    if export is None:
        return ""
    links = export["report_links"]
    return (
        f'<p>Explicit observation export: <a download href="{quote(links["csv"], safe="/.")}">CSV rows</a> '
        f'and <a download href="{quote(links["metadata"], safe="/.")}">JSON metadata</a> '
        f"({_number(export['n_rows'])} rows). Keep these separate files with the report; "
        "they contain observation values and identities.</p>"
    )


def _failure(report: dict[str, Any]) -> str:
    failure = report["context"].get("failed_analysis")
    if not isinstance(failure, dict) or failure.get("status") != "failed":
        return ""
    return (
        '<section class="notice"><h2>Recorded analysis failure</h2>'
        "<p>The caller supplied a failed-analysis record. No scores are inferred "
        "from that record. Any completed results are shown separately below.</p>"
        + _table(
            [
                (name.capitalize(), failure.get(name))
                for name in ("dataset", "target", "comparison")
            ]
        )
        + _fold_attempts(failure.get("fold_attempts"))
        + _details("Failure diagnostics", failure)
        + "</section>"
    )


def _section(record: dict[str, Any], index: int) -> str:
    metadata, response = record["metadata"], record["response"]
    config = metadata.get("configuration") or {}
    comparison = record["comparison"] or {}
    metrics = record["metrics"]
    cards = "".join(
        f'<div class="stat"><span>{label}</span><strong>{_number(value)}</strong></div>'
        for label, value in (
            ("Retained rows", record["n_observations"]),
            ("Effective scored rows", record["n_scored"]),
            ("Gain, nats / observation", metrics.get("delta_ll_per_unit")),
            ("R² gain", metrics.get("delta_r2")),
        )
    )
    choices = _table(
        [
            ("Dataset", record["dataset"]),
            ("Target", record["target"]),
            ("Baseline controls", comparison.get("baseline")),
            ("Added predictors", comparison.get("added")),
            ("Feature model", record["feature_model"]),
            ("Regression backend", config.get("backend")),
            ("Backend options", config.get("backend_options")),
            ("Response transform", config.get("response_transform")),
            ("Scoring scale", config.get("score_scale")),
            ("Residual scale estimated from", config.get("sigma_source")),
            ("Cross-validation", config.get("cv_splitter")),
            ("Cross-validation options", config.get("cv_options")),
            ("Scaler", config.get("scaler")),
            ("Training-fold preprocessing", config.get("preprocessing")),
            ("Response eligibility filter", config.get("outlier_filter")),
            ("Missing-feature policy", config.get("missing")),
            ("Predictive significance", config.get("predictive_significance")),
        ]
    )
    exclusions = metadata.get("row_exclusions", response.get("row_exclusions"))
    exclusion_html = (
        _table(list(exclusions.items()))
        if isinstance(exclusions, dict)
        else "<p>Row exclusions were not recorded.</p>"
    )
    availability = record["availability"]
    notices = "".join(
        f"<li>{escape(label)}: {_text(values)}</li>"
        for label, values in (
            ("Not recorded", availability["not_recorded"]),
            ("Options not declared by components", availability["undeclared_options"]),
            ("Metadata values omitted", availability["omitted_metadata_values"]),
        )
        if values
    )
    return (
        f'<section id="result-{index}"><h2>{_text(record["dataset"])} · {_text(record["target"])}</h2>'
        f'<p class="subtitle">{_text(record["comparison_name"])}</p><div class="stats">{cards}</div>'
        f'<h3 id="folds-{index}">Scored folds</h3>'
        + _folds(record)
        + (
            "<h3>Fold attempts</h3>" + _fold_attempts(metadata["fold_attempts"])
            if "fold_attempts" in metadata
            else ""
        )
        + "<h3>Fit diagnostics</h3>"
        + _fit_diagnostics(metadata)
        + "<h3>Prediction-error diagnostics</h3>"
        + _prediction_diagnostics(record)
        + "<h3>Retained observation fields</h3>"
        + _observation_fields(record)
        + f"<h3>Recorded choices</h3>{choices}"
        + "<h3>Observation flow</h3><p>Exclusions apply sequentially after the dataset loader. "
        "Counts do not describe rows discarded upstream of the loader output.</p>"
        + exclusion_html
        + "<h3>Diagnostics and provenance</h3>"
        + _details("Split warnings", metadata.get("split_warnings"))
        + _details("Conditioning warnings", metadata.get("conditioning_warnings"))
        + _details(
            "Feature measurements and sources",
            {
                "features": metadata.get("feature_measurements"),
                "sources": metadata.get("feature_sources"),
                "target": response.get("target_measurement"),
            },
        )
        + _details(
            "Coefficients — means across folds; SEs are mean within-fit SEs",
            {
                "by_predictor": record["coefficients"],
                "coefficient": record["coefficient"],
                "coefficient_mean_within_fit_se": record[
                    "coefficient_mean_within_fit_se"
                ],
            },
        )
        + _details(
            "All recorded result metadata",
            {"response": response, "metadata": metadata, "folds": record["folds"]},
        )
        + "<h3>Information available</h3>"
        + (
            f'<ul class="notice">{notices}</ul>'
            if notices
            else "<p>The fields checked above are present. This does not establish complete provenance.</p>"
        )
        + "</section>"
    )


def render(report: dict[str, Any]) -> str:
    """Render escaped result text and a portable JSON download in one document."""
    payload = json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    navigation = "".join(
        f'<a href="#result-{i}">{_text(record["dataset"])} · {_text(record["target"])} · {_text(record["comparison_name"])}</a>'
        for i, record in enumerate(report["results"])
    )
    sections = "".join(
        _section(record, i) for i, record in enumerate(report["results"])
    )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(report["title"])}</title>
<style>
:root {{color-scheme:light; font:16px/1.5 system-ui,sans-serif; color:#20323b; background:#f4f7f8}}
body {{margin:0}} main {{max-width:1000px;margin:auto;padding:32px 20px}}
h1 {{font-size:2rem;line-height:1.2}} h2 {{font-size:1.5rem;margin-bottom:0}} h3 {{margin-top:30px}}
section {{background:white;border:1px solid #d9e2e6;border-radius:10px;padding:24px;margin:24px 0}}
.subtitle, .note {{color:#53646e}} a {{color:#12676e}} nav {{display:flex;gap:16px;flex-wrap:wrap}}
.stats {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}}
.stat {{background:#edf5f5;padding:14px;border-radius:6px}} .stat span {{display:block;font-size:.8rem}}
.stat strong {{display:block;font-size:1.4rem;overflow-wrap:anywhere}}
table {{border-collapse:collapse;width:100%;font-size:.9rem}} th,td {{padding:9px;border-bottom:1px solid #dde5e8;text-align:left;vertical-align:top}}
th {{font-weight:600}} th[scope=row] {{width:32%}} pre {{font: .85rem/1.5 ui-monospace,monospace;margin:0;white-space:pre-wrap;overflow-wrap:anywhere}}
details {{border:1px solid #d9e2e6;border-radius:5px;padding:12px;margin:10px 0}} summary {{cursor:pointer;font-weight:600}} details>pre {{margin-top:12px}}
.notice {{background:#fff6e6;padding:16px 16px 16px 36px;overflow-wrap:anywhere}} .scroll {{overflow:auto}}
.fold-chart {{max-height:28rem;overflow:auto;padding:6px 0}}
.fold-bar {{display:grid;grid-template-columns:4.5rem minmax(50px,1fr) 6rem;gap:10px;align-items:center;font-size:.8rem;min-height:27px}}
.track {{position:relative;height:26px}} .bar {{position:absolute;top:5px;height:16px}} .zero {{position:absolute;top:0;height:26px;border-left:1px solid #63717b}}
.prediction-models {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}
.prediction-models h5 {{font-size:1rem;margin:16px 0 0}} .prediction-models p {{font-size:.85rem}}
.diagnostic-plot {{width:100%;height:auto;overflow:visible;background:#f8fafb}}
.diagnostic-plot text {{font:11px system-ui,sans-serif;fill:#20323b}} .diagnostic-plot .plot-title {{font-size:13px;font-weight:600}}
@media(max-width:800px) {{.prediction-models {{grid-template-columns:1fr}}}}
@media(max-width:600px) {{main {{padding:16px 10px}} section {{padding:14px}} .stats {{grid-template-columns:repeat(2,minmax(0,1fr))}} .stat {{padding:10px}} .stat strong {{font-size:1.1rem}}}}
@media print {{body {{background:white}} details {{break-inside:avoid}}}}
</style><main><header><p class="note">REPLICA · Analysis diagnostics</p><h1>{escape(report["title"])}</h1>
<p>Recorded decisions, observation counts and scored comparisons. Missing information stays explicit.
Dedicated per-observation arrays are omitted; caller metadata is included. This summary is not a replay checkpoint.</p>
<p><a download="analysis-report.json" href="data:application/json;base64,{encoded}">Download report data as JSON</a></p>
{_observation_links(report)}
{_failure(report)}
{_details("Caller-supplied run context", report["context"])}
{_details("Context values omitted", report["context_omissions"]) if report["context_omissions"] else ""}
<nav aria-label="Results">{navigation}</nav></header>{sections}</main></html>"""
