# Analysis reports

Use `write_analysis_report` after a scalar evaluation to inspect its choices,
observation counts, fold scores and available fit diagnostics:

```python
from replica import AnalysisConfig, evaluate, write_analysis_report

config = AnalysisConfig(per_fold=True, prediction_diagnostics=True)
results = evaluate(dataset, spec, config)
write_analysis_report(results, "analysis.html")
```

The HTML file opens offline, includes an embedded JSON download and needs no
plotting dependency. A report accepts one `AnalysisResult` or a sequence, including
results from different datasets and comparisons. Reporting reads existing results
without resolving features or fitting again. `per_fold=True` retains fold scores;
it does not change the fitted models. `EncodingResult` is not supported by this
scalar report.

Fit diagnostics are retained separately for each fold's baseline and full model.
The summary distinguishes reported convergence, unavailable convergence checks,
and direct fits where iterative convergence does not apply. Fixed-design rank,
random-effect singularity, selected tuning parameters and backend warnings are
shown when reported. A singular random-effect covariance is distinct from an
optimizer failure. Bayesian sampler checks are not a convergence guarantee.

`prediction_diagnostics=True` additionally records bounded summaries of the
scored prediction errors (`response - prediction`). Each fold has separate
baseline/full error-versus-prediction plots and error summaries. These use the
model response scale after transformations and fold preprocessing, including
when scores are reported on the original response scale. They use existing
predictions and do not fit again. Training/scoring overlap is recorded explicitly.

The normal-reference Q–Q plot divides errors by the supplied Gaussian scoring
scale. Without a supplied scale, it uses the finite scored errors' RMS and labels
that estimate; overflowed errors are excluded.
Backends that supply their own likelihood do not declare this Gaussian reference,
so their Q–Q plot is unavailable. Neither plot is a formal assumption test;
grouped or repeated observations can remain dependent.

Prediction diagnostics require `per_fold=True` and the built-in analysis. They
are off by default. Capture uses deterministic bins and selected order statistics,
with no random sampling. Retained size is bounded per fit and grows with the
number of folds; capture temporarily sorts scored values. The report records the
binning method, counts and any nonfinite errors. Summaries can reveal individual
values in small groups and at distribution extremes.

The report distinguishes the **feature model** that supplies predictors from the
**regression backend** that fits responses. It shows baseline controls and added
predictors, the response transform and scoring scale, declared backend and CV
options, response filters and training-fold preprocessing. Named designs and
feature provenance are available in the expandable metadata. Row exclusions are
sequential counts after the dataset loader, not counts of the original corpus.

Fold gains are descriptive. They are not independent replicates or confidence
intervals. Log-likelihood gains are in nats even when the response uses a log₂
transform. Headline metrics retain the analysis's aggregation convention:
repeated-CV totals may be normalized, and summed fold counts need not count unique
observations. A splitter that scores its training data is not an out-of-sample
evaluation. Coefficient SEs are the mean within-fit SE, not the uncertainty of
the across-fold mean.

Missing configuration, undeclared component options and unavailable diagnostics
are listed explicitly. GAM/BAM subprocesses do not currently return optimizer
diagnostics; custom backends can expose the [optional diagnostic contract](../../src/replica/analysis/README.md).
A report does not certify complete provenance or convergence.
Dedicated observation and loss arrays are omitted. Full response, prediction and
residual vectors are not retained for these plots; caller metadata is included
and may contain such data.
Arrays in metadata are summarized by shape and dtype, and unsupported objects
by type. Their omitted paths are listed; nonfinite numbers retain explicit tags.

For a separate machine-readable observation export, explicitly retain predictions
when evaluating and supply a CSV destination when reporting:

```python
config = AnalysisConfig(per_fold=True, retain_predictions=True)
results = evaluate(dataset, spec, config)
write_analysis_report(
    results, "analysis.html", observations_path="observations.csv"
)
```

This writes `observations.csv` and `observations.csv.json`. The JSON sidecar contains
the same structured report record as the HTML download, plus the CSV schema, row
count and SHA-256. The HTML shows a compact inventory of retained fields; it never
embeds the observation arrays. Retention uses existing scored responses and
predictions, without another prediction or fit, and requires the built-in analysis
with `per_fold=True`. It increases result memory in proportion to the number of
scored rows across folds. Every requested result and fold must have retained
observations before export; incomplete retention raises an error.

Each CSV row identifies its result, fold and position within that fold. Baseline
and full predictions occupy separate columns. `observed`, `prediction_baseline`,
`prediction_full`, `error_baseline` and `error_full` use the model response scale
after transformation and fold preprocessing. Errors are derived during export as
observed minus predicted. The sidecar records the original target measurement and
unit, response transform and preprocessing; transformed values are not described
as original-unit measurements. `in_train` identifies scored rows also present in
that fold's training data.

`row_key`, `row_id` and `factor:*` cells are JSON, preserving labels and types.
`row_key` is the exact canonical observation key, with dimensions declared in
`observation_capture.row_key_names`; `row_id` gives readable participant/stimulus/unit
labels and need not be a complete identity for custom observation anchors.
`prepared_row_index` identifies the row within the recorded prepared cohort.
Readable labels follow `observation_capture.row_id_columns`. Empty CSV cells mean a
field was not retained for that fold; JSON `null` is a retained missing identity.
Numeric nonfinite values use `nan`, `inf` and `-inf`. `delta_loglik` is populated
only when `retain_per_obs_delta_ll=True` and the backend supplies aligned losses;
it is full minus baseline in nats on the recorded scoring scale. These rows are
not normalized for repeated cross-validation and can repeat the same observation.
Do not interpret their sum as a repeated-CV-normalized headline metric or treat
repeated rows as independent replicates.

The CSV is streamed to disk; no row-sized lists are built for HTML. Export validates
array alignment and stages all files before replacing destinations. Keep the CSV
and sidecar with the HTML if its export links should remain usable. Observation
exports include response values and identifiers, so share them deliberately.

Failures keep their original exception. When an exception includes the compact
`analysis_diagnostics` attachment, it can be reported without inventing scores:

```python
try:
    results = evaluate(dataset, spec, config)
except Exception as error:
    failure = getattr(error, "analysis_diagnostics", None)
    if failure is not None:
        write_analysis_report(
            [], "failed-analysis.html", context={"failed_analysis": failure}
        )
    raise
```

An empty result sequence is accepted only with an explicit failed-analysis record
whose `status` is `"failed"`. The report labels it as caller supplied and retains
its diagnostic facts separately from completed results. It does not create
placeholder metrics or permit observation export when there are no results.

Applications can attach source and input revisions without changing the analysis:

```python
write_analysis_report(
    results,
    "analysis.html",
    title="Reading-time analysis",
    context={"source_commit": source_commit, "input_revision": input_revision},
)
```

Context is labelled as caller supplied. The function does not discover revisions,
capture environments or save model artifacts, so the report is not a replay
checkpoint. The destination's parent directory must exist. Existing files at the
destination are overwritten; custom metadata must be acyclic. Review metadata
before sharing a report, since applications may include paths or identifiers.
