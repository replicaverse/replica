# `replica.analysis`

`evaluate()` compares models on aligned observations. `AnalysisConfig` selects
scalar-response analyses; `EncodingConfig` selects sampled multichannel responses
such as fMRI, EEG, MEG, and ECoG. `benchmark()` applies the same entry point across
caller-supplied datasets and feature models.

| Package | Responsibility |
|---------|----------------|
| `specification/` | Feature roles, observation cohorts and fitting settings |
| `data/` | Indexed observation alignment, eligibility filters and event preparation |
| `contracts/` | Experimental designs, event and result records, and backend protocols |
| `fitting/` | Training-fold transforms, regression fitting, scoring, and fold arrays |
| `evaluation/` | Scalar and sampled-response evaluation, and benchmarking |
| `reporting/` | Portable summaries and visualizations of recorded scalar results |
| `models/` | Concrete regression backends, grouped by statistical family |
| `cv/` | Resampling splitters and training-fitted scalers |
| `inference/` | Predictive tests, uncertainty summaries, and effect sizes |
| `crossed/` | Frozen-table mixed-model fitting, inference, and verification |

Call `evaluate` and `benchmark` through `replica` or `replica.analysis`, or import
them from `replica.analysis.evaluation`. Lower-level components do not import
`evaluation`: observation preparation, backend fitting, and inference can be used
independently. Applications supply datasets, predictors, and comparison grids.

`write_analysis_report(results, "analysis.html")` writes an offline report of
scalar results, with a JSON download. It reads recorded choices, exclusions,
scores and diagnostics; it does not rerun evaluation. See the
[reporting guide](../../../docs/guides/analysis-reports.md) for usage and limits.

## Regression models

Backends implement one of three contracts:

| Contract | Configuration | Required methods |
|----------|---------------|------------------|
| `AnalysisModel` | `design=None` | `fit_predict(y_train, X_train, X_test)` supplies predictions for Gaussian scoring |
| `LikelihoodModel` | `design=None`, backend sets `owns_loglik=True` | `fit_score(y_train, X_train, X_test, y_test, is_full=...)` supplies its own score |
| `GroupedModel` | A named `Design` | `validate_design(design)` checks support; `fit_grouped(y_train, train, test)` returns a `GroupedFit` from named `GroupedFrame` inputs |

The matrix contracts live in `contracts/models.py`; named-frame contracts live
in `contracts/grouped.py`. A backend may implement more than one contract.
Designs and event records live in `contracts/design.py` and `contracts/events.py`;
these contracts do not import fitting configurations or concrete backends.
Named-design dispatch takes precedence. Otherwise `owns_loglik=True` selects
`fit_score`; this path does not require `fit_predict`. The scorer is called for
both baseline and full models and returns `FoldScore` or `(loglik, predictions)`.
`FoldScore.per_obs_loglik` is optional for scores without a pointwise decomposition.

Matrix backends can declare `needs_grouping=True` to receive participant and item
codes as two trailing design columns. Likelihood backends with `needs_events=True`
also receive `events`, `train_mask`, and `test_mask`. These flags default to false.

For Gaussian scoring, optional `fit_predict_with_sigma` returns
`(predictions, sigma)`; `StatsPredictor.fit_predict_with_stats` returns
`(predictions, beta, se, sigma)`. A backend with the latter method can set
`provides_stats=False` to use its prediction path. Without a scale method,
training residual variance uses the fixed-column count plus an intercept,
excluding grouping codes; other effective degrees of freedom require a
backend-provided scale. An empty control set uses a training-mean baseline.
Likelihood backends may expose `last_fixed_effects()` for their latest fit's
`(beta, se)`. Coefficients follow design-column order, with the intercept first.

Matrix and likelihood backends can implement `last_fit_diagnostics()` to return
a compact mapping for their latest successful fit, or `None` when unavailable.
Clear it before each fit attempt, and derive values from that fit's outputs.
The evaluator snapshots it immediately after each baseline/full call. Named
backends instead return the mapping in `GroupedFit.metadata["fit_diagnostics"]`.
Keep diagnostics separate from `configuration()`, which describes chosen options.
Provider exceptions and invalid return types are errors; they are not silently
treated as missing diagnostics.

Common diagnostic fields are `converged`, `convergence_applicable`, `singular`,
`rank`, `n_columns`, `selected_parameters`, `warnings` and `details`. Missing fields
remain unknown; set `convergence_applicable=False` for direct solutions. Rank and
column count must describe the same fitted fixed-effect design, including its
intercept and excluding group-ID columns. `diagnostic_checks_passed` describes
backend-specific checks, such as sampler diagnostics, without claiming convergence.
Use small JSON-compatible values; do not return fitted objects or observation
vectors. Custom fields remain available in the [analysis report](../../../docs/guides/analysis-reports.md).

| Backend group | Implementations | Optional requirements |
|--------|----------|-----------------------|
| `models/fixed_effects/` | NumPy OLS, statsmodels OLS, scikit-learn ridge/elastic net, R `lm` | R `lm` requires `replicaverse[r]` and R |
| `models/mixed_effects/` | lme4, statsmodels, lme-rs, brms, marginal and robust crossed models | R backends require their R packages; lme-rs requires `replicaverse[lmer-rs]` |
| `models/additive/` | mgcv GAM and BAM | `replicaverse[r]`, R and mgcv |
| `models/deconvolution/` | Native CDR and CDRNN | `replicaverse[models]` |
| `models/support/` | R conversion, residual-scale adapters, and scikit-learn support | Loaded by the selected backend |

`fixed_effects` and `mixed_effects` distinguish effect structure: both contain
linear regression models. The additive and deconvolution groups own specialized
implementations that can also include random effects.

R scripts live beside the backend that uses them. BAM runs in an Rscript
subprocess; brms requires brms, cmdstanr, posterior, jsonlite, and CmdStan.
Marginal LME additionally requires mvtnorm and optimx. Optional dependencies are
loaded when a backend fits, so importing `replica.analysis.models` does not
require R or lme-rs.

## Imports and checkpoints

The package-level exports from `replica`, `replica.analysis`, and
`replica.analysis.models` are available independently of the implementation tree.
Direct implementation imports follow the responsibilities listed above.

Inference restart manifests bind the analysis source, prepared data, and numerical
settings. A changed source hash prevents resuming an existing checkpoint. Resume
historical runs with their recorded source commit and environment, or start a new
checkpoint under the changed source.

## Response filtering

`AnalysisConfig(outlier_method=...)` accepts `"mad"`, `"sd"`, or a callable.
A custom callable receives finite raw responses for one participant group and
returns a boolean mask with the same shape. For example,
`lambda values: values <= np.quantile(values, 0.99)` selects a groupwise upper
quantile. Explicit `outlier_floor` and `outlier_ceiling` combine with that mask;
`outlier_k` applies only to MAD and SD. Without participant labels,
`OutlierFilter.keep_mask()` applies the rule globally.

Filtering precedes the response transform and cross-validation. It is an
explicit response-eligibility policy, not a training-fitted preprocessing step.
Configuration metadata records the callable's qualified name; applications must
record its captured parameters or source when reproducing a custom rule.
Inference checkpoints bind the complete prepared observations, including row
identities, rather than relying on that descriptive name.
