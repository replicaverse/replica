# Named factors and grouped inference

After evaluation, use the [analysis report](analysis-reports.md) to inspect
recorded choices, observation exclusions, fold scores and fit diagnostics.

`AnalysisConfig.design` declares categorical factors separately from numeric
predictors. Each factor has a data source and either an `experimental` role
(eligible for CV and permutation units) or a `model` role (model terms only).
No sampling unit is inferred from a random-effects formula.

The named `Design` default holds out **both participants and stimuli**. Its
crossed folds test new readers on new materials. Each observation is tested
once, in the intersection of its reader and material partitions; training
excludes both partitions. Rows sharing just one held-out identity are unused
in that fit. An explicit single `cv_unit` selects a different generalization
target. An explicit `design=None` selects the scalar model interface for
alternative recipes; the ordinary configuration uses the named design.
The default `permutation_unit=None` leaves permutation inference undeclared:
a one-axis sign-flip test is not supplied automatically for crossed CV.
Selecting a permutation unit that omits a declared CV factor requires
`allow_scope_mismatch=True` at the test call. The test still warns and records
both scopes; this acknowledgement does not validate a conditional test.

This example uses the working structure in [getting started](../tutorials/getting_started.ipynb): participant and
word-occurrence intercepts, with independent participant slopes for the focal
family. The full model adds surprisal's fixed effect and participant slope;
the predictive baseline uses no surprisal term. A source identifies a complete
held-out material and need not be the modeling item. Additional source effects,
control slopes and correlations are separate modeling choices.
Independent slopes impose zero intercept–slope and slope–slope covariances;
they are the prespecified default simplification, not the theoretical
reference structure or evidence that a correlated fit failed. Consider the
design-supported correlated structure first and document any simplification
after checking identification and convergence. Using zero expected random effects
in predicted means does not remove the covariance assumption from estimation.
An unseen word occurrence has zero expected adjustment to its predicted mean.
Its random-intercept variance contributes to the marginal score and also affects
estimation of fixed effects and variance components.
For texts with enough within-text predictor variation and independent source
replication, a fuller structure can additionally include
`RandomEffect("source", slopes=("surprisal",), correlated=True)`. Control slopes
and correlations require their own justification. Adding these terms changes
the fitted models and requires refitting; the example is not a maximal model.

```python
from replica import AnalysisConfig, Design, evaluate, predictive_test

# dataset is loaded; features declares predictors, controls and response measures.
results = evaluate(dataset, features)

# Declare reviewed choices through the same configuration, when needed.
config = AnalysisConfig(design=Design(comparison="predictor"), train_fraction=0.8)
results = evaluate(dataset, features, config)

# Inference refits the models and takes substantially more computation.
inference = predictive_test(dataset, features, config)
```

`AnalysisConfig()` supplies ML fitting, raw responses, training-only
standardization, reproducibly shuffled crossed CV (seed 42, nominal 90% per
axis), complete predictor removal and marginal scoring. Its default random
structure uses participant intercepts and slopes for the focal columns in
`EvaluationSpec.predictors`, including declared lags, with correlations fixed to
zero; occurrence intercepts are shared across readers. Baseline controls do not
receive default random slopes. Supply `Design(random_effects=...)` to replace
that structure, including its covariance assumptions.

`Factor("item", "occurrence", role="model")` identifies each position by
stimulus identity, analyzed layer and position. It is built into `Design()`;
word spellings are not occurrence IDs and no metadata mutation is needed.
The default treats each stimulus as a complete source. For connected pages or
paragraphs, declare a parent-source factor and hold it out instead:

```python
from replica import Factor

config = AnalysisConfig(
    design=Design(
        factors=(
            Factor("participant", "participant"),
            Factor("source", "stimulus:source_id"),
            Factor("item", "occurrence", role="model"),
        ),
        cv_unit=("participant", "source"),
        comparison="predictor",
    ),
)
```

Nonoverlap of IDs alone does not establish independent source sampling: pages
from one novel do not supply replication across novels. Use `transform_dv=np.log`
for a log response scale and `per_fold=True, retain_per_obs_delta_ll=True` when
row-level diagnostic scores are needed. Both evaluation and predictive inference
resolve the same configuration against the focal predictor family.

## Choose the comparison

`Design.comparison` declares what removing a predictor means:

| Choice | Baseline random slopes | Question |
|---|---|---|
| `"predictor"` | Only slopes for numeric predictors in that model's fixed effects | Does adding the predictor's fixed effect and participant/item variation improve prediction? |
| `"fixed_effect"` | All declared slopes, including those for omitted fixed effects | Does adding the population-average fixed effect improve the model, given the same random structure? |

For overall held-out predictive contribution, use complete predictor removal.
In the optional new-texts/known-reader analysis, useful reader-specific effects
also count toward that contribution: set `cv_unit="source"` explicitly for the
declaration above (or `"stimulus"` for the built-in factor name).
Use `condition_on=("participant",)` to include learned reader effects; the default
`()` does not. A slopes-only baseline already predicts with those effects and
measures the additional predictive value of the average coefficient. For an
in-sample likelihood-ratio test of the population-average fixed effect, use
`"fixed_effect"`: fit both models by ML to identical rows, retaining the same
random structure. Include all focal lags in the removed family. Removing only
the current-word term asks whether it helps beyond spillover; the two
baselines have different likelihoods and depend on the generalization target.

Set the known-reader options together:

```python
from dataclasses import replace

known_reader_design = replace(
    config.design,
    cv_unit="source",
    condition_on=("participant",),
    scoring="residual",
    unseen="error",
)
# Let the declared single axis select its splitter.
known_reader_config = replace(config, design=known_reader_design, cv=None)
```

This predicts new sources for the observed reader panel. Every test reader must
have training responses. Evaluation warns once per response/comparison when a
modeled reader factor overlaps training and test but is omitted from
`condition_on`. The messages are retained in `conditioning_warnings` on the
analysis and any saved folds. Intentional zero-effect predictions remain
available. REPLICA does not automatically condition on every factor outside
`cv_unit`: word occurrences inside a held-out source are unseen too.

Complete predictor removal is the default. Leaving `comparison=None` uses
`"predictor"` and warns before feature extraction in every named `Design`
evaluation, even if no random slopes would change. The warning asks you to
review both models and the inference method. Explicitly selecting either mode
acknowledges that choice and silences the warning. `evaluate()` warns once per
comparison, rather than once per fold; direct `GroupedFrame` construction also
warns when resolving an unspecified choice.

To reproduce the previous shared-slope behavior, explicitly set
`comparison="fixed_effect"`. Existing callers that leave the choice unset now
get removal and a warning. For workflows that require an explicit choice before
computation, promote this warning to an error:

```python
import warnings

warnings.filterwarnings("error", message="Design.comparison was not set")
```

This setting is separate from `EvaluationSpec.comparison`, which chooses joint
or individual comparisons of the predictor set.

In `"predictor"` mode, `design.for_model(fixed)` resolves the declared slopes
against one model's numeric fixed predictor names. `GroupedFrame` applies this
resolution and removes unused numeric input columns before fitting. Random
intercepts remain. If removing slopes empties a slope-only term, that term and
its prediction conditioning disappear; its factor labels and sampling roles
remain. If no random terms remain, the `RLmerModel` adapter uses R's `lm` for
that model, retaining the requested residual-scale convention: RSS / n for ML
and RSS / (n - rank) for REML are the residual variances. The reported coefficient
standard errors use that same scale.

For example, with controls C, surprisal S and entropy E, declare candidate
participant slopes for S and E and use `comparison="predictor"`:

| Model | Fixed effects | Participant slopes |
|---|---|---|
| M0 | C | None |
| MS | C + S | S |
| ME | C + E | E |
| MSE | C + S + E | S and E |

Keep participant and item intercepts in every model. Include each predictor's
lags in both its fixed terms and candidate slopes when the analysis uses lags.
Test surprisal's contribution with M0 versus MS, and entropy's additional
contribution with MS versus MSE. Surprisal correctly remains in the baseline
for entropy addition: put it in `EvaluationSpec.controls`, with entropy in
`predictors`. Comparing MS with ME asks about replacement and is nonnested.
Match observations, preprocessing, training-derived scaling and material folds
across the models being compared. `evaluate()` matches rows within each pair;
separate calls for different predictor sets need an explicit common-row plan.
Each `evaluate()` call validates slope names against its declared features.
When reusing a larger candidate design, first select that call's full predictor
set with `design.for_model(...)`.

Holding random slopes fixed is appropriate for the distinct fixed-effect-only
question. In that mode the baseline still uses the predictor through its random
slopes, including in conditional predictions. Even with `condition_on=()`, the
training model's covariance structure still depends on those predictor values.

A word-occurrence intercept is shared across readers of the same source word.
If that occurrence is seen during training and its effect enters prediction,
the intercept can absorb word-level variation even in a baseline without the
numeric predictor. Holding out whole source materials and excluding word
effects from conditioning avoids using learned effects for the held-out words.
The models can also differ in fitted residual scale, so a log-score change need
not come solely from a change in predicted means.

### Choose the training fractions

The default crossed splitter targets **90% of distinct groups on each held-out
axis** in training. This is a practical training-size choice, not a statistically
optimal fold count. Fractions describe groups, not observations. The smallest
integer K with nominal training fraction `1 - 1/K` at least the requested target
is used on each axis. Thus 80% gives five folds and 90% gives ten.

```python
CrossedGroupSplitter()  # nominal 90% on every axis
CrossedGroupSplitter(train_fraction={"participant": 0.8, "source": 0.9}, seed=42)
CrossedGroupSplitter(n_folds=(5, 10), seed=42)  # explicit alternative
```

Set either `train_fraction` or `n_folds`. A scalar applies to every axis; a tuple
follows `Design.cv_unit` order; named fractions match those factor names exactly.
With `cv=None`, `AnalysisConfig(train_fraction=...)` and `AnalysisConfig(n_folds=...)`
use the same resolver. Without either override, a crossed `Design` uses the 90%
target. Legacy and single-axis analyses retain their ten-fold default.

| Reader/source targets | Folds | Approximate training rows | Fits per model per cycle |
|---|---|---|---|
| 80% / 80% | 5 × 5 | 64% | 25 |
| 80% / 90% | 5 × 10 | 72% | 50 |
| 90% / 90% | 10 × 10 | 81% | 100 |

The row fractions assume a balanced, complete reader–source grid. Whole groups
make the targets approximate: with twelve sources and ten folds, each fit trains
on ten or eleven sources. With four sources, at most three can train a model
that holds out a source. The splitter caps folds at the available group count
and warns when the requested target is unattainable. Complete K-fold CV supports
targets from 50% up to, but excluding, 100%; lower fractions require another
split design.

Inspect the actual layout before fitting:

```python
from replica.analysis import prepare_observations

observations = prepare_observations(dataset, features, config)
splitter = config.resolve_cv()
preview = splitter.preview({
    name: observations.factors[name] for name in config.design.cv_units
})
print(preview)
```

The preview reports requested and resolved counts, achieved group and row
fraction ranges, and nonempty test intersections. Multiply its
`fits_per_model_per_cycle` by the number of fitted models and complete cycles
to estimate work. Empty training rectangles fail before regression begins;
nonempty rectangles still need enough information to estimate the chosen model.
`evaluate()` logs the fold count and row-fraction range and records the summary
in `AnalysisResult.metadata["crossed_cv"]`, even without retained fold results.

Without a seed, sorted identities receive deterministic round-robin assignments,
preserving existing explicit-fold configurations. A seed shuffles whole groups
reproducibly. For repeated CV, use independently seeded complete cycles and
average their observation-weighted gains. Keep the resolved counts fixed within
a bootstrap analysis, keep all copies of an original identity in the same fold,
and record the bootstrap partition policy separately. This splitter alone does
not implement bootstrap resampling or inference.

A custom `cv` still runs its declared protocol; splitting a declared CV unit
across training and test warns and is recorded in result/fold metadata.

### Predict and score unseen observations

`condition_on=()` predicts means with zero expected random effects. Adding `"participant"`
uses learned participant effects. If a word-type random effect has separately
been declared, conditioning on it uses learned effects for recurring types;
that is a different structure from the occurrence-intercept example above.
`unseen="zero"` assigns zero expected effects to unseen random levels;
`"error"` rejects unseen levels among the conditioned factors.

`scoring="marginal"` is the default for the new-reader/new-source analysis.
Each observation has mean `X beta` and variance `sigma² + z G z'`, summing the
contributions of all fitted random-effect blocks. Unseen effects have zero
expectation but nonzero variance. Correlated blocks include their covariance
terms. The score averages individual Gaussian log densities; it can improve
through better predictions of the mean, variability, or both. This follows from
the [Gaussian LMM formulation](https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf),
equations 2–3. It is not a joint held-out likelihood, and estimated model
parameters are treated as fixed when scoring. Dependence and fitting uncertainty
still require the inference procedure described below.

Report the mean paired full-minus-baseline log score in nats per held-out
response, averaging across complete CV cycles when partitions are repeated.
A positive gain favors the full model; it is an effect estimate, not a p-value.
A wider predictive distribution is not automatically better: increasing variance
reduces the squared-error penalty but increases the log-variance penalty.

Marginal scoring requires every modeled random-effect level in the test set to
be absent from training and requires `sigma_source="training"`. It fails on
overlapping levels rather than applying an unseen-group variance to known
groups. For the separate known-reader analysis, explicitly select
`scoring="residual"`; this preserves the earlier plug-in residual-scale score.
Report the convention and retain earlier results under their original label.
Calibration of one score does not validate inference for another.
`integrated_loglik=True` remains the separate training-likelihood API and is
rejected on this path.

`Factor` also reads scalar `stimulus:key` / `datum:key` metadata and aligned
`unit:key` annotation-layer metadata. `word` uses exact word strings, preserving
case and punctuation. For lemmas or normalized types, declare a unit metadata
source. Missing factor labels and mixed scalar types fail; normalize mixed
metadata explicitly to avoid merging IDs such as `1` and `"1"`. Factors and
source row identities follow
feature, outlier and transformed-response masks; inspect them with
`prepare_observations(...).factors`.

List fixed categorical controls in `Design.categorical`. lme4 uses explicit
treatment contrasts with lexically sorted training levels; the first is the
reference. Unseen fixed categories and single-level training categories fail.
Use `correlated=False` on a `RandomEffect` for independent intercept and slopes.
Declare structural terms in `Design` only; conflicting legacy model options fail.

## Inference and provenance

The primary question is predictive improvement for new readers and sources.
The full-refit crossed bootstrap below estimates predictive uncertainty.
Its finite-sample corrections and t-reference are approximations; assess their
calibration for the sampling design and fold structure used in an analysis.
`evaluate()` returns scores. An explicit `predictive_test(dataset, features,
config)` call defaults to the full-refit crossed bootstrap for two unseen
sampling axes, complete predictor removal and marginal Gaussian scoring.
Unsupported targets raise before feature extraction. This is the primary
procedure for predictive superiority. Saved-score alternatives and the crossed
jackknife have different uncertainty targets and assumptions; fixed-effect
tests address population-average association.

```python
from replica import CrossedBootstrapConfig, predictive_test

inference = CrossedBootstrapConfig()  # 500 draws/series, C=2, M=8, seed=42, one process
results = predictive_test(
    dataset, features, config, inference=inference,
    checkpoint_dir="results/predictive-bootstrap",
)
```

To preview the cost without fitting, prepare one response and inspect its
resolved groups. Preparation can still require predictor extraction:

```python
import math
from replica import prepare_observations

observations = prepare_observations(dataset, features, config, target_name=features.targets[0].name)
plan = config.resolve_cv().preview({name: observations.factors[name] for name in config.design.cv_units})
print(plan)
print(inference.fit_budget(math.prod(plan["resolved_folds"].values())))
```

The bound is per comparison and response, before retries; multiply by the
number of requested tests when planning a run. Actual runtime also depends on
model size and convergence. Group counts in the preview use eligible rows.

The default budget rounds the assessment's 499 draws to 500; it is configurable,
not a universal precision guarantee. `n_jobs` selects independent processes.
Checkpointing is on by default: `checkpoint_dir="auto"` creates a unique
REPLICA user-cache directory, logged at INFO level before fitting.
`predictive_test` records its parent as `checkpoint_root`; pass that path to
the same function to resume all comparisons. `crossed_bootstrap_test` records
its single-test directory as `checkpoint_dir`.
An explicit directory, as above, makes a complete multi-comparison request easy
to resume. `checkpoint_dir=None` disables persistence. Checkpointing binds completed cycles to inputs, method, model options and budgets,
while permitting a different worker count on resume. The LMM's cached
coefficients from an earlier evaluation are reporting state, so they do not
change checkpoint identity; optimizer and model options still do. No scientific failures are
redrawn. The result contains `gain`, `standard_error`, `pvalue`,
`confidence_interval`, `available`, `reason`, variance components and cycle
provenance. Tests are two-sided and unadjusted for multiplicity.

For prepared `ObservationTable` inputs, use `crossed_bootstrap_test(table,
ComparisonSpec(baseline, added), config, ...)`. Both entry points refit from
observations. Training-only scaling and feature residualization repeat within
every fold. Extra random grouping factors must be nested in a sampled axis;
bootstrap copies of crossed word types require a separately specified extension.
Model fitting uses the declared ML backend. Unresolved convergence, dropped
numeric coefficients, empty training sets or unsupported variances make inference
unavailable. Singular fits are recorded without changing the model structure.

For the **paired crossed reader-and-source bootstrap with refitting**,
resample readers and whole sources independently, preserve sequences, and rerun
preprocessing and both ML fits. Repeat independently drawn complete CV partitions
within each resample, keeping all copies of each original identity together in
every cycle. Leakage prevention does not require retaining the original folds.
Use the same resample and observation weighting for both models. The
estimator reports reader-only,
source-only and joint variance estimates, partition-variance and finite-sample
corrections, and the evaluated finite-partition version: a nonnegative lower
variance bound, a finite-sample upper bound, and an approximate two-sided
t-reference with `min(R - 1, T - 1, M - 1)` degrees of freedom.
R and T count original readers and sources; M counts original-data CV partitions.
Bootstrap repetitions do not increase these counts. This adaptation is distinct
from the authors' original bounds and reference; its confirmation verdict remains
unsupported. Report pointwise intervals and Holm-adjusted p-values over a
declared comparison family, after validating the procedure for the design.
The original estimate must also average CV partitionings, with uncertainty from
its finite number of repetitions accounted for. A fixed-partition run remains a
separate analysis. The API retains fold counts resolved on the original identity
pool, draws fresh assignments over that entire pool in each cycle, and sends all
copies of an identity to the same fold. Identities absent from a resample still
occupy their assigned slots. Empty test rectangles are skipped; an occupied test
rectangle without training data makes inference unavailable. Counts never adapt
inside a resample. Realized training fractions can differ from the nominal target.

**Reporting and implementation:** `evaluate()` supplies predictive scores, not
an automatic calibrated test for the crossed target. Report the mean paired
gain, its units, observation/reader/source counts, exclusions, model structures,
training fraction, conditioning and convergence diagnostics. A reportable
estimate does not require a significance claim. Label exploratory uncertainty
by its method and assumptions; do not fill missing predictive p-values with a
coefficient test. Check calibration for the fitted statistic and numerical
precision for the chosen repetition budget; successful model fitting alone
does not establish either.

Three bootstrap series with B draws, C partition cycles and K folds require about
`6 * B * C * K` model fits per baseline/full comparison: 240,000 for B=1,000,
C=2 and K=20, before optimizer retries and original-data CV repetitions.
Shared models across contrasts can reduce the total. Reusing
saved scores omits changes in learned coefficients, random effects, residual
scale and preprocessing, including their effects across overlapping CV training
sets. A crossed jackknife refits after deleting clusters
and needs a justified two-factor variance estimator; it is not a jackknife over
nominally independent CV folds.

The null of equal expected predictive performance differs from an uninformative
focal predictor. Extra estimation can reduce expected held-out performance even
when the predictor supplies no information. A reliable negative gain means the
augmented fitting procedure predicts worse under the stated training protocol;
it does not make the predictor intrinsically harmful. A matched-complexity
stand-in baseline would define a separate comparison.

Changing `Design.comparison` changes the models being compared, not the
significance-test implementation. For an in-sample average-effect LRT, retain
the slopes and compare integrated mixed-model likelihoods from ML fits; the
named-design plug-in scores from `evaluate()` cannot be substituted for them.
`FullDataSplitter` alone does not implement this LRT. A separately specified
in-sample test removing fixed effects and random slopes together tests their
joint contribution. That test involves variance parameters on the zero boundary;
the usual fixed-effect chi-square reference cannot simply be reused. For nested
ML fits, a
[parametric-bootstrap comparison](https://hojsgaard.github.io/pbkrtest/reference/pb-modcomp.html)
simulates under the reduced model and refits both models to calibrate the
statistic. This is separate from inference about held-out CV performance.
A replacement comparison such as MS versus ME does not support a nested LRT.
`Design` selects neither an LRT nor a bootstrap automatically.

`grouped_permutation_test` sums held-out ΔLL within each declared permutation
unit, then flips whole sums. Unequal group sizes retain observation weighting.
An explicit `unit=` override warns and is recorded in the returned result.
If the selected unit omits any declared CV factor, the call raises unless
`allow_scope_mismatch=True`; explicit `Design.permutation_unit` declarations
do not bypass this check. Acknowledged mismatches still warn. The returned
`inference_scope` records CV units, prediction conditioning, the sign-flip unit
and unresampled CV factors, without rewriting the original analysis declaration.
`replica.multiverse.curve.stats.grouped_family_wise_signflip([result, ...])`
aligns observations by source
identity, requires identical observation sets and factor labels, and uses the
same group signs across specifications. It applies the same scope check to
every input and records `inference_scopes` in input order. For example,
`grouped_permutation_test(result, unit="source", allow_scope_mismatch=True)`
acknowledges an omitted reader axis but does not certify a new-text/known-panel
test: joint sign symmetry and dependence across sources still need justification.
For genuinely known-reader inference, retain the panel and assess source
resampling with refitting, or a conditional source-block sign flip under its
own assumptions. These procedures are not interchangeable variance formulas.
Matching test and CV factor names also does not establish inferential validity.
Older results without recorded CV units have `cv_scope_recorded=False`; their
scope cannot be checked automatically. The lower-level
`per_datapoint_permutation_pvalue` and `family_wise_signflip` accept an aligned
`groups=` vector. Grouped tests count floating-point ties consistently with
exact enumeration. For family-wise inference, a constant nonzero group effect
has infinite standardized magnitude; an all-zero effect has zero magnitude.
Compensated group sums and internal scale normalization stabilize inference
under row reordering and changes of loss units. Nonrepresentable group sums,
invalid sampling budgets and invalid significance thresholds fail explicitly.
Legacy ungrouped test conventions remain unchanged.

Grouped inference rejects missing losses, missing row identities and repeated
scored rows. Aggregate repeated-CV predictions explicitly before using this
interface. Grouping preserves dependence within a unit; validity still requires
joint sign symmetry across units. Shared CV training data can violate this.
Exact enumeration describes the computation, not a guarantee of inferential
validity. Use Monte Carlo for more than 24 units.

A grouped reanalysis asks whether a focal predictor adds predictive power
beyond a specified baseline. The choices it fixes are the response level
(individual responses or aggregates), the baseline formula, known versus new
groups, training-only scaling, grouped folds, score aggregation and the
inferential question; the sections above describe each one.

`AnalysisResult.metadata["design"]` records the candidate design, requested `comparison`
(including None when unspecified), and `effective_comparison`, alongside the
splitter and scoring policy. `FoldResult.metadata["baseline"]["design"]`
and `["full"]["design"]` record each resolved design. Fold metadata also records
formulas, column mappings, fixed estimates,
training category levels, conditioning, requested and actual optimizer settings, convergence messages
and singularity.
Retained ΔLL vectors carry aligned `factor_ids` and `row_ids`. Keep these fields
together when exporting results; a bare loss vector cannot establish alignment.
Row identities contain text participant IDs, stimulus IDs and unit positions;
only the participant ID may be empty, for aggregate observations.

## Backend support

| Backend | Explicit named `Design` | Existing interface |
|---|---|---|
| `RLmerModel` | Arbitrary factors, numeric random slopes, categorical fixed controls, selected conditioning | Preserved |
| `RustLmerModel` | Rejected | Participant/item; fixed-only prediction; configurable iteration bounds and tolerance |
| `RGamModel` | Rejected | Participant/item smooth terms and existing prediction policies |
| `BrmsModel` | Rejected | Participant/item; existing posterior predictive scoring |
| `RobustCrossedLmer` | Rejected | Existing two-factor fallback sequence |
| `StatsmodelsLME` | Rejected | One selected participant/item group |
| `MarginalLMEModel` | Rejected | Participant marginal likelihood; no arbitrary crossed factors |
| Other models / `analysis_fn` | Rejected | Preserved when `design=None` |

Unsupported backends fail before feature extraction; no declared term is dropped
or silently routed through a different backend. No existing study recipe or
stored result is changed by introducing `Design`.

Rust exposes `max_iterations`, `max_inner_iterations` and `tolerance` through
[`lme_python.FitControl`](https://pypi.org/project/lme-python/0.2.6/).
These are optimizer bounds, not a wall-clock timeout or a numerical-accuracy
guarantee. Version 0.2.5's known combination of uncorrelated slopes and multiple
grouping factors is rejected before native fitting. Inspect `last_diagnostics`
and validate coefficients and predictions against lme4 before using a new
Rust configuration for scientific comparisons.

The executable synthetic example and direct R agreement checks are in
`tests/analysis/contracts/test_design.py`; grouped inference/alignment checks are in
`tests/analysis/inference/test_grouped_permutation.py`.

Backend extensions implement the typed `GroupedModel` protocol in
`replica.analysis.contracts.grouped`; the lme4 adapter lives separately in
`replica.analysis.models.mixed_effects.grouped`. Coefficient arrays retain the order
`[intercept, *train.fixed]`, including NaN positions for dropped terms.
Named fits require finite training responses and predictions, a positive finite
residual scale, and aligned coefficient and standard-error arrays. Numeric
predictors and categorical fixed controls must have distinct names.

## Analysis defaults and explicit recipes

`evaluate(dataset, features)` and `predictive_test(dataset, features)` use the
same `AnalysisConfig()` defaults. The first estimates predictive scores; the
second runs full-refit crossed bootstrap inference. Use the ordinary constructor
to override settings. `transform_dv=None` keeps raw responses, and a callable
selects a transformation.

Reproductions declare the paper's estimator, splitter, scaling and response
transform. For example, an OLS recipe with item-only folds and log2 responses is
explicit:

```python
import numpy as np
from replica.analysis import AnalysisConfig, ItemSplitter, OLS

config = AnalysisConfig(
    model=OLS(), design=None, cv=ItemSplitter(10), transform_dv=np.log2,
)
```

The named Design is the normal mixed-model interface. An explicit `design=None`
selects the scalar interface used by OLS and existing paper-specific backends.
A backend that cannot implement a named Design fails before feature extraction.

Reading-time loaders retain their historical `exclude_first_unit=True` policy.
For loaders using the base mask, setting `dataset.exclude_first_unit=False`
lets requested feature lags determine boundary eligibility. A loader override
can impose additional exclusions. Eye-tracking `drop_skips=True` remains an
explicit loader policy; changing it changes the response population.

Result metadata records row counts after loading and alignment, exclusions due
to the response/dataset mask, missing features, the outlier filter and nonfinite
transformed responses. These sequential counts reconcile to the retained rows;
loader-masked skips cannot be separately recovered from other missing responses
at this stage. Responses already averaged upstream cannot be disaggregated by
the evaluator.

Named evaluation and predictive inference share a fit-acceptance check. They
reject unresolved optimizer/convergence errors and dropped numeric fixed terms;
boundary singularity is recorded without changing the declared model. Compact
fit diagnostics are retained even when `per_fold=False`; absent backend
diagnostics are marked unknown. Top-level scores are labelled
`marginal_gaussian` or `residual_gaussian`, with response-transform, backend,
ML/REML, variance-convention and scaling metadata. Evaluation of a direct
crossed splitter is one cycle; the inference point estimate averages the
configured original-data cycles.

`coefficient_mean_within_fit_se` and `Coefficient.mean_within_fit_se` describe
an arithmetic mean of per-fit coefficient SEs, not an SE of the mean coefficient.
The old names `coefficient_se` and `standard_error` on `Coefficient` retain their
values for compatibility. In historical multiverse payloads, `fold_gain_sd`
names the SD of fold means; `delta_ll_per_unit_se` and `delta_ll_se` are legacy
aliases for it. None of these summaries supplies predictive significance.

`multiverse.results.analysis_results.row_from_analysis` flattens an `AnalysisResult`
into a descriptive table row. Supply the model identity explicitly with `lm=`.
The formatter performs no inference and writes no files by default; its p-value
and fold-mean interval columns are NaN. An optional `inference=` callback receives
the `AnalysisResult` and returns any of `pvalue`, `pvalue_per_datapoint`,
`fold_dll_per_unit_ci_lo` and `fold_dll_per_unit_ci_hi`. Choose a method whose
assumptions match the evaluation design; the formatter does not validate those
assumptions. Callbacks cannot replace identity or descriptive columns.
To save retained observation deltas, supply an explicit
`per_observation_path=Path(...)`. A `ContrastRecipe` can select its own row policy
through `summarize=`, using the formatter's call signature.

The old multiverse label `sig_test="replica"` is a deprecated alias for
`legacy_per_observation_signflip`; its `cv_splitter="replica"` label denotes
historical ten-fold item CV. These labels do not select the current crossed
procedure. The historical live-fit significance route rejects crossed designs
before fitting. Use `predictive_test` for the declared crossed test, and report
its unadjusted p-values with an explicitly chosen comparison family and
multiplicity policy.

Explicit random-effects terms preserve their declared covariance.
