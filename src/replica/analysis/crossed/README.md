# Crossed mixed-model analysis

This package provides fitting, prediction, and verification for frozen observation
tables and explicit model specifications. Dataset selection, predictor choices,
comparison families, and study-specific protocols belong to the calling study or
application.

| Package | Responsibility |
|---------|----------------|
| `specification/` | Random-effect identifiability audits and frozen holdout preparation |
| `fitting/` | R fitting and prediction engines for Gaussian, binomial, and ordinal models |
| `inference/` | Contrast families and held-out loss comparisons |
| `verification/` | Independent likelihood and prediction checks, plus fresh-process refit replay |
| `evaluation.py` | Evaluation of a caller-declared comparison inventory |

`verification` exposes `verify_fit()` and `verify_predictions()`. Its command line
selects the check: `python -m replica.analysis.crossed.verification fit --help`
or `python -m replica.analysis.crossed.verification predictions --help`.
`python -m replica.analysis.crossed --help` prepares holdouts or compares losses.

This package operates on frozen tables and explicit specifications. By contrast,
`analysis/cv/group_holdout.py` implements `CrossedGroupSplitter`, a reusable
resampling component for `evaluate()`. The two entry points have separate split
settings, described below.

Archive authentication, inference replay and export belong to the integration
repository's [archive tools](https://github.com/replicaverse/replica-dev/tree/main/scripts/archives/crossed).
Calibration simulation drivers live with the [analysis tests](../../../../tests/analysis/calibration).

## Observations and random effects

Use actual participant, material and item identities. Related passages and stimulus
versions share a material group. Rating-list positions are not participant IDs.
Retain participant intercepts and identifiable within-participant slopes, material
intercepts and within-material slopes, and item intercepts for repeated items.
A predictor constant within an item cannot have a by-item slope. Audit within-group
variation and pooled design rank before fitting; record requested, retained and
unidentified slopes. Random effects are diagonal, and paired models share the same
observations and random-effects design. Convergence failure never silently removes
slopes. Save variance estimates, singularity and optimizer diagnostics.

Gaussian models use maximum likelihood, binary models use a binomial logit GLMM,
and ordinal ratings use a cumulative-logit mixed model. The latter two use Laplace
likelihoods. Response scales and source controls are study-specific. Nested
fixed-effect association tests use the actual estimable rank difference and an
asymptotic chi-square likelihood-ratio reference. Reject nonnested designs, failed
convergence and worse full-model likelihoods. These tests are approximate,
especially with few groups or boundary variances; they are not parametric-bootstrap
or predictive-superiority tests. See [lme4 inference guidance](https://lme4.github.io/lme4/reference/pvalues.html).

## Independent holdouts

Frozen-table `split()` and `prepare()` accept `fraction` and `seed`; their defaults
hold out 25% of readers and whole materials using identity hashes with seed
20260927. The CLI exposes these as `prepare --fraction ... --seed ...`.
These settings are independent of the higher-level `evaluate()` workflow and
its `CrossedGroupSplitter` default of 90% training per axis.

Train on the remaining-reader/remaining-material rectangle and evaluate
only the held-out-reader/held-out-material rectangle; the two other rectangles are
unused. Audit random-slope identifiability again on training data. Learn scaling,
imputation and any data-derived selection on training observations only. Paired
models use identical observations and splits. Repeated folds or seeds are never
independent observations in a significance test.

Gaussian predictions use population fixed effects and squared error on the declared
response scale. Binary and ordinal predictions integrate unseen random effects and
use log loss. A single-book corpus cannot establish new-book predictive performance;
its conditional association analysis remains possible.

### Table schema

The observations table (TSV; `participant`, `material` and `item` are read as
strings) holds one row per observation with those three identity columns
and the response and predictor columns the specification's formulas name.
`participant` and `material` must be complete; `item` joins predictions back
to observations, so a `(participant, material, item)` key is unique. `split()`
adds `role` with the values `train`, `test` and `unused`.

The input `specification.json` declares `family` (the response family),
`models` (model name to formula terms) and `random_effect_audits` (entries
with `group` and `requested`). `prepare` writes `observations.tsv.gz`,
`training.tsv.gz`, `specification.json` (the input plus `random_terms`, the
audits recomputed on training rows, `bootstrap_samples` and `status`) and
`source.json` (protocol `independent-readers-materials-v1`, seed, fraction per
axis, held-out identities, row counts per role and file hashes).

The predictions table holds one row per test observation keyed by the three
identities, a `<model>_loss` column per declared model and the family's
prediction columns (for an ordinal response, `<model>_probability_<category>`).
`compare` reads `<baseline>_loss` and `<target>_loss` for every comparison
`{name, baseline, target}` in the comparisons file.

## Predictive uncertainty and multiplicity

Inference is conditional on the recorded training models. Resample held-out readers
and whole materials independently, retaining their intersection observations
(9,999 pigeonhole-bootstrap draws by default; seed 20260927). Declare a larger
budget when its minimum p-value would preclude any rejection in the complete
Holm family: 630 comparisons require more than 12,599 draws and use 99,999.
Report the realized budget, group counts and failed draws. The one-sided p-value is the larger of the centered-bootstrap value and a
Student reference using the bootstrap standard error and
`min(readers, materials) - 1` degrees of freedom. The two-sided interval encloses the
basic-bootstrap and Student intervals. This is an approximate finite-group
safeguard, not an exact test or an estimate of training-sample uncertainty.
See [Owen (2007)](https://doi.org/10.1214/07-AOAS122) and the
[multiway degrees-of-freedom reference](https://faculty.econ.ucdavis.edu/faculty/cameron/research/CGM_twoway_ALL_13sept2006.pdf).
The combined safeguard is an explicit implementation choice. Independent null
simulations in `tests.analysis.calibration.crossed` accompany inferential summaries with their
full draws and group counts; they do not establish universal calibration.

Freeze comparison and multiplicity families before inspecting their test results.
Apply Holm correction within each complete family; structurally unavailable tests
retain p=1 in its denominator. Set `alpha` (default .05) and the bootstrap `seed`
in the loss-evaluation call or through `compare --alpha ... --seed ...`.
Frozen manifests require `alpha` at the top level. Each available prediction cell
requires `bootstrap_draws` and a nonnegative integer `bootstrap_seed`; missing
settings are errors. Unavailable cells need no resampling settings.
The declared alpha governs draw-budget checks and family decisions; receipts
retain it and reject a changed alpha during summary.
Confidence intervals remain 95% pointwise intervals.
A descriptive ranking does not establish significance,
and predictive improvement alone does not establish a directional association.
Original-data analyses remain conditional on historical estimator/eligibility
selection: a later split cannot undo selection using the same outcomes. New-data
analyses freeze those choices.

## Verification and publication

Authenticate observations, specifications, producer code and output hashes. Refit
in a fresh process using the recorded historical engine; replay prediction and
inference from those new fits. Independently reconstruct Gaussian likelihoods and
fixed effects at the fitted variance parameters, and verify prediction arithmetic.
This numerical reconstruction does not independently optimize variance parameters.
The integration archive tools export complete families with authenticated fit
and inference replay receipts. Preserve unavailable cells and source limitations explicitly.

An interrupted production batch may resume completed models into a new output
directory using the optional fourth argument to `fit.R`. Reuse requires
matching transformed observations, formula, likelihood and optimizer settings;
convergence checks still apply. Keep the partial outputs and resumption receipt.
Independent verification starts fresh and must never reuse a production fit.

The optional fifth argument selects one declared model (`-` as the fourth
argument starts it from scratch). Scaling still uses the complete frozen design,
including all declared models and random slopes. A partial batch cannot run
paired bootstrap tests or close a comparison family. Assemble all component
models through the same strict validation before calculating comparisons.
Verification may also schedule models separately, provided every component is
optimized from scratch in a separate directory from production and every final
fitted state and inference result passes the recorded replay checks.

Published prerequisites include source identities, predictor settings and draws,
model specifications, split seeds, runtime versions, diagnostics, test arrays and
replay receipts. Preserve immutable historical evidence and its producer identity;
remove superseded mutable files only after checking active references.
