# Contributing

Create a branch from `main`, add the implementation and relevant validation,
and open a pull request. Follow the [development setup](requirements/README.md)
to install the declared dependencies in your environment. The same guide explains
how to reproduce the pinned Linux CI setup.
The [installation guide](docs/guides/environment.md) covers library capabilities
and study environments.
Run `replica check` and the applicable tiers in the [test guide](tests/README.md)
before submitting. New studies also need a reproduction test and a README in
their study module.

Keep workstation-specific ignore rules in Git's `info/exclude` file
(`git rev-parse --git-path info/exclude`). Shared generated files belong in
`.gitignore`.

## Docstrings

Use Google-style docstrings for the quartodoc API reference. Every module and
public class, function and method needs a descriptive summary. Functions with
parameters must document them in `Args:` without listing names absent from the
signature. Describe parameter meanings; types come from annotations. Use only
sections supported by the reference renderer.

`replica check` runs `scripts/checks/source/docstrings.py` for coverage and
structure, and `scripts/checks/source/references.py` for repository paths.
Keep comments and test descriptions concise: state the behavior or constraint
being checked. Implementation history belongs in Git. Cite primary sources for
scientific claims and keep study and dataset citations with their metadata.

## Reporting study results

A reconsideration revisits a published study along two axes: **Repetition**
versus **Reanalysis** describes how the data are analyzed, and **Reproduction**
versus **Replication** describes whether the data are original or new.
**Robustness** separately asks whether findings survive reasonable analytic
choices.

- **Original** records significance reported in the original paper, with the
  full tested cell set for the declared comparison. Numerical improvements,
  rankings and effect directions do not establish significance. An unavailable
  significance test is labelled **Not reported** and is excluded from the
  nonsignificant count.
  For a paper claim without a significance test, label its registered
  **Numerical evidence**, **Directional evidence** or **Descriptive evidence**
  explicitly. Missing significance does not block those comparisons.
- **Reproduction / repetition** recomputes the paper's experiments on the original
  data, retaining released
  defects. Reading released results is not recomputation. A refit conditional on
  released predictors must state that scope and any generation or selection
  steps that have not been repeated.
  Missing models, corpora or experiments require a partial-coverage label.
- **Reproduction / reanalysis** rigorously re-examines the original data,
  including fixing clear bugs, controlling confounds and correcting
  anti-conservative statistical practices. Keep the faithful
  repetition alongside the corrected result. Every registered correction must
  be implemented before describing a run as fully corrected; missing corrected
  evidence must never fall back to a defective result.
- **Replication / repetition** applies the released procedure to new corpora.
  Register fresh runs separately; results from a corrected procedure cannot fill it.
  A refit conditional on frozen predictors must identify those inputs and any
  predictor-generation changes, just as original-data repetition does.
- **Replication / reanalysis** applies the corrected procedure to new corpora, preserving
  the language model and checkpoint, predictor and estimator, response,
  preprocessing, controls, lags, regression, folds, statistical test and
  multiplicity correction. Changes to other analysis choices are **Robustness**
  checks, including transfers that also substitute a model or response.

In the registry and the study READMEs, the stage names map to the two axes as
follows: `repetition` maps to reproduction / repetition, `reproduction` to
reproduction / reanalysis, and `replication` to replication / reanalysis. The
website renders these results under both axes.

### Study organization and review

Use the study's canonical identifier in its test filenames and external entry
points. Name shared modules for their responsibility.

Keep reusable implementation in the study package and shared logic in existing
shared modules. Document each procedure in one concise `repetition/README.md` or
`reanalysis/README.md`. Bind execution archives and their retained source metadata
in `results/index.json`. Publish result payloads on HF under
`results/{reproduction,replication}/{repetition,reanalysis}/<analysis>/<record>/`.
Paper and released-output comparison targets go under
`results/references/{paper,released}/<analysis>/<record>/`. Every record has a
complete manifest in `results/index.json`. The checkout contains that index and
the generated results README; the indexed payload hierarchy lives on HF.
Descriptions are the result link labels. Name the calculation and distinguish
its inputs, measures or configuration from other records in the same section.
Descriptions must be unique within each data and analysis stage. Record IDs and
checksums remain in the manifest and download paths.
Declare necessary source inputs and documentation figures in the index's
`resources` mapping, with exact paths, roles, descriptions and SHA-256 digests.
Numerical outputs belong in indexed result records, including tables used to
draw figures. Package validation checks all tracked artifacts, including files
outside `results/`. Operational inventories and action tables belong with the
external coordination records.
A directory or an archived replay does not establish that its scientific scope is complete.

Store work logs, scheduling notes, handoffs and one-off audit output with the
external run or coordination record. They do not belong in the source package.
Preserve useful acquisition and replay instructions, exact source versions and
unresolved scientific limitations in the maintained documentation. Keep full
run payloads in the results store and reference their immutable revisions
and hashes; organization changes must not rewrite their producer identity.

Review the combined PR diff for naming, placement, duplication, working notes,
broken references and scientific classification, as well as numerical correctness
and tests. In both data axes, corrected predictors belong to **reanalysis**, even
when the statistical estimator is unchanged. State which corrections remain
unimplemented; a corrected predictor alone does not complete the reanalysis.

### Storing and publishing a run

A job's output directory is temporary. When a run is finished, save it into
the results store, publish it, and open a pull request with the index entry:

```bash
export REPLICA_RESULTS_STORE=/path/to/store   # durable, not purged
replica results save RUN_ID --study STUDY --from OUTPUT_DIR \
    --input corpus=SHA256 --command "python -m replica.studies.STUDY ..."
replica results publish RUN_ID --study STUDY \
    --run replication/reanalysis/reading_time_prediction/r0123456789abcdef \
    --description "Reading-time fits with corrected predictors" \
    --role evidence
```

`save` copies tables, intermediates and logs into `<store>/RUN_ID/` and writes
`MANIFEST.json` with a digest per file and the `produced_by` stamp: the
package commit, the digests of the inputs you name, and a footprint over the
study package and the `replica` modules it imports. A run whose content
depends on a few files only, such as frames frozen from the authors' code,
names them with `--depends-on studies/STUDY/freeze.py` (repeatable); the
footprint then covers those files and nothing else. `publish` uploads the
tables to the study index's `hub_repo` (default `replicaverse/replica-study-bundles`)
and records the run in the study's `results/index.json` with its
role, store id, hub digests and stamp. Code reads any of these files through
`replica.studies.records.files.results_file`, which looks in the checkout,
then the store, then the hub; `replica results fetch STUDY` warms the cache.
Omit `STUDY` to fetch and checksum-verify all indexed payloads. `replica results
path STUDY RUN [FILE]` prints the resolved cache path for another command.
`REPLICA_RESULTS_HF_REPO` selects an alternative repository for reads or an
unpublished index. Publishing an existing index cannot move its payloads to
another repository.
The claim registry cites indexed artifacts with their extraction rules and hashes.
Keep small synthetic or independent regression fixtures under `tests/`; the
repository policy check rejects tracked result payloads even with an index entry.

Store full frozen input tables in a study bundle on Hugging Face. In the study
index, `resources[PATH].archive` selects a digest in `archives`; its `reference`
pins the repository, revision and archive checksum, and `records` maps member
paths to their checksums. Load an input with
`replica.studies.records.resources.resource_file(STUDY, PATH)`, which verifies
the bytes and returns a local path, fetching the bundle when needed.
Record consumed resource checksums with `save --input LABEL=SHA256`; an index
binding describes available inputs, while a run records the inputs it used.

`replica results stale` compares current code with each run's latest source
fingerprint; the light test tier runs the same check. Rerun affected analyses
or assess source compatibility and record it with `replica results stamp STUDY
RUN --validate --input REVIEW=SHA256`. This preserves the original producer and
records the assessed source and evidence hashes. A validation stamp does not
certify a scientific rerun or numerical agreement. Staleness uses the footprint;
the recorded commit identifies the source revision.

Replication defaults to **every eligible corpus in the catalog**. Enumerate the
corpora and exclusions before fitting. Eligibility requires compatible stimulus
language, the claim's response and the inputs needed by the corrected procedure.
For an English GPT-2 recipe, use all eligible English corpora. For a multilingual
mGPT recipe, use all eligible corpora in languages covered by that same model
and its controls. Multilingual model support alone does not establish that
lexical controls or language-specific manipulations are valid.

Original corpora and duplicate releases do not count as new replications.
Separate cohorts sharing materials must be identified as such. Missing data,
missing predictors and unresolved input defects remain explicit gaps; they
cannot silently reduce the planned corpus universe. A corpus default response
must not replace the claim's response. Applying an English model and English
controls to another language is a robustness check.

Keep numerical comparisons beside significance counts, with the tested scope,
statistical criterion and denominator explicit. Do not infer significance from
a numerical match or calculate percentages against unavailable Original tests.
Study overviews retain one compact summary row with explicit yes, partial and
no claim totals, plus unavailable results. Do not hide partial results inside
a zero-success fraction. Expanded rows retain observed counts, units, coverage
and claim-level criteria; never combine heterogeneous comparison units or treat
a positive cell count as confirmation of the whole claim. Null outcomes stay in
the denominator. An unidentifiable effect stays in the
requested inventory as **Not estimable**, with a diagnostic reason and no
invented coefficient or p-value; list it outside the significance denominator.
Convergence failures remain incomplete until resolved.

Keep each headline claim at the scope of the paper's own assertion and test.
A pooled significance test does not establish separate subgroup significance;
subgroup calculations remain supplementary unless the paper makes and supports
that separate claim. Preserve a descriptive claim's numerical evidence when the
paper reports no significance test.

The registry's `comparand.significance: false` selects a comparison without a
significance test; it does not mean missing evidence. For statistical decisions,
the optional `by_cell` mapping specifies which values count as significant in
each cell. A descriptive direction or failure to reject cannot become a hit.
Label agreement counts as **numerical matches**, **directional matches** or
**decision matches**. `comparand.reporting_basis: sign` identifies directional
decisions stored as Booleans; otherwise the basis follows `comparand.match.kind`.
These labels do not change the comparison rule or its tolerance.

For corrected reading-time comparisons, keep observations from the same source
material in one test fold, including pages or versions of the same text. Paired
models must score matched observations. Report attainable test resolution and
account for dependence from overlapping training sets; enumerating fold signs
alone does not make the full CV inference exact. Preserve the released split
and test in repetition. Reusable design and sampling-unit interfaces are
documented in [the grouped inference guide](docs/guides/grouped-analysis.md).

## Evidence and public content

Organize new or reorganized study results by the website's two axes:
`results/reproduction/repetition`, `results/reproduction/reanalysis`,
`results/replication/repetition`, and `results/replication/reanalysis`.
Keep actively supported sensitivity analyses under named `results/robustness`
subdirectories. Retire superseded partial analyses from the public results,
execution code, tests and reporting hooks; preserve historical execution records
in the archive. Do not maintain a runnable legacy pipeline solely to
keep an old result available.
Record each artifact's description, role and pinned hub revision in its index.
`replica results publish` updates `results/README.md` automatically.
Keep execution dates and source hashes in provenance; do not require
readers to infer the method from a date-named directory. Moving an artifact
must preserve its bytes and update registry, generator and test paths.

Keep scientific methods, results, source citations and compact reproducibility
artifacts in the public repository. Internal audit reports and review procedures
belong in the experiments repository; public study pages must not link
to them.

Register claims and their exact paper locators in
the study's `claims.yaml` beside its code. Pin the paper
version and released code, record model and tokenizer revisions, and retain
artifact hashes and execution provenance. Each changed result must trace to an
actual run. Preserve the distinction between released result readers,
recomputation from released predictors and fresh predictor computation.

Validate against independent paper values, author outputs or a reference
calculation. Check row alignment, data leakage, missingness and complete cell
inventories. Never weaken checks or invent outcomes to fill missing cells.
Record unavailable prerequisites and remaining scope explicitly.

The registry, study README and generated website must agree. Keep changed-model,
changed-response and other robustness results outside replication totals.
Numerical matches, significant cells and successful claims are different counts
and must have visible labels. Keep partial coverage visible beside the observed
results.

## Maintaining reporting labels

Public Result labels are defined by `headline_claims.RESULT_LABELS`, with the
contract version in `LABEL_CONTRACT_VERSION`. Result, Scope, Evidence and Caveats
describe separate aspects of a claim. Corrected partial sources are compared
over their available cells; the registered scope and missing cells remain explicit.
Stage labels such as **Released results
only**, **Reference refit only**, **Robustness only** and **Review pending**
explain why existing evidence cannot supply a stage count.

The contract also includes **Corrected reference only** for computed
corrections whose original-generation fidelity is unresolved. The corresponding
**Fidelity unresolved** stage label, recorded through `reproduction_review`, keeps
those results visible without supplying a certified Reproduction count.

Scoped result entries retain their own values, artifact hashes, extraction rules
and declared scope. They must not fill the comparison cells or significance
counts of a broader registered claim. Completed narrower experiments use
`scoped_repetition` or `scoped_reproduction` evidence and display **Partial
repetition** or **Partial reproduction** on both study and claim rows. A scoped
corrected result must cover every correction key registered on the claim;
its `corrected_defects` cannot bypass an outstanding correction. Other scoped
refits remain **Reference refit only**. A known defect without a completed
corrected experiment remains **Correction pending**.

A verified `replication_review.corrected_baseline_reference` may identify a
scoped corrected baseline for new-data comparisons. It must map every registered
cell, pin independently computed evidence, and cover every registered defect.
This authorizes the declared transfer scope; it never fills missing original-data
cells or establishes the original cohort's result.

An upper bound below alpha certifies significance; a lower bound at or above
alpha certifies nonsignificance. An upper bound above alpha alone does not
settle the decision. Completing an implementation, repairing the REPLICA port or
substituting an unreleased test does not by itself establish bug-corrected
reproduction. A defect in shared experimental code must be annotated on every
affected study, retaining the uncorrected results as repetition evidence.

Changing a Result label requires a contract version increment, an updated exact
label test, updated study READMEs and website pages, and a pull-request
description explaining the change. New evidence mapped to an existing label
does not require a version increment. Run the relevant study, registry and
documentation checks, and inspect the rendered page before publishing new results.

## Preparing a package release

Maintainers run the manual `release` workflow from public `main` after the
quality workflow passes on the same commit. Leave `publish` unchecked to build
and inspect the source distribution and its rebuilt wheel. The workflow checks
packaged resources, distribution metadata and an installed package with base
dependencies, then saves both distributions in the `python-distributions`
artifact and records their SHA-256 checksums.

Confirm the version and authors before publishing. Selecting `publish` uploads
the checked distributions to PyPI using the `pypi` environment's trusted
publisher. Obtain the release owner's approval and configure the publisher
before selecting that option. The workflow never publishes in response to a
push or tag.
