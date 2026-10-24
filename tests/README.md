# Tests

CI runs two offline test tiers on every push to `main` on Linux using the tested
dependency constraints. Scheduled runs resolve current dependencies. Both tiers
are required; coverage is combined before enforcing the 50% minimum. The split gives quick
feedback without dropping published-evidence or statistical checks.

Base-wheel installation and core functionality are checked separately with
Python 3.11 and 3.12. Routine runs use Linux; the weekly schedule and manual
runs also check macOS and Windows. Those jobs resolve
dependencies from package metadata and exercise evaluation, packaged resources,
writable paths and the installed CLI. The full numerical and optional-backend
suites run on Linux.

## Offline tiers

Follow the [development setup](../requirements/README.md), including the
tokenizer resource needed by these tests. The quick
tier covers unit contracts and small local integrations, including synthetic
dataset loaders, fixation aggregation, segmentation and toy-model pipelines:

```bash
pytest tests/ -n 2 --disable-socket --allow-unix-socket \
  --offline-tier fast
```

The offline regression tier checks indexed evidence and rendered pages,
repeated-seed calibration, and bundled third-party resources. Prepare its result
cache once with `replica results fetch`; this verifies every file against the
pinned HF revision and checksum. Access-controlled HF repositories require a
read token through `hf auth login` or `HF_TOKEN`. Then run the tests offline:

```bash
pytest tests/ -n 2 --disable-socket --allow-unix-socket \
  --offline-tier regression
```

To run both together:

```bash
pytest tests/ -n 2 --disable-socket --allow-unix-socket \
  --offline-tier all
```

`--offline-tier` uses the shared selections in `tests/conftest.py`. An additional
`-m` expression narrows that selection; omitting the flag leaves ordinary pytest
selection unchanged. The flag selects tests but does not install dependencies or
block network access, so retain the socket flags above.

The public export runs both tiers too. Tests for one study travel with that
study's full package; recipe-only exports omit its evidence and tests. Shared
reference checks use the bibliography fixture in `tests/studies/fixtures/`.

The workflow builds and checks an export only in a checkout that carries the
release rules. The exported repository runs its ordinary quality jobs without
trying to invoke the development release tooling again.

Two workers bound memory use when importing PyTorch. CI sets empty corpus and model-cache roots, prepares a separate results cache
for the regression tier, disables model downloads and limits numerical libraries to
one thread. Runtime depends on the machine and its dependency caches; inspect
`--durations=20` and the uploaded JUnit reports for the current bottlenecks.

## Markers

| Marker | Meaning |
|---|---|
| `artifact` | Indexed study evidence, registries or generated public pages |
| `statistical` | Repeated-seed calibration or numerical convergence sweeps |
| `dependency` | A bundled third-party resource, such as wordfreq language tables |
| `slow` | Long-running work explicitly excluded from ordinary offline CI |
| `needs_model` | A real pretrained model checkpoint |
| `needs_lmer_rs` | Rust mixed-model extension and its system libraries |
| `resource_free(reason)` | Reviewed use of a resource API that rejects input before loading or uses a test double |
| `needs_r` | rpy2 and system R with the required packages |
| `needs_data` | Downloaded corpora, upstream checkouts or external result stores |
| `needs_network` | A live external service |

Markers describe individual tests, or entire modules when every test qualifies.
Directories do not assign markers automatically. Keep synthetic tests in mixed
modules unmarked unless they need another tier. Purpose markers (`artifact`,
`statistical`, `dependency`) can coexist with resource markers (`needs_*`);
resource requirements still exclude a test from the ordinary offline runs.

`--strict-markers` rejects unknown markers. Combine module markers in one
`pytestmark` list; assigning it twice discards the first assignment. The
stdlib-only `scripts/checks/source/test_policy.py` checks marker declarations and
tracked references alongside linting, and through `replica check`. The static scan remains conservative at module level. During collection,
`tests/_resource_policy.py` checks each item's effective module, class, function
and parameter markers, its resolved fixture dependencies, and direct resource
calls in test/fixture bodies. A marked sibling cannot exempt an unmarked test.
The `lm_cache` and `data_root` fixtures require resource markers even when reached
through another fixture; corpus data requires `needs_data` independently of models.

For a deliberate preflight test or stub that calls a resource API without loading
resources, use `@pytest.mark.resource_free("specific reason")` and test that the
resource loader is never reached. The check rejects empty reasons, stale markers
and exemptions of the real resource fixtures. Syntax inspection cannot infer all
dynamic aliases or calls hidden inside arbitrary helpers; keep the module scan,
explicit fixture contracts and offline socket enforcement together.

## Calibration experiments

The drivers in [`analysis/calibration/`](analysis/calibration) run explicit
simulation experiments for crossed held-out losses and material-level bootstrap
inference. They are separate from automatic pytest collection. From the checkout
root, use `python -m tests.analysis.calibration.crossed --help` or
`python -m tests.analysis.calibration.material --help` to choose group counts,
simulation budgets and an output destination.

## External and R tests

```bash
pytest -m needs_model
pytest -m needs_r
pytest -m needs_data --data-root /path/to/datasets
pytest -m needs_network --require-external
```

Use an environment containing each tier's dependencies. CI runs the R tier
on each push to `main`. Scheduled and manually
dispatched workflows also run selected external model and service checks.
`--require-external` makes a skipped selected test fail, so missing resources
cannot masquerade as successful validation.

Real-corpus tests resolve their root as `--data-root` > `$PSYCHO_DATA_DIR` >
the registry's default data directory. Fixtures and registry factories share
this root. Fetch public datasets with `replica fetch <name>`.

The dataset suite can run with `pip install -e ".[readers,zuco,frequency,test]"`.
The frequency extra includes wordfreq's Chinese, Japanese and Korean tokenizers;
it is not needed to load the reading datasets. Source timing checks are collected only
for corpora that provide the relevant timestamps or synthetic-time metadata.

## Shared fixtures

| Fixture | Scope | Purpose |
|---|---|---|
| `seed_rng` | function, autouse | Seeds NumPy to 42 |
| `data_root` | function | Resolves the dataset root |
| `_cache_dataset_loads` | session, autouse | Opt-in cache with `PSYCHO_CACHE_DATASETS=1`; copies saved and restored state |
| `constant_lm` | function | Toy model for local pipelines |
| `lm_cache` | session | Shares pretrained model instances; users must declare a resource marker |
| `claim_registry`, `replication_results` | function | Deep copies of evidence parsed once per worker |
| `study_pages` | function | A copied mapping of immutable pages rendered once per worker |
| `study_index`, `replications_page` | session | Immutable rendered page strings |

Published-evidence fixtures require an explicit `artifact` marker. They do not
patch production loaders. Tests of file loading use temporary files and call
the loader directly; tests changing claims receive independent mutable copies.

Checksums are skipped by default in tests through `PSYCHO_SKIP_CHECKSUMS=1`.
Tests of integrity checking must explicitly enable it. Dataset cache tests and
scientific roundtrips must preserve their independent reference assertions.

## Result cache in CI

The regression job prepares the pinned result cache with the repository secret
`REPLICA_RESULTS_HF_TOKEN`, a read-only HF token for the configured study bundle
repository (`replicaverse/replica-study-bundles` by default).
The download step uses `replica results fetch --verification` to include archives
whose index entry declares `verification_required: true`. It fails if a required
artifact is missing or its checksum differs.
Tests run with sockets disabled and without the token. The fast tier requires no
result downloads. Site rendering uses the committed claim registries and indexed
artifact links; cached HF tables are available to table generators.

## Test locations

Tests follow the source component they exercise: `tests/analysis/inference/`
checks `replica.analysis.inference`, `tests/feature/frequency/` checks
`replica.feature.frequency`, and `tests/predictor_store/` checks the store.
Shared fixtures stay at their narrowest common package. Study-specific fixtures
stay with their study. Two tests cited by source files retain their paths to
preserve recorded source fingerprints.

Cross-package workflows belong in `tests/integration/`. Documentation examples,
packaging contracts and pytest infrastructure live in `tests/docs/`,
`tests/packaging/` and `tests/harness/`. Contributor-tool tests mirror `scripts/`
under `tests/scripts/`. Test descriptions state the behavior being checked.

Static import/scanner policy tests live in `tests/architecture/`. Run them in a
pytest/pytest-socket environment without the root scientific fixtures:

```bash
PYTHONPATH=src:. python -m pytest --confcutdir=tests/architecture \
  tests/architecture --disable-socket -q
```

The ordinary full-suite invocation still collects them. Result indexing, bundles,
storage, extraction and provenance contracts live in `tests/studies/records/`;
run that directory to exercise the complete result-service suite. Resource and
purpose markers retain their usual meaning regardless of directory.
