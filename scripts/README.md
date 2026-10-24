# Repository tools

Installed commands use `replica`. In particular,
`replica compute` generates predictors and `replica compute merge-shards`
combines completed shards. Use `--help` for the available options.

Library services live under `src/replica`; scripts here check or maintain the
checkout. Dataset/model selections and campaign sweeps belong to study or
application repositories. The architecture guide explains the
[package boundaries](../docs/guides/architecture.md).

## Contributor checks

`replica check` runs the static checks below together with Ruff and mypy.
The [test guide](../tests/README.md) describes the behavioral test tiers.
Source checks read Python, test declarations, and import policies; distribution
checks read dataset metadata, wheel archives, or an installed package. They
report violations and return a failing exit status when a contract is broken.

`replica.feature.diagnostics` serves a different purpose: it accepts predictors,
reference models, and estimator callables, then returns numerical reports about
target agreement, bias, variance, and coverage. It is part of the installed
library and does not depend on these checkout tools.

| Tool | Purpose |
|---|---|
| `checks/source/import_boundaries.py` | Enforce import boundaries and reviewed cycle edges from `checks/source/architecture.toml` |
| `checks/source/references.py` | Check explicit repository paths in comments and docstrings |
| `checks/source/docstrings.py` | Check public-symbol documentation and parameter descriptions |
| `checks/source/test_policy.py` | Check resource markers and portable test references |
| `checks/distribution/dataset_metadata.py` | Check packaged metadata; `--check-urls` also checks upstream availability |
| `checks/distribution/wheel_contents.py` | Compare wheel code, typing files and resources with the source package |
| `checks/distribution/installed_wheel.py` | Exercise an installed wheel outside its checkout |

## Study documentation

Edit study READMEs directly. Start with the paper abstract, then describe the
predictors, usage and citation. Keep result links in `results/README.md` and
provenance in the indexed records.

Use `replica results fetch STUDY` to fetch indexed artifacts with checksum
verification. Dataset metadata generation is available through
`python -m replica.dataset.metadata.generate <dataset-directory>`.
