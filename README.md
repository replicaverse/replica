<p align="center">
  <img src="docs/assets/wordmark_replica.svg" alt="REPLICA" width="840">
</p>

REPLICA is a toolkit for replication and reproduction in computational
psycholinguistics. It combines dataset loading, feature extraction and
cross-validated model comparison through configurable interfaces.

[Website](https://replicaverse.org) ·
[Datasets](https://replicaverse.org/datasets) ·
[Tutorials](#tutorials) · [Architecture](docs/guides/architecture.md) ·
[Contributing](CONTRIBUTING.md)

## Install

Use Python 3.11 or newer:

```bash
python -m pip install replicaverse
```

The Python import and command are named `replica`. The base package supports
tabular analysis and precomputed predictors. The getting-started example uses
`replicaverse[models,r]` and requires R with `lme4` for the default mixed-effects
backend. See the [environment guide](docs/guides/environment.md) for setup,
optional dependencies and source installation.

## Tutorials

Start with [Getting started](docs/tutorials/getting_started.ipynb) to load a
dataset, evaluate a language-model predictor and interpret the results.

| Notebook | Topics |
|---|---|
| [Benchmarking](docs/tutorials/benchmarking.ipynb) | Compare datasets and models |
| [Loading datasets](docs/tutorials/loading_any_dataset.ipynb) | Catalog, variants and loader options |
| [Segmentation](docs/tutorials/segmentation.ipynb) | Character, word and sentence units |
| [Surprisal methods](docs/tutorials/surprisal_methods.ipynb) | Compare predictor calculations |
| [Adding a study](docs/tutorials/extending_replica.ipynb) | Integrate and validate a study recipe |

Dataset walkthroughs: [eye-tracking](docs/datasets/eye_tracking.ipynb),
[self-paced reading](docs/datasets/spr.ipynb), [EEG](docs/datasets/eeg.ipynb)
and [fMRI](docs/datasets/fmri.ipynb).

## Datasets and storage

Explore available datasets with `replica datasets` or the
[catalog](https://replicaverse.org/datasets). Download a supported source with
`replica fetch DATASET`; some sources require manual acquisition or licensed
materials, as described in their catalog entries.

Data and caches use automatic defaults; no path settings are required.
The [environment guide](docs/guides/environment.md#data-and-cache-locations)
explains how to change their locations. The
[predictor-store guide](docs/guides/predictor-store.md) covers precomputed predictors.

## Contributing

The [architecture guide](docs/guides/architecture.md) describes the package and
its extension points. Follow the [contributor guide](CONTRIBUTING.md) and
[test guide](tests/README.md) to develop and validate changes, then open a PR.

Related projects: [ModelBlocks](https://github.com/modelblocks/modelblocks-release),
[pymovements](https://pymovements.readthedocs.io/en/stable/) and
[EyeBench](https://github.com/EyeBench/eyebench).
