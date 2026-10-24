<p align="center">
  <img src="https://replicaverse.org/assets/wordmark_replica.svg" alt="REPLICA" width="840">
</p>

REPLICA is a toolkit for replication and reproduction in computational
psycholinguistics. It combines dataset loading, feature extraction and
cross-validated model comparison through configurable interfaces.

[Website](https://replicaverse.org) ·
[Datasets](https://replicaverse.org/datasets) ·
[Tutorials](#tutorials) · [Architecture](https://github.com/replicaverse/replica/blob/main/docs/guides/architecture.md) ·
[Contributing](https://github.com/replicaverse/replica/blob/main/CONTRIBUTING.md)

## Install

Use Python 3.11 or newer:

```bash
python -m pip install replicaverse
```

The Python import and command are named `replica`. The base package supports
tabular analysis and precomputed predictors. The getting-started example uses
`replicaverse[models,r]` and requires R with `lme4` for the default mixed-effects
backend. See the [environment guide](https://github.com/replicaverse/replica/blob/main/docs/guides/environment.md) for setup,
optional dependencies and source installation.

## Tutorials

Start with [Getting started](https://github.com/replicaverse/replica/blob/main/docs/tutorials/getting_started.ipynb) to load a
dataset, evaluate a language-model predictor and interpret the results.

| Notebook | Topics |
|---|---|
| [Benchmarking](https://github.com/replicaverse/replica/blob/main/docs/tutorials/benchmarking.ipynb) | Compare datasets and models |
| [Loading datasets](https://github.com/replicaverse/replica/blob/main/docs/tutorials/loading_any_dataset.ipynb) | Catalog, variants and loader options |
| [Segmentation](https://github.com/replicaverse/replica/blob/main/docs/tutorials/segmentation.ipynb) | Character, word and sentence units |
| [Surprisal methods](https://github.com/replicaverse/replica/blob/main/docs/tutorials/surprisal_methods.ipynb) | Compare predictor calculations |
| [Adding a study](https://github.com/replicaverse/replica/blob/main/docs/tutorials/extending_replica.ipynb) | Integrate and validate a study recipe |

Dataset walkthroughs: [eye-tracking](https://github.com/replicaverse/replica/blob/main/docs/datasets/eye_tracking.ipynb),
[self-paced reading](https://github.com/replicaverse/replica/blob/main/docs/datasets/spr.ipynb), [EEG](https://github.com/replicaverse/replica/blob/main/docs/datasets/eeg.ipynb)
and [fMRI](https://github.com/replicaverse/replica/blob/main/docs/datasets/fmri.ipynb).

## Datasets and storage

Explore available datasets with `replica datasets` or the
[catalog](https://replicaverse.org/datasets). Download a supported source with
`replica fetch DATASET`; some sources require manual acquisition or licensed
materials, as described in their catalog entries.

Data and caches use automatic defaults; no path settings are required.
The [environment guide](https://github.com/replicaverse/replica/blob/main/docs/guides/environment.md#data-and-cache-locations)
explains how to change their locations. The
[predictor-store guide](https://github.com/replicaverse/replica/blob/main/docs/guides/predictor-store.md) covers precomputed predictors.

## Contributing

The [architecture guide](https://github.com/replicaverse/replica/blob/main/docs/guides/architecture.md) describes the package and
its extension points. Follow the [contributor guide](https://github.com/replicaverse/replica/blob/main/CONTRIBUTING.md) and
[test guide](https://github.com/replicaverse/replica/blob/main/tests/README.md) to develop and validate changes, then open a PR.

Related projects: [ModelBlocks](https://github.com/modelblocks/modelblocks-release),
[pymovements](https://pymovements.readthedocs.io/en/stable/) and
[EyeBench](https://github.com/EyeBench/eyebench).
