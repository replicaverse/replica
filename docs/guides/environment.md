# Installation and environments

Install REPLICA in a Python 3.11 or newer environment. The distribution is
`replicaverse`; imports and the command are named `replica`. Supported dependency
ranges and optional capabilities are defined in [`pyproject.toml`](../../pyproject.toml).

## Base installation

Create an environment and install the package. On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install replicaverse
python -m pip check
replica version
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1` in PowerShell. An existing Conda environment
with a supported Python version works too; activate it before installing.
Use `python -m pip` so installation targets the interpreter you selected.

The base package supports tabular data and precomputed predictors with Python
analysis backends, such as `OLS`. The [getting-started notebook](../tutorials/getting_started.ipynb)
also uses language models and the default R mixed-effects backend.

For a source installation, clone the repository and install from its root:

```bash
git clone https://github.com/replicaverse/replica.git
cd replica
python -m pip install -e .
```

## Optional capabilities

Install extras in the same environment, for example:

```bash
python -m pip install "replicaverse[models,frequency,r]"
```

For a checkout, use `python -m pip install -e ".[models,frequency,r]"`.
The [dependency metadata](../../pyproject.toml) lists all extras and supported
versions. Common choices are:

| Extra | Support |
|---|---|
| `models` | PyTorch and Transformers |
| `readers` | R data files, neuroimaging and Excel |
| `zuco` | HDF5-based ZuCo 2.0 files |
| `frequency` | Word frequencies, including Chinese, Japanese and Korean tokenizers |
| `character-surprisal` | Character-level beam surprisal |
| `r` | R regression backends |
| `lmer-rs` | Rust mixed-effects backend |
| `dev` | Tests, linting and type checks |

Install the capabilities your analysis needs. `all` includes tools with
additional system requirements and is not the default setup.

- **Models:** select a suitable [PyTorch build](https://pytorch.org/get-started/locally/)
  for CPU or your GPU before installing `models`. REPLICA does not select a CUDA
  version. Model weights are downloaded separately when requested.
- **R:** `r` installs the Python bridge, not R itself. Install R and the packages
  used by the chosen backend. The default mixed model requires `lme4`; GAMs
  require `mgcv`. For the default backend:
  `Rscript -e 'install.packages("lme4", repos="https://cloud.r-project.org")'`.
  R must also be discoverable by `rpy2`; install the bridge version allowed by
  `pyproject.toml`.
- **Other statistical backends:** the Rust adapter requires a Rust toolchain
  and system CBLAS. The Bayesian backend runs a separate R process and requires `brms`,
  CmdStanR, `posterior`, `jsonlite` and CmdStan; see the
  [backend reference](../../src/replica/analysis/README.md).

`WordsProbabilityLM` needs an upstream commit unavailable in its published
package. From the checkout, use pip 25.1 or newer:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install -e . --group word-surprisal
```

This source-only dependency group is separate from the published extras,
including `all`. Its commit is pinned in `pyproject.toml`.

## Development

Use a dedicated environment for a checkout. Follow the
[contributor setup](../../CONTRIBUTING.md) and [test guide](../../tests/README.md)
for the dependencies and checks appropriate to your change. Python 3.12 on
Linux is the primary CI environment. Passing that suite does not establish
compatibility with every Python, operating-system or accelerator combination.

## Study environments

A study using library capabilities can use the same environment. Authors'
software may require an incompatible Python, R or model stack; install that
software separately and follow the study's instructions. Environment names
are local labels, not library configuration.

Study-specific requirements belong beside the study or its external application.
Record the source revision, command, input identities and actual Python, R and
system-library versions with each run's outputs. Use `python -m pip freeze` for
Python packages and record `sessionInfo()` from the analysis's R session after
loading its packages. Include model revisions and accelerator details when
relevant. These observations document the run; supported library dependencies
remain defined in `pyproject.toml`.

## Data and cache locations

No path configuration is required. REPLICA uses these writable defaults:

| Contents | Source checkout | Installed package |
|---|---|---|
| Downloaded datasets | `datasets/` | `~/.local/share/replica/datasets/` |
| Computed predictors | `compute/` | `~/.cache/replica/compute/` |
| External study inputs | checkout root | `~/.local/share/replica/` |

Installed packages use `XDG_DATA_HOME` and `XDG_CACHE_HOME`. To override an
individual location, set `PSYCHO_DATA_DIR`, `PSYCHO_COMPUTE_DIR` or
`PSYCHO_REPL_DATA`, respectively. For one dataset, pass `path=` to `load_dataset()`.
These locations are independent of the Python environment.

`REPLICA_CHECKOUT` selects a source checkout for developer commands when automatic
discovery is insufficient. Remote predictor-store selection and version pinning
are covered in the [predictor-store guide](predictor-store.md#reading-a-store-version).

## Troubleshooting

Check `python -m pip --version` to confirm the interpreter receiving packages,
then run `python -m pip check` for declared dependency conflicts. For a missing
optional package, install the extra named by that capability. For R failures,
check that `Rscript` starts and the required R package loads. Recreate a clean
environment when incompatible study requirements have been combined.
