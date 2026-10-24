# Development setup

Create and activate a virtual environment using the
[installation guide](../docs/guides/environment.md). Choose a PyTorch build for
your platform and accelerator as described there, then install the development
dependencies declared in [`pyproject.toml`](../pyproject.toml):

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install -e ".[dev]" --group word-surprisal
python -m pip check
python -m nltk.downloader -q punkt_tab
```

Run `replica check` and the relevant [test tiers](../tests/README.md). The NLTK
command downloads the tokenizer resource used by the tests. Corpora, model
weights, R and native extensions are separate prerequisites for their marked
tiers. Full numerical and optional-backend testing runs on Linux. Routine
base-wheel checks use Python 3.11 and 3.12 on Linux; the weekly schedule and
manual runs also check macOS and Windows.

## Linux CI constraints

[`ci-linux-py312.txt`](ci-linux-py312.txt) selects tested Python dependency versions
for Ubuntu 24.04, x86-64, Python 3.12 and CPU PyTorch. Contributors can use it to
reproduce that CI setup. It does not freeze OS libraries, R or accelerators.

From the repository root, create a fresh Python 3.12 environment and install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==26.0.1"
export PIP_CONSTRAINT="$PWD/requirements/ci-linux-py312.txt"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]" --group word-surprisal
unset PIP_CONSTRAINT
python -m pip check
python -m nltk.downloader -q punkt_tab
```

## Updating the CI constraints

Resolve updates in a new environment, never in a study's recorded environment.
Repeat the installation above without setting `PIP_CONSTRAINT`, then generate:

```bash
python -m pip freeze --all --exclude pip --exclude replicaverse > requirements/ci-linux-py312.txt
```

The editable checkout is deliberately excluded. The pinned source dependency
remains a commit-qualified URL. Review the dependency diff, recreate another
clean environment using the new constraints, and run the required test tiers
before opening a PR. Do not regenerate this file from an everyday environment
with unrelated packages installed. Build tools and package indexes remain
external prerequisites; this file is not an archival or hash-verified lock.

The shared [CI setup action](../.github/actions/setup-development/action.yml)
uses these commands for Linux test jobs. Push and manual quality runs use the
constraints; the scheduled quality run resolves current dependencies to detect
drift and logs installed versions. Constraint updates require a reviewed PR.
