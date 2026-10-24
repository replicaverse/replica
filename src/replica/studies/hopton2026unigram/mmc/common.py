"""Paths, digests and small writers every estimation stage shares."""

import argparse
import subprocess
from pathlib import Path

from replica.dataset.catalog.registry import REGISTRY
from replica.feature.backends.generation import declared_position_limit
from replica.feature.frequency import UnigramSurprisalLM
from replica.feature.frequency import pooled as pooled_mod
from replica.feature.frequency.registry import SOURCES

_PATHWISE = "mmc_pool"


_EE = "mmc_pool_ee"


_POOLED = frozenset({_PATHWISE, _EE})


def _version_for(source: str) -> str:
    """The estimator a pooled source publishes; the tag family follows it."""
    versions = {
        _PATHWISE: pooled_mod.ESTIMATOR_VERSION,
        _EE: pooled_mod.EE_ESTIMATOR_VERSION,
    }
    if source not in versions:
        raise SystemExit(f"{source!r} is not a pooled source")
    return versions[source]


def _write_text(path: Path, text: str) -> None:
    """Write through a temporary file so a killed task leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_npz(path: Path, save) -> None:
    """``save(tmp)`` then rename, for the numpy artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    save(tmp)
    tmp.replace(path)


def start_ids_arg(text: str) -> list[int]:
    """Parse a ``--start-ids`` value: one non-negative token id, as the sampler requires.

    Args:
        text: The option value.
    """
    parts = text.replace(",", " ").split()
    if len(parts) != 1 or not parts[0].isdigit():
        raise argparse.ArgumentTypeError(
            f"--start-ids takes exactly one non-negative token id, got {text!r}"
        )
    return [int(parts[0])]


def _done(path: Path, label: str) -> bool:
    if path.exists():
        print(f"[{label}] {path} exists, skipping", flush=True)
        return True
    return False


def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _configure_sampler(source) -> None:
    """Select the study's tokenizer prefix and original model position limit."""
    if source.start_ids is None:
        source.start_ids = tuple(pooled_mod.start_prefix(source._ensure_tokenizer()))
    if source.context_window is None:
        source.context_window = declared_position_limit(source._model_config())


def _git_sha() -> str:
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[3]),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def _surfaces_for_dataset(name: str, source: str) -> list[str]:
    """Every distinct surface in one dataset, under the source's lookup."""
    model = UnigramSurprisalLM(_Nowhere(), lookup=SOURCES[source].lookup)
    dataset = REGISTRY[name].factory()
    found: set[str] = set()
    for stimulus in dataset.stimuli():
        for surfaces in model._surfaces(stimulus, "words"):
            found.update(surfaces)
    return sorted(found)


class _Nowhere:
    """Stand-in source: ``_surfaces`` needs only the lookup."""

    language = "en"
    case_sensitive = True

    def probability(self, unit: str) -> float:
        raise AssertionError("surface collection must not estimate")
