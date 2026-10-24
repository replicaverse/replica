"""Shared test fixtures for analysis tests.

Provides reusable toy LMs, datasets, and helper functions used across
analysis and segmentation contracts.
"""

import numpy as np

from replica.core import Datum
from replica.dataset.base import SPRDataset
from replica.dataset.contracts import DatasetInfo
from replica.feature import FeatureModelOutput
from tests._stimuli import word_stimulus

# ---------------------------------------------------------------------------
# Toy feature models
# ---------------------------------------------------------------------------


class ConstantLM:
    """Toy FeatureModel returning constant log-probs.

    All positions after 0 get ``log_p`` and a uniform next_log_probs over 3
    words. Position 0 gets log_p = NaN.

    Counts its calls in ``call_count``.
    """

    vocab = ["a", "b", "c"]

    def __init__(self, name="constant", log_p=-1.0):
        self.name = name
        self.log_p = log_p
        self.call_count = 0

    def __call__(self, stimulus, extract=None, target_layer="words"):
        self.call_count += 1
        n = len(stimulus.layers[target_layer])
        outputs = []
        for t in range(n):
            lp = np.nan if t == 0 else self.log_p
            nlp = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
            outputs.append(
                FeatureModelOutput(
                    values={"log_p_observed": lp, "next_log_probs": nlp},
                )
            )
        return outputs


# ---------------------------------------------------------------------------
# Toy datasets
# ---------------------------------------------------------------------------


class ToyDataset(SPRDataset):
    """Minimal SPR dataset for unit testing."""

    info = DatasetInfo(name="toy")

    def load(self, path: str) -> None:
        pass


def make_toy_dataset(
    n_stimuli=8,
    n_words=5,
    n_participants=1,
    measures=None,
    dataset_name="toy",
    seed=7,
):
    """Create a toy SPR dataset with synthetic data.

    Args:
        n_stimuli: Number of stimuli to generate.
        n_words: Words per stimulus.
        n_participants: Participants per stimulus.
        measures: List of measure names (default ["spr_rt"]).
        dataset_name: Name for the DatasetInfo.
        seed: RNG seed for reproducibility.
    """
    if measures is None:
        measures = ["spr_rt"]

    class _DS(SPRDataset):
        info = DatasetInfo(name=dataset_name)

        def load(self, path):
            pass

    ds = _DS()
    rng = np.random.default_rng(seed)
    for i in range(n_stimuli):
        text = " ".join(f"w{i}_{j}" for j in range(n_words))
        stim = word_stimulus(text, f"{dataset_name}_{i}")
        ds._stimuli.append(stim)
        for p in range(n_participants):
            for measure in measures:
                rt_vals = rng.uniform(200, 800, size=n_words)
                ds._data.append(
                    Datum(
                        stimulus=stim,
                        measure_name=measure,
                        values=rt_vals,
                        participant_id=f"p{p}",
                    )
                )
    return ds


# ---------------------------------------------------------------------------
# Stimulus helpers
# ---------------------------------------------------------------------------


def make_stimulus(words=("The", "cat", "sat"), stimulus_id="s1"):
    """Build a Stimulus with a 'words' layer from a sequence of strings."""
    return word_stimulus(" ".join(words), stimulus_id)


class VaryingLM:
    """Toy FeatureModel whose surprisal varies with position and stimulus."""

    name = "varying"
    vocab = None

    def __call__(self, stimulus, extract=None, target_layer="words"):
        offset = sum(ord(c) for c in stimulus.stimulus_id) % 5
        n = len(stimulus.layers[target_layer])
        return [
            FeatureModelOutput(
                values={
                    "log_p_observed": (
                        np.nan if t == 0 else -(1.0 + ((3 * t + offset) % 7) / 2.0)
                    )
                }
            )
            for t in range(n)
        ]


def linear_data(n_train=60, n_test=20, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_train + n_test, 3))
    y = 2.0 + X @ np.array([1.0, -0.5, 0.25]) + 0.1 * rng.standard_normal(len(X))
    return y[:n_train], X[:n_train], X[n_train:]
