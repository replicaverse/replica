"""Alpha sweeps reconstruct stored scalars from expected similarity.

Alpha=1 matches the cell's surprisal and information value exactly. Zero
expected similarity produces positive infinity.
"""

import math
import pickle

import numpy as np
import pytest

from replica.dataset.catalog.registry import compute_dir
from replica.feature.distance.similarity_surprisal import (
    PAPER_ALPHAS,
    alpha_sweep_from_expected,
)
from tests._require import require

_CELL = compute_dir() / "gpt2_simadj_surprisal_noncontextual_provo.pkl"
needs_cell = require(_CELL.exists, "noncontextual simadj provo cell not on disk")

_A1 = PAPER_ALPHAS.index(1.0)


@pytest.mark.needs_data
@needs_cell
def test_alpha1_reproduces_the_stored_scalars():
    with open(_CELL, "rb") as fh:
        data = pickle.load(fh)
    checked = 0
    for outs in data.values():
        for out in outs:
            assert tuple(out.metadata["alphas"]) == PAPER_ALPHAS
            derived = alpha_sweep_from_expected(
                out.values["expected_similarity_tokens"]
            )
            for key in (
                "similarity_adjusted_surprisal",
                "next_word_information_value",
            ):
                want = float(out.values[key])
                got = float(derived[key][_A1])
                if math.isnan(want):
                    assert math.isnan(got)
                else:
                    assert got == want, (key, got, want)
                checked += 1
    assert checked > 1000


def test_zero_similarity_yields_inf_not_a_clip():
    m = np.array([[0.5, 0.0], [0.25, 0.1]])
    derived = alpha_sweep_from_expected(m)
    assert math.isinf(derived["similarity_adjusted_surprisal"][1])
    assert derived["similarity_adjusted_surprisal"][0] == pytest.approx(
        -math.log(0.5) - math.log(0.25)
    )
    assert derived["next_word_information_value"][1] == pytest.approx(1.9)


def test_nan_and_empty_words_are_nan():
    empty = alpha_sweep_from_expected(np.empty((0, 3)))
    assert np.isnan(empty["similarity_adjusted_surprisal"]).all()
    partial = alpha_sweep_from_expected(np.array([[0.5, np.nan]]))
    assert math.isfinite(partial["similarity_adjusted_surprisal"][0])
    assert math.isnan(partial["similarity_adjusted_surprisal"][1])
