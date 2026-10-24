"""Fixtures of the predictor store tests."""

import numpy as np
import pytest

from tests.predictor_store._cells import ToyCorpora


@pytest.fixture
def corpora():
    return ToyCorpora()


@pytest.fixture
def samples():
    """Deterministic sample arrays for the Monte Carlo cells."""
    rng = np.random.default_rng(0)
    return lambda sid, i: {"log_p_samples_nats": rng.normal(size=4)}
