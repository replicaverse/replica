"""Round-trip correctness for MECO_char's raw fixation API.

MECO_char's source CSV is per-fixation. The loader builds two views:
the per-character ``data()`` API and the ``fixations()`` API. The
corpus-specific test pins the contract: aggregating
``fixations()`` back to the ``"characters"`` layer reproduces
``data(measure="total_time")`` exactly.

Smoke / invariant tests come from :class:`FixationsTestBase`.
"""

import os

import pytest

from tests._require import load_or_skip
from tests.dataset._fixation_test_base import (
    FixationsTestBase,
    assert_fixation_round_trip,
)

pytestmark = pytest.mark.needs_data


@pytest.mark.slow
class TestMECOCharFixations(FixationsTestBase):
    min_fixations = 1000
    unit_layer = "characters"

    @pytest.fixture(autouse=True)
    def load(self, data_root):
        from replica.dataset.eye_tracking.meco.character import (
            MECO_char,
        )

        path = os.path.join(data_root, "eye-tracking", "processed", "meco_char")
        self.ds = load_or_skip(MECO_char(language="en"), path)

    def test_round_trip_reproduces_data(self):
        """Aggregating ``fixations()`` to the ``characters`` layer reproduces
        ``data(measure="total_time")`` exactly."""
        assert_fixation_round_trip(
            self.ds,
            source_measure="total_time",
            unit_layer="characters",
            tolerance=1e-3,
            min_compared=1000,
        )
