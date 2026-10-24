"""Round-trip + sanity correctness for BSC's raw fixation API.

BSC's ``BSC.EMD.txt`` is per-fixation but ships only ``dur`` (no
per-fixation start/end timestamps). The loader synthesizes trial-relative
timing via :class:`SyntheticTimer` and infers ``run_id`` per-IA via
:class:`RunIdInferrer`. The round-trip here verifies that aggregating
the raw fixations to the ``words`` layer reproduces the per-word
``data()`` API that BSC's load() built independently from the same EMD.

Smoke / invariant tests come from :class:`FixationsTestBase`.
"""

import os

import pytest

from replica.dataset import BSC
from tests._require import load_or_skip
from tests.dataset._fixation_test_base import (
    FixationsTestBase,
    SyntheticTimingChecks,
    assert_fixation_round_trip,
)

pytestmark = pytest.mark.needs_data


@pytest.mark.slow
class TestBSCFixations(FixationsTestBase, SyntheticTimingChecks):
    min_fixations = 50_000

    @pytest.fixture(autouse=True)
    def load(self, data_root):
        path = os.path.join(data_root, "eye-tracking", "original", "bsc")
        self.ds = load_or_skip(BSC(), path)

    def test_metadata_carries_sentence_num(self):
        f = self.ds.fixations()[0]
        assert f.metadata.get("sentence_num") is not None
        # Word position is carried as ``word_pos`` (the cross-loader
        # convention, e.g. ucl.py), not ``word_num``.
        assert f.metadata.get("word_pos") is not None

    @pytest.mark.parametrize(
        "source_measure", ["total_time", "first_run", "first_pass"]
    )
    def test_round_trip_reproduces_measure(self, source_measure):
        # Both APIs use the same EMD rows, so their measures must match exactly.
        assert_fixation_round_trip(
            self.ds, source_measure=source_measure, tolerance=0.0, min_compared=5_000
        )
