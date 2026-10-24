"""Replay verification rejects altered fitted state."""

import pytest

from replica.analysis.crossed.verification.replay import compare_value


def test_replay_rejects_changed_model_inventory_and_estimates():
    expected = {"model": {"loglik": -152.5, "coefficients": [0.5, -1.2]}}
    compare_value(expected, expected)
    with pytest.raises(ValueError, match="keys"):
        compare_value({"other": expected["model"]}, expected)
    with pytest.raises(AssertionError):
        compare_value(
            {"model": {"loglik": -152.5, "coefficients": [0.5, 1.2]}}, expected
        )
