"""FiniteLogitsProcessor masks non-finite logits and refuses empty rows."""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from replica.feature.backends.generation import (
    NonFiniteLogitsError,
    usable_position_limit,
)
from replica.feature.backends.logits import FiniteLogitsProcessor


def test_finite_rows_pass_through_unchanged():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    out = FiniteLogitsProcessor()(None, scores)
    assert torch.equal(out, scores)


def test_non_finite_entries_are_masked():
    scores = torch.tensor([[1.0, math.nan, math.inf, 2.0]])
    out = FiniteLogitsProcessor()(None, scores)
    assert out[0, 1] == -math.inf
    assert out[0, 2] == -math.inf
    assert out[0, 0] == 1.0


def test_fully_non_finite_row_raises():
    scores = torch.tensor([[math.nan, math.inf]])
    with pytest.raises(NonFiniteLogitsError):
        FiniteLogitsProcessor()(None, scores)


@pytest.mark.parametrize(
    "name",
    [
        "ai-forever/mGPT",
        "sberbank-ai/mGPT",
        "/cache/models--ai-forever--mGPT/snapshots/abc",
    ],
)
def test_mgpt_defaults_respect_the_architecture(name):
    assert usable_position_limit(name, SimpleNamespace(n_positions=2048)) == 512
    assert usable_position_limit(name, SimpleNamespace(n_positions=256)) == 256
    assert usable_position_limit(name, SimpleNamespace()) == 512


@pytest.mark.parametrize(
    "attribute", ["n_positions", "max_position_embeddings", "n_ctx"]
)
def test_generic_models_use_their_declared_limit(attribute):
    config = SimpleNamespace(**{attribute: 2048})
    assert usable_position_limit("other/mGPT", config) == 2048
    assert usable_position_limit("gpt2", config, context_window=128) == 128


@pytest.mark.parametrize("window", [128, 1024])
def test_different_mgpt_context_windows_warn(window):
    with pytest.warns(UserWarning, match="instead of REPLICA's default of 512"):
        assert (
            usable_position_limit(
                "ai-forever/mGPT",
                SimpleNamespace(n_positions=2048),
                context_window=window,
            )
            == window
        )


def test_selecting_default_context_or_overriding_an_unknown_model_is_quiet():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert usable_position_limit("ai-forever/mGPT", SimpleNamespace()) == 512
        assert (
            usable_position_limit(
                "ai-forever/mGPT", SimpleNamespace(n_positions=2048), context_window=512
            )
            == 512
        )
        assert (
            usable_position_limit("other/model", SimpleNamespace(), context_window=1024)
            == 1024
        )
    assert not caught


@pytest.mark.parametrize("window", [True, False, 0, 1, -1, 512.0, "512"])
def test_invalid_context_window_is_rejected(window):
    with pytest.raises(ValueError, match="integer of at least 2"):
        usable_position_limit(
            "ai-forever/mGPT", SimpleNamespace(), context_window=window
        )


def test_context_window_cannot_exceed_the_architecture():
    with pytest.raises(ValueError, match="exceeds.*position limit"):
        usable_position_limit(
            "ai-forever/mGPT", SimpleNamespace(n_positions=2048), context_window=2049
        )


@pytest.mark.parametrize("limit", [None, True, False, 0, 1, -1, 512.0, "512"])
def test_unknown_or_unusable_limits_require_an_explicit_window(limit):
    config = SimpleNamespace(n_positions=limit)
    with pytest.raises(ValueError, match="no usable context limit.*context_window"):
        usable_position_limit("new/model", config)
    assert usable_position_limit("new/model", config, context_window=64) == 64


def test_position_limit_uses_the_first_valid_config_field():
    config = SimpleNamespace(n_positions=0, max_position_embeddings=None, n_ctx=128)
    assert usable_position_limit("new/model", config) == 128
    config.max_position_embeddings = 64
    assert usable_position_limit("new/model", config) == 64
