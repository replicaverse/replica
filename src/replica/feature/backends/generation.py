"""Generation-time guards shared by the sampling feature models."""

from __future__ import annotations

import warnings
from numbers import Integral

from .model_defaults import model_defaults


class NonFiniteLogitsError(RuntimeError):
    """A sampling step produced no finite logits to draw from."""


def validate_context_window(context_window: int | None) -> int | None:
    """Validate an optional context window of at least two tokens.

    Args:
        context_window: Token count, or ``None`` for the model default.
    """
    if context_window is None:
        return None
    if (
        isinstance(context_window, bool)
        or not isinstance(context_window, Integral)
        or context_window < 2
    ):
        raise ValueError("context_window must be an integer of at least 2 tokens")
    return int(context_window)


def declared_position_limit(config) -> int | None:
    """Read the first integer position limit of at least two tokens, if present.

    Args:
        config: Model configuration with optional position-limit attributes.
    """
    for name in ("n_positions", "max_position_embeddings", "n_ctx"):
        limit = getattr(config, name, None)
        if isinstance(limit, Integral) and not isinstance(limit, bool) and limit >= 2:
            return int(limit)
    return None


def usable_position_limit(
    model_name: str, config, *, context_window: int | None = None
) -> int:
    """Resolve the context window within the model's declared position limit.

    Args:
        model_name: Hub id or local cache snapshot path.
        config: Model configuration declaring its position limit.
        context_window: Optional override; departures from model defaults warn.

    Raises:
        ValueError: No usable limit is known, or the requested window is invalid.
    """
    requested = validate_context_window(context_window)
    limit = declared_position_limit(config)
    default = model_defaults(model_name).context_window
    if requested is not None:
        if limit is not None and requested > limit:
            raise ValueError(
                f"context_window={requested} exceeds {model_name}'s declared "
                f"position limit of {limit}"
            )
        if default is not None and requested != default:
            reason = (
                "It exceeds the documented training sequence length and may "
                "affect estimates."
                if requested > default
                else "It changes the available context and may affect estimates."
            )
            warnings.warn(
                f"{model_name}: using a {requested}-token context instead of "
                f"REPLICA's default of {default}. {reason}",
                UserWarning,
                stacklevel=2,
            )
        return requested
    if default is not None:
        return default if limit is None else min(default, limit)
    if limit is None:
        raise ValueError(
            f"{model_name}: no usable context limit; provide context_window "
            "explicitly or a model configuration with a valid position limit"
        )
    return limit
