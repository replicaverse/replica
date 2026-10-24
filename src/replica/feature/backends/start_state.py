"""Resolve a causal-LM start token from an explicit request or model default."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from .hub import hub_id
from .model_defaults import model_defaults


@dataclass(frozen=True)
class StartState:
    """The resolved prefix: token ids, their strings and where they came from."""

    ids: tuple[int, ...]
    tokens: tuple[str, ...]
    basis: str

    def record(self) -> dict:
        """The identity record of this prefix."""
        return {"ids": list(self.ids), "tokens": list(self.tokens), "basis": self.basis}


def resolve_start_state(
    tokenizer,
    model_name: str | None = None,
    *,
    start_ids: Sequence[int] | None = None,
    add_bos: bool = True,
) -> StartState:
    """Resolve the prefix of *tokenizer*'s model.

    Args:
        tokenizer: The model's tokenizer; its ``name_or_path`` names the model
            when *model_name* is not given.
        model_name: Hub id or snapshot path of the model.
        start_ids: Optional single-token override, selected before default policy.
        add_bos: Whether a prefix is enabled.

    Returns:
        The explicit token, model default or tokenizer BOS. A missing BOS or
        ``add_bos=False`` gives an empty prefix.

    Raises:
        ValueError: The explicit prefix is invalid or a policy token is unknown.
    """
    explicit = validate_start_ids(start_ids, add_bos=add_bos)
    name = hub_id(str(model_name or getattr(tokenizer, "name_or_path", "") or ""))
    token = model_defaults(name).start_token
    if not add_bos:
        if token is not None:
            warnings.warn(
                f"{name}: disabling the start token overrides REPLICA's default "
                f"{token!r} and may affect estimates.",
                UserWarning,
                stacklevel=2,
            )
        return StartState((), (), "disabled")
    if explicit is not None:
        default_id = (
            tokenizer.convert_tokens_to_ids(token) if token is not None else None
        )
        if token is not None and (
            default_id is None
            or default_id == getattr(tokenizer, "unk_token_id", None)
            or explicit != (default_id,)
        ):
            warnings.warn(
                f"{name}: start token {_label(tokenizer, explicit[0])!r} "
                f"overrides REPLICA's default {token!r} and may affect estimates.",
                UserWarning,
                stacklevel=2,
            )
        return StartState(
            explicit, tuple(_label(tokenizer, i) for i in explicit), "explicit"
        )
    if token is not None:
        ident = tokenizer.convert_tokens_to_ids(token)
        if ident is None or ident == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(
                f"{name}: the start token {token!r} is not in the vocabulary"
            )
        return StartState((int(ident),), (token,), "override")
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return StartState((), (), "none")
    return StartState((int(bos),), (_label(tokenizer, int(bos)),), "tokenizer")


def _label(tokenizer, ident: int) -> str:
    """The token string of *ident*, or the id itself when the tokenizer has no vocabulary."""
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        return str(ident)
    tokens = convert([ident])
    return str(tokens[0] if isinstance(tokens, (list, tuple)) else tokens)


def start_ids(
    tokenizer,
    model_name: str | None = None,
    *,
    override: Sequence[int] | None = None,
    add_bos: bool = True,
) -> list[int]:
    """The prefix ids of *tokenizer*'s model; empty when it has none.

    Args:
        tokenizer: The model's tokenizer.
        model_name: Hub id or snapshot path; the tokenizer's name by default.
        override: Optional single-token prefix replacing the default policy.
        add_bos: Whether a prefix is enabled.
    """
    return list(
        resolve_start_state(
            tokenizer, model_name, start_ids=override, add_bos=add_bos
        ).ids
    )


def start_id(
    tokenizer,
    model_name: str | None = None,
    *,
    override: Sequence[int] | None = None,
    add_bos: bool = True,
) -> int | None:
    """The token the first text token is predicted after; ``None`` without a prefix.

    Args:
        tokenizer: The model's tokenizer.
        model_name: Hub id or snapshot path; the tokenizer's name by default.
        override: Optional single-token prefix replacing the default policy.
        add_bos: Whether a prefix is enabled.
    """
    ids = start_ids(tokenizer, model_name, override=override, add_bos=add_bos)
    return ids[-1] if ids else None


def single_token_prefix(ids: Sequence[int]) -> tuple[int, ...]:
    """Validate a nonnegative, single-token prefix.

    Args:
        ids: One integer token id.

    Returns:
        The id as a one-element tuple of ``int``; NumPy integers are accepted.

    Raises:
        ValueError: Other than one id, or an id that is a bool, a non-integer
            or negative.
    """
    values = tuple(ids)
    if len(values) != 1:
        raise ValueError(f"a prefix is exactly one token id, got {values!r}")
    ident = values[0]
    if isinstance(ident, bool) or not isinstance(ident, Integral) or ident < 0:
        raise ValueError(f"a prefix is exactly one token id, got {values!r}")
    return (int(ident),)


def validate_start_ids(
    ids: Sequence[int] | None, *, add_bos: bool = True
) -> tuple[int, ...] | None:
    """Validate an optional single-token prefix without loading a tokenizer.

    Args:
        ids: One nonnegative integer, or ``None`` for the backend default.
        add_bos: Whether a prefix is enabled; an explicit token requires it.

    Raises:
        ValueError: The override is invalid or contradicts ``add_bos=False``.
    """
    if ids is None:
        return None
    if not add_bos:
        raise ValueError("start_ids requires add_bos=True")
    try:
        return single_token_prefix(ids)
    except ValueError:
        raise ValueError(
            "start_ids requires exactly one nonnegative integer token id"
        ) from None
