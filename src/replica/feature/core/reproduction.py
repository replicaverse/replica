"""Warnings for predictors kept in their released form, for reproduction only."""

import warnings


class ReproductionOnlyWarning(UserWarning):
    """A predictor retains known errors solely to reproduce earlier results."""


_MESSAGES = {
    "first_token_entropy_buggy": (
        "first_token_entropy_buggy is provided only for reproduction. "
        "It sums entropies across subtokens instead of taking the first-token "
        "entropy, so it does not compute the intended quantity. "
        "Do not use it for new analyses. Use first_token_entropy (TokenEntropy)."
    ),
    "renyi_entropy_buggy": (
        "renyi_entropy_buggy is provided only for reproduction. "
        "It sums Rényi entropies across subtokens instead of taking the "
        "first-token entropy, so it does not compute the intended quantity. "
        "Do not use it for new analyses. Use token_renyi_entropy (TokenRenyiEntropy)."
    ),
    "continuation_entropy_buggy": (
        "continuation_entropy_buggy is provided only for reproduction. "
        "It samples from a top-k-truncated distribution and reshapes logits "
        "without transposing their sample and step axes, so it does not "
        "compute full-distribution continuation entropy. "
        "Do not use it for new analyses. Use continuation_entropy (ContinuationEntropy)."
    ),
}


def is_reproduction_only(method: str) -> bool:
    """Whether *method* is kept only to reproduce earlier results.

    Args:
        method: Compute method name.
    """
    return method in _MESSAGES


def check_reproduction_opt_in(name: str, *, allow_reproduction: bool) -> None:
    """Refuse a reproduction-only entropy method without explicit consent.

    Args:
        name: Compute method; ordinary methods need no opt-in.
        allow_reproduction: Explicit acknowledgement of the estimator's errors.
    """
    if name in _MESSAGES and allow_reproduction is not True:
        raise ValueError(
            _MESSAGES[name] + " To reproduce earlier results deliberately, pass "
            "allow_reproduction=True (CLI: --allow-reproduction)."
        )


def warn_reproduction_only(
    name: str, *, allow_reproduction: bool = False, stacklevel: int = 2
) -> None:
    """Require opt-in, then warn about a reproduction-only estimator.

    Args:
        name: Reproduction-only entropy method.
        allow_reproduction: Explicit acknowledgement of the estimator's errors.
        stacklevel: Warning stack depth, including this helper.
    """
    check_reproduction_opt_in(name, allow_reproduction=allow_reproduction)
    warnings.warn(
        _MESSAGES[name],
        ReproductionOnlyWarning,
        stacklevel=stacklevel,
    )
