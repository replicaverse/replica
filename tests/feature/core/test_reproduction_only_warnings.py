"""Known-incorrect entropy variants require opt-in and an actionable warning."""

import warnings
from functools import partial

import pytest

from replica.feature import (
    ContinuationEntropyBuggy,
    ContinuationEntropyLMBuggy,
    HuggingFaceLM,
    TokenEntropySum,
    TokenRenyiEntropySum,
)
from replica.feature.entropy import ReproductionOnlyWarning
from replica.feature.entropy._base import _BaseEntropyEstimatorLM


@pytest.mark.parametrize(
    ("constructor", "buggy_name", "corrected_name"),
    [
        (TokenEntropySum, "first_token_entropy_buggy", "first_token_entropy"),
        (TokenRenyiEntropySum, "renyi_entropy_buggy", "token_renyi_entropy"),
        (
            ContinuationEntropyBuggy,
            "continuation_entropy_buggy",
            "continuation_entropy",
        ),
        (
            ContinuationEntropyLMBuggy,
            "continuation_entropy_buggy",
            "continuation_entropy",
        ),
        (
            partial(HuggingFaceLM, token_entropy_sum=True),
            "first_token_entropy_buggy",
            "first_token_entropy",
        ),
        (
            partial(HuggingFaceLM, token_renyi_entropy_sum=True),
            "renyi_entropy_buggy",
            "token_renyi_entropy",
        ),
    ],
)
def test_opt_in_is_required_even_if_warnings_are_ignored(
    constructor, buggy_name, corrected_name, monkeypatch
):
    loaded = []
    monkeypatch.setattr(
        _BaseEntropyEstimatorLM,
        "__init__",
        lambda *args, **kwargs: loaded.append(True),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="allow_reproduction=True") as error:
            constructor()
    assert not loaded
    assert buggy_name in str(error.value)
    assert corrected_name in str(error.value)
    with pytest.warns(ReproductionOnlyWarning) as caught:
        constructor(allow_reproduction=True)
    message = str(caught[0].message)
    assert buggy_name in message
    assert "only for reproduction" in message
    assert "does not compute" in message
    assert "Do not use it for new analyses" in message
    assert corrected_name in message
