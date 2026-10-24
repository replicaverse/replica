"""WikiText103Source reproduces the authors' unigram lookup bit-for-bit.

Anchor: Pimentel et al. (2023)'s ``unigram.py`` (Moses tokenize-and-sum over
the released ``unigrams.csv``), run from their pinned checkout.
"""

import math
import os
import sys

import numpy as np
import pytest

from replica.external import repo_path
from tests._require import require
from tests._stimuli import s0_word_stimulus

pytestmark = pytest.mark.needs_data

pytest.importorskip("mosestokenizer")

_REPO = repo_path("anticipation-on-reading-times", fetch=False)
needs_repo = require(
    (_REPO / "corpora" / "unigrams.csv").exists,
    "anticipation-on-reading-times checkout not available",
)

WORDS = [
    "the",
    "ottoman",
    "keep,",
    "Aliens",
    "don't",
    "steep.",
    "BAFTA",
    "xylophonequark",
    "",
    "state-of-the-art",
    "1984",
    "words'",
]


def _source():
    from replica.feature.frequency.registry import SOURCES

    return SOURCES["wt103"].build()


@needs_repo
def test_matches_the_authors_lookup():
    sys.path.insert(0, str(_REPO / "src"))
    src = _source()
    import string

    # The authors' package does ``from utils.utils import ...``, so it needs
    # ``utils`` to resolve inside their own tree. Another study's test adds a
    # checkout whose top-level ``utils.py`` shadows that package, and the
    # import then fails on whichever test happens to run second.
    shadowed = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "utils" or name.startswith("utils.")
    }
    for name in shadowed:
        del sys.modules[name]

    # The authors' module opens ``corpora/unigrams.csv`` relative to the
    # repo root, so the comparison runs from there.
    cwd = os.getcwd()
    os.chdir(_REPO)
    try:
        from h01_data.models import unigram as authors

        for word in WORDS:
            # The authors' callers strip + lowercase before the lookup; the
            # source applies the same normalization to literal input surfaces.
            key = word.strip().strip(string.punctuation).lower()
            theirs = authors.frequency(key)
            p = src.probability(word)
            ours = -math.log(p) if not math.isnan(p) else float("nan")
            if math.isnan(theirs):
                assert math.isnan(ours), (word, ours)
            else:
                assert ours == pytest.approx(theirs, rel=1e-12), (word, ours, theirs)
    finally:
        os.chdir(cwd)
        sys.modules.update(shadowed)


@needs_repo
def test_feature_layer_emits_the_authors_quantity():
    from replica.feature import UnigramSurprisalLM

    text = "The ottoman is steep"
    stim = s0_word_stimulus(text)

    outs = UnigramSurprisalLM(_source())(stim)
    src = _source()
    for w, o in zip(text.split(), outs, strict=True):
        expected_nats = -math.log(src.probability(w.lower()))
        got_bits = float(np.ravel(o.values["unigram_surprisal"])[0])
        assert got_bits == pytest.approx(expected_nats / math.log(2), rel=1e-9)
        assert float(o.values["log_p_observed"]) == pytest.approx(
            -expected_nats, rel=1e-9
        )


@needs_repo
def test_oov_and_empty_yield_nan():
    src = _source()
    assert math.isnan(src.probability("xylophonequark"))
    assert math.isnan(src.probability("..."))
