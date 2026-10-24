"""The natural_stories sentence provider matches the authors' segmentation.

Anchor: Kuribayashi et al. (2025)'s release ships both the per-zone
``sent_id`` annotation the provider reads and ``data/NS/tokens.json``, the
sentence-grouped word lists their pipeline scored. Grouping our stimulus
words by the attached sentence layer must reproduce those groups exactly
(the release substitutes one word, ``peaked`` to ``peeked``, for MAZE
alignment; the comparison applies the same substitution).
"""

import json
from itertools import groupby

import pytest

from replica.external import repo_path
from tests._require import require

_REPO = repo_path("surprisal_internal_layers", fetch=False)
needs_release = require(
    (_REPO / "data" / "NS" / "tokens.json").exists,
    "surprisal_internal_layers release not available",
)

pytestmark = [pytest.mark.needs_data]


@pytest.fixture(scope="module")
def stimuli():
    from replica.dataset.catalog.registry import REGISTRY
    from replica.dataset.catalog.sentences import attach_sentences

    loaded = REGISTRY["natural_stories"].factory().stimuli()
    attach_sentences("natural_stories", loaded)
    return loaded


@needs_release
def test_sentence_groups_match_the_released_tokens(stimuli):
    from replica.context import SENTENCE_LAYER

    with (_REPO / "data" / "NS" / "tokens.json").open(encoding="utf-8") as fh:
        released = json.load(fh)
    assert len(stimuli) == len(released) == 10
    for stimulus in stimuli:
        article = stimulus.metadata["story_id"]
        words = [
            stimulus.text[span.start : span.end]
            for span in stimulus.layers["words"].spans
        ]
        ids = stimulus.layers[SENTENCE_LAYER].values
        groups = [
            [words[i].replace("peaked", "peeked") for i, _ in group]
            for _, group in groupby(enumerate(ids), key=lambda pair: pair[1])
        ]
        assert groups == released[str(article)], f"article {article}"


@needs_release
def test_sentence_ids_are_contiguous_from_zero(stimuli):
    from replica.context import SENTENCE_LAYER

    for stimulus in stimuli:
        ids = stimulus.layers[SENTENCE_LAYER].values
        assert ids[0] == 0
        steps = {b - a for a, b in zip(ids, ids[1:], strict=False)}
        assert steps <= {0, 1}, stimulus.stimulus_id
