"""Decoding cell tags into the index's model columns."""

import pytest

from replica.feature.frequency.registry import SOURCES, pickle_prefix
from replica.predictor_store.layout import check_ids, decode_tag
from tests.predictor_store._cells import DATASET

#: Frequency sources without a sampling estimator; each reads a frequency table.
_TABLE_SOURCES = sorted(
    name for name, source in SOURCES.items() if source.estimator is None
)


@pytest.mark.parametrize("sid", ["", 1, None])
def test_export_ids_must_be_nonempty_strings(sid):
    with pytest.raises(ValueError, match="nonempty strings"):
        check_ids([sid], "synthetic cell")


@pytest.mark.parametrize(
    "prefix",
    [pickle_prefix(name) for name in _TABLE_SOURCES]
    + ["wordfreq-en_unigram_surprisal"],
)
def test_a_frequency_table_cell_is_indexed_as_a_frequency_table(prefix):
    info = decode_tag(f"{prefix}_{DATASET}", DATASET)
    assert info.model_kind == "frequency_table"
    assert info.model == prefix.split("_", 1)[0]


def test_a_sampled_unigram_cell_is_indexed_under_its_language_model():
    info = decode_tag(
        f"{pickle_prefix('mmc', 'ai-forever/mGPT')}_{DATASET}",
        DATASET,
        model_id="ai-forever/mGPT",
    )
    assert (info.model, info.model_kind) == ("ai-forever/mGPT", "lm")


def test_a_model_slot_does_not_prove_a_model_identity():
    tag = f"org_team_model_name_surprisal_{DATASET}"
    info = decode_tag(tag, DATASET)
    assert info.model_slot == "org_team_model_name" and info.model is None
    for identity in ("org_team/model_name", "org/team_model_name"):
        assert decode_tag(tag, DATASET, model_id=identity).model == identity
