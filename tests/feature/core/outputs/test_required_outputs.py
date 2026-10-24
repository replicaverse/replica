"""Declared model-output keys, which let ``evaluate()`` ask a model for less.

One feature without a declaration turns the union into ``None``, and
``evaluate()`` then asks the model for every key it can compute.
"""

import pytest

from replica.feature import ItemMeasuresTable, ModelBlocksFeature, Surprisal
from replica.feature.core.outputs.requirements import required_outputs
from replica.feature.distance.extractors import (
    NextWordInformationValue,
    ResponsivityDistance,
    SimilarityAdjustedSurprisal,
)
from replica.feature.distance.information_value import InformationValue
from replica.feature.embedding.hidden_state import HiddenState


def test_table_column_reads_no_model_output(tmp_path):
    path = tmp_path / "t.itemmeasures"
    path.write_text("word sentid sentpos dlt\na 1 1 0\n")
    feature = ModelBlocksFeature(ItemMeasuresTable.from_file(path), column="dlt")
    assert required_outputs([Surprisal(), feature]) == ["log_p_observed"]


@pytest.mark.parametrize(
    "feature, key",
    [
        (HiddenState(3), "hidden_state_3"),
        (InformationValue("cosine", "mean"), "information_value_mean_cosine"),
        (ResponsivityDistance("expected"), "expected_decontextualised_distance"),
        (NextWordInformationValue("pos"), "next_word_information_value_pos"),
        (SimilarityAdjustedSurprisal(), "similarity_adjusted_surprisal"),
    ],
    ids=["hidden_state", "information_value", "responsivity", "nwiv", "simadj"],
)
def test_stored_value_features_declare_their_key(feature, key):
    assert required_outputs([Surprisal(), feature]) == sorted(["log_p_observed", key])
