"""Reject unidentified word slopes and retain genuine within-item variation."""

import pandas as pd
import pytest

from replica.analysis.crossed.specification.random_effects import (
    random_terms,
    slope_audit,
)


def test_word_predictor_is_constant_within_word_but_not_participant():
    frame = pd.DataFrame(
        {
            "participant": ["a", "a", "b", "b"],
            "item": ["one", "two", "one", "two"],
            "surprisal": [2.0, 5.0, 2.0, 5.0],
            "reader_control": [1.0, 3.0, 2.0, 5.0],
        }
    )
    subject = slope_audit(frame, "participant", ["surprisal"])
    item = slope_audit(frame, "item", ["surprisal", "reader_control"])
    assert subject["retained"] == ["surprisal"]
    assert item["retained"] == ["reader_control"]
    assert "surprisal" in item["omitted"]
    assert "(0 + surprisal | participant)" in random_terms([subject, item])
    assert "(0 + surprisal | item)" not in random_terms([subject, item])


def test_aliased_slopes_are_reported_before_fitting():
    frame = pd.DataFrame({"group": [1, 1, 2, 2], "a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    audit = slope_audit(frame, "group", ["a", "b"])
    assert audit["retained"] == ["a"]
    assert audit["omitted"]["b"] == "aliased with earlier within-group slopes"


def test_single_group_cannot_supply_random_effect_variance():
    frame = pd.DataFrame({"group": [1, 1], "x": [1, 2]})
    with pytest.raises(ValueError, match="at least two"):
        random_terms([slope_audit(frame, "group", ["x"])])


def test_one_sentence_per_material_has_only_one_item_intercept():
    frame = pd.DataFrame(
        {"material": [1, 1, 2, 2], "item": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]}
    )
    audits = [slope_audit(frame, group, ["x"]) for group in ("material", "item")]
    assert random_terms(audits) == ["(1 | material)", "(0 + x | material)"]
    assert audits[1]["alias_of"] == "material"
    # Similar group counts alone do not establish equivalent partitions.
    frame["item"] = ["a", "b", "a", "b"]
    audits = [slope_audit(frame, group, ["x"]) for group in ("material", "item")]
    assert "(1 | item)" in random_terms(audits)
