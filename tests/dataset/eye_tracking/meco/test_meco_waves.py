"""MECO wave loaders and cross-repository compatibility checks."""

import numpy as np
import pandas as pd
import pytest

from replica import (
    OLS,
    AnalysisConfig,
    EvaluationSpec,
    ItemSplitter,
    evaluate,
)
from replica.dataset.eye_tracking.meco.waves import (
    MECOL1W2,
    MECOL2W1,
    MECOL2W2,
)
from replica.feature import UnitLength


def _frame(*, l1: bool = False, tokenization_variant: bool = False) -> pd.DataFrame:
    rows = []
    for participant in ("site_1", "site_2"):
        for text_id in range(1, 13):
            # L1W2 sites do not share a numeric item-id origin. Russian, for
            # example, uses 2..13; the loader must normalize this to 1..12.
            item_id = text_id + 1 if l1 else text_id
            words = [f"text{text_id}", "alpha-beta"]
            if tokenization_variant and participant == "site_2" and text_id == 1:
                words = [f"text{text_id}", "alpha-", "beta"]
            for wordnum, word in enumerate(words, start=1):
                rows.append(
                    {
                        "uniform_id": participant,
                        "lang": "ru_mo" if l1 else "du",
                        "itemid": str(item_id),
                        # Deliberately wrong: presentation order is not text identity.
                        "trialid": 13 - text_id,
                        "wordnum": wordnum,
                        "word": word,
                        "firstfix.dur": 100.0 + wordnum,
                        "firstrun.dur": 200.0 + wordnum,
                        "firstrun.skip": 0,
                        "firstrun.gopast": 300.0 + wordnum,
                        "dur": 400.0 + wordnum,
                    }
                )
    frame = pd.DataFrame(rows)
    # Upstream L1W2/L2W1 contain exact duplicate rows; they must not create
    # duplicate word positions or overwrite values unpredictably.
    return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)


def test_l1w2_uses_site_language_and_normalizes_item_ids():
    ds = MECOL1W2(site="ru_mo")
    ds._load_frame(_frame(l1=True))

    assert ds.info.language == "ru"
    assert len(ds.stimuli()) == 12
    assert [s.metadata["source_item_id"] for s in ds.stimuli()] == list(range(2, 14))
    assert [s.metadata["text_id"] for s in ds.stimuli()] == list(range(1, 13))
    assert len(ds.data("first_run", source="released")) == 24
    assert np.array_equal(
        ds.data("first_run", source="released")[0].values, [201.0, 202.0]
    )
    assert np.array_equal(ds.data("first_pass")[0].values, [1.0, 1.0])
    assert np.array_equal(
        ds.data("first_run_in_first_pass", source="released")[0].values, [201.0, 202.0]
    )


def test_l1w2_blank_areas_keep_measurements_outside_lexical_targets():
    frame = _frame(l1=True)
    frame["lang"] = "ba"
    blank = frame[(frame.itemid == "4") & (frame.wordnum == 2)].copy()
    blank["wordnum"] = 19
    blank["word"] = " "
    blank["dur"] = 355.0
    ds = MECOL1W2(site="ba")
    ds._load_frame(pd.concat([frame, blank], ignore_index=True))
    stimulus = next(s for s in ds.stimuli() if s.metadata["source_item_id"] == 4)
    assert stimulus.metadata["source_nonword_numbers"] == [19]
    assert stimulus.unit_strings() == ["text3", "alpha-beta"]
    for datum in ds._data:
        if datum.stimulus is stimulus:
            assert datum.metadata["source_nonword_measures"][0]["total_time"] == 355
            assert len(datum.values) == 2


def test_l2w2_preserves_incompatible_upstream_tokenizations_as_variants():
    ds = MECOL2W2()
    ds._load_frame(_frame(tokenization_variant=True))

    text_one = [s for s in ds.stimuli() if s.metadata["text_id"] == 1]
    assert len(text_one) == 2
    assert sorted(len(s.layers["words"]) for s in text_one) == [2, 3]
    assert len(ds.participants) == 2
    assert all(
        len(d.values) == len(d.stimulus.layers["words"])
        for d in ds.data("first_run", source="released")
    )


@pytest.mark.parametrize(
    ("dataset", "frame"),
    [
        (MECOL1W2(site="ru_mo"), _frame(l1=True)),
        (MECOL2W1(), _frame()),
        (MECOL2W2(), _frame(tokenization_variant=True)),
    ],
    ids=("l1w2", "l2w1", "l2w2"),
)
def test_meco_wave_loaders_run_through_evaluate(dataset, frame):
    frame = frame.copy()
    frame["firstrun.dur"] += frame["itemid"].astype(int)
    dataset._load_frame(frame)
    spec = EvaluationSpec(
        predictors=[UnitLength()],
        controls=[],
        targets=[dataset.feature("first_run_in_first_pass", source="released")],
    )
    config = AnalysisConfig(
        design=None, transform_dv=np.log2, model=OLS(), cv=ItemSplitter(2), n_folds=2
    )

    [result] = evaluate(dataset, spec, config)

    assert result.n_observations > 0
    assert np.isfinite(result.delta_ll)
    assert result.response["source"] == "released"
    assert result.response["measure"] == "first_run_in_first_pass"
    assert {o["source_algorithm"] for o in result.response["origins"]} == {
        "popeye_episode@1"
    }


@pytest.mark.parametrize("site", ["ch_s", "ch_t"])
def test_chinese_word_areas_do_not_insert_spaces_into_the_stimulus(site):
    frame = _frame()
    frame["lang"] = site
    frame["word"] = frame.wordnum.map({1: "中文", 2: "閱讀。"})
    ds = MECOL1W2(site)
    ds._load_frame(frame)
    stimulus = ds.stimuli()[0]
    assert stimulus.text == "中文閱讀。"
    assert stimulus.unit_strings() == ["中文", "閱讀。"]
    assert [(s.start, s.end) for s in stimulus.layers["words"].spans] == [
        (0, 2),
        (2, 5),
    ]
    assert ds.data("first_run", source="released")[0].values.tolist() == [201.0, 202.0]
