"""Unit tests for the ModelBlocks .evmeasures dataset adaptor."""

import numpy as np
import pytest

from replica.dataset.measures import FIRST_PASS_RUN, with_derived
from replica.dataset.modelblocks import (
    DV_MEASURE_NAMES,
    ModelBlocksDataset,
    read_evmeasures,
)

# Two documents, two subjects. Subject s2 is missing one word in doc d1
# (sentid 1, sentpos 2) -> that word's DV must be NaN for s2.
_EVMEASURES = """\
word subject docid sentid sentpos fdurFP fdurGP wlen totsurp totsurpS1 prevwasfix
The s1 d1 1 1 200 220 3 5.0 0.0 0
cat s1 d1 1 2 250 250 3 7.0 5.0 1
sat s1 d1 1 3 180 300 3 6.0 7.0 1
Dogs s1 d1 2 1 210 210 4 8.0 0.0 0
run s1 d1 2 2 190 190 3 4.0 8.0 1
The s2 d1 1 1 205 205 3 5.0 0.0 0
sat s2 d1 1 3 175 175 3 6.0 7.0 1
Dogs s2 d1 2 1 215 240 4 8.0 0.0 0
run s2 d1 2 2 195 195 3 4.0 8.0 1
Birds s1 d2 3 1 230 230 5 9.0 0.0 0
fly s1 d2 3 2 160 160 3 3.0 9.0 1
Birds s2 d2 3 1 235 235 5 9.0 0.0 0
fly s2 d2 3 2 165 165 3 3.0 9.0 1
"""


@pytest.fixture
def evfile(tmp_path):
    p = tmp_path / "toy.evmeasures"
    p.write_text(_EVMEASURES)
    return p


def test_reads_and_builds_stimuli(evfile):
    ds = ModelBlocksDataset(evfile, dv_column="fdurFP")
    stims = {s.stimulus_id: s for s in ds.stimuli()}
    assert set(stims) == {"d1", "d2"}
    # d1 words ordered by (sentid, sentpos): The cat sat Dogs run
    assert stims["d1"].unit_strings() == ["The", "cat", "sat", "Dogs", "run"]
    assert stims["d2"].unit_strings() == ["Birds", "fly"]


def test_dv_measure_naming(evfile):
    ds = ModelBlocksDataset(evfile, dv_column="fdurFP")
    assert ds.default_measure == FIRST_PASS_RUN
    assert sorted(ds.measures) == sorted(with_derived(["first_pass", "first_run"]))
    assert (
        ModelBlocksDataset(evfile, dv_column="fdurGP").default_measure
        == "go_past_in_first_pass"
    )
    # SPR-style column maps to spr_rt
    assert DV_MEASURE_NAMES["fdur"] == "spr_rt"


def test_per_subject_alignment_and_missing(evfile):
    ds = ModelBlocksDataset(evfile, dv_column="fdurFP")
    d1 = next(s for s in ds.stimuli() if s.stimulus_id == "d1")
    data = {
        d.participant_id: d for d in ds.data(measure="first_run") if d.stimulus is d1
    }
    # s1 has all 5 words, all reached in first pass
    np.testing.assert_array_equal(data["s1"].values, [200, 250, 180, 210, 190])
    [flags] = [
        d.values
        for d in ds.data(measure="first_pass")
        if d.stimulus is d1 and d.participant_id == "s1"
    ]
    np.testing.assert_array_equal(flags, [1, 1, 1, 1, 1])
    [restricted] = [
        d.values
        for d in ds.data(measure=FIRST_PASS_RUN)
        if d.stimulus is d1 and d.participant_id == "s1"
    ]
    np.testing.assert_array_equal(restricted, [200, 250, 180, 210, 190])
    # s2 is missing "cat" (sentid1,sentpos2) -> NaN at index 1
    vals = data["s2"].values
    assert np.isnan(vals[1])
    np.testing.assert_array_equal(vals[[0, 2, 3, 4]], [205, 175, 215, 195])


def test_per_word_sentence_ids_global(evfile):
    ds = ModelBlocksDataset(evfile, dv_column="fdurFP")
    d1 = next(s for s in ds.stimuli() if s.stimulus_id == "d1")
    d2 = next(s for s in ds.stimuli() if s.stimulus_id == "d2")
    # d1: sentids 1,1,1,2,2 -> global 0,0,0,1,1 ; d2: sentid 3 -> global 2
    assert d1.metadata["per_word_sentence_ids"] == [0, 0, 0, 1, 1]
    assert d2.metadata["per_word_sentence_ids"] == [2, 2]


def test_predictor_model_aligned_per_word(evfile):
    ds = ModelBlocksDataset(evfile, dv_column="fdurFP")
    assert "totsurpS1" in ds.predictor_columns
    assert "prevwasfix" in ds.predictor_columns
    # fdurGP must not be a predictor (it's an alternative DV, varies by subject)
    assert "fdurGP" not in ds.predictor_columns
    model = ds.predictor_model(["totsurp", "totsurpS1", "wlen"])
    from replica.feature.extractors import PrecomputedFeature

    feat = PrecomputedFeature(name="totsurpS1", key="totsurpS1")
    d1 = next(s for s in ds.stimuli() if s.stimulus_id == "d1")
    outs = model(d1, extract=None, target_layer="words")
    arr = feat(d1, outs, target_layer="words")
    # totsurpS1 for d1 words The,cat,sat,Dogs,run = 0,5,7,0,8
    np.testing.assert_array_equal(arr, [0.0, 5.0, 7.0, 0.0, 8.0])


def test_subject_level_dv_not_in_predictors(evfile):
    # fdurFP varies across subjects -> auto predictor detection must exclude it
    ds = ModelBlocksDataset(evfile, dv_column="fdurFP")
    assert "fdurFP" not in ds.predictor_columns
    assert ds.participants == ["s1", "s2"]


def test_datum_columns_exposed_as_measures(evfile):
    # prevwasfix varies per subject -> request it as a datum-level measure.
    ds = ModelBlocksDataset(evfile, dv_column="fdurFP", datum_columns=["prevwasfix"])
    d1 = next(s for s in ds.stimuli() if s.stimulus_id == "d1")
    pf = {
        d.participant_id: d for d in ds.data(measure="prevwasfix") if d.stimulus is d1
    }
    # s1 d1 words The,cat,sat,Dogs,run -> prevwasfix 0,1,1,0,1
    np.testing.assert_array_equal(pf["s1"].values, [0, 1, 1, 0, 1])
    # s2 missing "cat" -> NaN at index 1, others 0,1,0,1 (The,sat,Dogs,run)
    v = pf["s2"].values
    assert np.isnan(v[1])
    np.testing.assert_array_equal(v[[0, 2, 3, 4]], [0, 1, 0, 1])


# Per-fixation rows: "cat" (sentid1,sentpos2) is fixated twice, first-pass
# duration on row 1, NaN on the refixation row (the ModelBlocks ET convention).
_EVMEASURES_REFIX = """\
word subject docid sentid sentpos fdurFP prevwasfix
The s1 d1 1 1 200 0
cat s1 d1 1 2 250 1
cat s1 d1 1 2 NaN 1
sat s1 d1 1 3 180 1
"""


def test_per_fixation_first_value_wins(tmp_path):
    p = tmp_path / "refix.evmeasures"
    p.write_text(_EVMEASURES_REFIX)
    ds = ModelBlocksDataset(p, dv_column="fdurFP")
    d1 = ds.stimuli()[0]
    assert d1.unit_strings() == ["The", "cat", "sat"]
    vals = ds.data(measure="first_run")[0].values
    # "cat" must keep its first-pass 250 after the trailing NaN row
    np.testing.assert_array_equal(vals, [200, 250, 180])


def test_read_evmeasures_roundtrip(evfile):
    df = read_evmeasures(evfile)
    assert list(df.columns)[:3] == ["word", "subject", "docid"]
    assert len(df) == 13


def test_literal_period_is_text_even_when_numeric_period_is_missing(tmp_path):
    path = tmp_path / "punctuation.evmeasures"
    path.write_text(
        "word subject docid sentid sentpos fdurFP\n. s1 17 1909 1 150\nNA s1 17 1909 2 .\n"
    )
    dataset = ModelBlocksDataset(path, dv_column="fdurFP")
    assert dataset.stimuli()[0].unit_strings() == [".", "NA"]
    np.testing.assert_equal(dataset.data("first_run")[0].values, [150, np.nan])


def test_source_run_duration_is_not_labelled_self_paced_reading(tmp_path):
    path = tmp_path / "runs.evmeasures"
    path.write_text(
        "word subject docid sentid sentpos fdurSPsummed\none s1 1 1 1 220\none s1 1 1 1 330\n"
    )
    dataset = ModelBlocksDataset(path, dv_column="fdurSPsummed")
    assert dataset.default_measure == "fdurSPsummed"
    assert dataset.measures == ["fdurSPsummed"]
    assert not dataset.additive_measures
    assert dataset.data("fdurSPsummed")[0].values.tolist() == [220]
