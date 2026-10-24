"""The alignment check shared by the export, the readback and evaluate()."""

import json
import pickle

import pytest

from replica.artifacts import write_manifest
from replica.context import corpus_digest
from replica.dataset.catalog.provenance import corpus_fingerprint
from replica.predictor_store.alignment import (
    RowCheck,
    check_rows,
    describe,
    same_text,
)
from replica.predictors import _check_pickle_corpus
from tests.predictor_store._cells import TEXTS, ToyCorpora, word_payload, write_cell

WORDS = {sid: text.split() for sid, text in TEXTS.items()}


def _rows(words):
    sids, idxs, texts = [], [], []
    for sid, units in words.items():
        for i, unit in enumerate(units):
            sids.append(sid)
            idxs.append(i)
            texts.append(unit)
    return sids, idxs, texts


def test_an_aligned_cell_has_no_problems():
    sids, idxs, texts = _rows(WORDS)
    assert check_rows(WORDS, sids, idxs, texts) == []
    assert check_rows(WORDS, sids, idxs) == []
    # Two predictors per unit count each on its own.
    assert check_rows(WORDS, sids * 2, idxs * 2, texts * 2, ["a"] * 5 + ["b"] * 5) == []


@pytest.mark.parametrize("column", ["texts", "keys"])
@pytest.mark.parametrize("size", [0, 2])
def test_optional_row_columns_must_match_the_row_count(column, size):
    with pytest.raises(ValueError, match=f"{column} must have one value"):
        check_rows({"s": ["x"]}, ["s"], [0], **{column: ["x"] * size})


def _without_first_unit(sid: str):
    """The toy rows with the first unit of *sid* missing and the rest shifted down."""
    sids, idxs, texts = [], [], []
    for other, units in WORDS.items():
        kept = units[1:] if other == sid else units
        for i, unit in enumerate(kept):
            sids.append(other)
            idxs.append(i)
            texts.append(unit)
    return sids, idxs, texts


def test_each_reason_is_found_and_reported_in_order():
    sids, idxs, texts = _rows(WORDS)
    extra = check_rows(WORDS, [*sids, "toy_9"], [*idxs, 0], [*texts, "x"])
    assert [p.reason for p in extra] == ["ids_extra"]
    assert extra[0].stimuli == ("toy_9",)

    shifted = check_rows(WORDS, *_without_first_unit("toy_1"))
    assert [p.reason for p in shifted] == ["unit_count"]
    assert shifted[0].example == "toy_1: 1 units stored, 2 in the loader"

    gap = check_rows(WORDS, sids, [i + 1 for i in idxs], texts)
    assert [p.reason for p in gap] == ["unit_index_gap"]

    null = check_rows(WORDS, sids, idxs, [None, *texts[1:]])
    assert [p.reason for p in null] == ["text_null"]

    other = check_rows(WORDS, sids, idxs, ["zz", *texts[1:]])
    assert [p.reason for p in other] == ["text_mismatch"]
    assert other[0].example == "toy_1 unit 0: stored 'zz', loader 'ab'"

    s2, i2, t2 = _without_first_unit("toy_1")
    both = check_rows(WORDS, [*s2, "toy_9"], [*i2, 0], [*t2, "x"])
    assert [p.reason for p in both] == ["ids_extra", "unit_count"]
    assert "ids_extra on 1 stimuli" in describe(both)


def test_a_duplicate_row_counts_against_the_unit_count():
    sids, idxs, texts = _rows(WORDS)
    doubled = check_rows(WORDS, [*sids, "toy_1"], [*idxs, 0], [*texts, "ab"])
    assert [p.reason for p in doubled] == ["unit_count"]


def test_duplicate_interior_positions_cannot_hide_a_missing_unit():
    check = RowCheck({"s": ["x"] * 4})
    for index in (0, 1, 1, 3):
        check.add("s", index, "x")
    assert [p.reason for p in check.problems()] == ["unit_index_gap"]


@pytest.mark.parametrize("index", [True, 1.5])
def test_noninteger_unit_positions_are_rejected(index):
    assert [p.reason for p in check_rows({"s": ["x"]}, ["s"], [index])] == [
        "unit_index_gap"
    ]


@pytest.mark.parametrize("unit", ["words", "characters"])
def test_corpus_fingerprint_matches_story_compute_without_mutating_inputs(unit):
    from copy import deepcopy

    from replica.context import PRECEDING_CONTEXT, attach_preceding_context
    from replica.feature.core.annotation.characters import attach_characters

    stimuli = ToyCorpora().stimuli("toyset")
    for position, stimulus in enumerate(stimuli, 1):
        stimulus.metadata["context_sequence"] = {"group": "story", "position": position}
    prepared = deepcopy(stimuli)
    if unit == "characters":
        attach_characters(prepared)
    attach_preceding_context(prepared)
    assert corpus_fingerprint("toyset", stimuli, unit, "story") == corpus_digest(
        prepared, unit, "story"
    )
    assert all(PRECEDING_CONTEXT not in s.metadata for s in stimuli)
    assert all("characters" not in s.layers for s in stimuli)


def test_unit_strings_compare_after_nfc():
    assert same_text("Café", "Café")
    assert not same_text("Cafe", "Café")
    check = RowCheck({"s": ["Café"]})
    check.add("s", 0, "Café")
    assert check.problems() == []


def test_a_text_mismatch_on_a_miscounted_stimulus_is_reported_once():
    """A shifted stimulus fails on its count; its strings differ as a consequence."""
    shifted = check_rows(WORDS, *_without_first_unit("toy_2"))
    assert [p.reason for p in shifted] == ["unit_count"]
    assert shifted[0].stimuli == ("toy_2",)


def test_a_pickle_without_a_manifest_is_refused(tmp_path):
    path = tmp_path / "gpt2_surprisal_toyset.pkl"
    path.write_bytes(pickle.dumps(word_payload()))
    with pytest.raises(ValueError, match="no provenance manifest"):
        _check_pickle_corpus(path, "toyset")


def test_a_pickle_digesting_another_corpus_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "replica.predictors._corpus_stimuli",
        lambda dataset: ToyCorpora().stimuli(dataset),
    )
    path = write_cell(
        tmp_path, "gpt2_surprisal_toyset", word_payload(), method="surprisal"
    )
    sidecar = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(sidecar.read_text())
    manifest["identity"]["corpus_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="another version of toyset"):
        _check_pickle_corpus(path, "toyset")


def test_a_pickle_digesting_the_loaders_corpus_is_read(tmp_path, monkeypatch):
    stimuli = ToyCorpora().stimuli("toyset")
    monkeypatch.setattr("replica.predictors._corpus_stimuli", lambda dataset: stimuli)
    path = write_cell(
        tmp_path, "gpt2_surprisal_toyset", word_payload(), method="surprisal"
    )
    identity = {
        "schema_version": 1,
        "method": "surprisal",
        "options": {"unit": "words", "context_scope": "stimulus"},
        "corpus_sha256": corpus_digest(stimuli, "words", "stimulus"),
    }
    write_manifest(path, identity, status="completed")
    _check_pickle_corpus(path, "toyset")


def test_the_store_reader_carries_each_rows_unit_string():
    import pyarrow as pa

    from replica.predictors import _reconstruct

    table = pa.table(
        {
            "tag": ["t"] * 3,
            "stimulus_id": ["s", "s", "s"],
            "unit_index": [0, 1, 1],
            "predictor": ["a", "a", "b"],
            "value": [1.0, 2.0, 3.0],
            "text_unit": ["x", "y", "y"],
        }
    )
    outputs = _reconstruct(table)["s"]
    assert [o.metadata["text_unit"] for o in outputs] == ["x", "y"]
    assert outputs[1].values == {"a": 2.0, "b": 3.0}
    assert _reconstruct(table.drop_columns(["text_unit"]))["s"][0].metadata == {}


def test_the_readback_refuses_a_word_row_without_its_unit_string(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from replica.predictor_store import layout
    from replica.predictor_store.verify import VerifyReport, _check_text_units

    stage = tmp_path / "stage"
    stage.mkdir()
    rows = {name: [] for name in layout.WORD_SCHEMA.names}
    rows.update(
        dataset=["toyset"],
        tag=["t"],
        predictor=["p"],
        stimulus_id=["toy_1"],
        unit_index=[0],
        text_unit=[None],
        value=[1.0],
        value_unit=["nats"],
        layer=[1],
        storage_class=["closed_form"],
        unit_rule=[None],
    )
    pq.write_table(pa.table(rows, schema=layout.WORD_SCHEMA), stage / "toyset.parquet")
    report = VerifyReport()
    _check_text_units(stage, report)
    assert report.failures == ["toyset.parquet: 1 rows without a unit string"]


@pytest.fixture
def whole_segmentation(monkeypatch):
    """``whole``: one unit per text, registered for one test."""
    from replica import segmentation
    from replica.core import Span

    class Whole(segmentation.Segmenter):
        def __init__(self):
            super().__init__("whole")

        def segment(self, text):
            return [Span(0, len(text))]

    monkeypatch.setattr(segmentation, "SEGMENTERS", dict(segmentation.SEGMENTERS))
    segmentation.register_segmenter("whole")(Whole)


def test_a_cell_at_a_registered_segmentation_is_staged_against_its_layer(
    tmp_path, whole_segmentation
):
    from replica.predictor_store.export import export
    from tests.predictor_store._cells import record, rows

    tag = "gpt2_surprisal_seg-whole_toyset"
    payload = {sid: [record({"log_p_observed": -1.0})] for sid in TEXTS}
    out = tmp_path / "out"
    write_cell(out, tag, payload, method="surprisal", unit="whole")
    stage = tmp_path / "stage"
    report = export(stage, [("w", out)], corpora=ToyCorpora())
    assert report.ok, report.failures
    staged = {r["stimulus_id"]: r["text_unit"] for r in rows(stage, "toyset.parquet")}
    assert staged == dict(TEXTS)
    index = rows(stage, "tags.parquet")[0]
    assert index["unit"] == "whole"
    # A payload with one record per word is off the ``whole`` grid.
    out2 = tmp_path / "out2"
    write_cell(out2, tag, word_payload(), method="surprisal", unit="whole")
    report = export(tmp_path / "stage2", [("w", out2)], corpora=ToyCorpora())
    assert not report.ok and "unit_count" in report.failures[0]


def test_export_grid_does_not_change_the_corpus_fingerprint(whole_segmentation):
    from replica.predictor_store.export import TextIndex

    index = TextIndex(ToyCorpora().stimuli("toyset"))
    assert index.grid("whole") == {sid: [text] for sid, text in TEXTS.items()}
    assert all("whole" not in stimulus.layers for stimulus in index.stimuli)


def test_a_pickle_at_a_registered_segmentation_is_checked_on_that_layer(
    tmp_path, monkeypatch, whole_segmentation
):
    from replica import segmentation

    stimuli = ToyCorpora().stimuli("toyset")
    monkeypatch.setattr("replica.predictors._corpus_stimuli", lambda dataset: stimuli)
    path = write_cell(
        tmp_path,
        "gpt2_surprisal_seg-whole_toyset",
        word_payload(),
        method="surprisal",
        unit="whole",
    )
    digested = ToyCorpora().stimuli("toyset")
    segmentation.attach_segmentation("whole", digested)
    identity = {
        "schema_version": 1,
        "method": "surprisal",
        "options": {"unit": "whole", "context_scope": "stimulus"},
        "corpus_sha256": corpus_digest(digested, "whole", "stimulus"),
    }
    write_manifest(path, identity, status="completed")
    _check_pickle_corpus(path, "toyset")
    identity["corpus_sha256"] = corpus_digest(digested, "words", "stimulus")
    write_manifest(path, identity, status="completed")
    with pytest.raises(ValueError, match="another version of toyset"):
        _check_pickle_corpus(path, "toyset")
