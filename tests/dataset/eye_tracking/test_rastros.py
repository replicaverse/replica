"""RastrOS source alignment, cloze semantics and ordinary Dataset access."""

import csv
from pathlib import Path

import numpy as np
import pytest

from replica.dataset import RastrOS, describe_dataset, load_dataset
from replica.dataset.catalog.sources import DATASETS
from replica.dataset.eye_tracking.rastros import (
    MEASURE_COLUMNS,
    NORMS_FILE,
    RESPONSE_FILE,
    read_words,
)
from replica.dataset.measures import EYELINK_MEASURE_NAMES, FIRST_PASS_RUN, with_derived
from replica.dataset.metadata.io import ChecksumMismatchError
from replica.feature.surprisal.cloze import ClozeLM


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def release(tmp_path):
    """Small complete paragraph with a shifted norm ID and a multiword region."""
    words = ['"Olá",', "de\u00a0repente", "chegou", "Ana."]
    responses = []
    for participant in ["B", "A"]:
        for index in [3, 1, 0, 2]:
            row = {
                "RECORDING_SESSION_LABEL": participant,
                "Word_Unique_ID": f"UID_2_{index + 1}",
                "Text_ID": 2,
                "Word": words[index],
                "IA_LABEL": words[index],
                "IA_SKIP": int(index == 1),
                "IA_TOP": 50,
            }
            row.update(
                {column: [100, 0, 220, 180][index] for column in MEASURE_COLUMNS}
            )
            responses.append(row)
    norms = []
    for position, target, alternatives in [
        (2, "de repente", [("de repente", 2), ("hoje", 3)]),
        (3, "chegou", [("saiu", 5)]),
        (4, "ana", [("ana", 3), ("maria", 2)]),
    ]:
        for response, count in alternatives:
            norms.append(
                {
                    "Word_Unique_ID": f"UID_2_{position}",
                    "Text_ID": 3 if position == 2 else 2,
                    "Sentence_Number": 1,
                    "Word": target,
                    "Response": response,
                    "Response_Count": count,
                    "Total_Response_Count": 5,
                }
            )
    _write(tmp_path / RESPONSE_FILE, responses)
    _write(tmp_path / NORMS_FILE, norms)
    return tmp_path, responses, norms


def test_paragraph_alignment_preserves_quotes_and_multiword_regions(release):
    path, _, _ = release
    ds = load_dataset("rastros", path=path)
    assert isinstance(ds, RastrOS)
    assert ds.info.language == "pt"
    (stimulus,) = ds.stimuli()
    assert stimulus.text == '"Olá", de\u00a0repente chegou Ana.'
    assert stimulus.stimulus_id == "rastros_p2"
    layer = stimulus.layers["words"]
    assert len(layer) == 4
    assert [
        stimulus.text[span.start : span.end] for span in layer.spans
    ] == layer.values
    assert len(ds.data(measure="first_run")) == 2
    for datum in ds.data(measure="first_run"):
        np.testing.assert_equal(datum.values, [100, np.nan, 220, 180])
    for datum in ds.data(measure="first_pass"):
        np.testing.assert_equal(datum.values, [1, 0, 1, 1])


def test_cloze_counts_distinguish_missing_from_zero_without_smoothing(release):
    path, _, _ = release
    ds = load_dataset("rastros", path=path)
    (stimulus,) = ds.stimuli()
    np.testing.assert_equal(
        stimulus.layers["human_cloze"].values, [np.nan, 0.4, 0, 0.6]
    )
    np.testing.assert_equal(
        stimulus.layers["human_cloze_hits"].values, [np.nan, 2, 0, 3]
    )
    np.testing.assert_equal(
        stimulus.layers["human_cloze_responses"].values, [np.nan, 5, 5, 5]
    )
    assert ds.word_table().context_id.eq(2).all()


def test_display_label_overrides_shifted_word_and_masks_wrong_cloze(release):
    path, responses, _ = release
    for row in responses:
        if row["Word_Unique_ID"] == "UID_2_3":
            row["Word"] = "de repente"
        if row["Word_Unique_ID"] == "UID_2_4":
            row["IA_LABEL"] = "\x97"
    _write(path / RESPONSE_FILE, responses)
    dataset = load_dataset("rastros", path=path)
    [stimulus] = dataset.stimuli()
    assert stimulus.unit_strings() == ['"Olá",', "de\u00a0repente", "chegou", "—"]
    np.testing.assert_equal(
        stimulus.layers["human_cloze"].values, [np.nan, 0.4, 0, np.nan]
    )
    np.testing.assert_equal(
        dataset.data("first_run")[0].values, [100, np.nan, 220, 180]
    )


def test_export_comma_repair_requires_matching_source_spelling(release):
    path, responses, _ = release
    for row in responses:
        if row["Word_Unique_ID"] == "UID_2_1":
            row["IA_LABEL"] = '"Olá".'
        elif row["Word_Unique_ID"] == "UID_2_3":
            row["Word"] = "diferente,"
            row["IA_LABEL"] = "chegou."
    _write(path / RESPONSE_FILE, responses)
    ds = load_dataset("rastros", path=path)
    assert ds.stimuli()[0].text == '"Olá", de\u00a0repente chegou. Ana.'
    # Keep the literal original report available for auditing.
    assert (
        ds.response_table()
        .loc[lambda f: f.Word_Unique_ID.eq("UID_2_1"), "IA_LABEL"]
        .eq('"Olá".')
        .all()
    )


def test_raw_tables_keep_trial_controls_and_are_independent_copies(release):
    path, _, _ = release
    ds = load_dataset("rastros", path=path)
    rows = ds.response_table()
    assert len(rows) == 8
    assert rows.IA_SKIP.sum() == 2
    assert (rows.IA_FIRST_RUN_DWELL_TIME == 0).sum() == 2
    rows.loc[:, "IA_TOP"] = -1
    assert ds.response_table().IA_TOP.eq(50).all()
    words = ds.word_table()
    words.loc[:, "word"] = "changed"
    assert ds.word_table().word.iloc[0] == '"Olá",'


@pytest.mark.parametrize(
    "defect", ["duplicate_trial", "missing_word", "conflicting_word"]
)
def test_corrupt_interest_area_mapping_fails(release, defect):
    path, responses, _ = release
    if defect == "duplicate_trial":
        responses.append(dict(responses[0]))
    elif defect == "missing_word":
        responses = [row for row in responses if row["Word_Unique_ID"] != "UID_2_3"]
    else:
        responses[0]["IA_LABEL"] = "different"
    _write(path / RESPONSE_FILE, responses)
    with pytest.raises(ValueError, match="RastrOS"):
        load_dataset("rastros", path=path)


@pytest.mark.parametrize(
    "defect", ["inconsistent_total", "exceeds_total", "fractional_count"]
)
def test_invalid_cloze_counts_fail(release, defect):
    path, _, norms = release
    if defect == "inconsistent_total":
        norms[0]["Total_Response_Count"] = 10
    elif defect == "exceeds_total":
        norms[0]["Response_Count"] = 8
    else:
        norms[0]["Response_Count"] = 2.5
    _write(path / NORMS_FILE, norms)
    with pytest.raises(ValueError, match="RastrOS"):
        read_words(path / NORMS_FILE, path / RESPONSE_FILE)


def test_unlisted_responses_do_not_shrink_the_published_denominator(release):
    path, _, norms = release
    norms[1]["Response_Count"] = 2
    _write(path / NORMS_FILE, norms)
    (stimulus,) = load_dataset("rastros", path=path).stimuli()
    assert stimulus.layers["human_cloze"].values[1] == 2 / 5
    assert stimulus.layers["human_cloze_responses"].values[1] == 5


def test_fetch_and_discovery_name_the_author_files():
    description = describe_dataset("rastros")
    assert description.download_keys == ("rastros",)
    assert description.default_subdir == "eye-tracking/original/rastros"
    assert set(description.measures) == with_derived(EYELINK_MEASURE_NAMES)
    assert description.default_measure == FIRST_PASS_RUN
    assert description.fixations.startswith("unavailable")
    assert DATASETS["rastros"]["files"] == [
        ("zr394", RESPONSE_FILE),
        ("7qcej", NORMS_FILE),
    ]
    with pytest.raises(NotImplementedError, match="word-level"):
        RastrOS().fixations()


def test_cloze_model_consumes_shared_annotation_without_changing_counts(release):
    path, _, _ = release
    (stimulus,) = load_dataset("rastros", path=path).stimuli()
    scores = [output.values["log_p_observed"] for output in ClozeLM()(stimulus)]
    np.testing.assert_equal(scores, [np.nan, np.log(0.4), np.log(1e-10), np.log(0.6)])
    assert stimulus.layers["human_cloze"].values[2] == 0


def test_registered_load_rejects_unverified_source_bytes(
    release, tmp_path, monkeypatch
):
    source, _, _ = release
    registered = tmp_path / "eye-tracking" / "original" / "rastros"
    registered.mkdir(parents=True)
    for filename in [RESPONSE_FILE, NORMS_FILE]:
        (registered / filename).write_bytes((source / filename).read_bytes())
    monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
    with pytest.raises(ChecksumMismatchError):
        load_dataset("rastros", path=registered)


def test_missing_release_file_gives_fetch_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="replica fetch rastros"):
        RastrOS().load(str(tmp_path))


@pytest.mark.needs_data
def test_published_release_counts_and_complete_alignment(data_root):
    """Check paper totals and the sentence labels in the actual source tables."""
    ds = load_dataset("rastros", path=Path(data_root) / "eye-tracking/original/rastros")
    assert len(ds.participants) == 37
    assert len(ds.stimuli()) == 50
    assert sum(len(s.layers["words"]) for s in ds.stimuli()) == 2494
    # The paper reports 120 sentences, but both released tables label 122.
    assert ds.word_table().sentid.nunique() == 122
    raw = ds.response_table()
    assert raw[["Text_ID", "Sentence_Number"]].drop_duplicates().shape[0] == 122
    assert len(ds.response_table()) == 87883
    assert set(
        ds.word_table().loc[
            ds.word_table().word.map(lambda word: any(char.isspace() for char in word)),
            "Word_Unique_ID",
        ]
    ) == {"UID_40_31", "UID_40_33"}
    for datum in ds.data():
        assert len(datum.values) == len(datum.stimulus.layers["words"])
    assert ds.info.license == "CC-BY-4.0"
