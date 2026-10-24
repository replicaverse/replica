"""Preserve ZuCo task identities when a released subject array compacts a block."""

from pathlib import Path

import numpy as np
import pytest

from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1


def _sentence(text, duration):
    return {
        "content": text,
        "word": {
            "content": text,
            "FFD": duration,
            "GD": duration,
            "GPT": duration,
            "TRT": duration,
            "fixPositions": 1,
        },
        "wordbounds": np.array([[0, 0, 20, 20]]),
        "allFixations": {"x": 10, "y": 10, "duration": duration},
    }


def _source_files(path, *, corruption=None):
    from scipy.io import savemat

    texts = [f"word{i}" for i in range(407)]
    texts[342] = texts[376] = "repeated"
    reference = [_sentence(text, 100 + i) for i, text in enumerate(texts)]
    compacted = [
        _sentence(texts[i if i < 314 else i + 48], 100 + i) for i in range(359)
    ] + [{"content": texts[i], "word": []} for i in range(359, 407)]
    if corruption == "text":
        compacted[362]["content"] = "different"
    elif corruption == "populated_tail":
        compacted[362] = reference[362]
    elif corruption == "length":
        compacted.pop()
    savemat(path / "resultsAAA_TSR.mat", {"sentenceData": reference})
    savemat(path / "resultsZKB_TSR.mat", {"sentenceData": compacted})


@pytest.mark.parametrize("include_reference", [False, True])
def test_compacted_trials_keep_task_identity_source_rows_and_event_measures(
    tmp_path, include_reference
):
    _source_files(tmp_path)
    if not include_reference:
        (tmp_path / "resultsAAA_TSR.mat").unlink()
    dataset = ZuCo1(task="TSR")
    dataset.load(str(tmp_path))
    assert len(dataset.stimuli()) == (407 if include_reference else 359)
    expected_indices = [*range(314), *range(362, 407)]
    for measure in dataset.measures:
        responses = dataset.data(measure, participant="ZKB")
        assert [
            d.stimulus.metadata["sentence_idx"] for d in responses
        ] == expected_indices
        assert [d.metadata["source_sentence_idx"] for d in responses] == list(
            range(359)
        )
        np.testing.assert_equal(
            [d.values[0] for d in responses], 2 * (np.arange(359) + 100)
        )
        moved = responses[314]
        assert moved.stimulus.stimulus_id == "zuco1_TSR_362"
        assert moved.stimulus.text == "word362"
        np.testing.assert_equal(
            dataset.reaggregate(moved, "words").values, moved.values
        )
    repeated = dataset.data("first_run", participant="ZKB")[328]
    assert repeated.stimulus.text == "repeated"
    assert repeated.stimulus.stimulus_id == "zuco1_TSR_376"
    assert len(dataset.fixations("ZKB", "zuco1_TSR_362")) == 1
    assert dataset.fixations("ZKB", "zuco1_TSR_314") == []


@pytest.mark.parametrize("corruption", ["text", "populated_tail", "length"])
def test_unknown_compacted_layout_is_rejected(tmp_path, corruption):
    _source_files(tmp_path, corruption=corruption)
    with pytest.raises(ValueError, match="unrecognized TSR trial layout"):
        ZuCo1(task="TSR").load(str(tmp_path))


@pytest.mark.needs_data
@pytest.mark.slow
def test_complete_tsr_cohort_preserves_compacted_source_trials(data_root):
    root = Path(data_root) / "eye-tracking/original/zuco1"
    if len(list(root.glob("results*_TSR.mat"))) != 10:
        pytest.skip("Complete ZuCo 1 TSR release unavailable")
    dataset = ZuCo1(task="TSR")
    dataset.load(str(root))
    assert len(dataset.stimuli()) == 407
    references = dataset.data("first_run", participant="ZKB")
    assert len(references) == 359
    assert [d.stimulus.metadata["sentence_idx"] for d in references] == [
        *range(314),
        *range(362, 407),
    ]
    assert references[314].stimulus.text == "He married Carolyn Bessette in 1996."
    for measure in dataset.measures:
        for datum in dataset.data(measure, participant="ZKB")[314:]:
            np.testing.assert_allclose(
                dataset.reaggregate(datum, "words").values,
                datum.values,
                rtol=0,
                atol=1e-6,
                equal_nan=True,
            )
