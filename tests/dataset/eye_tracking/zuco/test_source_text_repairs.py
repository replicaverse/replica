"""Verified text repairs must preserve response alignment and reject new damage."""

from pathlib import Path

import numpy as np
import pytest

from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1, _display_tokens


@pytest.mark.parametrize(
    "source,expected,task,text_source",
    [
        (
            "Hindman died in Baltimore, Maryland and is buried in St. Paul�s Burial Ground there.",
            "Hindman died in Baltimore, Maryland and is buried in St. Paul’s Burial Ground there.",
            "NR",
            "https://osf.io/tu6yx/",
        ),
        (
            "From 1775 and 1777, Hindman served in Maryland�s revolutionary government, as the state�s Treasurer for the Eastern Shore.",
            "From 1775 and 1777, Hindman served in Maryland’s revolutionary government, as the state’s Treasurer for the Eastern Shore.",
            "NR",
            "https://osf.io/tu6yx/",
        ),
        (
            "Stella Dorothea Gibbons (5 January 1902�19 December 1989) was an English novelist and poet.",
            "Stella Dorothea Gibbons (5 January 1902—19 December 1989) was an English novelist and poet.",
            "NR",
            "https://osf.io/tu6yx/",
        ),
        (
            "He maintained homes in Long Island, New York and Greenwich, Connecticut; the family compound at Kennebunkport, Maine; a 10,000 acre (40 km�) plantation in South Carolina; and an island retreat in Florida.",
            "He maintained homes in Long Island, New York and Greenwich, Connecticut; the family compound at Kennebunkport, Maine; a 10,000 acre (40 km˝) plantation in South Carolina; and an island retreat in Florida.",
            "TSR",
            "https://osf.io/2r95j/",
        ),
    ],
)
def test_zuco_repair_preserves_matlab_response_positions(
    tmp_path, source, expected, task, text_source
):
    from scipy.io import savemat

    words = source.split()
    fields = np.array(
        [(word, 100 + i, 200 + i, 300 + i, 400 + i) for i, word in enumerate(words)],
        dtype=[(field, object) for field in ("content", "FFD", "GD", "GPT", "TRT")],
    )
    savemat(
        tmp_path / f"resultsTEST_{task}.mat",
        {"sentenceData": {"content": source, "word": fields}},
    )
    dataset = ZuCo1(task=task)
    dataset.load(str(tmp_path))
    stimulus = dataset.stimuli()[0]
    assert stimulus.text == expected
    assert stimulus.metadata["source_text"] == source
    assert stimulus.metadata["text_source"] == text_source
    np.testing.assert_equal(
        dataset.data("first_fixation")[0].values, 2 * (np.arange(len(words)) + 100)
    )


def test_zuco_does_not_guess_unverified_punctuation():
    with pytest.raises(ValueError, match="unverified replacement"):
        _display_tokens(["Another", "person�s", "sentence."], "NR")
    assert _display_tokens(["Ordinary", "text."], "NR") == ["Ordinary", "text."]


@pytest.mark.needs_data
def test_zuco_release_text_repairs(data_root):
    root = Path(data_root) / "eye-tracking/original/zuco1"
    if not root.is_dir():
        pytest.skip("ZuCo 1 release unavailable")
    dataset = ZuCo1()
    dataset.load(str(root), max_subjects=1)
    repaired = [s for s in dataset.stimuli() if "source_text" in s.metadata]
    assert len(repaired) == 3
    assert all("\ufffd" not in s.text for s in dataset.stimuli())
