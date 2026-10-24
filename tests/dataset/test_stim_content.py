"""Check registered datasets for unexpected stimulus content issues.

Dataset-specific exceptions are declared below.
Run with ``pytest tests/dataset/test_stim_content.py --data-root PATH``.
"""

import pytest

from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.eye_tracking.meco.releases import LANGUAGES as MECO_LANGUAGES
from replica.dataset.validation import (
    stim_content_issues,
)

KNOWN_FALSE_POSITIVES: dict[str, dict[str, str]] = {
    # MECO's character release measures the spaces between words, so a
    # blank unit there is a measured unit.
    **{
        f"meco_char_{lang}": {"blank_unit": "the space characters are measured"}
        for lang in MECO_LANGUAGES
    },
    # OneStop shares paragraphs across (article, difficulty) conditions
    # by design.
    "onestop": {"duplicate_stim_text": "shared paragraphs across conditions"},
    # CELER's train/test split files share sentences from universal-deps.
    "celer": {"duplicate_stim_text": "shared sentences across CELER files"},
    # eMTeC has multiple LM continuations of the same prompt that
    # collide when topk == sampling produces identical output.
    "emtec": {"duplicate_stim_text": "multiple LM continuations per prompt"},
    # Brown SPR document d06 presents the same one-word sentence six times.
    "brown_spr": {"duplicate_stim_text": "a repeated one-word sentence in d06"},
    # CopCo parliamentary speeches repeat short stock openers verbatim
    # (e.g. "Tak for ordet.") across distinct (speech, paragraph) ids.
    "copco": {"duplicate_stim_text": "recurring stock phrases across speeches"},
    "szewczyk_n400": {
        "duplicate_stim_text": "separate source studies supply different item norms",
    },
    **{
        name: {"duplicate_stim_text": "the released NR inventory repeats sentences"}
        for name in ("zuco2", "zuco2_eeg")
    },
}


pytestmark = [pytest.mark.slow, pytest.mark.needs_data]


def _load_dataset(name: str, data_root: str, monkeypatch: pytest.MonkeyPatch):
    """Resolve the dataset's loader and call ``load(data_root)``.

    The factory uses ``DatasetRegistration.default_path()`` which honours
    ``PSYCHO_DATA_DIR``. When pytest is invoked with ``--data-root``
    the env var is overridden (via monkeypatch, so it auto-reverts after
    each test) so the factory finds the test fixtures.
    """

    monkeypatch.setenv("PSYCHO_DATA_DIR", data_root)
    return REGISTRY[name].factory()


@pytest.mark.parametrize("dataset_name", sorted(REGISTRY))
def test_stim_content_invariants(
    dataset_name: str, data_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every dataset's stims should be free of content issues, except
    for known-false-positive buckets per dataset."""
    try:
        ds = _load_dataset(dataset_name, data_root, monkeypatch)
    except FileNotFoundError as e:
        pytest.skip(f"{dataset_name} data not available: {e}")

    report = stim_content_issues(ds)
    allowed = KNOWN_FALSE_POSITIVES.get(dataset_name, {})

    unexpected: dict[str, int] = {
        bucket: count
        for bucket, count in report.counts.items()
        if count > 0 and bucket not in allowed
    }

    if unexpected:
        lines = [f"{dataset_name} has unexpected issues:"]
        for bucket, n in sorted(unexpected.items()):
            lines.append(f"  {bucket}: {n}")
            for ex in report.examples.get(bucket, [])[:3]:
                lines.append(f"    ↳ {ex}")
        pytest.fail("\n".join(lines))
