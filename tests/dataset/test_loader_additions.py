"""MECO language coverage, Narratives discovery and BSC fixation aggregation.

Tests use downloaded corpora and skip with a fetch hint when inputs are missing.
"""

from pathlib import Path

import numpy as np
import pytest

from replica.dataset.measures import FIRST_PASS_RUN

REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Helpers
# ============================================================================


def _require_dir(path: Path, hint: str):
    """Skip the test (with a fetch hint) when `path` is absent or empty."""
    if not path.exists() or not any(path.iterdir()):
        pytest.skip(f"{path} is empty or missing. {hint}")


# ============================================================================
# MECO_L1 across all 13 languages
# ============================================================================


@pytest.mark.needs_data
@pytest.mark.parametrize(
    "lang",
    [
        "de",
        "du",
        "ee",
        "en",
        "fi",
        "gr",
        "he",
        "it",
        "ko",
        "no",
        "ru",
        "sp",
        "tr",
    ],
)
def test_meco_l1_loads_each_language(lang, data_root):
    """MECO_L1 must work for every language listed in its LANGUAGES
    constant. Catches drift between the constant and what load()
    actually supports."""
    from replica.dataset.eye_tracking.meco.l1 import MECO_L1

    path = Path(data_root) / "eye-tracking/original/meco"
    _require_dir(path, "Run `replica fetch meco`.")

    ds = MECO_L1(language=lang)
    ds.load(str(path))
    stims = ds.stimuli()
    data = list(ds.data(measure=ds.default_measure))

    assert len(stims) == 12, (
        f"{lang}: expected 12 trials (MECO L1 wave 1 ships 12 texts per "
        f"language), got {len(stims)}"
    )
    assert len(data) > 0, f"{lang}: zero datums for default measure"
    # Each trial must contain words.
    for stim in stims:
        assert len(stim.layers["words"]) > 0, (
            f"{lang}: stimulus {stim.stimulus_id} has no words"
        )


# ============================================================================
# MECO_char across all 13 languages
# ============================================================================


@pytest.mark.needs_data
@pytest.mark.parametrize(
    "lang",
    [
        "de",
        "du",
        "ee",
        "en",
        "fi",
        "gr",
        "he",
        "it",
        "ko",
        "no",
        "ru",
        "sp",
        "tr",
    ],
)
def test_meco_char_loads_each_language(lang, data_root):
    """MECO_char must accept the same ISO 639-1 codes as MECO_L1. German
    is "de" externally and gets mapped to "ge" inside the loader."""
    from replica.dataset.eye_tracking.meco.character import MECO_char

    path = Path(data_root) / "eye-tracking/processed/meco_char"
    _require_dir(path, "Run `replica fetch meco_char`.")

    ds = MECO_char(language=lang)
    ds.load(str(path))

    stims = ds.stimuli()
    data = list(ds.data(measure=ds.default_measure))
    assert len(stims) == 12, f"{lang}: expected 12 trials, got {len(stims)}"
    assert len(data) > 0, f"{lang}: zero datums for default measure"
    for stim in stims:
        assert len(stim.layers["characters"]) > 0, (
            f"{lang}: stimulus {stim.stimulus_id} has no characters"
        )


def test_meco_l1_and_meco_char_have_identical_languages():
    """The two loaders accept the same set of language codes."""
    from replica.dataset.eye_tracking.meco.character import MECO_char
    from replica.dataset.eye_tracking.meco.l1 import MECO_L1

    assert sorted(MECO_L1.LANGUAGES) == sorted(MECO_char.LANGUAGES), (
        "MECO_L1.LANGUAGES and MECO_char.LANGUAGES disagree, they "
        "must share the same ISO 639-1 codes"
    )


@pytest.mark.needs_data
def test_meco_char_german_alias_is_iso(data_root):
    """``MECO_char(language="de")`` loads and ``"ge"``, the on-disk file
    code, raises."""
    from replica.dataset.eye_tracking.meco.character import MECO_char

    path = Path(data_root) / "eye-tracking/processed/meco_char"
    _require_dir(path, "Run `replica fetch meco_char`.")

    ds = MECO_char(language="de")
    ds.load(str(path))
    assert len(ds.stimuli()) == 12

    with pytest.raises(ValueError, match="Unknown MECO_char language 'ge'"):
        MECO_char(language="ge")


# ============================================================================
# MECO_L1 reading processed/meco_opedal2024context/ (per-language CSVs)
# ============================================================================
# ============================================================================
# Narratives stimulus_resources auto-discovery
# ============================================================================


@pytest.mark.needs_data
def test_narratives_finds_alignment_under_subdirectory(data_root):
    """Narratives finds alignment files in nested stimulus-resource directories."""
    from replica.dataset import Narratives

    path = Path(data_root) / "fmri/original/narratives"
    _require_dir(path, "Narratives data not present.")
    ds = Narratives(story="21styear")
    ds.load(str(path), max_subjects=1)
    assert len(ds.stimuli()) == 1
    stim = ds.stimuli()[0]
    assert "words" in stim.layers
    assert "word_times" in stim.layers


# ============================================================================
# BSC end-to-end
# ============================================================================


@pytest.mark.needs_data
def test_bsc_loads_with_published_counts(data_root):
    """BSC builds 150 sentence Stimuli from BSC.Word.Info.v2.xlsx,
    aggregates fixations from BSC.EMD.txt to per-word measures, and
    yields the published 60-participant count. Pins paper-reported
    counts and the canonical Chinese-word formatting (no spaces)."""
    from replica.dataset import BSC

    path = Path(data_root) / "eye-tracking/original/bsc"
    _require_dir(path, "Run `replica fetch bsc`.")
    ds = BSC()
    ds.load(str(path))

    stims = ds.stimuli()
    assert len(stims) == 150, f"Expected 150 sentences, got {len(stims)}"
    assert ds.default_measure == FIRST_PASS_RUN
    measures = list(ds.measures)
    assert {"first_fixation", "first_run", "first_pass", "total_time"}.issubset(
        measures
    )

    # Each sentence is a sequence of Chinese characters joined into the
    # raw text; the words layer holds the original BSC word boundaries.
    for stim in stims[:5]:
        assert len(stim.layers["words"]) > 0
        assert " " not in stim.text  # BSC concatenates Chinese words.

    pids = {d.participant_id for d in ds.data(measure="first_run")}
    assert len(pids) == 60, f"Expected 60 participants, got {len(pids)}"


# ============================================================================
# End-to-end gap tests: loading a dataset is one thing; running it
# through evaluate() with a real LM is another. Catch the case where
# the loader produces stimuli + data that are individually well-formed
# but don't compose with the regression pipeline (e.g.\ unit-layer name
# mismatches, datum.layer pointing somewhere evaluate() does not look).
# ============================================================================


@pytest.mark.needs_data
@pytest.mark.slow
def test_meco_char_runs_end_to_end_through_evaluate(data_root):
    """MECO_char stimuli use a 'characters' layer in place of the default
    'words'. Datums also point to that layer. evaluate() must follow
    the layer through correctly. Catches regressions in the data-layer-
    vs-target-layer plumbing."""
    from replica import (
        OLS,
        AnalysisConfig,
        EvaluationSpec,
        ItemSplitter,
        UnitLength,
        evaluate,
    )
    from replica.dataset.eye_tracking.meco.character import MECO_char

    path = Path(data_root) / "eye-tracking/processed/meco_char"
    _require_dir(path, "Run `replica fetch meco_char`.")
    ds = MECO_char(language="en")
    ds.load(str(path))

    spec = EvaluationSpec(
        predictors=[UnitLength()],
        controls=[],
        targets=[ds.feature("total_time")],
    )
    cfg = AnalysisConfig(
        design=None, transform_dv=np.log2, model=OLS(), cv=ItemSplitter(2), n_folds=2
    )
    results = evaluate(ds, spec, cfg)
    assert len(results) == 1
    assert np.isfinite(results[0].delta_ll)
    assert results[0].n_observations > 0


@pytest.mark.needs_data
def test_geco_carries_the_release_word_id(data_root):
    """Word positions skip values in some trials, so the release's own key
    cannot be rebuilt from the word index. A study joining GECO to its
    materials annotation needs it carried through."""
    import os

    from replica.dataset import GECO

    path = os.path.join(data_root, "eye-tracking", "original", "geco")
    if not os.path.isdir(path):
        pytest.skip(f"GECO data not found at {path}")
    ds = GECO(group="l1")
    ds.load(path)

    for stimulus in ds.stimuli():
        word_ids = stimulus.metadata["word_ids"]
        assert len(word_ids) == len(stimulus.layers["words"])
        part, trial = stimulus.metadata["part"], stimulus.metadata["trial"]
        assert all(w.startswith(f"{part}-{trial}-") for w in word_ids)

    # Non-contiguous positions are why the index cannot stand in for the key.
    gaps = [
        s
        for s in ds.stimuli()
        if [int(w.rsplit("-", 1)[1]) for w in s.metadata["word_ids"]]
        != list(range(1, len(s.metadata["word_ids"]) + 1))
    ]
    assert gaps, "expected trials whose word positions skip values"
