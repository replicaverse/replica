"""Smoke tests for every registered dataset.

Parametrized over the dataset registry, so a newly registered key is covered
without editing this module. Each test verifies basic loading, data alignment,
and a minimal evaluate() pipeline with a constant LM.

Run with: pytest tests/integration/test_all_datasets.py -v --data-root datasets
"""

import os
from pathlib import Path

import numpy as np
import pytest

from replica import (
    OLS,
    AnalysisConfig,
    EvaluationSpec,
    ItemSplitter,
    ModelFeature,
    evaluate,
)
from replica.analysis import AnalysisResult
from replica.dataset import Narratives
from replica.dataset.catalog import registry
from replica.feature import Surprisal, UnitLength

from .conftest import _ConstantLM

_READING_PARADIGMS = ("eye-tracking", "self-paced-reading", "maze")

READING_TIME_DATASETS = [
    name
    for name in registry.names()
    if registry.get(name).paradigm in _READING_PARADIGMS
]
EEG_DATASETS = [
    name for name in registry.names() if registry.get(name).paradigm == "eeg"
]
FMRI_DATASETS = ["narratives"]

# Load options that limit the ZuCo releases to two subjects.
_LOAD_KWARGS = {
    "zuco1": {"max_subjects": 2, "load_eeg": False},
    "zuco1_eeg": {"max_subjects": 2},
    "zuco2": {"max_subjects": 2, "load_eeg": False},
    "zuco2_eeg": {"max_subjects": 2},
}


def test_every_registered_dataset_has_a_smoke_test():
    covered = READING_TIME_DATASETS + EEG_DATASETS + FMRI_DATASETS
    assert sorted(covered) == sorted(registry.names())


def _load_registered(name, data_root):
    """Load one registry key, skipping only when its data directory is absent."""
    spec = registry.get(name)
    path = Path(data_root) / spec.default_subdir
    if not path.is_dir():
        pytest.skip(f"{name} data not found at {path}")
    dataset = spec.loader_cls()(**spec.init_kwargs)
    dataset.load(str(path), **_LOAD_KWARGS.get(name, {}))
    return dataset


def _unit_layers(ds):
    """Annotation layers the dataset's values align to."""
    return {datum.layer for datum in ds.data()}


def _smoke_evaluate(ds):
    """Run evaluate() with a constant LM + UnitLength baseline. Returns
    the AnalysisResult list. Raises if the pipeline does not produce a finite
    delta_ll on at least one DV.

    Special-cases datasets whose stimuli all contain a single unit (e.g.
    Brothers & Kuperberg (2021)'s critical-region SPR design): Surprisal can't
    be computed on those (no left-context distribution), so a
    UnitLength-only spec is used instead.
    """
    # Probe a few stimuli to detect single-unit datasets.
    [layer] = {d.layer for d in ds.data(measure=ds.default_measure)}
    sample = ds.stimuli()[:5]
    one_unit = bool(sample) and all(len(s.layers[layer]) == 1 for s in sample)
    if one_unit:
        spec = EvaluationSpec(
            predictors=[UnitLength()],
            controls=[],
            targets=[ds.feature(ds.default_measure)],
        )
    else:
        spec = EvaluationSpec(
            predictors=[ModelFeature(_ConstantLM(), Surprisal())],
            controls=[UnitLength()],
            targets=[ds.feature(ds.default_measure)],
        )
    cfg = AnalysisConfig(
        design=None, transform_dv=np.log2, model=OLS(), cv=ItemSplitter(2), n_folds=2
    )
    return evaluate(ds, spec, cfg)


# ---------------------------------------------------------------------------
# Reading-time (eye-tracking, SPR and maze) and EEG datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=READING_TIME_DATASETS + EEG_DATASETS)
def loaded_dataset(request, data_root):
    return _load_registered(request.param, data_root)


@pytest.mark.needs_data
class TestDatasetSmoke:
    """Smoke tests applied to every reading-time and EEG dataset loader."""

    def test_loads_with_content(self, loaded_dataset, request):
        """Loads with non-empty stimuli, units, and participants; a reading-time
        dataset also has measures."""
        assert len(loaded_dataset.stimuli()) > 0, "no stimuli"
        layers = _unit_layers(loaded_dataset)
        for stim in loaded_dataset.stimuli():
            for layer in layers:
                assert len(stim.layers[layer]) > 0, (
                    f"Stimulus {stim.stimulus_id} has no {layer}"
                )
        if request.node.callspec.params["loaded_dataset"] not in EEG_DATASETS:
            assert len(loaded_dataset.measures) > 0, "no measures"
        assert len(loaded_dataset.participants) > 0, "no participants"

    def test_data_values_aligned(self, loaded_dataset):
        """Each Datum's values array matches its stimulus unit count."""
        for datum in loaded_dataset.data():
            units = len(datum.stimulus.layers[datum.layer])
            assert len(datum.values) == units, (
                f"Datum for {datum.participant_id} on {datum.stimulus.stimulus_id}: "
                f"values length {len(datum.values)} != units {units}"
            )

    def test_evaluate_pipeline(self, loaded_dataset):
        """End-to-end: constant LM + OLS produces finite delta_ll."""
        results = _smoke_evaluate(loaded_dataset)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, AnalysisResult)
            assert np.isfinite(r.delta_ll)
            assert r.n_observations > 0


@pytest.mark.needs_data
@pytest.mark.parametrize("loaded_dataset", ["zuco1_eeg", "zuco2_eeg"], indirect=True)
def test_band_power_measures_are_eeg(loaded_dataset):
    """The ZuCo EEG views expose band power, one ``eeg_*`` measure per band."""
    for m in loaded_dataset.measures:
        assert m.startswith("eeg_"), f"Non-EEG measure found: {m}"


# ---------------------------------------------------------------------------
# fMRI datasets
# ---------------------------------------------------------------------------

# Relative to data_root (the datasets dir), matching the sibling fixtures.
_NARRATIVES_DERIV = os.path.join("fmri", "original", "narratives", "afni-nosmooth")
_NARRATIVES_ALIGN = os.path.join(
    "fmri",
    "original",
    "narratives",
    "lingprop-brain-alignment",
    "stimulus_resources",
)


@pytest.fixture
def narratives_dataset(data_root):
    """Load the Narratives fMRI dataset for the 21styear story."""
    deriv = os.path.join(data_root, _NARRATIVES_DERIV)
    align = os.path.join(data_root, _NARRATIVES_ALIGN)

    if not os.path.isdir(deriv) or not os.path.isdir(align):
        pytest.skip("Narratives fMRI data not available")
    # A datalad clone lists the surfaces as symlinks before their content is
    # downloaded, and ``exists`` is False for such a symlink.
    surfaces = Path(deriv).glob("sub-*/func/*task-21styear*_hemi-L_desc-clean.func.gii")
    if not any(surface.exists() for surface in surfaces):
        pytest.skip("Narratives BOLD content is not downloaded")

    ds = Narratives("21styear")
    ds.load(
        deriv,
        derivatives_path=deriv,
        align_path=align,
    )
    return ds


@pytest.mark.needs_data
class TestNarrativesSmoke:
    """Smoke tests for the Narratives fMRI dataset."""

    def test_loads_with_content(self, narratives_dataset):
        """Loads with non-empty stimuli, words, BOLD data, and participants."""
        assert len(narratives_dataset.stimuli()) > 0, "no stimuli"
        for stim in narratives_dataset.stimuli():
            assert len(stim.layers["words"]) > 0, "stimulus has no words"
        recordings = narratives_dataset.recordings(measure="bold", modality="fmri")
        assert recordings, "no BOLD recordings"
        assert narratives_dataset.participants, "no fMRI participants"

    def test_stimulus_has_word_times(self, narratives_dataset):
        """Stimulus has word_times annotation layer."""
        for stim in narratives_dataset.stimuli():
            assert "word_times" in stim.layers
            times = stim.layers["word_times"].values
            assert len(times) == len(stim.layers["words"])
            # Each entry is (onset, offset) in seconds
            for onset, offset in times:
                assert onset >= 0
                assert offset >= onset

    def test_bold_shape(self, narratives_dataset):
        """BOLD data has 2D shape (n_TRs, n_voxels)."""
        for recording in narratives_dataset.recordings(measure="bold", modality="fmri"):
            assert recording.values.ndim == 2
            assert recording.values.shape[0] > 0  # has TRs
            assert recording.values.shape[1] > 0  # has vertices

    def test_bold_metadata(self, narratives_dataset):
        """BOLD recordings retain sampling and trim metadata."""
        for recording in narratives_dataset.recordings(measure="bold", modality="fmri"):
            assert recording.sample_interval == 1.5
            assert recording.metadata["trim_start_samples"] == 14
            assert recording.metadata["trim_end_samples"] == 9
