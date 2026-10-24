"""Run docs/tutorials/extending_replica.ipynb and check its results.

Every code cell runs in order in one namespace, as a notebook kernel runs it,
so a change that breaks the extension points it teaches (a Dataset, FeatureModel,
Feature, FrequencySource and Study) fails here. The toy study
runs end to end with no external data and no registry change.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.docs._notebooks import run_code_cells

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "tutorials"
    / "extending_replica.ipynb"
)


@pytest.fixture(scope="module")
def tutorial():
    """The namespace the tutorial's code cells leave behind."""
    return run_code_cells(NOTEBOOK)


def test_demo_study_runs_end_to_end(tutorial):
    results = tutorial["results"]
    assert len(results) == 1
    r = results[0]
    assert r.feature_name == "vowel_group_count"
    assert r.dv_name == "spr_rt"
    assert np.isfinite(r.delta_ll)


def test_demo_study_metadata_renders_from_bibtex(tutorial):
    study = tutorial["study"]
    assert study.name == "demo2099reading"
    # The Study base derives ``paper`` from ``bibtex``.
    assert "Doe" in study.paper
    assert "2099" in study.paper


def test_custom_model_and_feature_run_through_evaluate(tutorial):
    results = tutorial["custom_results"]
    assert len(results) == 1
    result = results[0]
    assert result.feature_name == "surprisal"
    assert result.dv_name == "spr_rt"
    assert np.isfinite(result.delta_ll)


def test_custom_frequency_source_applies_base_in_the_feature(tutorial):
    table, stimulus = tutorial["MY_TABLE"], tutorial["frequency_stimulus"]
    source = tutorial["MyFreq"]()
    log_freq = tutorial["LogFrequency"](source)(stimulus, None)
    surprisal = tutorial["UnigramSurprisal"](source)(stimulus, None)
    assert np.isclose(log_freq[0], np.log10(table["the"]))
    assert np.isclose(log_freq[1], np.log10(table["cat"]))
    assert np.isclose(surprisal[1], -np.log2(table["cat"]))
