"""Lazy package root: exports and submodules resolve on attribute access."""

import types

import pytest

import replica as pg


def test_submodules_load_lazily_as_attributes():
    assert isinstance(pg.dataset, types.ModuleType)
    assert isinstance(pg.feature, types.ModuleType)
    assert isinstance(pg.analysis, types.ModuleType)


def test_exports_resolve_from_owning_subpackage():
    assert callable(pg.evaluate)
    assert pg.EncodingConfig is pg.analysis.EncodingConfig
    assert pg.Stimulus is pg.core.Stimulus
    assert pg.Recording is pg.core.Recording
    assert pg.Modality is pg.core.Modality


@pytest.mark.parametrize(
    "name",
    ["evaluate_encoding", "evaluate_continuous_encoding", "BoldData", "FMRIDataset"],
)
def test_retired_fmri_only_api_is_not_exported(name):
    with pytest.raises(AttributeError):
        getattr(pg, name)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no_such_thing"):
        _ = pg.no_such_thing


def test_dir_lists_exports_and_submodules():
    listing = dir(pg)
    assert "evaluate" in listing
    assert "dataset" in listing
