"""The canonical predictor catalog: completeness, field invariants, behavior.

This is the source both CI completeness gates derive from, so these tests pin
that every shipped predictor is accounted for.
"""

import numpy as np
import pytest

import replica.feature as feature_pkg
import replica.feature.extractors as extractors_pkg
from replica.feature.diagnostics import MarkovReferenceLM, fingerprint

from .support.catalog import (
    CATALOG,
    PredictorStatus,
    entry_by_name,
)

_EXTRACTOR_SURFACE = set(extractors_pkg.__all__)
_FEATURE_SURFACE = set(feature_pkg.__all__) - _EXTRACTOR_SURFACE

_FINGERPRINTED = [e for e in CATALOG if e.status == PredictorStatus.FINGERPRINTED]
_NEGATIVE = [e for e in CATALOG if e.status == PredictorStatus.NEGATIVE_CONTROL]
_FP_VARIANTS = [(e, kw, exp) for e in _FINGERPRINTED for kw, exp in e.variants]


@pytest.fixture(scope="module")
def ref():
    return MarkovReferenceLM()


class TestCompleteness:
    def test_extractor_surface_exactly_covered(self):
        catalog_ext = {e.name for e in CATALOG if e.surface == "extractor"}
        assert catalog_ext == _EXTRACTOR_SURFACE, {
            "missing from catalog": sorted(_EXTRACTOR_SURFACE - catalog_ext),
            "stale in catalog": sorted(catalog_ext - _EXTRACTOR_SURFACE),
        }

    def test_feature_surface_exactly_covered(self):
        catalog_feat = {e.name for e in CATALOG if e.surface == "feature"}
        assert catalog_feat == _FEATURE_SURFACE, {
            "missing from catalog": sorted(_FEATURE_SURFACE - catalog_feat),
            "stale in catalog": sorted(catalog_feat - _FEATURE_SURFACE),
        }

    def test_names_unique(self):
        names = [e.name for e in CATALOG]
        assert len(names) == len(set(names))

    def test_surface_values_valid(self):
        assert {e.surface for e in CATALOG} == {"extractor", "feature"}


@pytest.mark.parametrize("entry", CATALOG, ids=[entry.name for entry in CATALOG])
def test_catalog_entry_declares_its_verification(entry):
    assert entry.blurb.strip()
    if entry.status in (
        PredictorStatus.VERIFIED,
        PredictorStatus.NOT_A_PREDICTOR,
        PredictorStatus.NEGATIVE_CONTROL,
    ):
        assert entry.detail.strip()
    if entry.status in (
        PredictorStatus.FINGERPRINTED,
        PredictorStatus.NEGATIVE_CONTROL,
    ):
        assert entry.make is not None
        assert entry.surface == "extractor"
    if entry.status == PredictorStatus.FINGERPRINTED:
        assert entry.variants


class TestBehavior:
    @pytest.mark.parametrize(
        "entry,kwargs,expected",
        _FP_VARIANTS,
        ids=[f"{e.name}-{exp}" for e, _, exp in _FP_VARIANTS],
    )
    def test_fingerprinted_entry_hits_expected_spec(self, ref, entry, kwargs, expected):
        report = fingerprint(entry.make(**kwargs), ref)
        assert report.best.name == expected, report.to_text()
        assert report.best.tier == "exact", report.to_text()

    @pytest.mark.parametrize("e", _NEGATIVE, ids=[e.name for e in _NEGATIVE])
    def test_negative_control_matches_nothing(self, ref, e):
        report = fingerprint(e.make(), ref)
        assert report.degenerate or report.best.tier is None, report.to_text()

    def test_pure_noise_matches_nothing(self, ref):
        rng = np.random.default_rng(7)
        n = len(ref.probe_stimulus().layers["words"])
        report = fingerprint(rng.normal(5.0, 1.0, size=n), ref)
        assert report.best.tier is None, report.to_text()


def test_entry_by_name_round_trips():
    assert entry_by_name("Surprisal").status == PredictorStatus.FINGERPRINTED
    with pytest.raises(KeyError):
        entry_by_name("NotAThing")


def test_data_derived_controls_match_their_definition():
    # The "data-derived" controls are positively verifiable here: each equals the
    # data quantity it names (length / position).
    from replica.feature.diagnostics.references.markov import make_stimulus
    from replica.feature.extractors import (
        PrevUnitLength,
        UnitLength,
        UnitPosition,
    )

    words = ["a", "the", "cat", "hello", "x"]
    stim = make_stimulus(words)
    length = UnitLength()
    assert length.name == "unit_length"
    np.testing.assert_array_equal(
        np.asarray(length(stim, None)), [len(w) for w in words]
    )
    np.testing.assert_array_equal(
        np.asarray(UnitPosition()(stim, None)), np.arange(len(words))
    )
    prev = np.asarray(PrevUnitLength()(stim, None))
    assert np.isnan(prev[0])
    np.testing.assert_array_equal(
        prev[1:], [len(words[i - 1]) for i in range(1, len(words))]
    )


def test_paper_keys_are_registered():
    """Every paper an entry cites is a key of the reference table."""
    from .support.catalog import _REFS

    unknown = {
        e.name: set(e.paper) - set(_REFS) for e in CATALOG if set(e.paper) - set(_REFS)
    }
    assert not unknown, unknown
