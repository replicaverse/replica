"""Data-free tests for the unified public MECO entry point."""

import pytest

from replica.dataset import MECO, MECO_RELEASES
from replica.dataset.eye_tracking.meco.character import MECO_char
from replica.dataset.eye_tracking.meco.l1 import MECO_L1
from replica.dataset.eye_tracking.meco.waves import (
    MECOL1W2,
    MECOL2W1,
    MECOL2W2,
)


def test_meco_package_exposes_only_unified_public_api():
    from replica.dataset.eye_tracking import meco

    assert meco.MECO is MECO
    assert set(MECO_RELEASES) == {"l1_w1", "l1_w2", "l2_w1", "l2_w2"}
    assert set(meco.__all__) == {
        "LANGUAGES",
        "L1W2_SITE_LANGUAGES",
        "MECO",
        "MECODataset",
        "MECORelease",
        "MECORepresentation",
        "MECO_RELEASES",
    }
    assert not hasattr(meco, "MECO_L1")
    assert not hasattr(meco, "MECO_char")
    assert not hasattr(meco, "MECOL1W2")


@pytest.mark.parametrize(
    "kwargs, expected_type",
    [
        ({}, MECO_L1),
        ({"release": "l1_w1", "language": "de"}, MECO_L1),
        (
            {
                "release": "l1_w1",
                "language": "en",
                "representation": "character",
            },
            MECO_char,
        ),
        ({"release": "l1_w2", "site": "en_uk"}, MECOL1W2),
        ({"release": "l2_w1"}, MECOL2W1),
        ({"release": "l2_w2"}, MECOL2W2),
    ],
)
def test_meco_dispatches_to_focused_loader(kwargs, expected_type):
    assert isinstance(MECO(**kwargs), expected_type)


def test_meco_l1w1_defaults_to_english():
    assert MECO().language == "en"


def test_meco_l1w2_preserves_site_language_metadata():
    dataset = MECO("l1_w2", site="ru_mo")
    assert dataset.site == "ru_mo"
    assert dataset.info.language == "ru"


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        (("not_a_release",), {}, "Unknown MECO release"),
        (("l1_w1",), {"representation": "tokens"}, "representation"),
        (("l1_w1",), {"site": "en_uk"}, "uses language"),
        (("l1_w2",), {}, "requires site"),
        (("l1_w2",), {"language": "en"}, "collection site"),
        (
            ("l1_w2",),
            {"site": "en_uk", "representation": "character"},
            "only for release='l1_w1'",
        ),
        (("l2_w1",), {"language": "en"}, "fixed English"),
        (("l2_w2",), {"site": "en_uk"}, "fixed English"),
    ],
)
def test_meco_rejects_incompatible_options(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        MECO(*args, **kwargs)
