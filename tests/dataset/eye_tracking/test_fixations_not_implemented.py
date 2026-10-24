"""Pin the ``fixations()`` raise contract for ET loaders that don't expose
raw per-fixation data.

REPLICA standardizes on: every :class:`EyeTrackingDataset` either provides
raw fixations through :meth:`fixations`, or raises
``NotImplementedError`` with a corpus-specific reason. A silent ``[]`` is
refused, because a researcher cannot tell "no data" from "filtered to
zero".

These tests check the raise contract without requiring the data to be
present locally: the loader's ``fixations()`` method should raise as
soon as it is called, regardless of whether ``load()`` succeeded.
"""

import pytest

from replica.dataset.eye_tracking.devarda2023 import DeVarda2023
from replica.dataset.eye_tracking.geco import GECO
from replica.dataset.eye_tracking.rastros import RastrOS
from replica.dataset.eye_tracking.rsc import RSC


@pytest.mark.parametrize(
    "cls,reason_keyword",
    [
        (DeVarda2023, "pre-averaged"),
        (GECO, "word-level"),
        (RSC, "fixation reports"),
        (RastrOS, "word-level"),
    ],
)
def test_fixations_raises(cls, reason_keyword):
    """Each silent-empty loader must raise NotImplementedError when
    ``fixations()`` is called, with a message that explains why."""
    # No load() required, the raise is a class-level contract.
    ds = cls()
    with pytest.raises(NotImplementedError) as excinfo:
        ds.fixations()
    msg = str(excinfo.value).lower()
    assert reason_keyword.lower() in msg, (
        f"{cls.__name__}: error message should mention {reason_keyword!r}; "
        f"got: {excinfo.value!r}"
    )
