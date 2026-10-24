"""Every tracked run is indexed, stamped or declared unstamped, and current.

These read the checkout only, so they run in the light tier. A study module
that builds its own results path bypasses the resolver, so its files could
never move to the hub.
"""

import re

import pytest

from replica.studies.records import files as R

_STUDIES = R.studies_with_results()
_OWN_PATH = re.compile(
    r"""Path\(__file__\)\.(?:with_name\("results"\)|parent\s*/\s*"results")"""
)


@pytest.mark.parametrize("study", _STUDIES)
def test_index_is_sound(study: str):
    assert R.validate_index(study) == []


@pytest.mark.parametrize("study", _STUDIES)
def test_no_stamped_run_is_stale(study: str):
    stale = R.stale_runs(study)
    assert not stale, (
        f"{study}: regenerate and restamp, or `replica results stamp {study} RUN "
        f"--validate` after checking the recorded results: {stale}"
    )


def test_study_modules_read_results_through_the_resolver():
    offenders = sorted(
        path.relative_to(R.STUDIES_DIR).as_posix()
        for path in R.STUDIES_DIR.rglob("*.py")
        if _OWN_PATH.search(path.read_text())
    )
    assert not offenders, f"results paths built from __file__: {offenders}"
