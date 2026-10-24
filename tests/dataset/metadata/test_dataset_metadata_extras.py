"""Pin ``info.stimulus_type``, the ``first_pass`` flag on EyeLink loaders,
and the canonical ``paradigm`` strings.

These three live on the loader/yaml side and are easy to drift. The checks
here are cheap (only a couple actually load data; the rest just inspect
strings).
"""

import os

import numpy as np
import pytest

from replica.dataset import CELER, PROVO

pytestmark = pytest.mark.needs_data

# ---------------------------------------------------------------------------
# 1. info.stimulus_type is populated and uses the canonical vocabulary
# ---------------------------------------------------------------------------

_STIMULUS_TYPE_VOCAB = {"single-unit", "phrase", "sentence", "paragraph", "passage"}

_STIMULUS_TYPE_CASES = [
    # (factory_src, data_subpath, expected_stimulus_type)
    (
        "from replica.dataset import PROVO; ds = PROVO()",
        "eye-tracking/original/provo",
        "paragraph",
    ),
    (
        "from replica.dataset import CELER; ds = CELER()",
        "eye-tracking/original/celer",
        "sentence",
    ),
    (
        "from replica.dataset import BSC; ds = BSC()",
        "eye-tracking/original/bsc",
        "sentence",
    ),
    (
        "from replica.dataset import DeVarda2023; ds = DeVarda2023()",
        "eye-tracking/processed/devarda2023",
        "sentence",
    ),
    (
        "from replica.dataset import OneStop; ds = OneStop()",
        "eye-tracking/original/onestop",
        "passage",
    ),
    (
        "from replica.dataset import PoTeC; ds = PoTeC()",
        "eye-tracking/original/potec",
        "passage",
    ),
    (
        "from replica.dataset import CopCo; ds = CopCo()",
        "eye-tracking/original/copco",
        "paragraph",
    ),
    (
        "from replica.dataset import MECO; ds = MECO(language='en')",
        "eye-tracking/original/meco",
        "passage",
    ),
    (
        "from replica.dataset import BrothersKuperberg; ds = BrothersKuperberg()",
        "self-paced-reading/original/brothers_kuperberg",
        "sentence",
    ),
    (
        "from replica.dataset import NaturalStories; ds = NaturalStories()",
        "self-paced-reading/original/natural_stories",
        "passage",
    ),
    (
        "from replica.dataset import Frank2015ERP; ds = Frank2015ERP()",
        "eeg/original/frank2015_erp",
        "sentence",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("factory_src,subpath,expected", _STIMULUS_TYPE_CASES)
def test_info_stimulus_type_after_load(factory_src, subpath, expected, data_root):
    """``info.stimulus_type`` must be populated and match the audit value."""
    full = os.path.join(data_root, subpath)
    if not os.path.isdir(full):
        pytest.skip(f"data dir not present: {full}")
    ns = {}
    exec(factory_src, ns)
    ds = ns["ds"]
    ds.load(full)
    assert ds.info.stimulus_type == expected, (
        f"expected stimulus_type={expected!r}, got {ds.info.stimulus_type!r}"
    )
    assert ds.info.stimulus_type in _STIMULUS_TYPE_VOCAB


# ---------------------------------------------------------------------------
# 2. first_pass flag on EyeLink loaders
# ---------------------------------------------------------------------------


def _assert_first_pass_flag_restricts_first_run(ds):
    """The flag is stored beside ``first_run`` and ``first_run_in_first_pass``
    keeps the run where the flag is 1 and masks it where the flag is 0."""
    assert "first_pass" in ds.measures
    assert "first_run" in ds.measures
    # The flag is a condition, so it is read off the full view.
    flags = {
        (d.stimulus.stimulus_id, d.participant_id): d.values
        for d in ds.data()
        if d.measure_name == "first_pass"
    }
    first_run = {
        (d.stimulus.stimulus_id, d.participant_id): d.values
        for d in ds.data(measure="first_run")
    }
    assert len(flags) == len(first_run) > 0
    for flag in flags.values():
        assert set(np.unique(flag[~np.isnan(flag)])) <= {0.0, 1.0}
    for datum in ds.data(measure="first_run_in_first_pass"):
        key = (datum.stimulus.stimulus_id, datum.participant_id)
        flag, run = flags[key], first_run[key]
        np.testing.assert_array_equal(datum.values[flag == 1], run[flag == 1])
        assert np.isnan(datum.values[flag == 0]).all()


@pytest.mark.slow
def test_first_pass_flag_provo(data_root):
    """PROVO reads the flag off ``IA_SKIP``."""
    path = os.path.join(data_root, "eye-tracking/original/provo")
    if not os.path.isdir(path):
        pytest.skip(f"PROVO data not present: {path}")
    ds = PROVO()
    ds.load(path)
    _assert_first_pass_flag_restricts_first_run(ds)


@pytest.mark.slow
def test_first_pass_flag_celer(data_root):
    """CELER reads the flag off ``IA_SKIP``."""
    path = os.path.join(data_root, "eye-tracking/original/celer")
    if not os.path.isdir(path):
        pytest.skip(f"CELER data not present: {path}")
    ds = CELER()
    ds.load(path)
    _assert_first_pass_flag_restricts_first_run(ds)


# ---------------------------------------------------------------------------
# 3. Paradigm canonical vocabulary
# ---------------------------------------------------------------------------

_PARADIGM_VOCAB = {"eye-tracking", "self-paced-reading", "eeg", "fmri"}

_PARADIGM_CASES = [
    (
        "from replica.dataset import PROVO; ds = PROVO()",
        "eye-tracking/original/provo",
        "eye-tracking",
    ),
    (
        "from replica.dataset import NaturalStories; ds = NaturalStories()",
        "self-paced-reading/original/natural_stories",
        "self-paced-reading",
    ),
    (
        "from replica.dataset import BrothersKuperberg; ds = BrothersKuperberg()",
        "self-paced-reading/original/brothers_kuperberg",
        "self-paced-reading",
    ),
    (
        "from replica.dataset.spr.ucl import UCL_SPR; ds = UCL_SPR()",
        "self-paced-reading/original/ucl",
        "self-paced-reading",
    ),
    (
        "from replica.dataset import BrownSPR; ds = BrownSPR()",
        "self-paced-reading/processed/brown_spr_shain2024logarithmic",
        "self-paced-reading",
    ),
    (
        "from replica.dataset import Frank2015ERP; ds = Frank2015ERP()",
        "eeg/original/frank2015_erp",
        "eeg",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("factory_src,subpath,expected", _PARADIGM_CASES)
def test_info_paradigm_canonical(factory_src, subpath, expected, data_root):
    """``info.paradigm`` must use the canonical hyphenated vocabulary
    (e.g. ``self-paced-reading``).
    """
    full = os.path.join(data_root, subpath)
    if not os.path.isdir(full):
        pytest.skip(f"data dir not present: {full}")
    ns = {}
    exec(factory_src, ns)
    ds = ns["ds"]
    ds.load(full)
    assert ds.info.paradigm == expected
    assert ds.info.paradigm in _PARADIGM_VOCAB
