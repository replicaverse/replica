"""Pin ``info.language`` for every dataset loader.

``DatasetInfo.language`` defaults to ``"en"`` in ``dataset/contracts.py``. A
loader that omits ``language=`` silently inherits that default and mislabels
a non-English corpus (e.g. PoTeC as ``en`` for German, BSC as ``en`` for
Chinese), so cross-language benchmarks would load English models for them.

These tests do not load any data, they only inspect metadata declared
on the class, so they're cheap to run and don't depend on
``--data-root``.
"""

import os

import pytest

from replica.dataset import (
    BSC,
    CELER,
    MECO,
    PROVO,
    SBSAT,
    BrothersKuperberg,
    BrownSPR,
    CopCo,
    DeVarda2023,
    Dundee,
    EMTeC,
    Frank2015ERP,
    NaturalStories,
    OneStop,
    PoTeC,
    RastrOS,
    ZuCo2,
)
from replica.dataset.eye_tracking.geco import GECO
from replica.dataset.eye_tracking.ucl import UCL_ET
from replica.dataset.spr.ucl import UCL_SPR

# (loader factory, expected info.language). Use factories so loaders that
# require an __init__ argument are not constructed at module import.
_CASES = [
    # English ET corpora
    (lambda: PROVO, "en"),
    (lambda: Dundee, "en"),
    (lambda: CELER, "en"),
    (lambda: OneStop, "en"),
    (lambda: EMTeC, "en"),
    (SBSAT, "en"),
    (lambda: ZuCo2, "en"),
    (lambda: DeVarda2023, "en"),
    (lambda: UCL_ET, "en"),
    # Non-English ET corpora, these were the latent bugs
    (lambda: PoTeC, "de"),
    (lambda: BSC, "zh"),
    (lambda: CopCo, "da"),
    (lambda: RastrOS, "pt"),
    # Per-language ET (instance-driven)
    (lambda: MECO(language="de"), "de"),
    (lambda: MECO(language="fi"), "fi"),
    (lambda: MECO(language="ru", representation="character"), "ru"),
    (lambda: GECO(group="monolingual"), "en"),
    (lambda: GECO(group="l1"), "nl"),
    (lambda: GECO(group="l2"), "en"),
    # SPR corpora (all English)
    (lambda: NaturalStories, "en"),
    (lambda: BrownSPR, "en"),
    (lambda: BrothersKuperberg, "en"),
    (lambda: UCL_SPR, "en"),
    # EEG corpora
    (lambda: Frank2015ERP, "en"),
]


@pytest.mark.parametrize("factory,expected", _CASES)
def test_info_language_declared(factory, expected):
    """Every loader must declare ``info.language`` matching the corpus's
    actual language, so the silent ``language="en"`` default never mislabels
    a non-English corpus (which would make cross-language benchmarks load
    English models for it).
    """
    obj = factory()
    actual = obj.info.language
    assert actual == expected, (
        f"{obj.__class__.__name__ if not isinstance(obj, type) else obj.__name__}: "
        f"expected info.language={expected!r}, got {actual!r}"
    )


# Post-load checks: ``_load_metadata()`` overwrites ``self.info`` from the
# dataset's YAML. The class-level/init-time language gets clobbered. These
# cases pin the post-load value, which is what cross-language benchmarks
# actually see. Marked ``slow`` because they touch real data.
_POST_LOAD_CASES = [
    # (factory_src, data_path_relative_to_data_root, expected_post_load_language)
    (
        "from replica.dataset import PROVO; ds = PROVO()",
        "eye-tracking/original/provo",
        "en",
    ),
    (
        "from replica.dataset import PoTeC; ds = PoTeC()",
        "eye-tracking/original/potec",
        "de",
    ),
    (
        "from replica.dataset import BSC; ds = BSC()",
        "eye-tracking/original/bsc",
        "zh",
    ),
    (
        "from replica.dataset import CopCo; ds = CopCo()",
        "eye-tracking/original/copco",
        "da",
    ),
    (
        "from replica.dataset import MECO; ds = MECO(language='de')",
        "eye-tracking/original/meco",
        "de",
    ),
    (
        "from replica.dataset import MECO; ds = MECO(language='en')",
        "eye-tracking/original/meco",
        "en",
    ),
    (
        "from replica.dataset import MECO; "
        "ds = MECO(language='en', representation='character')",
        "eye-tracking/processed/meco_char",
        "en",
    ),
    (
        "from replica.dataset.eye_tracking.geco import GECO; ds = GECO(group='l1')",
        "eye-tracking/original/geco",
        "nl",
    ),
    (
        "from replica.dataset.eye_tracking.geco import GECO; ds = GECO(group='monolingual')",
        "eye-tracking/original/geco",
        "en",
    ),
]


@pytest.mark.needs_data
@pytest.mark.slow
@pytest.mark.parametrize("factory_src,subpath,expected", _POST_LOAD_CASES)
def test_info_language_after_load(factory_src, subpath, expected, data_root):
    """``info.language`` must still be correct after ``ds.load(...)``.

    Regression: ``_load_metadata()`` replaces ``self.info`` with the YAML's
    contents. Per-instance overrides (MECO/GECO language argument) and
    class-level declarations get clobbered. Loaders that need to preserve
    a per-instance language must re-pin ``self.info.language`` after
    ``_load_metadata()``; YAMLs whose corpus is single-language must
    declare ``language:`` explicitly.
    """
    full = os.path.join(data_root, subpath)
    if not os.path.isdir(full):
        pytest.skip(f"data dir not present: {full}")
    ns = {}
    exec(factory_src, ns)
    ds = ns["ds"]
    ds.load(full)
    actual = ds.info.language
    assert actual == expected, (
        f"after load(), expected info.language={expected!r}, got {actual!r}"
    )
