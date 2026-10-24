"""MECO loaders declare ISO 639-1 language codes.

The release spells Dutch ``du``, Estonian ``ee``, Greek ``gr`` and Spanish
``sp``; ``DatasetInfo.language`` feeds language-conditioned predictors
(wordfreq), so a site code silently breaks or mislabels them.
"""

import pytest

from replica.dataset.eye_tracking.meco.character import MECO_char
from replica.dataset.eye_tracking.meco.l1 import MECO_L1
from replica.dataset.eye_tracking.meco.releases import FILE_TO_ISO, LANGUAGES

#: Languages wordfreq genuinely lacks; their frequency cells are refused,
#: never mislabelled.
WORDFREQ_UNSUPPORTED = {"et"}


@pytest.mark.parametrize("code", LANGUAGES)
@pytest.mark.parametrize("cls", [MECO_L1, MECO_char])
def test_info_language_is_iso(cls, code):
    info_lang = cls(language=code).info.language
    assert info_lang == FILE_TO_ISO.get(code, code)
    assert info_lang not in ("du", "ee", "gr", "sp"), "site code leaked"


@pytest.mark.parametrize("code", LANGUAGES)
@pytest.mark.dependency
def test_declared_language_resolves_in_wordfreq_or_is_declared_unsupported(code):
    from wordfreq import word_frequency

    lang = MECO_L1(language=code).info.language
    if lang in WORDFREQ_UNSUPPORTED:
        with pytest.raises(LookupError):
            word_frequency("test", lang)
    else:
        word_frequency("test", lang)
