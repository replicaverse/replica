"""Correctness tests for the frequency-derived predictors.

Pins:

* :class:`WordfreqSource`, wraps the `wordfreq` PyPI package; returns the
  base-free probability ``p(word)`` per :class:`FrequencySource` Protocol.
* :class:`LogFrequency`, Feature applying ``log₁₀`` to the source probability.
* :class:`UnigramSurprisal`, Feature exposing ``-log₂ p`` (the
  standard psycholinguistic "unigram surprisal in bits" control).

The two Features are tied by an exact identity:
``UnigramSurprisal = -log₂(10) · LogFrequency``. We pin that against
hand-computed values for a tiny stimulus so future refactors can't
silently invert the sign or scramble the base conversion.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from replica.feature import (
    LogFrequency,
    SubtlexSource,
    UnigramSurprisal,
    UserTable,
    WordfreqSource,
)
from tests._stimuli import word_stimulus as _stim

# ---------------------------------------------------------------------------
# WordfreqSource: smoke + multi-language coverage
# ---------------------------------------------------------------------------


@pytest.mark.dependency
def test_wordfreq_source_returns_probability_for_known_word():
    """``probability('the', 'en')`` must equal ``word_frequency('the','en')``
    from the upstream wordfreq library; pin the base-free contract so the
    feature layer owns the log base."""
    import wordfreq

    src = WordfreqSource(language="en")
    p = wordfreq.word_frequency("the", "en", wordlist="best")
    assert p > 0
    assert math.isclose(src.probability("the"), p, rel_tol=1e-12)


@pytest.mark.dependency
def test_wordfreq_source_returns_nan_for_unknown_word():
    src = WordfreqSource(language="en")
    # ``flibbertigibbetzzz`` will not be in wordfreq's vocabulary.
    val = src.probability("flibbertigibbetzzz")
    assert math.isnan(val)


@pytest.mark.parametrize(
    ("lang", "word"),
    [("de", "die"), ("fr", "le"), ("zh", "的"), ("ru", "и")],
)
@pytest.mark.filterwarnings("ignore:pkg_resources is deprecated as an API:UserWarning")
@pytest.mark.dependency
def test_wordfreq_source_multilingual(lang, word):
    """Known native words must resolve in the multilingual MECO languages."""
    value = WordfreqSource(language=lang).probability(word)
    assert math.isfinite(value) and 0 < value <= 1


# ---------------------------------------------------------------------------
# LogFrequency + UnigramSurprisal: identity + sign + base
# ---------------------------------------------------------------------------


def test_unigram_surprisal_equals_negative_log2_of_p_for_user_table():
    """For a hand-coded probability ``p = 0.01``,
    ``UnigramSurprisal('word') = -log₂(0.01) ≈ 6.64385619 bits``.
    Verify exact equality with a synthetic UserTable so the test is
    independent of wordfreq's internal numbers."""
    # UserTable holds the base-free probability p(foo) = 0.01.
    # log₁₀(0.01) = -2, so ``LogFrequency`` returns -2.
    # -log₂(0.01) = log₂(100) ≈ 6.643856189774724 bits.
    table = UserTable({"foo": 0.01}, language="en")
    stim = _stim("foo bar", "t")

    log_freq = LogFrequency(table)(stim, outputs=None)
    surprisal = UnigramSurprisal(table)(stim, outputs=None)

    assert math.isclose(float(log_freq[0]), -2.0, abs_tol=1e-12)
    assert math.isclose(float(surprisal[0]), 6.643856189774724, abs_tol=1e-9)
    # "bar" is not in the table → NaN under both features.
    assert math.isnan(log_freq[1])
    assert math.isnan(surprisal[1])


@pytest.mark.dependency
def test_a_fused_interest_area_sums_its_words_log_probabilities():
    """The compute model scores a unit of two whitespace words as the sum of
    their unigram log probabilities: p(foo) p(bar) = 0.01 * 0.1 = 0.001."""
    from replica.core import AnnotationLayer, Span, Stimulus
    from replica.feature.frequency.model import UnigramSurprisalLM

    table = UserTable({"foo": 0.01, "bar": 0.1}, language="en")
    text = "foo bar baz"
    stim = Stimulus(text=text, stimulus_id="t")
    stim.add_layer(
        AnnotationLayer("words", [Span(0, 7), Span(8, 11)], ["foo bar", "baz"])
    )

    outputs = UnigramSurprisalLM(table)(stim)

    assert len(outputs) == 2
    assert math.isclose(
        outputs[0].values["log_p_observed"], math.log(0.001), abs_tol=1e-12
    )
    assert math.isclose(
        outputs[0].values["unigram_surprisal"], -math.log2(0.001), abs_tol=1e-9
    )
    assert math.isnan(outputs[1].values["unigram_surprisal"])


@pytest.mark.parametrize("feature", [LogFrequency, UnigramSurprisal])
def test_user_table_keeps_distinct_case_keys_through_features(feature):
    source = UserTable({"The": 0.2, "the": 0.1})
    actual = feature(source)(_stim("The the", "t"), outputs=None)
    expected = np.log10([0.2, 0.1]) if feature is LogFrequency else -np.log2([0.2, 0.1])
    np.testing.assert_allclose(actual, expected)
    assert source.probability("The") == 0.2


def test_user_table_rejects_non_probabilities():
    """UserTable values are p(word) in (0, 1]; passing log-frequencies
    (negative) or counts (>1) must raise. Log-frequencies would otherwise
    yield NaN predictors (log10 of a negative)."""
    for bad in (-2.0, 5.0, 0.0):
        with pytest.raises(ValueError, match="probabilit"):
            UserTable({"x": bad})
    UserTable({"x": 0.01})  # valid
    UserTable({"x": float("nan")})  # NaN allowed (unknown unit)


def test_wikitext_source_owns_lowercasing_without_changing_feature_values(tmp_path):
    from replica.feature.frequency.model import UnigramSurprisalLM
    from replica.feature.frequency.sources import WikiText103Source

    path = tmp_path / "unigrams.csv"
    path.write_text("the,-2\ncat,-3\n")
    source = WikiText103Source(str(path))
    source._tokenize = str.split
    source._detokenize = " ".join
    assert source.probability("The!") == pytest.approx(math.exp(-2))
    outputs = UnigramSurprisalLM(source)(_stim("The cat", "t"))
    np.testing.assert_allclose([o.values["log_p_observed"] for o in outputs], [-2, -3])


def _write_tiny_subtlex(tmp_path):
    """Build a minimal SUBTLEX-shaped TSV with hand-coded SUBTLWF values."""
    content = (
        "Word\tFREQcount\tCDcount\tFREQlow\tCdlow\tSUBTLWF\tLg10WF\tSUBTLCD\tLg10CD\n"
        "the\t1500000\t8000\t1400000\t8000\t30000.00\t6.18\t100.00\t3.92\n"
        "cat\t750\t500\t700\t490\t15.00\t2.88\t6.10\t0.79\n"
        "galleon\t6\t6\t6\t6\t0.12\t0.78\t0.07\t-1.14\n"
    )
    fp = tmp_path / "subtlex_en.txt"
    fp.write_text(content, encoding="utf-8")
    return fp


def test_subtlex_source_parses_standard_format(tmp_path):
    """Parse a hand-coded SUBTLEX TSV and verify the probability
    conversion is ``p = SUBTLWF · 10⁻⁶``."""
    fp = _write_tiny_subtlex(tmp_path)
    src = SubtlexSource(fp, language="en")
    # the:     SUBTLWF=30000  → p = 30000e-6 = 0.03
    # cat:     SUBTLWF=15     → p = 15e-6
    # galleon: SUBTLWF=0.12   → p = 0.12e-6
    assert math.isclose(src.probability("the"), 30000 * 1e-6, rel_tol=1e-9)
    assert math.isclose(src.probability("cat"), 15.0 * 1e-6, rel_tol=1e-9)
    assert math.isclose(src.probability("galleon"), 0.12 * 1e-6, rel_tol=1e-9)
    # Case-insensitive lookup
    assert src.probability("THE") == src.probability("the")
    # Unknown word → NaN
    assert math.isnan(src.probability("flibbertigibbet"))


def test_subtlex_source_uses_per_language_column_defaults(tmp_path):
    """Per-language SUBTLEX files name different columns *and* publish
    different quantities. The DE release's ``WFfreqcount`` is a raw
    occurrence count, so it must be normalized by the summed count."""
    de_content = "Word\tWFfreqcount\tOtherCol\nund\t40000\tfoo\nHund\t10000\tbar\n"
    fp = tmp_path / "subtlex_de.txt"
    fp.write_text(de_content, encoding="utf-8")

    # No column override → must use the DE convention, counts included.
    src = SubtlexSource(fp, language="de")
    total = 40000 + 10000
    assert math.isclose(src.probability("und"), 40000 / total, rel_tol=1e-9)
    assert math.isclose(src.probability("hund"), 10000 / total, rel_tol=1e-9)
    # Reading these as per-million would have put p far below the truth.
    assert src.probability("und") > 40000 * 1e-6


def test_subtlex_source_count_column_yields_a_distribution(tmp_path):
    """An explicit ``count_col`` normalizes to probabilities summing to 1."""
    content = "W\tN\napple\t1\nbanana\t3\n"
    fp = tmp_path / "counts.tsv"
    fp.write_text(content, encoding="utf-8")
    src = SubtlexSource(fp, language="xx", word_col="W", count_col="N")
    assert math.isclose(src.probability("apple"), 0.25, rel_tol=1e-12)
    assert math.isclose(src.probability("banana"), 0.75, rel_tol=1e-12)


def test_subtlex_source_rejects_a_count_column_passed_as_per_million(tmp_path):
    """Counts interpreted on the wrong scale reject probabilities above one."""
    content = "Word\tSUBTLWF\nund\t4000000\n"
    fp = tmp_path / "mislabelled.tsv"
    fp.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="not per-million"):
        SubtlexSource(fp, language="en")


def test_subtlex_source_rejects_both_column_kinds(tmp_path):
    """The column kind is declared by which argument is passed, so
    passing both is ambiguous."""
    fp = _write_tiny_subtlex(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        SubtlexSource(fp, freq_per_million_col="SUBTLWF", count_col="SUBTLWF")


def test_subtlex_source_explicit_column_override(tmp_path):
    """Custom files with non-standard column names work when columns
    are passed explicitly."""
    custom = "MY_WORD\tFREQ_PM\nfoo\t100.0\n"
    fp = tmp_path / "custom.tsv"
    fp.write_text(custom, encoding="utf-8")
    src = SubtlexSource(
        fp,
        language="xx",
        word_col="MY_WORD",
        freq_per_million_col="FREQ_PM",
    )
    assert math.isclose(src.probability("foo"), 100 * 1e-6, rel_tol=1e-9)


def test_subtlex_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="SUBTLEX"):
        SubtlexSource(tmp_path / "does_not_exist.txt")


@pytest.mark.needs_data
def test_subtlex_source_real_file_if_present():
    """If the official SUBTLEX-US text file is present in the repo,
    spot-check a couple of high-frequency words. Skips otherwise.
    """
    p = Path("noisy_channel_model/data/SUBTLEXus74286wordstextversion.txt")
    if not p.exists():
        pytest.skip("real SUBTLEX file not present")
    src = SubtlexSource(p, language="en")
    # "the" should have very high frequency: p ≈ 3% (10⁻¹·⁵)
    assert 0.01 < src.probability("the") < 0.1
    # "galleon" should be rare
    g = src.probability("galleon")
    if not math.isnan(g):
        assert g < 1e-5
    # Unigram-surprisal sign check: "galleon" surprisal > "the" surprisal
    feat = UnigramSurprisal(src)
    stim = _stim("the galleon", "t")
    s = feat(stim, outputs=None)
    if not math.isnan(s[0]) and not math.isnan(s[1]):
        assert s[1] > s[0]


def test_log_frequency_and_unigram_surprisal_return_nan_without_source():
    """No source set → all-NaN array, same length as the stimulus's
    unit layer. The pipeline must not crash."""
    stim = _stim("hello world", "t")
    lf = LogFrequency(source=None)(stim, outputs=None)
    us = UnigramSurprisal(source=None)(stim, outputs=None)
    assert lf.shape == (2,) and np.isnan(lf).all()
    assert us.shape == (2,) and np.isnan(us).all()
