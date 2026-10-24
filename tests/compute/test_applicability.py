"""The applicability contract: which (method, corpus) cells are defined.

One function decides for the compute driver and the coverage table; this
matrix pins its answer for every storage class against every segmentation.
"""

import pytest

from replica.compute.cache import TOKEN_SENSITIVE, UNIT_SENSITIVE
from replica.compute.units import (
    AGGREGATED,
    BOUND_LAYERS,
    DEFINED,
    METHOD_STORAGE,
    MULTIWORD_RULES,
    POSITIONAL,
    TOKEN_UNIT,
    UNDEFINED,
    UNIT_BOUND,
    applicability,
    malformed_units,
)
from replica.core import AnnotationLayer, Span, Stimulus

from .._stimuli import word_stimulus

METHODS = sorted(METHOD_STORAGE)
WORD_BOUND = sorted(
    m
    for m, s in METHOD_STORAGE.items()
    if (s.storage_class == UNIT_BOUND and m not in BOUND_LAYERS) or m in MULTIWORD_RULES
)
TOKEN_POSITIONAL = sorted(
    m
    for m, s in METHOD_STORAGE.items()
    if s.storage_class == POSITIONAL and m not in MULTIWORD_RULES
)


def _layered(text, spans, name="words"):
    stimulus = Stimulus(text=text, stimulus_id="s0")
    stimulus.add_layer(
        AnnotationLayer(name, spans, [text[s.start : s.end] for s in spans])
    )
    return stimulus


def _fused():
    return _layered(
        "of the cat in a mat", [Span(0, 6), Span(7, 10), Span(11, 15), Span(16, 19)]
    )


def _empty_unit():
    return _layered("of the cat", [Span(0, 6), Span(6, 6), Span(7, 10)])


def _single():
    return [word_stimulus("ab cd ef")]


@pytest.mark.parametrize("method", METHODS)
def test_character_and_token_grains_need_a_method_that_scores_them(method):
    chars = applicability(method, "characters", _single())
    tokens = applicability(method, TOKEN_UNIT, _single())
    assert chars.defined == (method in UNIT_SENSITIVE)
    assert tokens.defined == (method in TOKEN_SENSITIVE)
    if not chars.defined:
        assert "character-unit methods" in chars.reason
    if not tokens.defined:
        assert "token-unit methods" in tokens.reason


@pytest.mark.parametrize("method", TOKEN_POSITIONAL)
def test_a_positional_word_cell_needs_word_spacing_except_surprisal(method):
    assert applicability(method, "words", _single()).status == DEFINED
    assert applicability(method, "words", [_fused()]).status == DEFINED
    fit = applicability(method, "words", _single(), word_spacing=False)
    if method == "surprisal" or method not in TOKEN_SENSITIVE:
        # Surprisal derives from characters; a layer scorer needs no tokens.
        assert fit.status == DEFINED
    else:
        assert fit.status == UNDEFINED and "no word spacing" in fit.reason
        assert f"--unit {TOKEN_UNIT}" in fit.reason


def test_a_character_scorer_sums_its_bytes_over_any_layer():
    stimulus = word_stimulus("ab cd ef", "s1")
    stimulus.add_layer(
        AnnotationLayer("pairs", [Span(0, 5), Span(6, 8)], ["ab cd", "ef"])
    )
    for unit in ("words", "pairs", "characters"):
        assert (
            applicability("char_beam", unit, [stimulus], word_spacing=False).status
            == DEFINED
        )
    with pytest.raises(KeyError):
        applicability("no_such_method", "characters", _single())


@pytest.mark.parametrize("method", WORD_BOUND)
def test_a_word_bound_method_on_single_words_is_defined(method):
    fit = applicability(method, "words", _single())
    assert fit == applicability(method, "words", _single(), unit_rule="x")
    assert fit.status == DEFINED and fit.multiword_units == 0


@pytest.mark.parametrize("method", [m for m in WORD_BOUND if m != "unigram_surprisal"])
def test_a_whitespace_bound_method_without_word_spacing_is_undefined(method):
    fit = applicability(method, "words", _single(), word_spacing=False)
    assert fit.status == UNDEFINED
    assert fit.reason.startswith(f"{method} is bound to the units of the words layer")
    assert fit.reason.endswith(
        "its loader declares no word spacing"
        if METHOD_STORAGE[method].storage_class == POSITIONAL
        else "pass --unit-rule NAME to name the boundary rule the cell is computed under"
    )
    named = applicability(method, "words", _single(), word_spacing=False, unit_rule="x")
    assert named.status == DEFINED


def test_lexical_lookup_accepts_unspaced_segments_but_not_empty_units():
    stimulus = _layered("北京大学", [Span(0, 2), Span(2, 4)])
    assert (
        applicability(
            "unigram_surprisal", "words", [stimulus], word_spacing=False
        ).status
        == DEFINED
    )
    empty = applicability(
        "unigram_surprisal", "words", [_empty_unit()], word_spacing=False
    )
    assert empty.status == UNDEFINED and "units are empty" in empty.reason
    assert "no word spacing" not in empty.reason


@pytest.mark.parametrize("method", WORD_BOUND)
def test_multiword_units_take_the_methods_rule_or_are_undefined(method):
    segmentation = "layer" if method == "word_information_value" else "whitespace"
    fit = applicability(method, "words", [_fused()], segmentation=segmentation)
    if method in MULTIWORD_RULES:
        rule = MULTIWORD_RULES[method]
        assert fit.status == (AGGREGATED if rule == "sum" else DEFINED)
        assert fit.rule == rule and fit.multiword_units == 2
        named = applicability(
            method, "words", [_fused()], unit_rule="x", segmentation=segmentation
        )
        assert named == fit
    else:
        assert fit.status == UNDEFINED
        assert (
            "2 of 4 words units contain whitespace (for example 'of the', 'in a')"
            in fit.reason
        )
        named = applicability(method, "words", [_fused()], unit_rule="x")
        assert named.status == DEFINED


@pytest.mark.parametrize("method", WORD_BOUND)
def test_an_empty_unit_is_undefined_under_every_rule(method):
    fit = applicability(method, "words", [_empty_unit()])
    assert fit.status == UNDEFINED
    if method in MULTIWORD_RULES:
        assert "1 of 3 words units are empty (for example '')" in fit.reason
    else:
        assert "2 of 3 words units are empty or contain whitespace" in fit.reason


def test_information_value_segmentation_follows_the_corpus_units():
    fused = applicability("word_information_value", "words", [_fused()])
    assert fused.status == UNDEFINED
    assert fused.reason == (
        "word_information_value iterates whitespace words under the reference "
        "recipe, but 2 units hold several; pass --segmentation layer to score "
        "the units"
    )
    plain = applicability(
        "word_information_value", "words", _single(), segmentation="layer"
    )
    assert plain.status == UNDEFINED and "drop --segmentation layer" in plain.reason
    assert applicability("word_information_value", "words", _single()).status == DEFINED
    # An unspaced script takes the layer segmentation under a named rule.
    unspaced = applicability(
        "word_information_value",
        "words",
        _single(),
        word_spacing=False,
        unit_rule="corpus_units",
        segmentation="layer",
    )
    assert unspaced.status == DEFINED and unspaced.rule is None


def test_a_registered_segmentation_is_checked_at_its_own_layer():
    stimulus = word_stimulus("ab cd ef", "s1")
    stimulus.add_layer(
        AnnotationLayer("pairs", [Span(0, 5), Span(6, 8)], ["ab cd", "ef"])
    )
    fit = applicability("unigram_surprisal", "pairs", [stimulus])
    assert fit.status == AGGREGATED and fit.multiword_units == 1
    fit = applicability("similarity_surprisal_word", "pairs", [stimulus])
    assert fit.status == UNDEFINED and "pairs units contain whitespace" in fit.reason
    fit = applicability("unigram_surprisal", "pairs", [stimulus], word_spacing=False)
    assert fit.status == AGGREGATED and fit.rule == "sum"
    # A whitespace boundary needs word spacing whatever the segmentation.
    fit = applicability("mc_word_entropy", "pairs", [stimulus], word_spacing=False)
    assert not fit.defined and "no word spacing" in fit.reason
    assert applicability("surprisal", "pairs", [stimulus]).status == DEFINED


def test_a_sentence_bound_method_checks_the_sentences_layer():
    stimulus = word_stimulus("One two. Three four.")
    words = stimulus.layers["words"]
    stimulus.add_layer(AnnotationLayer("sentences", list(words.spans), [1, 2, 1, 2]))
    fit = applicability("information_value", "words", [stimulus])
    assert fit.status == UNDEFINED
    assert (
        "bound to the units of the sentences layer under the rule spacy_sentence"
        in fit.reason
    )
    assert "2 of 4 sentences units are malformed" in fit.reason
    stimulus.add_layer(AnnotationLayer("sentences", list(words.spans), [1, 1, 2, 2]))
    assert applicability("information_value", "words", [stimulus]).status == DEFINED
    assert applicability(
        "information_value", "words", [stimulus], word_spacing=False
    ).defined
    stimulus.add_layer(AnnotationLayer("pairs", [Span(0, 8), Span(9, 20)], ["a", "b"]))
    fit = applicability("information_value", "pairs", [stimulus])
    assert fit.status == UNDEFINED
    assert "--unit pairs names another segmentation" in fit.reason


def test_malformed_units_counts_multiword_units():
    offending, total, multiword = malformed_units([_fused(), _empty_unit()], "words")
    assert (total, multiword) == (7, 3)
    assert offending == ["of the", "in a", "of the", ""]


def test_an_unknown_method_raises():
    with pytest.raises(KeyError):
        applicability("no_such_method", "words", _single())
