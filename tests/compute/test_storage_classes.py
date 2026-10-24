"""Storage classes, the layer field and the unit-bound guard of compute cells.

Every method the driver accepts is either positional (stored per token,
units derived under the nesting guard) or unit-bound (stored at the unit it
was computed for, with the boundary rule named). A unit-bound word cell is
refused on a corpus whose word units are not the model's own words unless
the caller names the rule, and the manifest records the rule either way.
"""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from replica import artifacts as manifests
from replica.compute import artifacts, methods, runner
from replica.compute.cache import TOKEN_SENSITIVE, UNIT_SENSITIVE
from replica.compute.methods import METHOD_SPECS, METHODS, ComputeContext
from replica.compute.units import (
    AGGREGATION_RULES,
    DEFAULT_UNIT_RULE,
    METHOD_STORAGE,
    MULTIWORD_RULES,
    POSITIONAL,
    STORAGE_CLASSES,
    UNIT_BOUND,
    bound_layer,
    boundary_rule,
)
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature import FeatureModelOutput

from .._stimuli import word_stimulus


def _context():
    return ComputeContext("gpt2", 5, 64, 10, 1.0, 0, True, "en", None)


def test_every_method_has_a_storage_class():
    assert set(STORAGE_CLASSES) == set(METHODS)
    assert set(STORAGE_CLASSES.values()) == {POSITIONAL, UNIT_BOUND}


def test_layer_two_methods_name_a_study_and_layer_one_methods_do_not():
    assert set(METHOD_STORAGE) == set(METHODS)
    for method, storage in METHOD_STORAGE.items():
        assert storage.layer in (1, 2), method
        assert (storage.layer == 2) == (storage.study is not None), method
    # Method provenance remains exact even when the study package is private.
    assert {m: s.study for m, s in METHOD_STORAGE.items() if s.layer == 2} == {
        "modelblocks_surprisal": "clark2025wordentropy",
        "wordsprobability_surprisal_buggy": "pimentel2024probability",
        "first_token_entropy_buggy": "pimentel2023anticipation",
        "renyi_entropy_buggy": "pimentel2023anticipation",
        "continuation_entropy_buggy": "giulianelli2024generalized",
    }


def test_bound_layer_is_words_unless_the_rule_binds_another_layer():
    assert bound_layer("restricted_first_token_entropy") == "words"
    assert bound_layer("information_value") == "sentences"
    assert bound_layer("surprisal") is None


def test_token_and_character_methods_are_positional_with_a_rule():
    for method in TOKEN_SENSITIVE | UNIT_SENSITIVE:
        assert STORAGE_CLASSES[method] == POSITIONAL
        assert method in AGGREGATION_RULES
    for method in AGGREGATION_RULES:
        assert STORAGE_CLASSES[method] == POSITIONAL


def test_boundary_rule_defaults_and_refuses_positional_methods():
    assert boundary_rule("restricted_first_token_entropy", None) == DEFAULT_UNIT_RULE
    assert boundary_rule("unigram_surprisal", None) == "corpus_segment_lookup"
    assert boundary_rule("mc_word_entropy", "corpus_units") == "corpus_units"
    assert boundary_rule("information_value", None) == "spacy_sentence"
    assert boundary_rule("surprisal", None) is None
    with pytest.raises(ValueError, match="positional"):
        boundary_rule("surprisal", "corpus_units")


def test_identity_names_class_layer_and_rule():
    context = _context()
    stimulus = word_stimulus("ab cd")
    positional = artifacts.artifact_identity("surprisal", context, [stimulus], {})
    assert positional["storage_class"] == POSITIONAL
    assert positional["layer"] == 1
    assert "unit_rule" not in positional and "unit_layer" not in positional
    assert "study" not in positional
    bound = artifacts.artifact_identity(
        "restricted_first_token_entropy", context, [stimulus], {}
    )
    assert bound["storage_class"] == UNIT_BOUND
    assert bound["unit_rule"] == DEFAULT_UNIT_RULE
    assert bound["unit_layer"] == "words"
    sentences = artifacts.artifact_identity(
        "information_value", context, [stimulus], {}
    )
    assert sentences["unit_rule"] == "spacy_sentence"
    assert sentences["unit_layer"] == "sentences"
    released = artifacts.artifact_identity(
        "modelblocks_surprisal", context, [stimulus], {}
    )
    assert released["layer"] == 2 and released["study"] == "clark2025wordentropy"
    named = artifacts.artifact_identity(
        "restricted_first_token_entropy", context, [stimulus], {}, rule="corpus_units"
    )
    assert named["unit_rule"] == "corpus_units"
    assert named != bound


# ---- the guard in the driver -----------------------------------------------


class _UnitModel:
    """Positional or unit-bound double: one constant value per unit."""

    name = "unit-double"
    vocab = None

    def __init__(self, key):
        self.key = key

    def __call__(self, stimulus, extract=None, target_layer="words"):
        return [
            FeatureModelOutput(
                values={
                    self.key: float(index + 1)
                    * (-1 if self.key == "log_p_observed" else 1)
                },
                metadata={"model_name": self.name},
            )
            for index, _ in enumerate(stimulus.layers[target_layer].spans)
        ]


def _spaced_units():
    """Five units, two of them holding a space inside the unit."""
    text = "of the cat sat in a mat"
    stimulus = Stimulus(text=text, stimulus_id="s0")
    stimulus.add_layer(
        AnnotationLayer(
            "words",
            [Span(0, 6), Span(7, 10), Span(11, 14), Span(15, 19), Span(20, 23)],
            ["of the", "cat", "sat", "in a", "mat"],
        )
    )
    return stimulus


def _install(monkeypatch, tmp_path, stimuli, *, word_spacing=True):
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda name: runner._LoadedDataset(
            deepcopy(stimuli), len(stimuli), "en", word_spacing
        ),
    )
    monkeypatch.setattr(
        runner,
        "resolve_model_snapshot",
        lambda method, context: (
            context.model_id,
            {"source": "unit-double", "revision": "r" * 40},
        ),
    )
    for method, key in (
        ("restricted_first_token_entropy", "restricted_first_token_entropy"),
        ("char_beam", "log_p_observed"),
        ("information_value", "information_value_mean_cosine"),
        ("word_information_value", "infovalue_layer0"),
    ):
        monkeypatch.setitem(
            methods.METHOD_SPECS,
            method,
            replace(
                METHOD_SPECS[method],
                build_model=lambda _c, key=key: _UnitModel(key),
                extract_keys=lambda _c, key=key: [key],
            ),
        )
    return dict(
        model_id="unit-double",
        K=5,
        n_samples=64,
        max_tokens=10,
        alpha=1.0,
        seed=None,
        out_dir=tmp_path,
        add_bos=True,
        force=False,
    )


def _manifest(path):
    return json.loads(manifests.manifest_path(path).read_text())


def test_unit_bound_word_cell_is_refused_where_units_hold_whitespace(
    tmp_path, monkeypatch
):
    # A unit-bound method without a multi-word rule stops before any model loads.
    assert "similarity_surprisal_word" not in MULTIWORD_RULES
    common = _install(
        monkeypatch, tmp_path, [_spaced_units(), word_stimulus("x y", "s1")]
    )
    with pytest.raises(ValueError) as error:
        runner.compute_for("corpus", method="similarity_surprisal_word", **common)
    message = str(error.value)
    assert message.startswith(
        "corpus: similarity_surprisal_word is bound to the units of the words layer "
        "under the rule whitespace_continuation"
    )
    assert "2 of 7 words units contain whitespace" in message
    assert "'of the', 'in a'" in message
    assert "--unit-rule" in message
    assert "no word spacing" not in message
    assert not list(tmp_path.glob("*.pkl"))


def test_multiword_units_are_combined_by_the_methods_rule_and_recorded(
    tmp_path, monkeypatch
):
    common = _install(
        monkeypatch, tmp_path, [_spaced_units(), word_stimulus("x y", "s1")]
    )
    outcome = runner.compute_for(
        "corpus", method="restricted_first_token_entropy", **common
    )
    assert outcome.status == "completed", outcome.error
    manifest = _manifest(outcome.path)
    assert manifest["identity"]["unit_rule"] == DEFAULT_UNIT_RULE
    assert manifest["multiword"] == {
        "over": "whitespace_words",
        "rule": "first",
        "multiword_units": 2,
    }


def test_a_corpus_of_single_words_records_no_multiword_block(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path, [word_stimulus("ab cd ef")])
    outcome = runner.compute_for(
        "corpus", method="restricted_first_token_entropy", **common
    )
    assert outcome.status == "completed", outcome.error
    assert "multiword" not in _manifest(outcome.path)


def test_the_reference_information_value_recipe_refuses_fused_units(
    tmp_path, monkeypatch
):
    common = _install(monkeypatch, tmp_path, [_spaced_units()])
    with pytest.raises(ValueError) as error:
        runner.compute_for("corpus", method="word_information_value", **common)
    assert str(error.value) == (
        "corpus: word_information_value iterates whitespace words under the "
        "reference recipe, but 2 units hold several; pass --segmentation layer "
        "to score the units"
    )
    outcome = runner.compute_for(
        "corpus", method="word_information_value", segmentation="layer", **common
    )
    assert outcome.status == "completed", outcome.error
    assert outcome.path.name.endswith("_layer_corpus.pkl")
    assert _manifest(outcome.path)["multiword"]["rule"] == "unit"


def test_layer_segmentation_is_refused_where_the_reference_recipe_applies(
    tmp_path, monkeypatch
):
    common = _install(monkeypatch, tmp_path, [word_stimulus("ab cd ef")])
    with pytest.raises(ValueError, match="drop --segmentation layer"):
        runner.compute_for(
            "corpus", method="word_information_value", segmentation="layer", **common
        )
    # An unspaced script names the boundary rule its units follow, as any
    # unit-bound method does there; the layer segmentation then scores them.
    unspaced = _install(
        monkeypatch, tmp_path, [word_stimulus("ab cd ef")], word_spacing=False
    )
    outcome = runner.compute_for(
        "corpus",
        method="word_information_value",
        segmentation="layer",
        unit_rule="corpus_units",
        **unspaced,
    )
    assert outcome.status == "completed", outcome.error
    assert "multiword" not in _manifest(outcome.path)


def test_an_empty_unit_is_refused_under_a_multiword_rule(tmp_path, monkeypatch):
    text = "of the cat"
    stimulus = Stimulus(text=text, stimulus_id="s0")
    stimulus.add_layer(
        AnnotationLayer(
            "words", [Span(0, 6), Span(6, 6), Span(7, 10)], ["of the", "", "cat"]
        )
    )
    common = _install(monkeypatch, tmp_path, [stimulus])
    with pytest.raises(ValueError) as error:
        runner.compute_for("corpus", method="restricted_first_token_entropy", **common)
    assert "1 of 3 words units are empty (for example '')" in str(error.value)
    assert "--unit-rule" in str(error.value)


def test_a_named_rule_still_records_the_multiword_block(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path, [_spaced_units()])
    outcome = runner.compute_for(
        "corpus",
        method="restricted_first_token_entropy",
        unit_rule="corpus_units",
        **common,
    )
    assert outcome.status == "completed", outcome.error
    manifest = _manifest(outcome.path)
    assert manifest["identity"]["unit_rule"] == "corpus_units"
    assert manifest["multiword"]["multiword_units"] == 2


def test_a_positional_recipe_on_an_unspaced_corpus_gets_no_rule_advice(
    tmp_path, monkeypatch
):
    common = _install(
        monkeypatch, tmp_path, [word_stimulus("ab cd")], word_spacing=False
    )
    with pytest.raises(ValueError) as error:
        runner.compute_for(
            "corpus", method="wordsprobability_surprisal_buggy", **common
        )
    message = str(error.value)
    assert message.endswith("its loader declares no word spacing")
    assert "--unit-rule" not in message


def test_unit_bound_word_cell_is_refused_without_word_spacing(tmp_path, monkeypatch):
    common = _install(
        monkeypatch, tmp_path, [word_stimulus("ab cd")], word_spacing=False
    )
    with pytest.raises(ValueError, match="its loader declares no word spacing"):
        runner.compute_for("corpus", method="restricted_first_token_entropy", **common)


def test_named_rule_lifts_the_refusal_and_is_recorded(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path, [_spaced_units()])
    outcome = runner.compute_for(
        "corpus",
        method="restricted_first_token_entropy",
        unit_rule="corpus_units",
        **common,
    )
    assert outcome.status == "completed", outcome.error
    identity = _manifest(outcome.path)["identity"]
    assert identity["storage_class"] == UNIT_BOUND
    assert identity["unit_rule"] == "corpus_units"
    assert identity["layer"] == 1


def test_default_rule_is_recorded_on_a_corpus_of_model_words(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path, [word_stimulus("ab cd ef")])
    outcome = runner.compute_for(
        "corpus", method="restricted_first_token_entropy", **common
    )
    assert outcome.status == "completed", outcome.error
    identity = _manifest(outcome.path)["identity"]
    assert identity["unit_rule"] == DEFAULT_UNIT_RULE
    assert identity["storage_class"] == UNIT_BOUND


def test_positional_methods_take_no_rule_and_pass_the_guard(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path, [_spaced_units()])
    outcome = runner.compute_for("corpus", method="char_beam", **common)
    assert outcome.status == "completed", outcome.error
    identity = _manifest(outcome.path)["identity"]
    assert identity["storage_class"] == POSITIONAL and "unit_rule" not in identity
    with pytest.raises(ValueError, match="positional"):
        runner.compute_for("corpus", method="char_beam", unit_rule="x", **common)


def _sentenced(ids):
    stimulus = word_stimulus("One two. Three four.")
    words = stimulus.layers["words"]
    stimulus.add_layer(AnnotationLayer("sentences", list(words.spans), ids))
    return stimulus


def test_sentence_bound_method_checks_sentences_and_records_the_layer(
    tmp_path, monkeypatch
):
    common = _install(
        monkeypatch, tmp_path, [_spaced_units(), _sentenced([1, 1, 2, 2])]
    )
    outcome = runner.compute_for("corpus", method="information_value", **common)
    assert outcome.status == "completed", outcome.error
    identity = _manifest(outcome.path)["identity"]
    assert identity["unit_layer"] == "sentences"
    assert identity["unit_rule"] == "spacy_sentence"
    with pytest.raises(ValueError, match="words units contain whitespace"):
        runner.compute_for("corpus", method="similarity_surprisal_word", **common)


@pytest.mark.parametrize(
    "stimuli,detail",
    [
        ([_sentenced([1, 2, 1, 2])], "2 of 4 sentences units are malformed"),
        ([_sentenced([1, None, 2, 2])], "word without a sentence id"),
        ([Stimulus(text="   ", stimulus_id="blank")], "blank text"),
    ],
)
def test_malformed_sentence_units_are_refused(tmp_path, monkeypatch, stimuli, detail):
    for stimulus in stimuli:
        stimulus.layers.setdefault("words", AnnotationLayer("words", [], []))
    common = _install(monkeypatch, tmp_path, stimuli)
    with pytest.raises(ValueError) as error:
        runner.compute_for("corpus", method="information_value", **common)
    message = str(error.value)
    assert (
        "bound to the units of the sentences layer under the rule spacy_sentence"
        in message
    )
    assert detail in message
