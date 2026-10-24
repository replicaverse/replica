"""Storage-class and unit-rule edges of the compute driver.

A positional method takes no boundary rule and its per-token table
aggregates under one fixed rule per method; a unit-bound method has no
aggregation rule and no token or character table. A derived word cell trusts
only a token table whose manifest digest matches the payload. An option a
method's model ignores must not reach the identity of a cell whose name does
not carry it.
"""

import json
import pickle
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from replica import artifacts as manifests
from replica.compute import methods, runner
from replica.compute.cache import (
    SCOPE_SENSITIVE,
    TOKEN_SENSITIVE,
    cache_filename,
    resolve_context_scope,
)
from replica.compute.methods import METHODS, ComputeContext
from replica.compute.units import (
    AGGREGATION_RULES,
    METHOD_STORAGE,
    MULTIWORD_RULES,
    POSITIONAL,
    STORAGE_CLASSES,
    TOKEN_UNIT,
    UNIT_BOUND,
    boundary_rule,
    derivation_plan,
    derive_units,
)
from replica.context import SENTENCE_LAYER
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.core.annotation.alignment import nested_members
from replica.utils.digest import file_sha256
from tests.feature.core.annotation.test_token_tables import _lm

from .._stimuli import word_stimulus

UNIT_BOUND_METHODS = sorted(m for m, c in STORAGE_CLASSES.items() if c == UNIT_BOUND)
POSITIONAL_METHODS = sorted(m for m, c in STORAGE_CLASSES.items() if c == POSITIONAL)


@pytest.mark.parametrize("method", UNIT_BOUND_METHODS)
def test_unit_bound_methods_have_no_aggregation_rule(method):
    assert method not in AGGREGATION_RULES
    assert method not in TOKEN_SENSITIVE
    with pytest.raises(KeyError):
        derivation_plan(method, ["x"])
    assert boundary_rule(method, None) is not None
    assert boundary_rule(method, "corpus_units") == "corpus_units"


@pytest.mark.parametrize("method", sorted(MULTIWORD_RULES))
def test_multiword_rules_belong_to_word_scoring_methods(method):
    # A multi-word rule combines whitespace words an estimator scored itself,
    # so it never meets a token table's aggregation rule; the one positional
    # method with a rule is a released recipe that scores whitespace words.
    assert MULTIWORD_RULES[method] in ("sum", "unit", "first")
    assert method not in AGGREGATION_RULES
    assert method not in TOKEN_SENSITIVE
    storage = METHOD_STORAGE[method]
    assert storage.storage_class == UNIT_BOUND or storage.layer == 2


def test_multiword_rules_match_the_estimators():
    expected = {
        "unigram_surprisal": "sum",
        "wordsprobability_surprisal": "sum",
        "wordsprobability_surprisal_buggy": "sum",
        "modelblocks_surprisal": "sum",
        "mc_word_entropy": "unit",
        "forward_word_lookahead_entropy": "unit",
        "mmc_unigram_surprisal": "unit",
        "word_information_value": "unit",
        "restricted_first_token_entropy": "first",
        "restricted_first_token_renyi_entropy": "first",
    }
    assert MULTIWORD_RULES == expected


@pytest.mark.parametrize("method", POSITIONAL_METHODS)
def test_positional_methods_refuse_a_rule(method):
    assert boundary_rule(method, None) is None
    with pytest.raises(ValueError, match="positional"):
        boundary_rule(method, "whitespace_continuation")


def test_log_probabilities_sum_and_entropies_take_the_first_token():
    assert derivation_plan("surprisal", ["log_p_observed"]) == (
        {"log_p_observed": "sum"},
        "surprisal_bits",
    )
    assert derivation_plan("first_token_entropy", ["token_entropy"]) == (
        {"token_entropy": "first"},
        None,
    )
    assert derivation_plan("token_renyi_entropy", ["token_renyi_entropy_alpha0.5"]) == (
        {"token_renyi_entropy_alpha0.5": "first"},
        None,
    )
    # Two keys in one table: the bits finisher is tied to the lone log_p key.
    plan, finisher = derivation_plan("surprisal", ["log_p_observed", "token_entropy"])
    assert plan == {"log_p_observed": "sum", "token_entropy": "sum"}
    assert finisher is None


def test_derive_units_refuses_an_unknown_rule_and_source_unit():
    stimulus = word_stimulus("ab cd")
    rows = _lm().score_tokens(stimulus)
    with pytest.raises(ValueError, match="unknown aggregation rule 'mean'"):
        derive_units(
            stimulus,
            rows,
            unit=TOKEN_UNIT,
            layer="words",
            key="log_p_observed",
            rule="mean",
        )
    with pytest.raises(ValueError, match="unknown source unit 'bytes'"):
        derive_units(
            stimulus,
            rows,
            unit="bytes",
            layer="words",
            key="log_p_observed",
            rule="sum",
        )


# ---- the driver --------------------------------------------------------------

STIMULI = [word_stimulus("The cats sat on a mat.", "s0"), word_stimulus("A dog.", "s1")]


def _install(monkeypatch, tmp_path, stimuli=STIMULI, *, word_spacing=True):
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
            {"source": "test-tokens", "revision": "r" * 40},
        ),
    )
    for method in TOKEN_SENSITIVE:
        monkeypatch.setitem(
            methods.METHOD_SPECS,
            method,
            replace(
                methods.METHOD_SPECS[method],
                build_model=lambda context: _lm(bos=context.add_bos),
            ),
        )
    return dict(
        model_id="test-tokens",
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


@pytest.mark.parametrize("unit", ["tokens", "characters"])
def test_unit_bound_method_has_no_fine_grained_table(tmp_path, monkeypatch, unit):
    common = _install(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=f"--unit {unit} is not supported"):
        runner.compute_for(
            "corpus", method="restricted_first_token_entropy", unit=unit, **common
        )
    with pytest.raises(ValueError, match=f"--unit {unit} is not supported"):
        runner.compute_for(
            "corpus",
            method="restricted_first_token_entropy",
            unit=unit,
            unit_rule="corpus_units",
            **common,
        )
    assert not list(tmp_path.iterdir())


def test_positional_rule_is_refused_before_the_corpus_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "_load_dataset", lambda name: pytest.fail("dataset loaded")
    )
    with pytest.raises(ValueError, match="positional"):
        runner.compute_for(
            "corpus",
            method="surprisal",
            unit_rule="x",
            model_id="m",
            K=5,
            n_samples=1,
            max_tokens=1,
            alpha=1.0,
            seed=None,
            out_dir=tmp_path,
            add_bos=True,
            force=False,
        )


def test_tampered_token_table_is_not_aggregated(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path)
    outcome = runner.compute_for("corpus", method="surprisal", **common)
    assert outcome.status == "completed", outcome.error
    source = tmp_path / _manifest(outcome.path)["derivation"]["source"]
    before = _manifest(outcome.path)["derivation"]["source_sha256"]
    table = pickle.loads(source.read_bytes())
    table["s0"][0].values["log_p_observed"] = -50.0
    source.write_bytes(pickle.dumps(table))
    context = ComputeContext("test-tokens", 5, 64, 10, 1.0, None, True, "en", None)
    with pytest.raises(ValueError, match="invalid artifact manifest"):
        runner._aggregate_cell(
            "corpus",
            "surprisal",
            context,
            deepcopy(STIMULI),
            source,
            TOKEN_UNIT,
            tmp_path,
            "",
            force=True,
        )
    # Through the driver the tampered table is stale and is recomputed; the
    # recomputed bytes equal the ones the word cell was derived from, so the
    # word cell is reused.
    again = runner.compute_for("corpus", method="surprisal", **common)
    assert again.status == "cached", again.error
    assert file_sha256(source) == before
    assert _manifest(again.path)["derivation"]["source_sha256"] == before
    assert pickle.loads(source.read_bytes())["s0"][0].values["log_p_observed"] != -50.0


def test_reissued_token_table_rederives_the_word_cell(tmp_path, monkeypatch):
    """A source table re-issued under the same identity with other numbers
    (another device, another build) must not leave the word cell behind."""
    common = _install(monkeypatch, tmp_path)
    outcome = runner.compute_for("corpus", method="surprisal", **common)
    assert outcome.status == "completed", outcome.error
    source = tmp_path / _manifest(outcome.path)["derivation"]["source"]
    old_words = pickle.loads(outcome.path.read_bytes())
    table = pickle.loads(source.read_bytes())
    for rows in table.values():
        for row in rows:
            row.values["log_p_observed"] -= 1.0
    source.write_bytes(pickle.dumps(table))
    manifest = json.loads(manifests.manifest_path(source).read_text())
    manifest["payload_sha256"] = file_sha256(source)
    manifests.manifest_path(source).write_text(json.dumps(manifest))
    again = runner.compute_for("corpus", method="surprisal", **common)
    assert again.status == "completed", again.error
    assert again.path == outcome.path
    assert (
        _manifest(again.path)["derivation"]["source_sha256"]
        == manifest["payload_sha256"]
    )
    new_words = pickle.loads(again.path.read_bytes())
    for stimulus in STIMULI:
        rows = table[stimulus.stimulus_id]
        tokens_per_word = [
            len(members)
            for members in nested_members(
                stimulus, "words", [Span(*r.metadata["span"]) for r in rows]
            )
        ]
        shift = np.array(
            [n.values["log_p_observed"] for n in new_words[stimulus.stimulus_id]]
        ) - np.array(
            [o.values["log_p_observed"] for o in old_words[stimulus.stimulus_id]]
        )
        np.testing.assert_allclose(shift, -np.array(tokens_per_word, dtype=float))


def test_no_bos_leaves_the_first_word_nan_in_the_derived_cell(tmp_path, monkeypatch):
    common = _install(monkeypatch, tmp_path) | {"add_bos": False}
    outcome = runner.compute_for("corpus", method="surprisal", **common)
    assert outcome.status == "completed", outcome.error
    assert outcome.path.name == "test-tokens_surprisal_nobos_corpus.pkl"
    words = pickle.loads(outcome.path.read_bytes())
    for stimulus in STIMULI:
        values = np.array([w.values["surprisal"] for w in words[stimulus.stimulus_id]])
        assert np.isnan(values[0]) and np.isfinite(values[1:]).all()


def test_unspaced_script_has_no_token_route_for_entropies(tmp_path, monkeypatch):
    from replica.feature.core.annotation.characters import attach_characters

    unspaced = Stimulus(text="我们今天", stimulus_id="zh")
    unspaced.add_layer(
        AnnotationLayer("words", [Span(0, 2), Span(2, 4)], ["我们", "今天"])
    )
    common = _install(monkeypatch, tmp_path, [unspaced], word_spacing=False)
    for method in ("first_token_entropy", "token_renyi_entropy", "responsivity"):
        with pytest.raises(ValueError, match="no word spacing"):
            runner.compute_for("corpus", method=method, **common)
    assert not list(tmp_path.glob("*.pkl"))
    with pytest.raises(ValueError, match="not whitespace-normalized"):
        attach_characters([word_stimulus("a  b")])


def _sentenced(text, ids):
    stimulus = word_stimulus(text, "s0")
    words = stimulus.layers["words"]
    stimulus.add_layer(AnnotationLayer(SENTENCE_LAYER, list(words.spans), ids))
    return stimulus


@pytest.mark.parametrize("scope", ["sentence", "story"])
def test_custom_dataset_declares_context_without_loading_it(monkeypatch, scope):
    from replica.dataset.catalog.registry import REGISTRY, DatasetRegistration

    monkeypatch.setitem(
        REGISTRY,
        "custom_corpus",
        DatasetRegistration(
            name="custom_corpus",
            loader_module="must_not_be_imported",
            loader_class="CustomCorpus",
            default_subdir="custom",
            default_context_scope=scope,
        ),
    )

    assert resolve_context_scope("custom_corpus", "surprisal") == scope
    assert resolve_context_scope("custom_corpus", "surprisal", "stimulus") == "stimulus"
    assert resolve_context_scope("custom_corpus", "unigram_surprisal") == "stimulus"


def test_sentence_scope_is_refused_for_a_method_whose_model_ignores_it(
    tmp_path, monkeypatch
):
    """The tag drops ``_sent`` for these methods, so the stimulus-scope cell
    and a sentence-scope request share one file name; accepting the request
    would record a scope the model never applied."""
    ignoring = sorted(set(METHODS) - SCOPE_SENSITIVE)
    assert "first_token_entropy" in ignoring
    for method in ignoring:
        assert cache_filename(
            model_id="m", method=method, dataset="d", context_scope="sentence"
        ) == cache_filename(model_id="m", method=method, dataset="d")
        with pytest.raises(ValueError, match="entence context is not supported"):
            resolve_context_scope("corpus", method, "sentence")
    for method in sorted(SCOPE_SENSITIVE):
        assert resolve_context_scope("corpus", method, "sentence") == "sentence"
    common = _install(
        monkeypatch, tmp_path, [_sentenced("One two. Three four.", [1, 1, 2, 2])]
    )
    with pytest.raises(ValueError, match="entence context is not supported"):
        runner.compute_for(
            "corpus", method="first_token_entropy", context_scope="sentence", **common
        )
    assert not list(tmp_path.iterdir())
