"""Story context must preserve page observations and causal prefix alignment."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica import artifacts as manifests
from replica.compute import artifacts, methods, runner
from replica.compute.cache import (
    STORY_SENSITIVE,
    cache_filename,
    parse_cache_tag,
    resolve_context_scope,
    split_cache_filename,
)
from replica.context import (
    CONTEXT_SEQUENCE,
    PRECEDING_CONTEXT,
    attach_preceding_context,
    context_input,
)
from replica.feature import HuggingFaceLM, PrecedingContextModel
from tests._stimuli import word_stimulus


def _page(text, page=1, book="dickens"):
    return word_stimulus(
        text,
        f"sbsat_{book}_{page}",
        metadata={
            "book_name": book,
            "page": page,
            CONTEXT_SEQUENCE: {"group": book, "position": page},
        },
    )


class _Tokenizer:
    bos_token_id = 0
    vocab_size = 11
    name_or_path = "test-story"

    def __call__(self, text, **kwargs):
        # Split every word into character tokens: outputs must still map back
        # to page words, including multi-token words at page boundaries.
        positions = [(i, c) for i, c in enumerate(text) if not c.isspace()]
        return {
            "input_ids": [1 + ord(c) % 10 for _, c in positions],
            "offset_mapping": [(i, i + 1) for i, _ in positions],
        }

    def decode(self, ids):
        return "".join(f"[{i}]" for i in ids)


class _CausalModel:
    def __init__(self, limit):
        self.config = SimpleNamespace(max_position_embeddings=limit, vocab_size=11)
        self.inputs = []

    def __call__(self, input_ids):
        assert input_ids.shape[-1] <= self.config.max_position_embeddings
        self.inputs.append(input_ids.tolist())
        prefix = input_ids.float().cumsum(-1)
        vocabulary = torch.arange(11, dtype=torch.float32)
        logits = -(vocabulary - (prefix.remainder(7) + 1).unsqueeze(-1)).square() / 8
        return SimpleNamespace(logits=logits)


class _StoryLM(HuggingFaceLM):
    """Exercise the backend with local doubles and reject pretrained loading."""

    def _ensure_loaded(self):
        assert isinstance(self._model, _CausalModel)
        assert isinstance(self._tokenizer, _Tokenizer)


def _lm(*, bos=True, limit=1024):
    lm = _StoryLM(
        "test-story",
        add_bos=bos,
        provide_word_dist=False,
        token_level_entropy=True,
        token_renyi_alphas=[0.5],
        device="cpu",
    )
    lm._tokenizer = _Tokenizer()
    lm._model = _CausalModel(limit)
    return lm


def _values(outputs, key):
    return np.array([o.values[key] for o in outputs])


@pytest.mark.parametrize("bos", [False, True])
@pytest.mark.parametrize("limit", [8, 1024])
def test_story_surprisal_and_entropy_match_independent_prefix(bos, limit):
    pages = [_page("ab cd"), _page("ef gh", 2), _page("ij kl", 3)]
    original = deepcopy(pages)
    attach_preceding_context(list(reversed(pages)))
    lm = _lm(bos=bos, limit=limit)
    model = PrecedingContextModel(lm)
    for i, page in enumerate(pages):
        expected_input = _page(" ".join(p.text for p in pages[: i + 1]))
        expected = lm(expected_input)[2 * i :]
        got = model(page)
        for key in ("log_p_observed", "token_entropy", "token_renyi_entropy_alpha0.5"):
            np.testing.assert_allclose(_values(got, key), _values(expected, key))
        assert len(got) == 2
        assert page.text == original[i].text
        assert page.layers == original[i].layers
        assert page.stimulus_id == original[i].stimulus_id
    # A page-initial word has a real prefix even when no BOS is requested.
    assert np.isfinite(model(pages[1])[0].values["log_p_observed"])


def test_preceding_text_changes_scores_but_later_pages_and_other_stories_do_not():
    pages = [
        _page("aa bb"),
        _page("cc dd", 2),
        _page("ee ff", 3),
        _page("gg hh", book="genome"),
    ]
    attach_preceding_context(pages)
    model = PrecedingContextModel(_lm())
    before = _values(model(pages[1]), "log_p_observed")
    page_only = _values(_lm()(pages[1]), "log_p_observed")
    assert not np.allclose(before, page_only)
    pages[2] = _page("Different later text", 3)
    pages[3] = _page("Different other story", book="genome")
    attach_preceding_context(pages)
    np.testing.assert_array_equal(before, _values(model(pages[1]), "log_p_observed"))
    pages[0] = _page("xx yz")
    attach_preceding_context(pages)
    assert not np.allclose(before, _values(model(pages[1]), "log_p_observed"))


def test_numeric_page_order_and_separate_stories():
    pages = [_page(f"word{i}", i) for i in range(1, 12)]
    other = _page("other", book="genome")
    attach_preceding_context([other, *reversed(pages)])
    combined, start = context_input(pages[10])
    assert combined.text == " ".join(f"word{i}" for i in range(1, 12))
    assert start == 10
    assert context_input(other)[0].text == "other"


def test_later_words_on_the_current_page_do_not_change_earlier_predictions():
    pages = [_page("ab cd"), _page("ef gh", 2)]
    attach_preceding_context(pages)
    model = PrecedingContextModel(_lm())
    before = model(pages[1])[0].values
    pages[1] = _page("ef entirely different later words", 2)
    attach_preceding_context(pages)
    after = model(pages[1])[0].values
    for key in ("log_p_observed", "token_entropy"):
        assert before[key] == after[key]


@pytest.mark.parametrize(
    "pages,match",
    [
        ([_page("one"), _page("three", 3)], "Missing preceding"),
        ([_page("one"), _page("one again")], "Duplicate stimulus id"),
        ([_page("two", 2)], "Missing preceding"),
        ([_page("")], "Empty context stimulus"),
    ],
)
def test_incomplete_or_ambiguous_context_is_rejected(pages, match):
    with pytest.raises(ValueError, match=match):
        attach_preceding_context(pages)


def test_unsupported_context_is_not_silently_ignored():
    page = _page("one")
    del page.metadata[CONTEXT_SEQUENCE]
    with pytest.raises(ValueError, match="Missing context sequence"):
        attach_preceding_context([page])
    with pytest.raises(ValueError, match="Preceding context missing"):
        PrecedingContextModel(_lm())(_page("one"))
    page = _page("one")
    attach_preceding_context([page])
    with pytest.raises(ValueError, match="Missing target layer"):
        PrecedingContextModel(_lm())(page, target_layer="characters")
    lm = _lm()
    lm.context_scope = "sentence"
    with pytest.raises(ValueError, match="stimulus-scope"):
        PrecedingContextModel(lm)


@pytest.mark.parametrize("method", sorted(STORY_SENSITIVE))
def test_story_cache_names_roundtrip_and_cannot_reuse_page_only(method):
    story = cache_filename(
        model_id="gpt2", method=method, dataset="sbsat", context_scope="story"
    )
    page = cache_filename(
        model_id="gpt2", method=method, dataset="sbsat", context_scope="stimulus"
    )
    assert story != page
    _, recovered_method, options = split_cache_filename(
        story.removesuffix(".pkl"), "sbsat"
    )
    assert recovered_method == method
    assert options["context_scope"] == "story"


def test_default_context_is_resolved_before_naming():
    assert resolve_context_scope("sbsat", "surprisal") == "story"
    assert resolve_context_scope("sbsat", "mc_word_entropy") == "story"
    assert resolve_context_scope("sbsat", "surprisal", "stimulus") == "stimulus"
    assert resolve_context_scope("sbsat", "unigram_surprisal") == "stimulus"
    assert resolve_context_scope("copco", "surprisal") == "stimulus"
    assert parse_cache_tag("surprisal") == ("surprisal", {})
    with pytest.raises(ValueError, match="not supported"):
        resolve_context_scope("sbsat", "unigram_surprisal", "story")
    with pytest.raises(ValueError, match="unresolved"):
        cache_filename(
            model_id="gpt2",
            method="first_token_entropy",
            dataset="sbsat",
            context_scope="auto",
        )


def _context():
    return methods.ComputeContext(
        "test-story", 5, 64, 10, 1, 0, True, "en", None, context_scope="story"
    )


def test_shard_identity_includes_preceding_text_and_spans():
    pages = [_page("ab cd"), _page("ef gh", 2)]
    attach_preceding_context(pages)
    identity = artifacts.artifact_identity("surprisal", _context(), [pages[1]], {})
    pages[0] = _page("xy zz")
    attach_preceding_context(pages)
    changed = artifacts.artifact_identity("surprisal", _context(), [pages[1]], {})
    assert identity["corpus_sha256"] != changed["corpus_sha256"]
    pages[1].metadata[PRECEDING_CONTEXT]["layers"]["words"]["spans"][0] = (0, 1)
    remapped = artifacts.artifact_identity("surprisal", _context(), [pages[1]], {})
    assert changed["corpus_sha256"] != remapped["corpus_sha256"]


def test_compute_shard_keeps_prefix_and_explicit_page_mode(tmp_path, monkeypatch):
    pages = [_page("ab cd"), _page("ef gh", 2), _page("ij kl", 3)]
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda name: runner._LoadedDataset(deepcopy(pages), 3, "en"),
    )
    monkeypatch.setattr(
        runner, "resolve_model_snapshot", lambda *args: ("test-story", {})
    )
    monkeypatch.setitem(
        methods.METHOD_SPECS,
        "surprisal",
        replace(methods.METHOD_SPECS["surprisal"], build_model=lambda context: _lm()),
    )
    common = dict(
        method="surprisal",
        model_id="test-story",
        K=5,
        n_samples=64,
        max_tokens=10,
        alpha=1,
        seed=0,
        out_dir=tmp_path,
        add_bos=True,
        force=False,
        stim_slice=(1, 2),
    )
    story = runner.compute_for("sbsat", **common)
    page = runner.compute_for("sbsat", **common, context_scope="stimulus")
    assert story.status == page.status == "completed"
    assert story.path != page.path
    import pickle

    with story.path.open("rb") as f:
        stored = pickle.load(f)
    assert list(stored) == [pages[1].stimulus_id]
    expected = _lm()(_page("ab cd ef gh"))[2:]
    np.testing.assert_allclose(
        _values(stored[pages[1].stimulus_id], "log_p_observed"),
        _values(expected, "log_p_observed"),
    )
    manifest = manifests.validate_artifact(story.path)
    assert manifest["options"]["context_scope"] == "story"
    with page.path.open("rb") as f:
        page_values = pickle.load(f)
    np.testing.assert_allclose(
        _values(page_values[pages[1].stimulus_id], "log_p_observed"),
        _values(_lm()(pages[1]), "log_p_observed"),
    )


def test_character_compute_shard_uses_preceding_character_layers(tmp_path, monkeypatch):
    from replica.feature import FeatureModelOutput

    class SpanModel:
        name = "span-prefix"
        vocab = None

        def __call__(self, stimulus, extract=None, target_layer="words"):
            return [
                FeatureModelOutput(values={"log_p_observed": -float(span.start)})
                for span in stimulus.layers[target_layer].spans
            ]

    pages = [_page("ab cd"), _page("ef gh", 2)]
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda name: runner._LoadedDataset(deepcopy(pages), 2, "en"),
    )
    monkeypatch.setattr(runner, "resolve_model_snapshot", lambda *args: ("test", {}))
    monkeypatch.setitem(
        methods.METHOD_SPECS,
        "char_beam",
        replace(
            methods.METHOD_SPECS["char_beam"], build_model=lambda context: SpanModel()
        ),
    )
    result = runner.compute_for(
        "sbsat",
        method="char_beam",
        model_id="test",
        K=5,
        n_samples=64,
        max_tokens=10,
        alpha=1,
        seed=0,
        out_dir=tmp_path,
        add_bos=True,
        force=False,
        stim_slice=(1, 2),
        unit="characters",
        context_scope="story",
    )
    assert result.status == "completed", result.error
    import pickle

    with result.path.open("rb") as stream:
        outputs = pickle.load(stream)[pages[1].stimulus_id]
    np.testing.assert_equal(_values(outputs, "log_p_observed"), -np.arange(6, 11))
    assert manifests.validate_artifact(result.path)["options"]["unit"] == "characters"
