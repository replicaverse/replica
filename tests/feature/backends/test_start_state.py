"""The prefix a model's empty context is scored and sampled from."""

import numpy as np
import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.backends.model_defaults import ModelDefaults, model_defaults
from replica.feature.backends.start_state import (
    StartState,
    resolve_start_state,
    start_id,
    start_ids,
)
from replica.feature.core.annotation.alignment import aligned_token_ids
from replica.feature.frequency.sampled import _encode_samples, start_prefix


class _Tokenizer:
    """A tokenizer stub with a name, special ids and a tiny vocabulary."""

    unk_token_id = 3

    def __init__(self, name, bos, eos, vocab):
        self.name_or_path = name
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        if isinstance(ids, list):
            return [inverse.get(i, str(i)) for i in ids]
        return inverse.get(ids, str(ids))

    def encode(self, text, add_special_tokens=True):
        return [10 + i for i, _ in enumerate(text.split())]


def _mgpt(name="ai-forever/mGPT", vocab=None):
    return _Tokenizer(name, 0, 5, vocab or {"<s>": 0, "<unk>": 3, "<|endoftext|>": 5})


def test_an_override_names_the_token_by_hub_id():
    assert model_defaults("ai-forever/mGPT") == ModelDefaults("<|endoftext|>", 512)
    state = resolve_start_state(_mgpt())
    assert state == StartState((5,), ("<|endoftext|>",), "override")
    assert start_ids(_mgpt()) == [5] and start_id(_mgpt()) == 5


def test_a_snapshot_path_resolves_to_its_hub_id():
    path = "/cache/hub/models--ai-forever--mGPT/snapshots/0123abcd"
    assert resolve_start_state(_mgpt(path)).basis == "override"
    assert resolve_start_state(_mgpt("elsewhere"), model_name=path).basis == "override"


def test_the_tokenizer_bos_is_the_default():
    gpt2 = _Tokenizer("gpt2", 50256, 50256, {"<|endoftext|>": 50256})
    assert resolve_start_state(gpt2) == StartState(
        (50256,), ("<|endoftext|>",), "tokenizer"
    )


def test_a_model_without_bos_gets_no_prefix():
    qwen = _Tokenizer("Qwen/Qwen3-0.6B", None, 151645, {"<|im_end|>": 151645})
    assert resolve_start_state(qwen) == StartState((), (), "none")
    assert start_ids(qwen) == [] and start_id(qwen) is None


def test_an_override_token_missing_from_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match="not in the vocabulary"):
        resolve_start_state(_mgpt(vocab={"<s>": 0, "<unk>": 3}))


def test_the_record_lists_ids_tokens_and_basis():
    assert resolve_start_state(_mgpt()).record() == {
        "ids": [5],
        "tokens": ["<|endoftext|>"],
        "basis": "override",
    }


@pytest.mark.parametrize(
    "method,add_bos,ids,basis",
    [
        ("surprisal", True, [5], "override"),
        ("surprisal", False, [], "disabled"),
        ("char_beam", False, [5], "override"),
        ("wordsprobability_surprisal", True, [5], "override"),
        ("wordsprobability_surprisal_buggy", True, [0], "upstream_tokenizer"),
        ("logit_lens_surprisal", True, [0], "model_config"),
        ("modelblocks_surprisal", True, [0], "model_config"),
    ],
)
def test_compute_records_the_methods_effective_prefix(
    monkeypatch, method, add_bos, ids, basis
):
    from types import SimpleNamespace

    from replica.compute.artifacts import _start_record
    from replica.compute.methods import ComputeContext

    class Tokenizer(_Tokenizer):
        def decode(self, ident):
            return self.convert_ids_to_tokens([ident])[0]

        def __call__(self, text, **kwargs):
            return {"input_ids": [self.convert_tokens_to_ids(text)]}

    tokenizer = Tokenizer("ai-forever/mGPT", 0, 5, {"<s>": 0, "<|endoftext|>": 5})
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_: tokenizer
    )
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_: SimpleNamespace(bos_token_id=0),
    )
    context = ComputeContext("ai-forever/mGPT", 5, 4, 2, 1.0, 0, add_bos, "en", None)
    record = _start_record("ai-forever/mGPT", method, context)
    assert record["ids"] == ids and record["basis"] == basis


def test_compute_records_recipe_input_policies_and_sampling_fallback(monkeypatch):
    from replica.compute.artifacts import _start_record
    from replica.compute.methods import ComputeContext

    class Tokenizer(_Tokenizer):
        def __call__(self, text):
            assert text == " "
            return {"input_ids": [227]}

    tokenizer = Tokenizer("no-bos", None, 0, {"eos": 0})
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_: tokenizer
    )
    context = ComputeContext("no-bos", 5, 4, 2, 1.0, 0, True, "en", None)
    sentence = _start_record("no-bos", "information_value", context)
    assert sentence == {
        "basis": "recipe",
        "add_special_tokens": True,
        "empty_context_text": " ",
        "empty_context_ids": [227],
    }
    word = _start_record("no-bos", "word_information_value", context)
    assert word == {
        "basis": "recipe",
        "generation_add_special_tokens": True,
        "embedding_add_special_tokens": False,
        "first_unit": "unscored",
    }
    assert _start_record("no-bos", "mmc_unigram_surprisal", context) == {
        "ids": [0],
        "tokens": ["eos"],
        "basis": "eos_fallback",
    }


def test_the_sampler_keeps_the_tokenizer_convention_unless_given_a_prefix():
    """The resolver is the compute layer's; the sampler's default is bos, else eos."""
    assert start_prefix(_mgpt()) == [0]
    assert _encode_samples(_mgpt(), ["a b", ""]) == [[0, 10, 11], [0]]
    gpt2 = _Tokenizer("gpt2", 50256, 50256, {"<|endoftext|>": 50256})
    assert start_prefix(gpt2) == [50256]
    no_bos = _Tokenizer("Qwen/Qwen3-0.6B", None, 7, {})
    assert start_prefix(no_bos) == [7]


def test_an_explicit_prefix_is_one_token_and_is_what_samples_are_encoded_after():
    assert start_prefix(_mgpt(), start_ids(_mgpt())) == [5]
    assert _encode_samples(_mgpt(), ["a b", ""], [5]) == [[5, 10, 11], [5]]
    with pytest.raises(ValueError, match="exactly one non-negative integer"):
        start_prefix(_mgpt(), [5, 9])
    with pytest.raises(ValueError, match="exactly one non-negative integer"):
        start_prefix(_mgpt(), [])


def test_the_former_hub_id_resolves_to_the_same_entry():
    assert resolve_start_state(_mgpt("sberbank-ai/mGPT")).basis == "override"
    assert start_ids(_mgpt("sberbank-ai/mGPT")) == [5]


@pytest.mark.parametrize("name", ["other/mGPT", "ai-forever/mGPT-13B", "mgpt", "gpt2"])
def test_model_defaults_only_apply_to_the_named_model(name):
    assert model_defaults(name) == ModelDefaults()
    assert resolve_start_state(_mgpt(name)).ids == (0,)


def test_changing_or_disabling_the_mgpt_start_token_warns():
    with pytest.warns(
        UserWarning, match="start token '<s>'.*default '<\\|endoftext\\|>'"
    ):
        assert resolve_start_state(_mgpt(), start_ids=[0]).ids == (0,)
    with pytest.warns(UserWarning, match="disabling the start token"):
        assert resolve_start_state(_mgpt(), add_bos=False) == StartState(
            (), (), "disabled"
        )
    with pytest.raises(ValueError, match="start_ids requires add_bos=True"):
        resolve_start_state(_mgpt(), start_ids=[5], add_bos=False)


def test_selecting_the_mgpt_default_or_overriding_an_unknown_model_is_quiet():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_start_state(_mgpt()).ids == (5,)
        assert resolve_start_state(_mgpt(), start_ids=[5]).ids == (5,)
        assert resolve_start_state(_mgpt("other/model"), start_ids=[5]).ids == (5,)
        assert start_id(_mgpt("other/model"), add_bos=False) is None
    assert not caught


class _OffsetTokenizer(_Tokenizer):
    """The stub extended with offsets, so alignment can run on it."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids, offsets, pos = [], [], 0
        for i, word in enumerate(text.split()):
            start = text.index(word, pos)
            ids.append(10 + i)
            offsets.append((start, start + len(word)))
            pos = start + len(word)
        return {"input_ids": ids, "offset_mapping": offsets}


def _offset_tokenizer(name):
    tok = _OffsetTokenizer(name, 0, 5, {"<s>": 0, "<unk>": 3, "<|endoftext|>": 5})
    return tok


def _words(text, stimulus_id):
    """A stimulus with its whitespace words as a layer."""
    stimulus = Stimulus(text=text, stimulus_id=stimulus_id)
    spans, pos = [], 0
    for word in text.split():
        start = text.index(word, pos)
        spans.append(Span(start, start + len(word)))
        pos = start + len(word)
    stimulus.add_layer(AnnotationLayer(name="words", spans=spans, values=text.split()))
    return stimulus


def test_alignment_prepends_the_given_prefix_or_the_tokenizer_bos_and_reports_its_length():
    stimulus = _words("one two", "s")
    tok = _offset_tokenizer("ai-forever/mGPT")
    ids, _, offset, n_units = aligned_token_ids(
        stimulus, tok, "words", True, start_ids(tok)
    )
    assert ids == [5, 10, 11] and offset == 1 and n_units == 2
    other = _offset_tokenizer("other")
    ids, _, offset, _ = aligned_token_ids(
        _words("one two", "t"), other, "words", True, start_ids(other)
    )
    assert ids == [0, 10, 11] and offset == 1
    ids, _, offset, _ = aligned_token_ids(_words("one two", "w"), other, "words", True)
    assert ids == [0, 10, 11] and offset == 1
    no_bos = _offset_tokenizer("other")
    no_bos.bos_token_id = None
    ids, _, offset, _ = aligned_token_ids(_words("one two", "x"), no_bos, "words", True)
    assert ids == [10, 11] and offset == 0
    ids, _, offset, _ = aligned_token_ids(
        _words("one two", "u"),
        _offset_tokenizer("other"),
        "words",
        False,
    )
    assert ids == [10, 11] and offset == 0
    ids, _, offset, _ = aligned_token_ids(
        _words("one two", "v"),
        _offset_tokenizer("other"),
        "words",
        True,
        start_ids=[5, 9],
    )
    assert ids == [5, 9, 10, 11] and offset == 2


@pytest.mark.parametrize(
    "model,override,expected",
    [
        ("ai-forever/mGPT", None, 5),
        ("gpt2", None, 0),
        ("ai-forever/mGPT", (0,), 0),
        ("gpt2", (5,), 5),
    ],
)
def test_character_beam_passes_the_resolved_prefix_to_its_backend(
    monkeypatch, model, override, expected
):
    import asyncio
    import sys
    from types import SimpleNamespace

    from replica.feature.surprisal.char_beam import CharBeamLM

    received = []

    class Backend:
        def __init__(self, weights, tokenizer):
            self.tokenizer = tokenizer
            self.timer = None

        async def next_token_logprobs(self, context):
            received.append(context)

        def reset_async_queries(self):
            pass

    class TokenState:
        @classmethod
        def initial(cls, model, initial_context=None):
            return SimpleNamespace(
                model=model,
                context=initial_context
                if initial_context is not None
                else [model.tokenizer.bos_token_id],
            )

    class TrieState:
        def __init__(self, *, lm_state, **kwargs):
            self.lm_state = lm_state

        @classmethod
        def initial(cls, model, trie, **kwargs):
            return cls(lm_state=TokenState.initial(model))

        async def materialize(self):
            await self.lm_state.model.next_token_logprobs(self.lm_state.context)
            return self

        async def cleanup(self):
            pass

    modules = {
        "genlm.backend.llm.hf": SimpleNamespace(AsyncTransformer=Backend),
        "genlm.bytes": SimpleNamespace(
            BeamParams=lambda **kwargs: None,
            ByteBeamState=lambda states, params: states[0],
        ),
        "genlm.bytes.byte_lm.lm_state": SimpleNamespace(StatefulTokenizedLM=TokenState),
        "genlm.bytes.byte_lm.trie_state": SimpleNamespace(
            LazyTrieState=TrieState, TrieMode=SimpleNamespace(WITH_EOS="eos")
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    scorer = object.__new__(CharBeamLM)
    scorer._requested_start_ids = override
    scorer._hf_model = None
    scorer._hf_tokenizer = _mgpt(model)
    scorer.model_id = model
    scorer.context_window = 512

    async def cleanup():
        pass

    scorer._async_trie = SimpleNamespace(trie=SimpleNamespace(root=0), cleanup=cleanup)
    scorer.K, scorer.heal, scorer.prune_threshold = 5, False, 0.0
    asyncio.run(scorer._byte_log_p(b""))
    assert received == [[expected]]
    import numpy as np

    async def byte_scores(text):
        return np.zeros(len(text))

    scorer.name = "charbeam-test"
    monkeypatch.setattr(scorer, "_byte_log_p", byte_scores)
    outputs = scorer(_words("one two", "s"))
    assert all(output.metadata["start_ids"] == [expected] for output in outputs)


@pytest.mark.parametrize("prefix", [(0,), (5,)])
@pytest.mark.resource_free(
    "Lazy construction and tokenizer doubles; no checkpoint loads"
)
def test_explicit_prefix_reaches_hf_alignment_and_metadata(monkeypatch, prefix):
    from dataclasses import replace

    from replica.compute.artifacts import _start_record
    from replica.compute.methods import ComputeContext
    from replica.feature.backends.huggingface import HuggingFaceLM

    tok = _offset_tokenizer("ai-forever/mGPT")
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_: tok)
    context = ComputeContext("ai-forever/mGPT", 5, 4, 2, 1.0, 0, True, "en", None)
    record = _start_record(
        "ai-forever/mGPT", "surprisal", replace(context, start_ids=prefix)
    )
    scorer = HuggingFaceLM("ai-forever/mGPT", start_ids=prefix)
    scorer._tokenizer = tok
    prepared = scorer._prepare_input(_words("one two", "s"), "words")
    assert prepared.token_ids[0] == prefix[0]
    assert prepared.bos_offset == 1
    assert record == {
        "ids": list(prefix),
        "tokens": tok.convert_ids_to_tokens(list(prefix)),
        "basis": "explicit",
    }


@pytest.mark.parametrize("prefix", [(), (0, 5), (-1,), (True,), (1.2,)])
@pytest.mark.resource_free(
    "Lazy construction and tokenizer doubles; no checkpoint loads"
)
def test_invalid_single_token_prefix_is_refused_before_loading(prefix):
    from replica.feature.backends.huggingface import HuggingFaceLM

    with pytest.raises(ValueError, match="exactly one nonnegative integer"):
        HuggingFaceLM("unused", start_ids=prefix)


@pytest.mark.resource_free(
    "Lazy construction and tokenizer doubles; no checkpoint loads"
)
def test_explicit_prefix_cannot_be_silently_disabled():
    from replica.feature.backends.huggingface import HuggingFaceLM

    with pytest.raises(ValueError, match="requires add_bos=True"):
        HuggingFaceLM("unused", add_bos=False, start_ids=(0,))
    scorer = HuggingFaceLM("unused", add_bos=False)
    assert scorer._start_prefix() == []


@pytest.mark.parametrize(
    "method",
    [
        "wordsprobability_surprisal_buggy",
        "modelblocks_surprisal",
        "logit_lens_surprisal",
        "information_value",
        "word_information_value",
        "unigram_surprisal",
    ],
)
def test_fixed_input_recipes_refuse_a_compute_prefix_override(method):
    from replica.compute.methods import ComputeContext, build_model

    context = ComputeContext(
        "unused", 5, 4, 2, 1.0, 0, True, "en", None, start_ids=(0,)
    )
    with pytest.raises(ValueError, match="fixed input convention"):
        build_model(method, context)


@pytest.mark.parametrize(
    ("value", "accepted"),
    [(np.int64(5), True), (7, True), (True, False), (1.5, False), (-1, False)],
)
@pytest.mark.resource_free(
    "Constructor validation only; WordsProbabilityLM loads its model lazily"
)
def test_every_prefix_entry_point_applies_the_same_single_token_rule(value, accepted):
    from replica.feature.backends.start_state import validate_start_ids as backend
    from replica.feature.frequency.sampled import validate_start_ids as sampler
    from replica.feature.surprisal.wordsprob import WordsProbabilityLM

    def scalar():
        return (WordsProbabilityLM("gpt2-small", start_token_id=value).start_token_id,)

    for entry in (lambda: backend((value,)), lambda: sampler((value,)), scalar):
        if accepted:
            assert entry() == (int(value),)
        else:
            with pytest.raises(ValueError):
                entry()
