"""Sampling prefixes, generated token preservation and generation stop tokens."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica.feature.frequency import sampled
from replica.feature.frequency.sampled import SampledUnigramSource, _drawn_ids
from tests.feature.frequency._sampled_stub import _Model, _Tokenizer


def test_drawn_ids_stop_at_the_first_end_token_and_keep_a_drawn_pad():
    tok = _Tokenizer()
    assert _drawn_ids([5, 6, 2, 0, 0], tok) == [5, 6]
    assert _drawn_ids([7, 0, 9], tok) == [7, 0, 9]
    assert _drawn_ids([7, 0, 2, 0], tok) == [7, 0]
    assert _drawn_ids([8, 9], tok) == [8, 9]
    assert _drawn_ids([], tok) == []


def test_the_sampler_keeps_the_ids_its_texts_decode_from():
    source = SampledUnigramSource(
        model_name="stub", n_samples=3, max_len=5, device="cpu"
    )
    source._tokenizer = _Tokenizer()
    source._model = _Model()
    samples = source._generate_samples()
    assert samples == ["t5 t6", "t7", "t8 t9 t10 t11 t12"]
    assert source._sample_ids == [[5, 6], [7], [8, 9, 10, 11, 12]]


def test_the_priming_fingerprint_loads_the_tokenizer_itself(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda name: _Tokenizer()
    )
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda name: SimpleNamespace(n_positions=2048),
    )
    source = SampledUnigramSource(model_name="stub", n_samples=3, device="cpu")
    assert source._tokenizer is None
    assert source._priming_fingerprint({"model": "stub"})["start"] == [1]
    assert source._model is None


@pytest.mark.parametrize("adds_specials", [False, True])
def test_scored_samples_include_exactly_one_start_prefix(adds_specials):
    class Tokenizer(_Tokenizer):
        def encode(self, text, add_special_tokens=True):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            if adds_specials and add_special_tokens:
                return [self.bos_token_id, *ids, self.eos_token_id]
            return ids

    assert sampled._encode_samples(Tokenizer(), ["t5 t6", ""]) == [[1, 5, 6], [1]]


def test_sampling_without_a_start_or_end_token_is_refused():
    tokenizer = _Tokenizer()
    tokenizer.bos_token_id = tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="a bos token or an eos token"):
        sampled.start_prefix(tokenizer)
    tokenizer.eos_token_id = 0
    assert sampled.start_prefix(tokenizer) == [0]


@pytest.mark.parametrize("batched", [False, True])
def test_unknown_model_requires_a_window_before_mmc_scoring(batched):
    from tests.feature.frequency._authors_mmc_reference import mmc_estimate as reference
    from tests.feature.frequency._pooled_stub import stub_pair

    tokenizer, model = stub_pair()
    model.config = SimpleNamespace()
    estimate = sampled.mmc_estimate_batch if batched else sampled.mmc_estimate
    args = ([" a"] if batched else " a", [" a"], model, tokenizer)
    with pytest.raises(ValueError, match="no usable context limit.*context_window"):
        estimate(*args)
    result = estimate(*args, context_window=64)
    result = result[" a"] if batched else result
    assert result == pytest.approx(reference(" a", [" a"], model, tokenizer))


class _PrefixModel(_Model):
    """The stub's generate, recording the prefix it was called with."""

    def __init__(self):
        self.prefixes = []

    def generate(self, input_ids, **kwargs):
        self.prefixes.append(input_ids[0].tolist())
        return super().generate(input_ids, **kwargs)


def test_samples_are_drawn_after_the_tokenizer_bos_by_default_and_an_explicit_prefix_when_given():
    for start_ids, prefix in ((None, [1]), ((2,), [2])):
        source = SampledUnigramSource(
            model_name="stub", n_samples=3, max_len=5, device="cpu", start_ids=start_ids
        )
        source._tokenizer = _Tokenizer()
        source._model = _PrefixModel()
        assert source._generate_samples() == ["t5 t6", "t7", "t8 t9 t10 t11 t12"]
        assert source._model.prefixes == [prefix]
        assert source._priming_fingerprint({"model": "stub"})["start"] == prefix
    with pytest.raises(ValueError, match="exactly one non-negative integer"):
        SampledUnigramSource(
            model_name="stub", n_samples=3, device="cpu", start_ids=(1, 2)
        )


def test_a_prefix_id_must_be_a_non_negative_integer():
    tok = _Tokenizer()
    for bad in ([1.0], [-1], [True], [np.float64(2.0)], ["2"]):
        with pytest.raises(ValueError, match="non-negative integer"):
            sampled.start_prefix(tok, bad)
    assert sampled.start_prefix(tok, [np.int64(2)]) == [2]
    assert sampled.start_prefix(tok, np.array([2], dtype=np.int32)) == [2]


def test_the_constructor_validates_the_prefix_before_storing_it():
    for bad in ([1.9], [True], [-1], [1, 2], []):
        with pytest.raises(ValueError, match="exactly one non-negative integer"):
            SampledUnigramSource(
                model_name="stub", n_samples=3, device="cpu", start_ids=bad
            )
    source = SampledUnigramSource(
        model_name="stub",
        n_samples=3,
        device="cpu",
        start_ids=np.array([2], dtype=np.int32),
    )
    assert source.start_ids == (2,) and type(source.start_ids[0]) is int
    assert (
        SampledUnigramSource(model_name="stub", n_samples=3, device="cpu").start_ids
        is None
    )


@pytest.mark.parametrize("via_registry", [False, True])
@pytest.mark.parametrize("requested, expected", [(None, 2), ((2,), 2), ((1,), 1)])
def test_mgpt_sampling_scoring_and_checkpoint_share_one_resolved_prefix(
    monkeypatch, via_registry, requested, expected
):
    from replica.feature.frequency.registry import SOURCES

    class Tokenizer(_Tokenizer):
        def convert_tokens_to_ids(self, token):
            assert token == "<|endoftext|>"
            return self.eos_token_id

    build = SOURCES["mmc"].build if via_registry else SampledUnigramSource
    source = build(model_name="ai-forever/mGPT", n_samples=3, start_ids=requested)
    source._tokenizer = Tokenizer()
    source._model = _PrefixModel()
    scored = []

    def estimate(words, *args, start_ids, **kwargs):
        scored.append(start_ids)
        return {word: 0.5 for word in words}

    monkeypatch.setattr(sampled, "mmc_estimate_batch", estimate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source._samples = source._generate_samples()
        assert source.estimate_words(["t5"]) == {"t5": 0.5}
        fingerprint = source._priming_fingerprint({"model": "stub"})
    assert source._model.prefixes == [[expected]]
    assert scored == [[expected]]
    assert fingerprint["start"] == [expected]
    assert len(caught) == (expected != 2)
    if caught:
        assert "overrides REPLICA's default" in str(caught[0].message)


@pytest.mark.parametrize("source_name", ["mmc_pool", "mmc_pool_ee"])
def test_sampling_registry_refuses_context_window_instead_of_ignoring_it(source_name):
    from replica.feature.frequency.registry import SOURCES

    with pytest.raises(ValueError, match="does not support context_window"):
        SOURCES[source_name].build(
            model_name="stub",
            context_window=1024,
            selection=object(),
            allocation=object(),
            k_eval=1,
        )


@pytest.mark.parametrize("prefix", [None, (1,), (2,)])
def test_a_pooled_source_warns_once_without_changing_its_archive_prefix(prefix):
    from replica.feature.frequency.pooled import PooledUnigramSource

    class Tokenizer(_Tokenizer):
        def convert_tokens_to_ids(self, token):
            assert token == "<|endoftext|>"
            return self.eos_token_id

    selection = SimpleNamespace(start=prefix)
    source = PooledUnigramSource(
        selection=selection, allocation=object(), k_eval=1, model_name="ai-forever/mGPT"
    )
    source._tokenizer = Tokenizer()
    source._model = _Model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source._ensure()
        source._ensure()
    assert selection.start == prefix
    assert len(caught) == (prefix != (2,))
    if caught:
        assert "pool selection uses start prefix [1]" in str(caught[0].message)


@pytest.fixture
def mgpt_pair():
    from tests.feature.frequency._pooled_stub import tiny_pair

    tokenizer, model = tiny_pair()
    tokenizer.name_or_path = "ai-forever/mGPT"
    tokenizer.convert_tokens_to_ids = lambda token: tokenizer.eos_token_id
    model.config = SimpleNamespace(n_positions=2048)
    return tokenizer, model


@pytest.mark.parametrize("batched", [False, True])
def test_public_mmc_default_and_explicit_study_prefix_match_independent_reference(
    mgpt_pair, batched
):
    from tests.feature.frequency._authors_mmc_reference import mmc_estimate as reference

    tokenizer, model = mgpt_pair
    samples, word = [" a", " bc"], " ac"
    old = reference(word, samples, model, tokenizer)
    estimate = sampled.mmc_estimate_batch if batched else sampled.mmc_estimate
    args = ([word] if batched else word, samples, model, tokenizer)
    default = estimate(*args)
    explicit = estimate(*args, start_ids=(tokenizer.eos_token_id,))
    with pytest.warns(UserWarning, match="overrides REPLICA's default"):
        study = estimate(*args, start_ids=(tokenizer.bos_token_id,))
    if batched:
        default, explicit, study = default[word], explicit[word], study[word]
    assert default == explicit
    assert study == pytest.approx(old, rel=1e-14)
    assert default != pytest.approx(study, rel=1e-4)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("sample_tokens, word", [(512, " a"), (511, " ac")])
def test_complete_sample_and_continuation_contexts_are_refused_without_truncation(
    mgpt_pair, batched, sample_tokens, word
):
    tokenizer, model = mgpt_pair
    lengths = []

    class RecordingModel:
        config = model.config

        def parameters(self):
            return model.parameters()

        def __call__(self, input_ids, **kwargs):
            lengths.append(input_ids.shape[1])
            return model(input_ids, **kwargs)

    estimate = sampled.mmc_estimate_batch if batched else sampled.mmc_estimate
    args = (
        [word] if batched else word,
        [" a" * sample_tokens],
        RecordingModel(),
        tokenizer,
    )
    with pytest.raises(ValueError, match="complete context of 513 tokens"):
        estimate(*args)
    assert not lengths or max(lengths) <= 512
    lengths.clear()
    with pytest.warns(UserWarning, match="using a 1024-token context"):
        result = estimate(*args, context_window=1024)
    assert max(lengths) == 513
    assert (result[word] if batched else result) > 0


def test_generation_budget_and_checkpoint_use_the_selected_complete_context():
    tokenizer = _Tokenizer()
    tokenizer.convert_tokens_to_ids = lambda token: tokenizer.eos_token_id
    source = SampledUnigramSource(
        model_name="ai-forever/mGPT", n_samples=1, max_len=512
    )
    source._tokenizer = tokenizer
    source._model = _PrefixModel()
    source._model.config = SimpleNamespace(n_positions=2048)
    with pytest.raises(ValueError, match="complete context of 513 tokens"):
        source._generate_samples()
    assert source._model.prefixes == []

    source = SampledUnigramSource(
        model_name="ai-forever/mGPT", n_samples=1, max_len=512, context_window=1024
    )
    source._tokenizer = tokenizer
    source._model = _PrefixModel()
    source._model.config = SimpleNamespace(n_positions=2048)
    with pytest.warns(UserWarning, match="using a 1024-token context"):
        source._generate_samples()
    assert source._model.prefixes == [[tokenizer.eos_token_id]]
    assert source._priming_fingerprint({})["context_window"] == 1024


class _ConfiguredModel(_Model):
    """A model whose generation config decides where rows end, with fixed draws."""

    def __init__(self, eos, rows):
        if eos != "absent":
            self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.rows = rows

    def generate(self, input_ids, **kwargs):
        prefix = input_ids[0].tolist()
        rows = [prefix + r for r in self.rows]
        return torch.tensor(rows[: kwargs["num_return_sequences"]])


@pytest.mark.parametrize(
    ("eos", "drawn", "kept"),
    [
        ([2, 6], [3, 6, 0, 0], [3]),
        ([6], [3, 2, 4, 6], [3, 2, 4]),
        (6, [3, 2, 4, 6], [3, 2, 4]),
        (None, [3, 2, 4, 0], [3, 2, 4, 0]),
        (2, [3, 0, 4, 2], [3, 0, 4]),
        ("absent", [3, 2, 4, 6], [3]),
    ],
    ids=str,
)
def test_the_generation_config_decides_where_a_sample_ends(eos, drawn, kept):
    """The config `generate` reads is authoritative: one id, a list, or none;
    the tokenizer's end token applies only without a generation config."""
    tok = _Tokenizer()
    source = SampledUnigramSource(
        model_name="stub", n_samples=1, max_len=4, device="cpu"
    )
    source._tokenizer = tok
    source._model = _ConfiguredModel(eos, [drawn])
    assert source._generate_samples() == [" ".join(f"t{i}" for i in kept)]
    assert source._sample_ids == [kept]
