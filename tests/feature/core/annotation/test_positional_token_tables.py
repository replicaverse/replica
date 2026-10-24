"""Per-token tables of the lens, continuation entropy, responsivity and
subword similarity methods, and their derivation to words.

The sampled estimators run on local doubles whose logits depend on the
current token, so draws matter and the per-text seeding is observable; the
lens runs on a small checkpoint. A word table derived under the nesting
guard must equal the unit-level call wherever the two draw the same numbers:
exactly for the lens and the exact similarity path, and for the sampled
methods wherever every word is one token, since both paths then draw at the
same positions in the same order.
"""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica import artifacts as manifests
from replica.compute import methods, runner
from replica.compute.units import (
    derivation_plan,
    derive_records,
    derived_keys,
    token_keys,
    verify_token_rows,
)
from replica.context import SENTENCE_LAYER
from replica.core import AnnotationLayer, Span
from replica.feature import FeatureModelOutput, ResponsivityLM
from replica.feature.core.annotation.alignment import nested_members
from replica.feature.core.outputs.information import LN2
from replica.feature.distance.similarity_surprisal import (
    SimilarityAdjustedSurprisalEmbeddingLM,
    similarity_unit_values,
)
from replica.feature.entropy import ContinuationEntropyLM
from tests._stimuli import word_stimulus
from tests.feature.core.annotation.test_token_tables import (
    TEXT,
    _CausalModel,
    _Tokenizer,
)
from tests.feature.entropy._entropy_mocks import build_estimator

ONE_TOKEN_TEXT = "The cat sat on a mat"


def _values(outputs, key):
    return np.array([o.values[key] for o in outputs], dtype=float)


def _spans(rows):
    return [Span(*r.metadata["span"]) for r in rows]


def _derived(stimulus, rows, method, keys):
    plan, finisher = derivation_plan(method, keys)
    return derive_records(
        stimulus, rows, unit="tokens", layer="words", plan=plan, finisher=finisher
    )


def _first_indices(stimulus, rows):
    members = nested_members(
        stimulus, "words", _spans(rows), [r.metadata["token"] for r in rows]
    )
    return [indices[0] for indices in members]


# ---- continuation entropy --------------------------------------------------


class _StubCache(tuple):
    def to_legacy_cache(self):
        return tuple(self)


class _TokenCausalLM(torch.nn.Module):
    """Logits set by the current token alone, so the cache carries nothing."""

    def __init__(self, vocab_size=64, n_positions=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(
            max_position_embeddings=n_positions, n_positions=n_positions
        )

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        vocabulary = torch.arange(self.vocab_size, dtype=torch.float32)
        target = (input_ids.float() * 7).remainder(self.vocab_size)
        logits = -(vocabulary - target.unsqueeze(-1)).square() / 9
        return SimpleNamespace(logits=logits, past_key_values=_StubCache())


def _cont_lm(n_samples=6, max_tokens=3, seed=0, add_bos=True, tokenizer=None):
    return build_estimator(
        ContinuationEntropyLM,
        _TokenCausalLM(),
        tokenizer=_Tokenizer() if tokenizer is None else tokenizer,
        model_id="test-tokens",
        name=f"cont_entropy_test_N{n_samples}_L{max_tokens}",
        n_samples=n_samples,
        seed=seed,
        add_bos=add_bos,
        max_tokens=max_tokens,
        _device_name="cpu",
        _torch_version=torch.__version__,
    )


CONT_KEYS = ["continuation_entropy", "log_p_samples_nats"]


def test_continuation_token_rows_tile_the_text_and_carry_the_draws():
    lm = _cont_lm()
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus)
    verify_token_rows(stimulus, rows)
    assert [r.metadata["token_index"] for r in rows] == list(range(len(rows)))
    assert len(rows) > len(stimulus.layers["words"])
    entropy = _values(rows, "continuation_entropy")
    assert np.isfinite(entropy).all()
    for row in rows:
        draws = row.values["log_p_samples_nats"]
        assert draws.shape == (6,)
        assert row.values["continuation_entropy"] == -float(np.mean(draws)) / LN2
    assert rows[0].metadata["seed"] == 0
    assert lm.sampling()["positions"] == "all" and lm.sampling()["seed"] == 0
    assert lm.window() == {
        "rule": "half_windows",
        "max_positions": 1024,
        "half_size": 509,
        "continuation": 3,
    }


def test_continuation_without_bos_leaves_the_first_token_nan():
    rows = _cont_lm(add_bos=False).score_tokens(word_stimulus(TEXT))
    entropy = _values(rows, "continuation_entropy")
    assert np.isnan(entropy[0]) and np.isfinite(entropy[1:]).all()
    assert rows[0].values["log_p_samples_nats"].shape == (0,)


def test_continuation_token_draws_depend_on_the_text_alone():
    first = word_stimulus("The cat sat on the mat")
    second = word_stimulus("Dogs bark loudly at night")
    # The tokenizer double numbers pieces by first sight, so one instance
    # that has seen both texts serves every model compared here.
    tokenizer = _Tokenizer()
    tokenizer(first.text)
    tokenizer(second.text)
    lm = _cont_lm(tokenizer=tokenizer)
    lm.score_tokens(first)
    after_first = _values(lm.score_tokens(second), "continuation_entropy")
    alone = _values(
        _cont_lm(tokenizer=tokenizer).score_tokens(second), "continuation_entropy"
    )
    np.testing.assert_array_equal(after_first, alone)
    # The word layer does not reach the token table.
    relabelled = word_stimulus("Dogs bark loudly at night")
    words = relabelled.layers["words"]
    relabelled.add_layer(
        AnnotationLayer(
            "words", [Span(0, 9), *words.spans[2:]], ["Dogs bark", *words.values[2:]]
        )
    )
    np.testing.assert_array_equal(
        _values(
            _cont_lm(tokenizer=tokenizer).score_tokens(relabelled),
            "continuation_entropy",
        ),
        alone,
    )
    other = _values(
        _cont_lm(seed=1, tokenizer=tokenizer).score_tokens(second),
        "continuation_entropy",
    )
    assert not np.array_equal(alone, other)


def test_continuation_derived_words_take_the_first_token():
    lm = _cont_lm()
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus)
    derived = _derived(stimulus, rows, "continuation_entropy", CONT_KEYS)
    assert len(derived) == len(stimulus.layers["words"])
    assert derived_keys("continuation_entropy", CONT_KEYS) == CONT_KEYS
    for record, index in zip(derived, _first_indices(stimulus, rows), strict=True):
        assert set(record.values) == set(CONT_KEYS)
        assert (
            record.values["continuation_entropy"]
            == rows[index].values["continuation_entropy"]
        )
        np.testing.assert_array_equal(
            record.values["log_p_samples_nats"],
            rows[index].values["log_p_samples_nats"],
        )
        assert record.metadata == {"model_name": lm.name}


# ---- responsivity ------------------------------------------------------------


def _resp_lm(add_bos=True, seed=3, n_samples=8):
    lm = ResponsivityLM.__new__(ResponsivityLM)
    lm.name = "resp-double"
    lm.model_name = "test-tokens"
    lm.n_samples = n_samples
    lm.importance_temp = 1.0
    lm.device = "cpu"
    lm.seed = seed
    lm.add_bos = add_bos
    lm.expected_is_correction = "unbiased"
    lm._model = _CausalModel()
    lm._tokenizer = _Tokenizer()
    generator = torch.Generator().manual_seed(0)
    lm._token_embeds = torch.randn(64, 8, generator=generator)
    lm._wpe0 = torch.zeros(1, 8)
    return lm


RESP_KEYS = ["decontextualised_distance", "expected_decontextualised_distance"]


def test_responsivity_token_rows_tile_the_text():
    lm = _resp_lm()
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus)
    verify_token_rows(stimulus, rows)
    assert set(rows[0].values) == set(RESP_KEYS)
    for key in RESP_KEYS:
        assert np.isfinite(_values(rows, key)).all()
    assert np.isnan(
        _resp_lm(add_bos=False).score_tokens(stimulus)[0].values[RESP_KEYS[0]]
    )
    assert lm.window() == {
        "rule": "sliding",
        "max_positions": 1024,
        "stride": 512,
        "fresh_from": "start + max_positions - stride",
    }
    assert lm.sampling()["seed"] == 3 and lm.sampling()["positions"] == "all"


def test_responsivity_derived_words_equal_the_unit_call_on_one_token_words():
    """Both paths draw at the same positions in the same order here."""
    lm = _resp_lm()
    stimulus = word_stimulus(ONE_TOKEN_TEXT)
    rows = lm.score_tokens(stimulus)
    assert len(rows) == len(stimulus.layers["words"])
    derived = _derived(stimulus, rows, "responsivity", RESP_KEYS)
    expected = lm(stimulus)
    for key in RESP_KEYS:
        np.testing.assert_array_equal(_values(derived, key), _values(expected, key))


def test_responsivity_token_path_draws_at_every_position():
    lm = _resp_lm()
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus)
    derived = _derived(stimulus, rows, "responsivity", RESP_KEYS)
    expected = lm(stimulus)
    # Draws advance once per scored position, so a word after a multi-token
    # word sees a different draw than under the word-level call.
    assert not np.array_equal(
        _values(derived, RESP_KEYS[0]), _values(expected, RESP_KEYS[0])
    )
    np.testing.assert_array_equal(
        _values(rows, RESP_KEYS[0]), _values(lm.score_tokens(stimulus), RESP_KEYS[0])
    )


# ---- subword similarity, exact path ----------------------------------------


class _EmbeddingModel:
    """Causal double with an input embedding matrix, no cache."""

    def __init__(self):
        self.config = SimpleNamespace(n_positions=1024, vocab_size=64)
        self._embeddings = torch.nn.Embedding(64, 8)
        generator = torch.Generator().manual_seed(1)
        with torch.no_grad():
            self._embeddings.weight.copy_(torch.randn(64, 8, generator=generator))
        self._logits = _CausalModel()

    def get_input_embeddings(self):
        return self._embeddings

    def __call__(self, input_ids, **_):
        return SimpleNamespace(
            logits=self._logits(input_ids).logits,
            hidden_states=None,
            past_key_values=None,
        )


def _sim_lm(add_bos=True):
    lm = SimilarityAdjustedSurprisalEmbeddingLM.__new__(
        SimilarityAdjustedSurprisalEmbeddingLM
    )
    lm.model_name = "test-tokens"
    lm.device = "cpu"
    lm.seed = 0
    lm.add_bos = add_bos
    lm.revision = None
    lm._model = _EmbeddingModel()
    lm._tokenizer = _Tokenizer()
    lm._device = "cpu"
    lm.contextual = False
    lm.n_samples = 50
    lm.alphas = (1.0, 2.0)
    lm.include_wpe0 = False
    lm.position_chunk = 4
    lm.name = "simadj-double"
    lm._emb_norm = None
    return lm


SIM_TOKEN_KEYS = [
    "similarity_adjusted_surprisal",
    "next_word_information_value",
    "expected_similarity_tokens",
]


def test_similarity_token_rows_carry_the_expectation_per_alpha():
    lm = _sim_lm()
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus, extract=SIM_TOKEN_KEYS)
    verify_token_rows(stimulus, rows)
    assert set(rows[0].values) == set(SIM_TOKEN_KEYS)
    for row in rows:
        m = row.values["expected_similarity_tokens"]
        assert m.shape == (2,) and np.isfinite(m).all()
        assert row.values["similarity_adjusted_surprisal"] == -np.log(m[0])
        assert row.values["next_word_information_value"] == 1.0 - m[0]
    assert rows[0].metadata["alphas"] == (1.0, 2.0)
    assert lm.window()["fresh_from"] == "start + stride"
    assert lm.sampling() == {
        "positions": "all",
        "seed": None,
        "seeding": "exact, no draws",
    }


def test_similarity_window_caps_mgpt_at_its_trained_positions():
    lm = _sim_lm()
    lm._model.config.n_positions = 2048
    assert lm._max_ctx() == 2048
    lm = _sim_lm()
    lm._model.config.n_positions = 2048
    lm.model_name = "ai-forever/mGPT"
    assert lm._max_ctx() == 512
    assert lm.window() == {
        "rule": "sliding",
        "max_positions": 512,
        "stride": 256,
        "fresh_from": "start + stride",
    }


def test_similarity_derived_words_equal_the_unit_call():
    lm = _sim_lm()
    stimulus = word_stimulus(TEXT)
    rows = lm.score_tokens(stimulus, extract=SIM_TOKEN_KEYS)
    derived = _derived(
        stimulus, rows, "similarity_surprisal_noncontextual", SIM_TOKEN_KEYS
    )
    expected = lm(stimulus)
    keys = derived_keys("similarity_surprisal_noncontextual", SIM_TOKEN_KEYS)
    assert keys == [
        "similarity_adjusted_surprisal",
        "next_word_information_value",
        "similarity_adjusted_surprisal_first",
        "next_word_information_value_first",
        "expected_similarity_tokens",
    ]
    assert set(derived[0].values) == set(keys)
    for key in keys[:4]:
        np.testing.assert_array_equal(_values(derived, key), _values(expected, key))
    for record, want in zip(derived, expected, strict=True):
        np.testing.assert_array_equal(
            record.values["expected_similarity_tokens"],
            want.values["expected_similarity_tokens"],
        )
    assert derived[0].metadata == {"model_name": lm.name, "alphas": (1.0, 2.0)}


def test_similarity_unit_values_from_stacked_expectations():
    m = np.array([[0.5, 0.25], [0.25, 0.0625]])
    values = similarity_unit_values(m, (1.0, 2.0))
    assert values["similarity_adjusted_surprisal"] == -np.log(0.5) + -np.log(0.25)
    assert values["next_word_information_value"] == 0.5 + 0.75
    assert values["similarity_adjusted_surprisal_first"] == -np.log(0.5)
    assert values["next_word_information_value_first"] == 0.5
    for value in similarity_unit_values(np.empty((0, 2)), (1.0, 2.0)).values():
        assert np.isnan(value)
    assert np.isnan(
        similarity_unit_values(np.array([[np.nan, 1.0]]), (1.0, 2.0))[
            "similarity_adjusted_surprisal"
        ]
    )


# ---- the derivation rules over arrays ---------------------------------------


def test_derive_records_stacks_and_picks_arrays():
    stimulus = word_stimulus("ab cd")
    rows = [
        FeatureModelOutput(
            values={"vector": np.array([index, index + 0.5]), "scalar": float(index)},
            metadata={
                "model_name": "m",
                "token_index": index,
                "span": span,
                "token": "t",
            },
        )
        for index, span in enumerate([(0, 1), (1, 2), (3, 5)])
    ]
    derived = derive_records(
        stimulus,
        rows,
        unit="tokens",
        layer="words",
        plan={"vector": "stack", "scalar": "first"},
    )
    np.testing.assert_array_equal(derived[0].values["vector"], [[0, 0.5], [1, 1.5]])
    np.testing.assert_array_equal(derived[1].values["vector"], [[2, 2.5]])
    assert [d.values["scalar"] for d in derived] == [0.0, 2.0]
    picked = derive_records(
        stimulus, rows, unit="tokens", layer="words", plan={"vector": "first"}
    )
    np.testing.assert_array_equal(picked[0].values["vector"], [0, 0.5])
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        derive_records(
            stimulus, rows, unit="tokens", layer="words", plan={"scalar": "mean"}
        )


def test_verify_token_rows_can_exempt_the_text_between_units():
    stimulus = word_stimulus("ab - cd")
    stimulus.add_layer(AnnotationLayer("words", [Span(0, 2), Span(5, 7)], ["ab", "cd"]))
    rows = [
        FeatureModelOutput(values={}, metadata={"token_index": i, "span": span})
        for i, span in enumerate([(0, 2), (5, 7)])
    ]
    with pytest.raises(ValueError, match="covered by no token"):
        verify_token_rows(stimulus, rows)
    verify_token_rows(stimulus, rows, "words")


# ---- the driver on a sampled positional method ------------------------------


def _install(monkeypatch, tmp_path, stimuli):
    monkeypatch.setattr(
        runner,
        "_load_dataset",
        lambda name: runner._LoadedDataset(deepcopy(stimuli), len(stimuli), "en", True),
    )
    monkeypatch.setattr(
        runner,
        "resolve_model_snapshot",
        lambda method, context: (
            context.model_id,
            {"source": "test-tokens", "revision": "r" * 40},
        ),
    )
    monkeypatch.setitem(
        methods.METHOD_SPECS,
        "responsivity",
        replace(
            methods.METHOD_SPECS["responsivity"],
            build_model=lambda context: _resp_lm(add_bos=context.add_bos),
        ),
    )
    monkeypatch.setitem(
        methods.METHOD_SPECS,
        "similarity_surprisal_noncontextual",
        replace(
            methods.METHOD_SPECS["similarity_surprisal_noncontextual"],
            build_model=lambda context: _sim_lm(add_bos=context.add_bos),
        ),
    )
    return dict(
        model_id="test-tokens",
        K=5,
        n_samples=8,
        max_tokens=10,
        alpha=1.0,
        seed=3,
        out_dir=tmp_path,
        add_bos=True,
        force=False,
    )


def _manifest(path):
    return json.loads(manifests.manifest_path(path).read_text())


def test_driver_derives_a_sampled_word_cell_and_records_the_sampling(
    tmp_path, monkeypatch
):
    stimuli = [word_stimulus(TEXT, "s0"), word_stimulus(ONE_TOKEN_TEXT, "s1")]
    common = _install(monkeypatch, tmp_path, stimuli)
    outcome = runner.compute_for("corpus", method="responsivity", **common)
    assert outcome.status == "completed", outcome.error
    assert outcome.path.name == "test-tokens_responsivity_N8_s3_corpus.pkl"
    manifest = _manifest(outcome.path)
    assert manifest["identity"]["storage_class"] == "positional"
    assert manifest["identity"]["layer"] == 1
    assert manifest["identity"]["units"] == dict.fromkeys(RESP_KEYS, "dimensionless")
    derivation = manifest["derivation"]
    assert derivation["source"] == "test-tokens_responsivity_N8_s3_tokens_corpus.pkl"
    assert derivation["rule"] == "first"
    assert derivation["keys"] == dict.fromkeys(RESP_KEYS, "first")
    assert derivation["finisher"] is None
    token_manifest = _manifest(tmp_path / derivation["source"])
    assert token_manifest["identity"]["options"]["unit"] == "tokens"
    assert token_manifest["sampling"]["positions"] == "all"
    assert token_manifest["sampling"]["seed"] == 3
    assert token_manifest["window"]["rule"] == "sliding"
    assert "sampling" not in manifest
    # Words of one token draw as the unit-level call does. The tokenizer
    # double numbers pieces by first sight, so the direct model's tokenizer
    # sees the corpus in the driver's order.
    tokenizer = _Tokenizer()
    tokenizer(TEXT)
    tokenizer(ONE_TOKEN_TEXT)
    direct_model = _resp_lm()
    direct_model._tokenizer = tokenizer
    direct = direct_model(stimuli[1])
    import pickle

    with outcome.path.open("rb") as stream:
        words = pickle.load(stream)["s1"]
    np.testing.assert_array_equal(
        _values(words, RESP_KEYS[0]), _values(direct, RESP_KEYS[0])
    )


# ---- the lens on a small checkpoint -------------------------------------------

LENS_TEXTS = ["The cat sat on the mat.", "A naïve “quoted” 你好 word"]


@pytest.mark.needs_model
def test_lens_token_rows_derive_the_word_values_exactly():
    from replica.feature.surprisal.logit_lens import LogitLensLM

    lm = LogitLensLM("sshleifer/tiny-gpt2", device="cpu")
    for text in LENS_TEXTS:
        stimulus = word_stimulus(text)
        expected = lm(stimulus)
        keys = [k for k in expected[0].values if k.startswith("logit_lens_log_p_L")]
        rows = lm.score_tokens(stimulus, extract=keys)
        verify_token_rows(stimulus, rows, "words")
        assert set(rows[0].values) == set(keys)
        tokenizer = lm._tokenizer
        joined = tokenizer.decode(lm._bos_id) + " " + " ".join(stimulus.unit_strings())
        assert [r.metadata["token_id"] for r in rows] == tokenizer(
            joined, add_special_tokens=False
        )["input_ids"][1:]
        assert len(rows) > len(expected)
        derived = _derived(stimulus, rows, "logit_lens_surprisal", keys)
        for key in keys:
            np.testing.assert_array_equal(_values(derived, key), _values(expected, key))
        assert np.isfinite(_values(rows, keys[-1])).all()
    assert lm.window() == {
        "rule": "sliding",
        "max_positions": 1024,
        "stride": 512,
        "fresh_from": "start + max_positions - stride",
    }


@pytest.mark.needs_model
def test_lens_token_rows_under_sentence_scope():
    from replica.feature.surprisal.logit_lens import LogitLensLM

    lm = LogitLensLM("sshleifer/tiny-gpt2", device="cpu", context_scope="sentence")
    stimulus = word_stimulus("The cat sat on the mat. A dog barked loudly outside.")
    words = stimulus.layers["words"]
    stimulus.add_layer(
        AnnotationLayer(SENTENCE_LAYER, list(words.spans), [1] * 6 + [2] * 5)
    )
    expected = lm(stimulus)
    keys = [k for k in expected[0].values if k.startswith("logit_lens_log_p_L")]
    rows = lm.score_tokens(stimulus, extract=keys)
    verify_token_rows(stimulus, rows, "words")
    # The second sentence starts its own sequence, so its first word keeps
    # the lens's leading space and its span points into the full text.
    second = stimulus.text.index("A dog")
    assert [r.metadata["token"] for r in rows if r.metadata["span"][0] == second] == [
        " A"
    ]
    derived = _derived(stimulus, rows, "logit_lens_surprisal", keys)
    for key in keys:
        np.testing.assert_array_equal(_values(derived, key), _values(expected, key))


def test_driver_token_manifest_names_only_per_token_keys(tmp_path, monkeypatch):
    extracted = methods.METHOD_SPECS["similarity_surprisal_noncontextual"].extract_keys(
        None
    )
    assert token_keys(extracted) == SIM_TOKEN_KEYS
    common = _install(monkeypatch, tmp_path, [word_stimulus(TEXT, "s0")])
    outcome = runner.compute_for(
        "corpus", method="similarity_surprisal_noncontextual", **common
    )
    assert outcome.status == "completed", outcome.error
    manifest = _manifest(outcome.path)
    token_manifest = _manifest(tmp_path / manifest["derivation"]["source"])
    assert list(token_manifest["identity"]["units"]) == SIM_TOKEN_KEYS
    assert list(manifest["identity"]["units"]) == derived_keys(
        "similarity_surprisal_noncontextual", SIM_TOKEN_KEYS
    )
    assert manifest["derivation"]["keys"] == {"expected_similarity_tokens": "stack"}
    assert manifest["derivation"]["finisher"] == "similarity_unit_values"
    with outcome.path.open("rb") as stream:
        import pickle

        words = pickle.load(stream)["s0"]
    assert set(words[0].values) == set(manifest["identity"]["units"])
