"""Character-beam token limits using the real beam and a tiny local model."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica.feature.surprisal.char_beam import CharBeamLM
from tests._stimuli import word_stimulus
from tests.feature.backends.test_start_state import _mgpt

pytestmark = pytest.mark.resource_free(
    "Local tokenizer and model doubles; the installed byte beam loads no checkpoint"
)


@pytest.fixture
def make_beam(monkeypatch):
    pytest.importorskip("genlm.bytes")
    from genlm.backend.llm.hf import AsyncTransformer
    from genlm.backend.tokenization import Token
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vocabulary = [b"<s>", b"<pad>", b"a", b"aa", b"b", b"<|endoftext|>"]
    tokens = [Token(token_id=i, byte_string=b) for i, b in enumerate(vocabulary)]
    monkeypatch.setattr("genlm.backend.llm.base.decode_vocab", lambda _: (tokens, []))
    monkeypatch.setattr(
        "genlm.backend.tokenization.bytes.get_byte_vocab", lambda _: tokens
    )
    monkeypatch.setattr(
        "replica.feature.surprisal.char_beam.resolve_device", lambda: "cpu"
    )
    original_init = AsyncTransformer.__init__
    original_batch = AsyncTransformer.batch_evaluate_queries

    def track_backend(self, model, tokenizer):
        original_init(self, model, tokenizer)
        model.backends.append(self)

    def track_batch(self):
        self.model.batch_evaluations += 1
        return original_batch(self)

    monkeypatch.setattr(AsyncTransformer, "__init__", track_backend)
    monkeypatch.setattr(AsyncTransformer, "batch_evaluate_queries", track_batch)

    class Model:
        device = torch.device("cpu")
        config = SimpleNamespace(n_positions=2048)

        def __init__(self):
            self.calls = []
            self.backends = []
            self.batch_evaluations = 0

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, input_ids, attention_mask, **kwargs):
            self.calls.extend(
                tuple(row[mask.bool()].tolist())
                for row, mask in zip(input_ids, attention_mask, strict=True)
            )
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, len(tokens)))

    def make(model_id="gpt2", context_window=None):
        tokenizer = _mgpt(model_id)
        tokenizer.pad_token_id = 1
        model = Model()
        monkeypatch.setattr(
            AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
        )
        monkeypatch.setattr(
            AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model
        )
        return CharBeamLM(model_id, K=5, context_window=context_window), model

    return make


@pytest.mark.parametrize("length", [4, 5])
def test_every_beam_branch_obeys_the_token_limit_and_cleans_up(make_beam, length):
    scorer, model = make_beam(context_window=4)

    async def score():
        if length == 5:
            with pytest.raises(
                ValueError, match="prefix of 5 tokens.*context_window=4"
            ):
                await scorer._byte_log_p(b"a" * length)
        else:
            constrained = await scorer._byte_log_p(b"a" * length)
            wider, _ = make_beam(context_window=16)
            np.testing.assert_array_equal(
                constrained, await wider._byte_log_p(b"a" * length)
            )
        batch_evaluations = model.batch_evaluations
        assert all(backend.timer is None for backend in model.backends)
        await asyncio.sleep(2 * max(backend.timeout for backend in model.backends))
        assert model.batch_evaluations == batch_evaluations
        assert scorer._async_trie._task is None
        assert asyncio.all_tasks() == {asyncio.current_task()}
        assert np.isfinite(await scorer._byte_log_p(b"aa")).all()

    asyncio.run(score())
    assert max(map(len, model.calls)) == 4
    assert (0, 2, 2) in model.calls and (0, 3) in model.calls


def test_mgpt_default_and_explicit_window_are_recorded(make_beam):
    scorer, model = make_beam("ai-forever/mGPT")
    output = scorer(word_stimulus("a"))[0]
    assert output.metadata["context_window"] == 512
    assert output.metadata["start_ids"] == [5]
    assert model.calls[0] == (5,)
    with pytest.warns(UserWarning, match="1024-token context.*default of 512"):
        override, _ = make_beam("ai-forever/mGPT", context_window=1024)
    assert override.context_window == 1024
    with pytest.raises(ValueError, match="exceeds.*declared position limit"):
        make_beam("ai-forever/mGPT", context_window=2049)


def test_timer_batch_failure_reaches_caller_and_allows_reuse(make_beam, monkeypatch):
    scorer, model = make_beam(context_window=16)
    original = type(model).__call__
    calls = 0

    def forward(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("model batch failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "__call__", forward)

    async def score():
        callback_errors = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: callback_errors.append(context)
        )
        with pytest.raises(RuntimeError, match="model batch failed"):
            await asyncio.wait_for(scorer._byte_log_p(b"aaa"), timeout=2.0)
        assert calls == 2
        assert all(backend.timer is None for backend in model.backends)
        assert scorer._async_trie._task is None
        assert asyncio.all_tasks() == {asyncio.current_task()}
        assert np.isfinite(await scorer._byte_log_p(b"aa")).all()
        assert not callback_errors

    asyncio.run(score())
