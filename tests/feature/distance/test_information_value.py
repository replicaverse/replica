"""Tests for the native information-value predictor of Giulianelli et al. (2023).

``test_scorer_matches_authors_code`` verifies the
:class:`InformationValueScorer` reproduces the authors' ``scorer.py`` +
``compute_information_value.py`` (``get_distances_fast`` followed by
mean/min/max over the cross distances) to ``rtol=1e-6`` on shared inputs.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from replica.external import repo_path
from replica.feature import FeatureModelOutput
from replica.feature.distance.information_value import (
    InformationValue,
    InformationValueLM,
    InformationValueScorer,
)
from tests._require import require
from tests._stimuli import word_stimulus as _build_stim

_AUTHORS_CODE = repo_path("information-value", fetch=False) / "code"


def _spacy_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("spacy") is not None


needs_authors = require(
    lambda: _spacy_available() and (_AUTHORS_CODE / "scorer.py").exists(),
    "information-value authors' code or spaCy not available",
)
needs_spacy = require(_spacy_available, "spaCy not installed")

_TARGET = "The quick brown fox jumps over the lazy dog."
_ALTERNATIVES = [
    "A fast brown fox leaps over a sleepy dog.",
    "The cat sat quietly on the warm windowsill.",
    "She poured herself a cup of strong black coffee.",
    "The quick brown fox jumps over the lazy dog!",
    "Stock markets tumbled sharply amid recession fears today.",
]


@pytest.mark.needs_model
@pytest.mark.needs_data
@needs_authors
def test_scorer_matches_authors_code():
    """REPLICA InformationValueScorer == authors' get_distances_fast aggregation."""

    sys.path.insert(0, str(_AUTHORS_CODE))
    from compute_information_value import get_distances_fast
    from scorer import Scorer

    # Keep reference arithmetic on CPU; float32 GPU batching changes rounding.
    with patch("torch.cuda.is_available", return_value=False):
        ref_scorer = Scorer()
    dists = get_distances_fast(_ALTERNATIVES, _TARGET, ref_scorer, max_len=1000000)
    ref = {}
    for k, v in dists.items():
        if k.endswith("_cross"):
            name = k[: -len("_cross")]
            arr = np.asarray(v, dtype=float)
            ref[f"mean_{name}"] = float(np.mean(arr))
            ref[f"min_{name}"] = float(np.min(arr))
            ref[f"max_{name}"] = float(np.max(arr))

    REPLICA = InformationValueScorer(device="cpu").score(_TARGET, _ALTERNATIVES)
    REPLICA = {k[len("information_value_") :]: v for k, v in REPLICA.items()}

    assert set(ref) <= set(REPLICA)
    for k, r in ref.items():
        np.testing.assert_allclose(
            REPLICA[k], r, rtol=1e-6, atol=1e-9, err_msg=f"metric {k} diverges"
        )

    scorer = InformationValueScorer(device="cpu")
    only_target = scorer.score_in_context(_TARGET, _ALTERNATIVES)
    assert set(only_target) == {f"information_value_{k}" for k in ref}
    for key, value in only_target.items():
        np.testing.assert_allclose(
            value, REPLICA[key.removeprefix("information_value_")], rtol=0, atol=0
        )


@pytest.mark.needs_model
@pytest.mark.needs_data
@needs_authors
def test_self_and_out_of_context_families_match_authors_code():
    """The expected/deviation and ooc/context-informativeness columns too.

    These come from the alternative-to-alternative distances, which the
    authors' ``__main__`` turns into ``expected_surprise_*`` and, against a
    second set of alternatives sampled from an empty context,
    ``surprise_ooc_*`` and ``context_informativeness_*``.
    """
    sys.path.insert(0, str(_AUTHORS_CODE))
    from compute_information_value import get_distances_fast
    from scorer import Scorer

    ooc = [
        "It was a bright cold day in April.",
        "Nobody expected the committee to reconvene so soon.",
        "Rain fell steadily on the empty parking lot.",
    ]
    # Keep reference arithmetic on CPU; float32 GPU batching changes rounding.
    with patch("torch.cuda.is_available", return_value=False):
        ref_scorer = Scorer()
    in_ctx = get_distances_fast(_ALTERNATIVES, _TARGET, ref_scorer, max_len=1000000)
    out_ctx = get_distances_fast(ooc, _TARGET, ref_scorer, max_len=1000000)

    ref: dict[str, float] = {}
    for k, v in in_ctx.items():
        arr = np.asarray(v, dtype=float)
        if k.endswith("_cross"):
            ref[f"mean_{k[: -len('_cross')]}"] = float(np.mean(arr))
        if k.endswith("_self"):
            name = k[: -len("_self")]
            ref[f"expected_{name}"] = float(np.mean(arr))
            ref[f"deviation_{name}"] = abs(ref[f"mean_{name}"] - float(np.mean(arr)))
    for k, v in out_ctx.items():
        arr = np.asarray(v, dtype=float)
        if k.endswith("_cross"):
            name = k[: -len("_cross")]
            ref[f"ooc_mean_{name}"] = float(np.mean(arr))
            ref[f"context_informativeness_{name}"] = float(np.mean(arr)) - float(
                np.mean(np.asarray(in_ctx[k], dtype=float))
            )
        if k.endswith("_self"):
            name = k[: -len("_self")]
            ref[f"ooc_expected_{name}"] = float(np.mean(arr))
            ref[f"expected_context_informativeness_{name}"] = float(
                np.mean(arr)
            ) - float(np.mean(np.asarray(in_ctx[k], dtype=float)))

    mine = InformationValueScorer(device="cpu").score(
        _TARGET, _ALTERNATIVES, ooc_alternatives=ooc
    )
    mine = {k[len("information_value_") :]: v for k, v in mine.items()}

    assert set(ref) <= set(mine)
    for k, r in ref.items():
        np.testing.assert_allclose(
            mine[k], r, rtol=1e-6, atol=1e-9, err_msg=f"metric {k} diverges"
        )


def test_pair_truncation_follows_the_authors_positional_cut(monkeypatch):
    """An explicit distance limit reproduces the source script's pair cut."""
    values = [10.0, 20.0, 30.0, 40.0, 45.0]
    pairs = [
        abs(values[i] - values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    target = [1.0, 2.0, 3.0, 4.0, 5.0]
    scorer = InformationValueScorer()
    monkeypatch.setattr(scorer, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        scorer,
        "_distances",
        lambda *args, **kwargs: ({"length": target}, {"length": pairs}),
    )
    out = scorer.score("target", ["alt"] * len(values), distance_limit=3)
    # (0,1), (0,2), (0,3): the pairs among the first three alternatives,
    # (0,1), (0,2), (1,2), would average 40 / 3.
    assert out["information_value_expected_length"] == np.mean([10.0, 20.0, 30.0])
    assert out["information_value_mean_length"] == np.mean(target[:3])
    assert out["information_value_deviation_length"] == abs(2.0 - 20.0)


@pytest.mark.resource_free("Synthetic distances and alternatives; no model loading")
def test_model_uses_all_pairs_by_default_and_records_explicit_truncation(monkeypatch):
    alternatives = ["1", "4", "9", "11"]

    def distances(target, samples):
        values = [float(value) for value in samples]
        return (
            {"length": [abs(float(target) - value) for value in values]},
            {
                "length": [
                    abs(left - right)
                    for i, left in enumerate(values)
                    for right in values[i + 1 :]
                ]
            },
        )

    def score(order, limit=None):
        model = InformationValueLM(
            n_samples=len(order), out_of_context=False, distance_limit=limit
        )
        monkeypatch.setattr(model, "_sentences", lambda text: [(0, len(text))])
        monkeypatch.setattr(model, "_sample_alternatives", lambda context: order)
        monkeypatch.setattr(model.scorer, "_ensure_loaded", lambda: None)
        monkeypatch.setattr(model.scorer, "_distances", distances)
        return model(_build_stim("5"))[0]

    forward, backward = score(alternatives), score(alternatives[::-1])
    expected = np.mean([3, 8, 10, 5, 7, 2])
    assert forward.values == backward.values
    assert forward.values["information_value_expected_length"] == expected
    assert forward.values["information_value_mean_length"] == 3.75
    assert forward.values["information_value_deviation_length"] == abs(3.75 - expected)
    assert forward.metadata["distance_limit"] is None
    truncated = score(alternatives, limit=len(alternatives))
    assert truncated.values["information_value_expected_length"] == np.mean(
        [3, 8, 10, 5]
    )
    assert truncated.values["information_value_expected_length"] != expected
    assert truncated.metadata["distance_limit"] == len(alternatives)
    with pytest.raises(ValueError, match="positive integer"):
        InformationValueLM(distance_limit=0)


def test_information_value_extractor_reads_outputs():
    """The InformationValue feature pulls the right key from outputs."""
    outputs = [
        FeatureModelOutput(
            values={
                "information_value_mean_cosine": 0.42,
                "information_value_min_3gram_pos": 0.1,
            }
        )
    ]
    stim = _build_stim("hello world")
    # one output per unit; here a single sentence-level output broadcast
    outs = outputs * len(stim.layers["words"])
    feat = InformationValue(distance="cosine", aggregation="mean")
    np.testing.assert_array_equal(feat(stim, outs), [0.42, 0.42])
    feat2 = InformationValue(distance="3gram_pos", aggregation="min")
    np.testing.assert_array_equal(feat2(stim, outs), [0.1, 0.1])


@pytest.mark.needs_model
@needs_spacy
def test_information_value_lm_end_to_end():
    """InformationValueLM samples alternatives from gpt2 and scores them."""
    lm = InformationValueLM(model_name="gpt2", n_samples=4, max_new_tokens=8, seed=0)
    stim = _build_stim("The dog ran across the field.")
    outputs = lm(stim, target_layer="words")
    assert len(outputs) == len(stim.layers["words"])
    iv = outputs[0].values["information_value_mean_cosine"]
    assert np.isfinite(iv) and 0.0 <= iv <= 2.0
    feat = InformationValue(distance="cosine", aggregation="mean")
    vals = feat(stim, outputs)
    assert np.all(np.isfinite(vals)) and len(vals) == len(stim.layers["words"])


@pytest.mark.resource_free("Stubbed generator and tokenizer; no checkpoint loads")
@pytest.mark.parametrize("temperature", [1.0, 0.75, 1.25])
@pytest.mark.parametrize("typical_p", [1.0, 0.3])
def test_sample_batch_splits_generation_rounds(monkeypatch, temperature, typical_p):
    """Chunked sampling issues ceil(n/batch) generate calls of the same total."""
    lm = InformationValueLM(
        n_samples=10,
        sample_batch=4,
        device="cpu",
        temperature=temperature,
        typical_p=typical_p,
    )
    calls = []

    class _Gen:
        def __init__(self, n):
            self.shape = (n, 1)

        def __iter__(self):
            import torch

            return iter([torch.tensor([0])] * self.shape[0])

    class _LM:
        def generate(self, input_ids, **kw):
            calls.append(kw["num_return_sequences"])
            assert kw.get("temperature", 1.0) == temperature
            assert ("temperature" in kw) == (temperature != 1.0)
            assert kw.get("typical_p", 1.0) == typical_p
            assert ("typical_p" in kw) == (typical_p != 1.0)
            return _Gen(kw["num_return_sequences"])

        class config:
            n_positions = 1024

    class _Tok:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            import torch

            class R(dict):
                input_ids = torch.tensor([[1, 2]])

            return R()

        def decode(self, ids, skip_special_tokens=True):
            return "word"

    lm._lm = _LM()
    lm._tok = _Tok()
    lm.truncate_to_sentence = False
    out = lm._sample_alternatives("a context")
    assert calls == [4, 4, 2]
    assert len(out) == 10


@pytest.mark.resource_free("Stubbed generator and tokenizer; no checkpoint loads")
@pytest.mark.parametrize(
    "model_name,position_limit,budget,prompt_length",
    [
        ("gpt2", None, 3, None),
        ("gpt2", 8, np.int64(7), 1),
        ("gpt2", 8, 8, None),
        ("gpt2", 8, 9, None),
        ("ai-forever/mGPT", 2048, 511, 1),
        ("ai-forever/mGPT", 2048, 512, None),
        ("ai-forever/mGPT", 2048, 513, None),
        ("gpt2", 64, None, 4),
    ],
)
def test_sampling_budget_reserves_prompt_space(
    model_name, position_limit, budget, prompt_length
):
    import torch

    kwargs = {} if budget is None else {"max_new_tokens": budget}
    lm = InformationValueLM(model_name=model_name, n_samples=2, device="cpu", **kwargs)
    prompts = []

    class _LM:
        config = SimpleNamespace(n_positions=position_limit)

        def generate(self, input_ids, **kw):
            prompts.append(input_ids.tolist()[0])
            assert kw["max_new_tokens"] == (60 if budget is None else budget)
            limit = 512 if model_name == "ai-forever/mGPT" else position_limit
            assert 1 <= input_ids.shape[1] <= limit - kw["max_new_tokens"]
            return [torch.cat([input_ids[0], torch.tensor([9])])] * kw[
                "num_return_sequences"
            ]

    class _Tok:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=torch.arange(10)[None, :])

        def decode(self, ids, skip_special_tokens=True):
            return "word"

    lm._lm = _LM()
    lm._tok = _Tok()
    lm.truncate_to_sentence = False
    if prompt_length is None:
        with pytest.raises(ValueError, match="no usable context limit|prompt token"):
            lm._sample_alternatives("a context")
        assert prompts == []
    else:
        assert lm._sample_alternatives("a context") == ["word", "word"]
        assert prompts == [list(range(10 - prompt_length, 10))]


@pytest.mark.resource_free("Invalid configuration fails before any model loading")
@pytest.mark.parametrize("budget", [0, -1, 1.5, 2.0, "2", True, None])
def test_sampling_budget_rejects_nonpositive_or_noninteger_values(budget):
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        InformationValueLM(max_new_tokens=budget)
