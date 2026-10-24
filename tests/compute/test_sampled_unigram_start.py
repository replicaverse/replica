"""The sampled unigram source draws and scores under the identity's resolved start state."""

from types import SimpleNamespace

import pytest

from replica.compute import methods


def _context(identity):
    return methods.ComputeContext(
        model_id="stub",
        beam_width=1,
        n_samples=1,
        max_tokens=1,
        alpha=1.0,
        seed=None,
        add_bos=True,
        dataset_language="en",
        language_override=None,
        model_identity=identity,
    )


class _Tokenizer:
    name_or_path = "ai-forever/mGPT"
    bos_token_id = 0
    eos_token_id = 5
    unk_token_id = 3

    def convert_tokens_to_ids(self, token):
        return {"<s>": 0, "<|endoftext|>": 5}.get(token, 3)

    def convert_ids_to_tokens(self, ids):
        return ["<s>"] if ids == [0] else ["<|endoftext|>"]


def test_the_identitys_start_ids_reach_the_sampler(monkeypatch):
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(language="en")

    monkeypatch.setitem(
        methods.SOURCES, "mmc", SimpleNamespace(build=build, lookup=lambda *a: None)
    )
    monkeypatch.setattr(methods, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(methods, "UnigramSurprisalLM", lambda source, lookup: source)
    methods._build_sampled_unigram(
        _context({"start": {"ids": [5], "tokens": ["<|endoftext|>"]}}), source="mmc"
    )
    assert seen["start_ids"] == [5]
    with pytest.raises(ValueError, match="resolved start state is empty"):
        methods._build_sampled_unigram(_context({"start": {"ids": []}}), source="mmc")


def test_without_an_identity_the_resolver_decides(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda name: _Tokenizer()
    )
    context = _context(None)
    context = methods.replace(context, model_id="ai-forever/mGPT")
    assert methods._resolved_start_ids(context) == [5]
    assert methods._resolved_start_ids(_context({"start": {"ids": [0]}})) == [0]


def test_requested_prefix_cannot_disagree_with_the_saved_identity():
    context = methods.replace(_context({"start": {"ids": [5]}}), start_ids=(0,))
    with pytest.raises(ValueError, match="disagree"):
        methods._resolved_start_ids(context)


def test_explicit_zero_overrides_default_without_an_identity(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_: _Tokenizer()
    )
    context = methods.replace(_context(None), start_ids=(0,))
    assert methods._resolved_start_ids(context) == [0]


@pytest.mark.parametrize("explicit", [None, (0,)])
def test_every_resolved_builder_accepts_and_forwards_the_requested_token(
    monkeypatch, explicit
):
    import inspect

    seen = {}

    def recorder(constructor):
        signature = inspect.signature(constructor)

        def build(*args, **kwargs):
            signature.bind(*args, **kwargs)
            seen.update(kwargs)
            return object()

        return build

    for name in (
        "HuggingFaceLM",
        "ContinuationEntropyLM",
        "ContinuationEntropyLMBuggy",
        "MCWordEntropyLM",
        "ForwardMarginalEntropyLM",
        "ForwardWordLookaheadEntropyLM",
        "ResponsivityLM",
        "SimilarityAdjustedSurprisalEmbeddingLM",
        "SimilarityAdjustedSurprisalWordLM",
        "surprisal_model",
    ):
        monkeypatch.setattr(methods, name, recorder(getattr(methods, name)))
    context = methods.replace(
        _context({"start": {"ids": [0]}}), start_ids=explicit, allow_reproduction=True
    )
    for name, spec in methods.METHOD_SPECS.items():
        if spec.start_policy != "resolved" or name == "unigram_surprisal":
            continue
        seen.clear()
        methods.build_model(name, context)
        if name == "wordsprobability_surprisal":
            assert seen["start_token_id"] == 0
        else:
            assert seen["start_ids"] == (0,), name


def test_cli_forwards_the_explicit_token_and_records_the_request(tmp_path, monkeypatch):
    from replica.compute import cli
    from replica.compute.runner import ComputeOutcome

    received = []

    def compute(dataset, **kwargs):
        received.append(kwargs)
        return ComputeOutcome(dataset, kwargs["method"], "completed")

    monkeypatch.setattr(cli, "expand_datasets", lambda _: ["example"])
    monkeypatch.setattr(cli, "compute_for", compute)
    cli.main(
        [
            "--dataset",
            "example",
            "--lm",
            "example",
            "--method",
            "surprisal",
            "--start-ids",
            "0",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert received[0]["start_ids"] == (0,)


@pytest.mark.parametrize("explicit", [None, (0,)])
def test_recorded_zero_is_the_scorers_actual_prefix(explicit):
    context = methods.replace(
        _context({"start": {"ids": [0]}}),
        model_id="ai-forever/mGPT",
        start_ids=explicit,
    )
    scorer = methods.build_model("surprisal", context)
    scorer._tokenizer = _Tokenizer()
    assert scorer._start_prefix() == [0]
    assert context.start_ids == explicit


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        {"ids": None},
        {"ids": [True]},
        {"ids": [1.5]},
        {"ids": ["0"]},
        {"ids": [-1]},
        {"ids": [0, 5]},
    ],
)
def test_invalid_recorded_prefix_is_rejected_before_loading(monkeypatch, record):
    monkeypatch.setattr(
        methods, "surprisal_model", lambda *a, **kw: pytest.fail("model loaded")
    )
    with pytest.raises(ValueError, match="start.ids|nonnegative integer"):
        methods.build_model("surprisal", _context({"start": record}))


def test_empty_record_requires_a_matching_effective_empty_prefix(monkeypatch):
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_: tokenizer
    )
    context = methods.replace(
        _context({"start": {"ids": []}}), model_id="ai-forever/mGPT"
    )
    with pytest.raises(ValueError, match="empty model identity prefix disagrees"):
        methods.build_model("surprisal", context)
    tokenizer.name_or_path = "no-bos"
    tokenizer.bos_token_id = None
    context = methods.replace(context, model_id="no-bos")
    scorer = methods.build_model("surprisal", context)
    scorer._tokenizer = tokenizer
    assert scorer.add_bos is True and scorer._start_prefix() == []
    seen = {}
    monkeypatch.setattr(methods, "surprisal_model", lambda *a, **kw: seen.update(kw))
    methods.build_model("char_beam", context)
    assert seen["start_ids"] is None


def test_disabled_empty_record_needs_no_tokenizer_and_rejects_nonempty(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_: pytest.fail("tokenizer loaded"),
    )
    context = methods.replace(_context({"start": {"ids": []}}), add_bos=False)
    scorer = methods.build_model("surprisal", context)
    assert scorer._start_prefix() == []
    context = methods.replace(context, model_identity={"start": {"ids": [0]}})
    with pytest.raises(ValueError, match="contradicts add_bos=False"):
        methods.build_model("surprisal", context)


def test_fixed_recipe_record_can_omit_a_simple_prefix(monkeypatch):
    seen = {}
    monkeypatch.setattr(methods, "InformationValueLM", lambda *a, **kw: seen.update(kw))
    methods.build_model("information_value", _context({"start": {"basis": "recipe"}}))
    assert "start_ids" not in seen
