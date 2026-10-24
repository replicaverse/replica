"""Offline model bundles exercise identity checks and actual entropy generation."""

import hashlib
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from replica.compute import artifacts, methods
from replica.core import AnnotationLayer, Span, Stimulus
from replica.feature.backends.hub import local_model_digest, resolve_model_bundle
from replica.feature.entropy import (
    ContinuationEntropyLM,
    ContinuationEntropyLMBuggy,
    ForwardMarginalEntropyLM,
    ForwardWordLookaheadEntropyLM,
    MCWordEntropyLM,
    _base,
)

REVISION = "a" * 40
FAMILIES = {
    "continuation_entropy": ContinuationEntropyLM,
    "forward_marginal_entropy": ForwardMarginalEntropyLM,
    "forward_word_lookahead_entropy": ForwardWordLookaheadEntropyLM,
    "mc_word_entropy": MCWordEntropyLM,
    "continuation_entropy_buggy": ContinuationEntropyLMBuggy,
}
POSITION_METHODS = tuple(name for name in FAMILIES if not name.endswith("buggy"))


@pytest.fixture
def bundles(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    monkeypatch.setattr(_base, "resolve_device", lambda: "cpu")
    result = {}
    for seed, revision in enumerate((REVISION, "b" * 40)):
        path = tmp_path / f"bundle-{seed}"
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = GPT2LMHeadModel(
                GPT2Config(
                    n_layer=1,
                    n_head=1,
                    n_embd=4,
                    vocab_size=5,
                    n_positions=32,
                    bos_token_id=1,
                    eos_token_id=None,
                    pad_token_id=0,
                )
            )
        model.save_pretrained(path)
        backend = Tokenizer(
            models.WordLevel(
                {"<unk>": 0, "<bos>": 1, "Ġa": 2, "Ġb": 3, "c": 4}, unk_token="<unk>"
            )
        )
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<unk>",
            pad_token="<unk>",
        )
        tokenizer.save_pretrained(path)
        result[revision] = path
    return result


@pytest.fixture
def hub(bundles, tmp_path, monkeypatch):
    import huggingface_hub

    state = SimpleNamespace(root=tmp_path / "cache-one", requests=[], pinned=REVISION)

    def info(repo_id, revision=None, files_metadata=False):
        pinned = revision or state.pinned
        entries = []
        for path in bundles[pinned].iterdir():
            payload = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            lfs = (
                SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest())
                if path.suffix == ".safetensors"
                else None
            )
            entries.append(SimpleNamespace(rfilename=path.name, blob_id=blob, lfs=lfs))
        return SimpleNamespace(
            id="org/model",
            sha=pinned,
            siblings=entries
            + [
                SimpleNamespace(rfilename="pytorch_model.bin"),
                SimpleNamespace(rfilename="README.md"),
            ],
        )

    def download(repo_id, *, revision, allow_patterns):
        state.requests.append((repo_id, revision, tuple(allow_patterns)))
        assert "pytorch_model.bin" not in allow_patterns
        path = state.root / "models--org--model" / "snapshots" / revision
        path.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            if not (path / name).exists():
                shutil.copy2(bundles[revision] / name, path / name)
        return str(path)

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=info)
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return state


@pytest.fixture
def scorer(bundles):
    def build(model_id=None, method="continuation_entropy", **kwargs):
        cls = FAMILIES[method]
        return cls(str(model_id or bundles[REVISION]), n_samples=4, seed=7, **kwargs)

    return build


def stimulus(text="a b", identifier="s"):
    return Stimulus(
        text,
        {"words": AnnotationLayer("words", [Span(0, 1), Span(2, 3)], text.split())},
        identifier,
    )


def draws(model, stim=None):
    model._seed_for_stimulus(stim or stimulus(), "words")
    return model._uniform_block(("unit", 0), 0, model.n_samples, 6).numpy()


@pytest.mark.parametrize("method", POSITION_METHODS)
def test_relocated_cache_keeps_public_direct_and_compute_streams_equal(
    hub, scorer, tmp_path, method
):
    context = methods.ComputeContext("org/model", 5, 4, 2, 1.0, 7, True, "en", None)
    path, identity = artifacts.resolve_model_snapshot(method, context)
    original = methods.build_model(
        method, replace(context, model_id=path, model_identity=identity)
    )
    expected = draws(original)
    hub.root = tmp_path / "cache-two"
    for model_id in ("org/model", "model-alias", path):
        direct = scorer(model_id, method)
        np.testing.assert_array_equal(draws(direct), expected)
        assert direct._provenance["model_identity"]["sha256"] == identity["sha256"]
    assert identity["revision"] == REVISION
    assert hub.requests


def test_actual_local_generation_repeats_after_relocation_and_resume(
    scorer, bundles, tmp_path
):
    relocated = tmp_path / "relocated"
    shutil.copytree(bundles[REVISION], relocated)
    first, resumed = scorer(), scorer(relocated)
    initial = first(stimulus())
    first(stimulus("b a"))
    for actual in (first(stimulus(identifier="renamed")), resumed(stimulus())):
        for left, right in zip(initial, actual, strict=True):
            np.testing.assert_equal(left.values, right.values)
    np.testing.assert_array_equal(draws(first), draws(resumed))
    np.testing.assert_array_equal(
        first._uniform_block(("unit", 0), 2, 2, 6).numpy(), draws(first)[2:]
    )


def test_distinct_loaded_model_contents_select_distinct_streams(scorer, bundles, hub):
    first = scorer("org/model")
    second = scorer("org/model", model_identity={"revision": "b" * 40})
    assert not np.array_equal(draws(first), draws(second))
    assert first._provenance["model_identity"]["revision"] == REVISION
    assert second._provenance["model_identity"]["revision"] == "b" * 40
    assert not torch.equal(
        scorer(bundles[REVISION])._uniform_block(("p", 2), 0, 4, 6),
        scorer(bundles["b" * 40])._uniform_block(("p", 2), 0, 4, 6),
    )


def test_local_hash_claim_is_verified_and_tokenizer_changes_invalidate_it(
    scorer, bundles
):
    path = bundles[REVISION]
    digest = local_model_digest(path)
    assert (
        scorer(model_identity={"sha256": digest})._provenance["model_identity"][
            "sha256"
        ]
        == digest
    )
    with pytest.raises(ValueError, match="sha256"):
        scorer(model_identity={"sha256": "b" * 64})
    config = path / "tokenizer_config.json"
    value = json.loads(config.read_text())
    value["model_max_length"] = 16
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="sha256"):
        scorer(model_identity={"sha256": digest})


def test_local_revision_and_remote_hash_claims_are_not_accepted(
    scorer, bundles, tmp_path
):
    for directory in (
        bundles[REVISION],
        tmp_path / "models--org--model" / "snapshots" / REVISION,
    ):
        if not directory.exists():
            shutil.copytree(bundles[REVISION], directory)
        assert scorer(directory)._provenance["model_identity"]["revision"] is None
        with pytest.raises(ValueError, match="verified Hub source"):
            scorer(directory, model_identity={"revision": REVISION})
    with pytest.raises(ValueError, match="remote hash assertion"):
        scorer("org/model", model_identity={"sha256": "b" * 64})


def test_asserted_hub_source_is_checked_against_actual_local_files(
    scorer, hub, bundles
):
    path, identity = resolve_model_bundle("org/model")
    scorer(path, model_identity=identity)
    identity["revision"] = "b" * 40
    with pytest.raises(ValueError, match="Local files differ"):
        scorer(path, model_identity=identity)


def test_extra_cache_files_cannot_silently_change_a_hub_load(hub, scorer):
    path, _ = resolve_model_bundle("org/model")
    from pathlib import Path

    (Path(path) / "extra_config.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned Hub inventory"):
        scorer("org/model")


def test_mutable_identity_and_unknown_hub_revision_are_not_reported_as_pinned(
    scorer, monkeypatch
):
    import huggingface_hub

    for revision in ("main", "v1", "abc123"):
        with pytest.raises(ValueError, match="full Hub revision"):
            scorer(model_identity={"revision": revision})
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *args, **kwargs: SimpleNamespace(sha=None)
        ),
    )
    with pytest.raises(ValueError, match="immutable model revision"):
        scorer("org/model")


def test_released_word_entropy_retains_its_advancing_generator(scorer):
    first = scorer(method="mc_word_entropy", max_iter=2)
    second = scorer(method="mc_word_entropy", max_iter=2)
    first.release_rng_stream = second.release_rng_stream = True
    before = first._rng.get_state().clone()
    a, b = first(stimulus()), second(stimulus())
    for left, right in zip(a, b, strict=True):
        np.testing.assert_equal(left.values, right.values)
    assert not torch.equal(before, first._rng.get_state())
    state = first._rng.get_state().clone()
    first._seed_for_stimulus(stimulus(), "words")
    assert torch.equal(state, first._rng.get_state())


def test_buggy_continuation_uses_actual_generate_and_reports_global_seed(
    scorer, monkeypatch
):
    model = scorer(
        method="continuation_entropy_buggy", max_tokens=2, allow_reproduction=True
    )
    calls = []
    generate = model._model.generate

    def record(*args, **kwargs):
        result = generate(*args, **kwargs)
        calls.append((torch.initial_seed(), result.sequences.clone()))
        return result

    monkeypatch.setattr(model._model, "generate", record)
    first = model(stimulus())
    first_sequences = [sequence for _, sequence in calls]
    calls.clear()
    torch.manual_seed(123)
    second = model(stimulus())
    assert len(calls) == len(first_sequences) == 2
    assert all(seed == 7 for seed, _ in calls)
    for (_, sequence), expected in zip(calls, first_sequences, strict=True):
        assert torch.equal(sequence, expected)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_equal(left.values, right.values)
        assert right.metadata["sampler"] == "transformers.generate"
        assert right.metadata["seed_scheme"] == "global-per-stimulus"


def test_modified_cached_weights_do_not_keep_a_verified_hub_revision(hub, scorer):
    from pathlib import Path

    path, _ = resolve_model_bundle("org/model")
    weights = Path(path) / "model.safetensors"
    payload = weights.read_bytes()
    weights.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match="pinned Hub content"):
        scorer("org/model")


def test_offline_cached_name_loads_by_content_without_claiming_hub_verification(
    bundles, scorer, monkeypatch
):
    import huggingface_hub
    from huggingface_hub.errors import OfflineModeIsEnabled

    def unavailable(*args, **kwargs):
        raise OfflineModeIsEnabled("offline")

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=unavailable)
    )
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *args, **kwargs: str(bundles[REVISION] / "config.json"),
    )
    model = scorer("org/model")
    assert model._provenance["model_identity"]["revision"] is None
    assert model._provenance["model_identity"]["sha256"] == local_model_digest(
        bundles[REVISION]
    )
    with pytest.raises(ValueError, match="cannot be verified offline"):
        scorer("org/model", model_identity={"revision": REVISION})
    with pytest.raises(ValueError, match="cannot be verified offline"):
        artifacts.resolve_model_snapshot(
            "continuation_entropy",
            methods.ComputeContext(
                "org/model",
                5,
                4,
                2,
                1.0,
                7,
                True,
                "en",
                None,
                model_revision=REVISION,
            ),
        )


@pytest.mark.parametrize("mutation", ["tokenizer", "inventory"])
def test_bundle_changes_during_loading_are_rejected(
    bundles, scorer, monkeypatch, mutation
):
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained

    def load_then_change(*args, **kwargs):
        tokenizer = original(*args, **kwargs)
        path = bundles[REVISION]
        if mutation == "tokenizer":
            config = path / "tokenizer_config.json"
            config.write_text(config.read_text() + "\n")
        else:
            (path / "added_config.json").write_text("{}")
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_then_change)
    with pytest.raises(ValueError, match="changed during model/tokenizer loading"):
        scorer()


def test_sharded_inventory_selects_one_complete_weight_family():
    from replica.feature.backends.hub import _bundle_inventory

    index = {
        "weight_map": {
            "first": "model-00001-of-00002.safetensors",
            "second": "model-00002-of-00002.safetensors",
        }
    }
    names = [
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        *index["weight_map"].values(),
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "README.md",
    ]
    selected = _bundle_inventory(names, lambda _: index)
    assert selected == sorted(
        [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            *index["weight_map"].values(),
        ]
    )
    with pytest.raises(ValueError, match="missing weight files"):
        _bundle_inventory(
            [name for name in names if name != "model-00002-of-00002.safetensors"],
            lambda _: index,
        )
    assert _bundle_inventory(
        ["config.json", "tokenizer.json", "pytorch_model.bin"], lambda _: None
    ) == ["config.json", "pytorch_model.bin", "tokenizer.json"]


@pytest.mark.parametrize("prefix", [(0,), (1,), None])
@pytest.mark.parametrize(
    "family",
    ["entropy", "huggingface", "responsivity", "similarity", "word_similarity"],
)
@pytest.mark.resource_free("Tiny locally saved weights; no external model downloads")
def test_direct_outputs_record_the_effective_prefix(bundles, family, prefix):
    from replica.feature import (
        HuggingFaceLM,
        ResponsivityLM,
        SimilarityAdjustedSurprisalEmbeddingLM,
        SimilarityAdjustedSurprisalWordLM,
    )

    constructors = {
        "entropy": lambda path, **kw: ContinuationEntropyLM(
            path, n_samples=2, max_tokens=2, seed=7, **kw
        ),
        "huggingface": HuggingFaceLM,
        "responsivity": lambda path, **kw: ResponsivityLM(path, n_samples=2, **kw),
        "similarity": lambda path, **kw: SimilarityAdjustedSurprisalEmbeddingLM(
            path, n_samples=2, **kw
        ),
        "word_similarity": lambda path, **kw: SimilarityAdjustedSurprisalWordLM(
            path, n_samples=2, max_iter=2, pos=False, **kw
        ),
    }
    model = constructors[family](
        str(bundles[REVISION]), add_bos=prefix is not None, start_ids=prefix
    )
    expected = list(prefix) if prefix is not None else []
    outputs = model(stimulus())
    assert outputs
    assert all(output.metadata["start_ids"] == expected for output in outputs)
    if family in {"huggingface", "responsivity", "similarity"}:
        tokens = model.score_tokens(stimulus())
        assert tokens
        assert all(output.metadata["start_ids"] == expected for output in tokens)
    if family == "entropy":
        assert "start_ids" not in model._model_identity


@pytest.mark.parametrize("method", ["surprisal", "continuation_entropy"])
@pytest.mark.resource_free("Tiny locally saved weights and tokenizer; no Hub access")
def test_resolved_empty_prefix_preserves_no_bos_model_values(bundles, method):
    from transformers import AutoTokenizer

    from replica.feature import HuggingFaceLM

    path = str(bundles[REVISION])
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.bos_token = None
    tokenizer.save_pretrained(path)
    _, identity = resolve_model_bundle(path)
    identity["start"] = {"ids": [], "basis": "none"}
    context = methods.ComputeContext(
        path, 5, 2, 2, 1.0, 7, True, "en", None, model_identity=identity
    )
    direct = (
        HuggingFaceLM(path)
        if method == "surprisal"
        else ContinuationEntropyLM(path, n_samples=2, max_tokens=2, seed=7)
    )
    computed = methods.build_model(method, context)
    for expected, actual in zip(direct(stimulus()), computed(stimulus()), strict=True):
        assert actual.metadata["start_ids"] == []
        assert set(actual.values) == set(expected.values)
        for key in expected.values:
            np.testing.assert_array_equal(actual.values[key], expected.values[key])


@pytest.mark.parametrize("method", POSITION_METHODS)
def test_entropy_context_override_controls_dispatch_and_provenance(scorer, method):
    options = {"max_iter": 2} if "word" in method else {"max_tokens": 2}
    if method == "forward_word_lookahead_entropy":
        options["max_words"] = 1
    model = scorer(method=method, context_window=8, **options)
    lengths = []

    def record_length(_module, args, kwargs):
        prefix = kwargs.get("past_key_values")
        cached = prefix.get_seq_length() if prefix is not None else 0
        lengths.append(args[0].shape[-1] + cached)

    handle = model._model.register_forward_pre_hook(record_length, with_kwargs=True)
    text = " ".join(["a", "b"] * 8)
    stim = Stimulus(
        text,
        {
            "words": AnnotationLayer(
                "words", [Span(i, i + 1) for i in range(0, len(text), 2)], text.split()
            )
        },
        "windowed",
    )
    try:
        outputs = model(stim)
    finally:
        handle.remove()
    assert lengths and max(lengths) <= 8
    assert all(output.metadata["context_window"] == 8 for output in outputs)
    assert all(output.metadata["start_ids"] == [1] for output in outputs)


@pytest.mark.parametrize("family", ["responsivity", "similarity", "word_similarity"])
@pytest.mark.resource_free("Tiny local model exercises configured distance windows")
def test_distance_context_override_reaches_outputs(bundles, family):
    from replica.feature import (
        ResponsivityLM,
        SimilarityAdjustedSurprisalEmbeddingLM,
        SimilarityAdjustedSurprisalWordLM,
    )

    constructors = {
        "responsivity": ResponsivityLM,
        "similarity": SimilarityAdjustedSurprisalEmbeddingLM,
        "word_similarity": lambda path, **kw: SimilarityAdjustedSurprisalWordLM(
            path, max_iter=2, pos=False, **kw
        ),
    }
    model = constructors[family](
        str(bundles[REVISION]),
        context_window=8,
        start_ids=(0,),
        add_bos=True,
        n_samples=2,
        device="cpu",
    )
    outputs = model(stimulus())
    assert outputs
    assert all(output.metadata["context_window"] == 8 for output in outputs)
    assert all(output.metadata["start_ids"] == [0] for output in outputs)


@pytest.mark.parametrize("family", ["entropy", "responsivity", "similarity"])
def test_mgpt_estimator_settings_resolve_once(family):
    import warnings

    from replica.feature.distance.responsivity import ResponsivityLM
    from replica.feature.distance.similarity_surprisal import _BaseSimilaritySurprisalLM

    cls = {
        "entropy": _base._BaseEntropyEstimatorLM,
        "responsivity": ResponsivityLM,
        "similarity": _BaseSimilaritySurprisalLM,
    }[family]
    model = cls.__new__(cls)
    model.model_id = model.model_name = "ai-forever/mGPT"
    model.add_bos = True
    model.context_window = 1024
    model._requested_start_ids = (0,)
    model._tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: 5,
        convert_ids_to_tokens=lambda ids: ["<s>" for _ in ids],
    )
    model._model = SimpleNamespace(config=SimpleNamespace(n_positions=2048))
    context = model._model_max_ctx_size if family == "entropy" else model._max_ctx
    prefix = model._resolved_start_ids if family == "entropy" else model._start_prefix
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            assert context() == 1024
            assert prefix() == [0]
    assert len(caught) == 2
    assert "512" in str(caught[0].message)
    assert "start token" in str(caught[1].message)


def test_reproduction_continuation_rejects_context_without_prefix_budget(scorer):
    from replica.feature.core.reproduction import ReproductionOnlyWarning

    with pytest.warns(ReproductionOnlyWarning):
        model = scorer(
            method="continuation_entropy_buggy",
            context_window=2,
            max_tokens=2,
            allow_reproduction=True,
        )
    with pytest.raises(ValueError, match="context_window must exceed max_tokens"):
        model(stimulus())


@pytest.mark.parametrize(
    "record, options, error",
    [
        ({"start": {"ids": [0]}}, {}, "start_ids disagree"),
        ({"start": {"ids": [1]}}, {"start_ids": (0,)}, "start_ids disagree"),
        ({"start": {}}, {}, "start.ids sequence"),
        (
            {"start": {"ids": [False]}},
            {},
            "start_ids requires exactly one nonnegative integer",
        ),
        ({"context_window": 8}, {}, "context_window disagrees"),
        ({"context_window": 8}, {"context_window": 16}, "context_window disagrees"),
        ({"context_window": None}, {}, "context_window disagrees"),
        ({"context_window": True}, {}, "integer of at least 2"),
    ],
)
def test_direct_entropy_rejects_inconsistent_settings_identity(
    bundles, scorer, record, options, error
):
    _, identity = resolve_model_bundle(str(bundles[REVISION]))
    with pytest.raises(ValueError, match=error):
        scorer(model_identity={**identity, **record}, **options)


@pytest.mark.parametrize("prefix", [(0,), None])
def test_direct_entropy_accepts_matching_settings_identity(bundles, scorer, prefix):
    _, identity = resolve_model_bundle(str(bundles[REVISION]))
    identity.update(start={"ids": list(prefix or ())}, context_window=16)
    model = scorer(
        model_identity=identity,
        start_ids=prefix,
        add_bos=prefix is not None,
        context_window=16,
    )
    assert model._provenance["start_ids"] == list(prefix or ())
    assert model._provenance["context_window"] == 16
