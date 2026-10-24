"""Sample sidecars, pool identities and resumable token stores."""

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from replica.feature.frequency import sampled
from replica.feature.frequency.sampled import SampledUnigramSource
from replica.studies.hopton2026unigram.mmc import pool
from tests.feature.frequency._sampled_stub import _Model, _Tokenizer


@pytest.mark.parametrize("stage", ["samples", "shard"])
@pytest.mark.parametrize(
    "bos, requested, expected", [(1, None, 1), (None, None, 2), (1, [2], 2)]
)
def test_study_sampling_and_scoring_keep_the_original_prefix(
    tmp_path, monkeypatch, stage, bos, requested, expected
):
    from replica.studies.hopton2026unigram.mmc import estimate

    class Tokenizer(_Tokenizer):
        bos_token_id = bos

        def convert_tokens_to_ids(self, token):
            assert token == "<|endoftext|>"
            return self.eos_token_id

    seen = []
    sources = []

    class Model(_Model):
        config = SimpleNamespace(n_positions=2048)

        def generate(self, input_ids, **kwargs):
            seen.append(input_ids[0].tolist())
            return super().generate(input_ids, **kwargs)

    def build(**kwargs):
        source = SampledUnigramSource(**kwargs)
        source._tokenizer = Tokenizer()
        source._model = Model()
        sources.append(source)
        return source

    def score(words, *args, start_ids, **kwargs):
        seen.append(start_ids)
        return {word: 0.5 for word in words}

    monkeypatch.setitem(pool.SOURCES, "mmc", SimpleNamespace(build=build))
    monkeypatch.setattr(sampled, "mmc_estimate_batch", score)
    args = SimpleNamespace(
        model_name="ai-forever/mGPT",
        device="cpu",
        start_ids=requested,
        out=tmp_path / "out.json",
        ids_out=tmp_path / "ids.npz",
        shard_samples=3,
        n_samples=3,
        seed_base=7,
        shard_index=0,
        samples=tmp_path / "samples.json",
        start=0,
        end=1,
    )
    args.samples.write_text('["t5"]', encoding="utf-8")
    with warnings.catch_warnings(record=True):
        if stage == "samples":
            assert pool.cmd_samples(args) == 0
        else:
            assert estimate._shard_exhaustive(args, ["t5"]) == 0
    assert sources[0].start_ids == (expected,)
    assert sources[0].context_window == 2048
    assert seen == [[expected]]


def test_the_samples_stage_writes_a_sidecar_the_pool_reads_back(tmp_path, monkeypatch):
    source = SampledUnigramSource(
        model_name="stub", n_samples=3, max_len=5, device="cpu"
    )
    source._tokenizer = _Tokenizer()
    source._model = _Model()
    monkeypatch.setitem(
        pool.SOURCES, "mmc", SimpleNamespace(build=lambda **kwargs: source)
    )
    out = tmp_path / "shard_0000.json"
    ids_out = tmp_path / "shard_0000.ids.npz"
    args = SimpleNamespace(
        out=out,
        ids_out=ids_out,
        model_name="stub",
        device="cpu",
        shard_samples=3,
        n_samples=3,
        seed_base=7,
        shard_index=2,
    )
    assert pool.cmd_samples(args) == 0
    samples = json.loads(out.read_text(encoding="utf-8"))
    with np.load(ids_out) as blob:
        assert blob["ids"].tolist() == [5, 6, 7, 8, 9, 10, 11, 12]
        assert blob["offsets"].tolist() == [0, 2, 3, 8]
        assert blob["start"].tolist() == [1]
    rows, binding, start_binding = pool._shard_token_ids(
        out, samples, _Tokenizer(), tmp_path
    )
    assert (binding, start_binding) == ("digest", "sidecar")
    assert rows == [[1, 5, 6], [1, 7], [1, 8, 9, 10, 11, 12]]
    assert rows == sampled._encode_samples(_Tokenizer(), samples)
    with pytest.raises(ValueError, match="does not re-encode"):
        pool._shard_token_ids(out, samples, _Tokenizer(), tmp_path / "elsewhere")


def _args(tmp_path, source, monkeypatch):
    monkeypatch.setitem(
        pool.SOURCES, "mmc", SimpleNamespace(build=lambda **kwargs: source)
    )
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    return SimpleNamespace(
        out=tmp_path / "shard_0000.json",
        ids_out=tmp_path / "shard_0000.ids.npz",
        model_name="stub",
        device="cpu",
        shard_samples=3,
        n_samples=3,
        seed_base=7,
        shard_index=0,
    )


def _stub_source():
    source = SampledUnigramSource(
        model_name="stub", n_samples=3, max_len=5, device="cpu"
    )
    source._tokenizer = _Tokenizer()
    source._model = _Model()
    return source


def test_an_interrupted_sidecar_write_leaves_no_complete_shard(tmp_path, monkeypatch):
    """The text is written last, so a failed ids write leaves nothing a resume could mistake for done."""
    source = _stub_source()
    args = _args(tmp_path, source, monkeypatch)

    def fail(path, save):
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(pool, "_write_npz", fail)
    with pytest.raises(OSError):
        pool.cmd_samples(args)
    assert not args.out.exists() and not args.ids_out.exists()
    monkeypatch.undo()
    args = _args(tmp_path, _stub_source(), monkeypatch)
    assert pool.cmd_samples(args) == 0
    assert args.out.exists() and args.ids_out.exists()
    assert pool._samples_done(
        args.out, args.ids_out, pool._sampling_request(source), False
    )


def test_a_shard_without_its_requested_ids_is_refused_rather_than_skipped(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, _stub_source(), monkeypatch)
    args.out.write_text(json.dumps(["t5 t6", "t7", "t8"]), encoding="utf-8")
    with pytest.raises(SystemExit, match="not recoverable"):
        pool.cmd_samples(args)
    assert not args.ids_out.exists()


def test_a_corrupt_sidecar_is_refused_on_resume(tmp_path, monkeypatch):
    args = _args(tmp_path, _stub_source(), monkeypatch)
    assert pool.cmd_samples(args) == 0
    np.savez(
        args.ids_out,
        ids=np.array([5, 6], dtype=np.int32),
        offsets=np.array([0, 2], dtype=np.int64),
    )
    with pytest.raises(SystemExit, match="id rows for 3 samples"):
        pool.cmd_samples(args)


def test_the_pool_fingerprint_binds_the_drawn_ids(tmp_path, monkeypatch):
    """Two pools over identical text but different sidecar ids get different fingerprints."""
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    fingerprints = []
    for variant, ids in enumerate(([11, 12], [31, 32])):
        root = tmp_path / f"pool{variant}"
        shards = root / "shards"
        shards.mkdir(parents=True)
        (shards / "shard_0000.json").write_text(json.dumps(["t1 t2"]), encoding="utf-8")
        np.savez(
            shards / "shard_0000.ids.npz",
            ids=np.array(ids, dtype=np.int32),
            offsets=np.array([0, 2], dtype=np.int64),
        )
        args = SimpleNamespace(
            model_name="stub",
            shards_dir=shards,
            ids_dir=shards,
            expected_shards=1,
            seed_base=7,
            out=root / "out",
        )
        assert pool.cmd_pool(args) == 0
        manifest = json.loads((root / "out" / "manifest.json").read_text())
        assert manifest["ids_sha256"] and manifest["encoding"]["start"] == [1]
        fingerprints.append(manifest["pool_fingerprint"])
        tokens = np.fromfile(root / "out" / "tokens.i32", dtype=np.int32).tolist()
        assert tokens == [1, *ids]
    assert fingerprints[0] != fingerprints[1]


@pytest.mark.parametrize(
    ("ids", "offsets", "n", "message"),
    [
        ([1.5, 2.5], [0, 2], 1, "ids must be one-dimensional integers"),
        ([[1, 2]], [0, 2], 1, "ids must be one-dimensional integers"),
        ([1, -2], [0, 2], 1, "non-negative"),
        ([1, 2], [0.0, 2.0], 1, "offsets must be one-dimensional integers"),
        ([1, 2], [[0, 2]], 1, "offsets must be one-dimensional integers"),
        ([1, 2], [0, 2, 2], 1, "id rows for 1 samples"),
        ([1, 2], [1, 2], 1, "must rise"),
        ([1, 2, 3], [0, 2, 1, 3], 3, "must rise"),
        ([1, 2], np.array([0, 3, 2], dtype=np.uint64), 2, "must rise"),
        ([1, 2], [0, 5], 1, "must rise"),
        ([2**31], [0, 1], 1, "int32 range"),
    ],
)
def test_a_malformed_sidecar_is_refused(tmp_path, ids, offsets, n, message):
    path = tmp_path / "shard_0000.ids.npz"
    np.savez(path, ids=np.array(ids), offsets=np.array(offsets))
    with pytest.raises(ValueError, match=message):
        pool._read_sidecar(path, n)


def test_a_sidecar_drawn_with_other_text_is_refused(tmp_path, monkeypatch):
    """The same sample count is not enough: the sidecar names the text it was drawn with."""
    args = _args(tmp_path, _stub_source(), monkeypatch)
    assert pool.cmd_samples(args) == 0
    other = json.dumps(["x5 x6", "x7", "x8"])
    args.out.write_text(other, encoding="utf-8")
    with pytest.raises(SystemExit, match="different text"):
        pool.cmd_samples(args)
    with pytest.raises(ValueError, match="different text"):
        pool._shard_token_ids(
            args.out, json.loads(other), _Tokenizer(), tmp_path, other
        )


def test_a_legacy_sidecar_is_classified_and_not_resumed(tmp_path, monkeypatch):
    """A sidecar without a text digest reads as legacy in the pool and cannot complete a resume."""
    args = _args(tmp_path, _stub_source(), monkeypatch)
    samples = ["t5 t6", "t7"]
    text = json.dumps(samples)
    args.out.write_text(text, encoding="utf-8")
    np.savez(
        args.ids_out,
        ids=np.array([5, 6, 7], dtype=np.int32),
        offsets=np.array([0, 2, 3], dtype=np.int64),
    )
    rows, binding, start_binding = pool._shard_token_ids(
        args.out, samples, _Tokenizer(), tmp_path, text
    )
    assert (binding, start_binding) == ("legacy", "requested")
    assert rows == [[1, 5, 6], [1, 7]]
    with pytest.raises(SystemExit, match="legacy sidecar"):
        pool.cmd_samples(args)


def test_a_shard_that_is_not_a_list_of_strings_is_refused(tmp_path, monkeypatch):
    args = _args(tmp_path, _stub_source(), monkeypatch)
    assert pool.cmd_samples(args) == 0
    args.out.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="list of strings"):
        pool.cmd_samples(args)
    args.out.write_text(json.dumps(["t5 t6", 7, "t8"]), encoding="utf-8")
    with pytest.raises(SystemExit, match="list of strings"):
        pool.cmd_samples(args)


def _pool_args(root, **extra):
    args = SimpleNamespace(
        model_name="stub",
        shards_dir=root / "shards",
        ids_dir=root / "shards",
        expected_shards=1,
        seed_base=7,
        out=root / "out",
    )
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def _request(**overrides):
    request = {
        "model_name": "stub",
        "seed": 7,
        "n_samples": 1,
        "max_len": 5,
        "gen_batch_size": 128,
        "start": [1],
    }
    request.update(overrides)
    return request


def _write_shard(root, ids, start=None, request=None):
    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    text = json.dumps(["t1 t2"])
    (shards / "shard_0000.json").write_text(text, encoding="utf-8")
    arrays = {
        "ids": np.array(ids, dtype=np.int32),
        "offsets": np.array([0, len(ids)], dtype=np.int64),
        "text_sha256": np.array(pool.text_sha256(text), dtype="U64"),
    }
    if start is not None:
        arrays["start"] = np.array(start, dtype=np.int64)
    if request is not None:
        arrays["request"] = np.array(json.dumps(request, sort_keys=True))
    np.savez(shards / "shard_0000.ids.npz", **arrays)


def test_the_pool_binds_its_prefix_to_the_sidecars_record(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "recorded"
    _write_shard(root, [11, 12], start=[2])
    with pytest.raises(
        ValueError, match=r"drawn after \[2\], the pool is built under \[1\]"
    ):
        pool.cmd_pool(_pool_args(root))
    assert pool.cmd_pool(_pool_args(root, start_ids=[2])) == 0
    manifest = json.loads((root / "out" / "manifest.json").read_text())
    assert manifest["encoding"]["start"] == [2]
    assert manifest["start_binding"] == ["sidecar"]
    assert np.fromfile(root / "out" / "tokens.i32", dtype=np.int32).tolist() == [
        2,
        11,
        12,
    ]
    legacy = tmp_path / "legacy"
    _write_shard(legacy, [11, 12])
    assert pool.cmd_pool(_pool_args(legacy, start_ids=[2])) == 0
    manifest = json.loads((legacy / "out" / "manifest.json").read_text())
    assert manifest["start_binding"] == ["requested"]
    assert manifest["encoding"]["start"] == [2]


def test_an_existing_manifest_is_reused_only_for_the_same_inputs_and_prefix(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "pool"
    _write_shard(root, [11, 12], start=[1])
    assert pool.cmd_pool(_pool_args(root)) == 0
    assert pool.cmd_pool(_pool_args(root)) == 0
    with pytest.raises(SystemExit, match="other inputs or under another request"):
        pool.cmd_pool(_pool_args(root, start_ids=[2]))
    with pytest.raises(SystemExit, match="other inputs or under another request"):
        pool.cmd_pool(_pool_args(root, ids_dir=None))
    _write_shard(root, [11, 13], start=[1])
    with pytest.raises(SystemExit, match="other inputs or under another request"):
        pool.cmd_pool(_pool_args(root))


def test_omitting_the_sidecars_keeps_the_text_reencoding_layout(tmp_path, monkeypatch):
    """Without sidecars the pool re-encodes the decoded text, the layout older pools have."""
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "pool"
    _write_shard(root, [11, 12], start=[1])
    args = _pool_args(root, start_ids=[2])
    del args.ids_dir
    assert pool.cmd_pool(args) == 0
    manifest = json.loads((root / "out" / "manifest.json").read_text())
    assert manifest["encoding"]["format"] == pool.POOL_FORMAT_REENCODED
    assert manifest["ids_binding"] == ["reencoded"]
    assert manifest["start_binding"] == ["requested"]
    assert manifest["ids_sha256"] == []
    assert np.fromfile(root / "out" / "tokens.i32", dtype=np.int32).tolist() == [
        2,
        1,
        2,
    ]


def test_a_sidecar_recording_a_malformed_start_is_refused(tmp_path):
    path = tmp_path / "shard_0000.ids.npz"
    for start in ([1, 2], [[1]], [1.5]):
        np.savez(
            path, ids=np.array([1, 2]), offsets=np.array([0, 2]), start=np.array(start)
        )
        with pytest.raises(ValueError, match="one non-negative integer id"):
            pool._read_sidecar(path, 1)


def test_a_shard_is_resumed_only_for_the_request_that_drew_it(tmp_path, monkeypatch):
    source = _stub_source()
    args = _args(tmp_path, source, monkeypatch)
    assert pool.cmd_samples(args) == 0
    with np.load(args.ids_out) as blob:
        recorded = json.loads(str(blob["request"]))
    assert recorded == pool._sampling_request(source)
    assert set(recorded) == {*pool.REQUEST_FIELDS, "start", "generation"}
    assert recorded["start"] == [1]
    assert recorded["generation"] == sampled.GENERATION_CONTRACT
    assert pool.cmd_samples(args) == 0
    for change in (
        dict(start_ids=(2,)),
        dict(model_name="other"),
        dict(seed=8),
        dict(n_samples=4),
        dict(max_len=6),
    ):
        other = _stub_source()
        for key, value in change.items():
            setattr(other, key, value)
        monkeypatch.setitem(
            pool.SOURCES,
            "mmc",
            SimpleNamespace(build=lambda other=other, **kwargs: other),
        )
        with pytest.raises(SystemExit, match="records the request"):
            pool.cmd_samples(args)


def test_a_text_only_shard_resumes_the_plain_request_and_refuses_an_explicit_prefix(
    tmp_path, monkeypatch
):
    """Without a sidecar the text alone completes the invocation older pools were
    drawn with; an explicit prefix cannot be verified against it."""
    source = _stub_source()
    args = _args(tmp_path, source, monkeypatch)
    args.ids_out = None
    assert pool.cmd_samples(args) == 0
    assert pool.cmd_samples(args) == 0
    explicit = _stub_source()
    explicit.start_ids = (1,)
    monkeypatch.setitem(
        pool.SOURCES, "mmc", SimpleNamespace(build=lambda **kwargs: explicit)
    )
    args.start_ids = [1]
    with pytest.raises(SystemExit, match="cannot be verified against the explicit"):
        pool.cmd_samples(args)


def test_the_pool_refuses_a_sidecar_drawn_by_another_producer(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "model"
    _write_shard(root, [11, 12], start=[1], request=_request(model_name="other"))
    with pytest.raises(
        ValueError, match="records model_name='other', the pool expects 'stub'"
    ):
        pool.cmd_pool(_pool_args(root))
    root = tmp_path / "seed"
    _write_shard(root, [11, 12], start=[1], request=_request(seed=9))
    with pytest.raises(ValueError, match="records seed=9, the pool expects 7"):
        pool.cmd_pool(_pool_args(root))
    root = tmp_path / "same"
    _write_shard(root, [11, 12], start=[1], request=_request())
    assert pool.cmd_pool(_pool_args(root)) == 0


def test_a_sidecar_with_a_malformed_request_is_refused(tmp_path):
    path = tmp_path / "shard_0000.ids.npz"
    np.savez(
        path, ids=np.array([1, 2]), offsets=np.array([0, 2]), request=np.array("[1]")
    )
    with pytest.raises(ValueError, match="must be a JSON object"):
        pool._read_sidecar(path, 1)
    np.savez(
        path, ids=np.array([1, 2]), offsets=np.array([0, 2]), request=np.array("{")
    )
    with pytest.raises(ValueError, match="not JSON"):
        pool._read_sidecar(path, 1)


@pytest.mark.parametrize(
    ("start", "recorded", "message"),
    [
        ([1], _request(start=[9]), "disagrees with the recorded start"),
        ([-1], _request(start=[-1]), "non-negative integer id"),
        (None, _request(), "names a start but the sidecar records none"),
        ([1], _request(n_samples=2), "names 2 samples for 1 rows"),
        ([1], _request(start=[1, 2]), "start is invalid"),
        ([1], _request(start=[1.0]), "start is invalid"),
        ([1], _request(seed=True), "seed is not an integer"),
        ([1], _request(max_len=0), "counts are not integers"),
        ([1], _request(gen_batch_size=0), "gen_batch_size is invalid"),
        ([1], _request(model_name=""), "model_name is not a name"),
        ([1], {k: v for k, v in _request().items() if k != "seed"}, "lacks"),
    ],
)
def test_a_request_is_reconciled_with_the_sidecar_it_describes(
    tmp_path, start, recorded, message
):
    """The recorded request has to agree with the arrays beside it, for a
    resume and for a fresh pool alike."""
    _write_shard(tmp_path, [11, 12], start=start, request=recorded)
    path = tmp_path / "shards" / "shard_0000.ids.npz"
    with pytest.raises(ValueError, match=message):
        pool._read_sidecar(path, 1)


def test_a_pool_resume_is_bound_to_its_seed_and_token_store(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "pool"
    _write_shard(root, [11, 12], start=[1], request=_request())
    assert pool.cmd_pool(_pool_args(root)) == 0
    assert pool.cmd_pool(_pool_args(root)) == 0
    with pytest.raises(SystemExit, match="under another request"):
        pool.cmd_pool(_pool_args(root, seed_base=8))
    tokens = root / "out" / "tokens.i32"
    kept = tokens.read_bytes()
    tokens.unlink()
    with pytest.raises(SystemExit, match="token store is missing"):
        pool.cmd_pool(_pool_args(root))
    tokens.write_bytes(kept[:-4])
    with pytest.raises(SystemExit, match="differs from the digest"):
        pool.cmd_pool(_pool_args(root))
    tokens.write_bytes(kept)
    assert pool.cmd_pool(_pool_args(root)) == 0


def test_a_text_only_shard_skips_without_resolving_the_tokenizer(tmp_path, monkeypatch):
    """The invocation older pools were drawn with resumes offline: no model, no tokenizer."""

    def unavailable(name):
        raise OSError("offline")

    monkeypatch.setattr(pool.common, "_tokenizer", unavailable)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", unavailable)
    monkeypatch.setitem(
        pool.SOURCES, "mmc", SimpleNamespace(build=lambda **kwargs: unavailable("x"))
    )
    out = tmp_path / "shard_0000.json"
    out.write_text(json.dumps(["t5 t6"]), encoding="utf-8")
    args = SimpleNamespace(
        out=out,
        ids_out=None,
        model_name="stub",
        device="cpu",
        shard_samples=1,
        n_samples=1,
        seed_base=7,
        shard_index=0,
    )
    assert pool.cmd_samples(args) == 0


def test_a_pool_store_is_bound_by_content_and_a_legacy_store_by_structure(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "pool"
    _write_shard(root, [11, 12], start=[1], request=_request())
    assert pool.cmd_pool(_pool_args(root)) == 0
    manifest_path = root / "out" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["store_sha256"]) == {"tokens.i32", "tokens.off.i64"}
    tokens = root / "out" / "tokens.i32"
    offsets = root / "out" / "tokens.off.i64"
    kept = tokens.read_bytes()
    np.array([1, 11, 13], dtype=np.int32).tofile(tokens)
    with pytest.raises(SystemExit, match="differs from the digest"):
        pool.cmd_pool(_pool_args(root))
    tokens.write_bytes(kept)
    assert pool.cmd_pool(_pool_args(root)) == 0
    assert "content verified" in capsys.readouterr().out
    legacy = {k: v for k, v in manifest.items() if k != "store_sha256"}
    manifest_path.write_text(json.dumps(legacy))
    for bad, message in (
        (np.array([1, 11, -1], dtype=np.int32).tobytes(), "negative token ids"),
        (kept + b"\x01", "not a whole number of entries"),
    ):
        tokens.write_bytes(bad)
        with pytest.raises(SystemExit, match=message):
            pool.cmd_pool(_pool_args(root))
    tokens.write_bytes(kept)
    kept_offsets = offsets.read_bytes()
    for bad, message in (
        (np.array([0, 0], dtype=np.int64).tobytes(), "non-empty rows"),
        (np.array([0, 4], dtype=np.int64).tobytes(), "non-empty rows"),
        (np.array([0, 2, 3], dtype=np.int64).tobytes(), "2 rows for 1 samples"),
        (kept_offsets + b"\x00", "not a whole number of entries"),
    ):
        offsets.write_bytes(bad)
        with pytest.raises(SystemExit, match=message):
            pool.cmd_pool(_pool_args(root))
    offsets.write_bytes(kept_offsets)
    np.array([1, 11, 13], dtype=np.int32).tofile(tokens)
    assert pool.cmd_pool(_pool_args(root)) == 0
    assert "structure verified only" in capsys.readouterr().out


def test_the_store_array_validator_refuses_empty_rows_before_indexing():
    tokens = np.array([1, 2, 1, 4], dtype=np.int32)
    assert pool._store_arrays_problem(tokens, np.array([0, 2, 4]), 2, [1]) is None
    assert "non-empty rows" in pool._store_arrays_problem(
        tokens, np.array([0, 0, 4]), 2, [1]
    )
    assert "non-empty rows" in pool._store_arrays_problem(
        tokens, np.array([0, 4, 4]), 2, [1]
    )
    assert "recorded prefix" in pool._store_arrays_problem(
        tokens, np.array([0, 2, 4]), 2, [9]
    )
    assert "2 rows for 3" in pool._store_arrays_problem(
        tokens, np.array([0, 2, 4]), 3, [1]
    )


@pytest.mark.parametrize(
    "recorded",
    [
        {},
        None,
        "abc",
        ["tokens.i32", "tokens.off.i64"],
        {"tokens.i32": "0" * 64},
        {"tokens.off.i64": "0" * 64},
        {"tokens.i32": "0" * 64, "tokens.off.i64": "0" * 64, "other": "0" * 64},
        {"tokens.i32": "0" * 64, "tokens.off.i64": "xyz"},
        {"tokens.i32": "0" * 64, "tokens.off.i64": 7},
    ],
    ids=str,
)
def test_a_present_store_digest_field_must_name_both_files(
    tmp_path, monkeypatch, recorded
):
    """Only an absent field means a store written before digests; anything
    else present has to be the complete mapping, and is checked."""
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "pool"
    _write_shard(root, [11, 12], start=[1], request=_request())
    assert pool.cmd_pool(_pool_args(root)) == 0
    manifest_path = root / "out" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["store_sha256"] = recorded
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="store_sha256 is not a mapping of both"):
        pool.cmd_pool(_pool_args(root))


def test_a_partial_digest_mapping_cannot_pass_a_mutated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "pool"
    _write_shard(root, [11, 12], start=[1], request=_request())
    assert pool.cmd_pool(_pool_args(root)) == 0
    manifest_path = root / "out" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    good = dict(manifest["store_sha256"])
    np.array([1, 11, 13], dtype=np.int32).tofile(root / "out" / "tokens.i32")
    manifest["store_sha256"] = {"tokens.off.i64": good["tokens.off.i64"]}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="store_sha256 is not a mapping of both"):
        pool.cmd_pool(_pool_args(root))
    manifest["store_sha256"] = good
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="differs from the digest"):
        pool.cmd_pool(_pool_args(root))


def test_a_shard_drawn_under_an_earlier_generation_contract_is_not_resumed(
    tmp_path, monkeypatch
):
    """A sidecar whose request names no generation contract, or another one,
    came from an earlier producer: automatic resume is rejected."""
    source = _stub_source()
    args = _args(tmp_path, source, monkeypatch)
    assert pool.cmd_samples(args) == 0
    with np.load(args.ids_out) as blob:
        arrays = {k: blob[k] for k in blob.files}
    recorded = json.loads(str(arrays["request"]))
    for stale in (
        {k: v for k, v in recorded.items() if k != "generation"},
        {**recorded, "generation": "replica/mmc-generate/2"},
    ):
        arrays["request"] = np.array(json.dumps(stale, sort_keys=True))
        np.savez(args.ids_out, **arrays)
        with pytest.raises(SystemExit, match="records the request"):
            pool.cmd_samples(args)
    arrays["request"] = np.array(json.dumps(recorded, sort_keys=True))
    np.savez(args.ids_out, **arrays)
    assert pool.cmd_samples(args) == 0


def test_an_archived_sidecar_without_a_generation_contract_is_still_pooled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pool.common, "_tokenizer", lambda name: _Tokenizer())
    root = tmp_path / "pool"
    _write_shard(root, [11, 12], start=[1], request=_request())
    assert "generation" not in _request()
    assert pool.cmd_pool(_pool_args(root)) == 0
    manifest = json.loads((root / "out" / "manifest.json").read_text())
    assert manifest["start_binding"] == ["sidecar"]
    other = tmp_path / "bad"
    _write_shard(other, [11, 12], start=[1], request=_request(generation=7))
    with pytest.raises(ValueError, match="generation is not a name"):
        pool.cmd_pool(_pool_args(other))
