"""Checkpointed priming of :class:`SampledUnigramSource`.

A job killed while ``estimate_words`` primes the cache resumes from the file it
left and ends with the estimates of one uninterrupted run over the same sampled
contexts. The stub models are CPU functions of the prefix, so every comparison
is exact.
"""

import math
import pickle
import random

import numpy as np
import pytest

from replica.feature.frequency import sampled
from replica.feature.frequency.sampled import SampledUnigramSource, mmc_estimate_batch
from replica.utils.atomic import atomic_pickle
from tests.feature.frequency._pooled_stub import tiny_pair
from tests.feature.frequency._priming_stub import (
    C1,
    C2,
    Killed,
    refuse_sampling,
    stub_source,
)

OWNER = {"method": "mmc_unigram_surprisal", "corpus_sha256": "0" * 64}
#: Two single-token words, then multi-token words of two and three tokens.
WORDS = ["a", " a", " the", " dog", " zq"]


def _load(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


@pytest.fixture(autouse=True)
def _write_every_group(monkeypatch):
    monkeypatch.setattr(sampled, "PRIMING_WRITE_INTERVAL", 0.0)


@pytest.fixture
def saved(tmp_path):
    """The file an uninterrupted checkpointed run over C1 leaves, and its payload."""
    path = tmp_path / "cell.pkl.primed"
    stub_source().estimate_words(WORDS, checkpoint=path, owner=OWNER)
    return path, _load(path)


def test_checkpointed_priming_matches_the_plain_call(tmp_path):
    """One batched call serves the whole request, so the column sums of the
    single-token words see the same matrix as the call without a checkpoint.
    Calling the estimator once per word moves the last bits of some of them.
    """
    tokenizer, model = tiny_pair()
    pieces = [" a", " b", " ac", " bc", " acc"]
    rng = random.Random(3)
    contexts = ["".join(rng.choice(pieces) for _ in range(12)) for _ in range(400)]
    words = [" a", " b", "c", " ac", " bc"]

    def source():
        built = SampledUnigramSource(samples=contexts, model_name="tiny")
        built._tokenizer, built._model = tokenizer, model
        return built

    direct = mmc_estimate_batch(words, contexts, model, tokenizer, 64)
    plain = source().estimate_words(words)
    checkpointed = source().estimate_words(
        words, checkpoint=tmp_path / "cell.pkl.primed", owner=OWNER
    )
    assert plain == direct
    assert checkpointed == direct
    assert _load(tmp_path / "cell.pkl.primed")["estimates"] == direct


def test_resume_matches_an_uninterrupted_run_at_every_kill_point(tmp_path, monkeypatch):
    """The resumed run reuses the saved contexts without sampling and requests
    only the words the killed run left unfinished.
    """
    whole = stub_source()
    expected = whole.estimate_words(WORDS)
    total = whole._model.calls
    assert total > 10

    requested: list[list[str]] = []

    def spy(words, *args, **kwargs):
        requested.append(list(words))
        return mmc_estimate_batch(words, *args, **kwargs)

    monkeypatch.setattr(sampled, "mmc_estimate_batch", spy)
    saved_counts = []
    for limit in range(total):
        path = tmp_path / f"kill{limit}.pkl.primed"
        with pytest.raises(Killed):
            stub_source(kill_after=limit).estimate_words(
                WORDS, checkpoint=path, owner=OWNER
            )
        saved = _load(path)
        assert saved["contexts"] == C1
        saved_counts.append(len(saved["estimates"]))

        requested.clear()
        resumed = stub_source(generate=refuse_sampling).estimate_words(
            WORDS, checkpoint=path, owner=OWNER
        )
        assert resumed == expected, f"killed after {limit} forward calls"
        assert requested == [[w for w in WORDS if w not in saved["estimates"]]]
        assert _load(path)["estimates"] == expected

    # Killed in the sweep, nothing is finished; killed in the last word, every
    # other word is on disk.
    assert saved_counts[0] == 0
    assert saved_counts[-1] == len(WORDS) - 1
    assert saved_counts == sorted(saved_counts)


def _edit(**fields):
    return lambda payload: {**payload, **fields}


def _estimate(value):
    return lambda payload: {
        **payload,
        "estimates": {**payload["estimates"], " the": value},
    }


def _setting(name, value):
    return pytest.param({name: value}, None, f"settings differ ({name})", id=name)


def _damage(case, rewrite, reason):
    return pytest.param({}, rewrite, reason, id=case)


#: Files a run ignores: the settings and owner of the run that finds the file,
#: how the saved payload or its bytes are rewritten, and the reason logged.
_IGNORED = [
    _setting("owner", {**OWNER, "corpus_sha256": "1" * 64}),
    _setting("n_samples", len(C1) + 1),
    _setting("max_len", 20),
    _setting("seed", 1),
    _setting("gen_batch_size", 2),
    _setting("batch_size", 3),
    _setting("prob_floor", 1e-4),
    _setting("max_continuation_boundaries", 3),
    _setting("context_window", 64),
    pytest.param({"samples": C2}, None, "contexts differ", id="supplied_samples"),
    _damage("truncated", lambda payload: pickle.dumps(payload)[:-40], "unreadable"),
    _damage("garbage", lambda _: b"not a pickle", "unreadable"),
    _damage("empty", lambda _: b"", "unreadable"),
    _damage("pickled_list", list, "not a priming checkpoint"),
    _damage(
        "fingerprint_not_a_dict",
        _edit(fingerprint=None),
        "settings differ (owner, start, context_window, n_samples, max_len, seed, gen_batch_size, "
        "batch_size, prob_floor, max_continuation_boundaries)",
    ),
    _damage(
        "extra_setting",
        lambda payload: {**payload, "fingerprint": {**payload["fingerprint"], "x": 1}},
        "settings differ (extra keys)",
    ),
    _damage("wrong_format", _edit(format="other/1"), "not a priming checkpoint"),
    _damage(
        "legacy_encoding",
        _edit(format="replica/mmc-priming/1"),
        "not a priming checkpoint",
    ),
    _damage(
        "previous_generation_contract",
        _edit(format="replica/mmc-priming/2"),
        "not a priming checkpoint",
    ),
    _damage("contexts_not_strings", _edit(contexts=[1, 2, 3, 4]), "list of strings"),
    _damage("missing_context", _edit(contexts=C1[:-1]), "3 contexts, expected 4"),
    _damage("infinite_estimate", _estimate(math.inf), "estimates are not"),
    _damage("negative_estimate", _estimate(-0.5), "estimates are not"),
    _damage("non_float_estimate", _estimate("0.1"), "estimates are not"),
]


@pytest.mark.parametrize(("change", "rewrite", "reason"), _IGNORED)
def test_a_foreign_or_damaged_file_is_replaced(saved, caplog, change, rewrite, reason):
    path, payload = saved
    if rewrite is not None:
        damaged = rewrite(payload)
        path.write_bytes(
            damaged if isinstance(damaged, bytes) else pickle.dumps(damaged)
        )
    settings = dict(change)
    owner = settings.pop("owner", OWNER)
    fresh = stub_source(C2, **settings).estimate_words(WORDS)
    assert fresh != payload["estimates"]

    source = stub_source(C2, **settings)
    assert source.estimate_words(WORDS, checkpoint=path, owner=owner) == fresh
    assert reason in caplog.text
    assert _load(path) == {
        "format": payload["format"],
        "fingerprint": source._priming_fingerprint(owner),
        "scorer": payload["scorer"],
        "contexts": C2,
        "estimates": fresh,
    }


def _moved(payload, **fields):
    """*payload* with its estimates moved in their last digits, as another
    scorer's are, so a resumed estimate is told apart from a recomputed one.
    """
    estimates = {w: p * (1 + 1e-6) for w, p in payload["estimates"].items()}
    assert estimates != payload["estimates"]
    return {**payload, "estimates": estimates, **fields}


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(_edit(), id="as_saved"),
        # The estimator returns NaN where the model's probabilities are NaN.
        pytest.param(_estimate(math.nan), id="nan_estimate"),
        # A key the reader does not use, such as token ids, is ignored.
        pytest.param(_edit(context_ids=[[0, 1]]), id="unknown_key"),
    ],
)
def test_a_file_from_the_same_scorer_is_resumed_as_saved(saved, edit):
    path, payload = saved
    resumable = edit(_moved(payload))
    path.write_bytes(pickle.dumps(resumable))

    source = stub_source(generate=refuse_sampling)
    result = source.estimate_words(WORDS, checkpoint=path, owner=OWNER)
    assert source._samples == C1
    np.testing.assert_equal(result, resumable["estimates"])


def test_estimates_from_another_scorer_are_recomputed_on_its_contexts(saved):
    path, payload = saved
    other = {"device": "cuda", "name": "Another GPU", "cuda": "12.1"}
    path.write_bytes(pickle.dumps(_moved(payload, scorer=other)))

    source = stub_source(generate=refuse_sampling)
    result = source.estimate_words(WORDS, checkpoint=path, owner=OWNER)
    assert result == payload["estimates"]
    assert _load(path) == payload


def test_writes_before_the_sweep_and_after_the_last_word(tmp_path, monkeypatch):
    """With a write interval longer than the run, the file is written with the
    contexts alone and then with every estimate, each write flushed to disk.
    """
    monkeypatch.setattr(sampled, "PRIMING_WRITE_INTERVAL", math.inf)
    writes: list[tuple[int, bool]] = []

    def spy(payload, path, *, fsync=False):
        writes.append((len(payload["estimates"]), fsync))
        atomic_pickle(payload, path, fsync=fsync)

    monkeypatch.setattr(sampled, "atomic_pickle", spy)
    path = tmp_path / "cell.pkl.primed"
    expected = stub_source().estimate_words(WORDS, checkpoint=path, owner=OWNER)
    assert writes == [(0, True), (len(WORDS), True)]
    assert _load(path)["estimates"] == expected


def test_scorer_key_names_the_gpu_and_cuda_build(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "GPU X")
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    assert sampled._scorer_key("cuda") == {
        "device": "cuda",
        "name": "GPU X",
        "cuda": "12.9",
    }
    assert sampled._scorer_key("cpu") == {
        "device": "cpu",
        "capability": torch.backends.cpu.get_cpu_capability(),
    }


def test_a_checkpoint_needs_its_owner(tmp_path):
    with pytest.raises(ValueError, match="owner"):
        stub_source().estimate_words(WORDS, checkpoint=tmp_path / "cell.pkl.primed")


def test_a_checkpoint_of_the_previous_contract_is_regenerated_and_a_current_one_resumed(
    tmp_path,
):
    """Contexts drawn under the previous generation contract are not adopted
    even for the same owner and seed; a current checkpoint is, without drawing."""
    path = tmp_path / "cell.pkl.primed"
    calls = []

    def generate():
        calls.append(1)
        return list(C1)

    first = stub_source(generate=generate).estimate_words(
        WORDS, checkpoint=path, owner=OWNER
    )
    assert calls == [1] and _load(path)["format"] == sampled._PRIMING_FORMAT
    stale = {**_load(path), "format": "replica/mmc-priming/2"}
    atomic_pickle(stale, path)
    again = stub_source(generate=generate).estimate_words(
        WORDS, checkpoint=path, owner=OWNER
    )
    assert calls == [1, 1] and again == first
    assert _load(path)["format"] == sampled._PRIMING_FORMAT
    resumed = stub_source(generate=generate).estimate_words(
        WORDS, checkpoint=path, owner=OWNER
    )
    assert calls == [1, 1] and resumed == first


def test_supplied_historical_samples_recompute_to_the_same_values(tmp_path):
    """Explicitly supplied samples are replayed as text: a rejected old
    checkpoint costs a recomputation and changes no value."""
    path = tmp_path / "cell.pkl.primed"
    direct = stub_source(samples=list(C1)).estimate_words(WORDS)
    checkpointed = stub_source(samples=list(C1)).estimate_words(
        WORDS, checkpoint=path, owner=OWNER
    )
    assert checkpointed == direct
    atomic_pickle({**_load(path), "format": "replica/mmc-priming/2"}, path)
    recomputed = stub_source(samples=list(C1)).estimate_words(
        WORDS, checkpoint=path, owner=OWNER
    )
    assert recomputed == direct
    assert _load(path)["format"] == sampled._PRIMING_FORMAT
