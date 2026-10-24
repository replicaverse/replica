"""Caller-defined partitions survive every stage of the pooled estimator."""

from dataclasses import replace

import numpy as np
import pytest

from replica.feature.frequency import AllocationSpec, Stratification
from replica.feature.frequency import pooled as P
from replica.feature.frequency.registry import SOURCES
from tests.feature.frequency._pooled_stub import (
    _MORE_SAMPLES,
    _SAMPLES,
    _STUB_WORDS,
    stub_pair,
    stub_selection,
)


def _policy(n=3):
    labels = tuple(f"length-{i}" for i in range(n))
    return Stratification(
        "test-length/1", labels, 13, lambda text: labels[len(text) % n]
    )


def _pipeline(tmp_path, policy, per_replicate=(128, 128)):
    tok, model = stub_pair()
    samples = _SAMPLES + _MORE_SAMPLES
    first_ids = np.array(sorted({tok.encode(w)[0] for w in _STUB_WORDS}))
    capacity = np.broadcast_to(
        per_replicate, (len(first_ids), len(policy.labels), 2)
    ).copy()
    indices, partials, paths = [], [], []
    for shard, chunk in enumerate((_SAMPLES, _MORE_SAMPLES)):
        index = P.boundary_index(
            chunk, tok, sample_offset=4 * shard, stratification=policy
        )
        path = tmp_path / f"index-{shard}.npz"
        index.save(path)
        index = P.BoundaryIndex.load(path)
        partial = P.sweep_shard(
            chunk,
            index,
            model,
            tok,
            first_ids=first_ids,
            mt_first_ids=first_ids,
            k_store=capacity,
            k_shard=capacity,
            rng_seed=7,
            shard_index=shard,
        )
        path = tmp_path / f"sweep-{shard}.npz"
        partial.save(path)
        indices.append(index)
        partials.append(partial)
        paths.append(path)
    index = P.merge_index(indices, expected_shards=2)
    kwargs = dict(
        expected_shards=2,
        n_pool_samples=len(samples),
        n_boundaries=index.n_boundaries,
        bos_p0=P.bos_first_probs(model, tok, first_ids),
        k_store=capacity,
        first_ids=first_ids,
        mt_first_ids=first_ids,
        pool_fingerprint="test",
        rng_seed=7,
    )
    selection = P.merge_sweep(paths, **kwargs)
    tokens, offsets = P._token_store(samples, tok)
    selection = P.attach_prefixes(selection, tokens, offsets)
    path = tmp_path / "selection.npz"
    selection.save(path)
    selection = P.PoolSelection.load(path)
    return selection, indices, partials, paths, kwargs, tok, model


@pytest.mark.parametrize("n", [1, 3, 5])
def test_caller_partition_reaches_the_exhaustive_estimate_after_disk_round_trips(
    tmp_path, n
):
    policy = _policy(n)
    selection, indices, partials, _, _, tok, model = _pipeline(tmp_path, policy)
    for artifact in (*indices, *partials, selection):
        assert artifact.strata == policy.labels
        assert artifact.stratification_id == policy.identity
    assert selection.colsum.shape[0] == selection.k_store.shape[1] == n
    samples = _SAMPLES + _MORE_SAMPLES
    allocation = AllocationSpec(
        strata=policy.labels,
        k_eval=256,
        k_floor=16,
        k_home=32,
        replicates=2,
        home_weights={w: {policy.labels[-1]: 1.0} for w in _STUB_WORDS},
    )
    exact = P.exhaustive_estimate_batch(_STUB_WORDS, samples, model, tok, 8)
    records = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=256,
        unit_counts=P.unit_counts(_STUB_WORDS, samples),
        take_store=P.take_store_from_shards([samples], tok, selection.first_ids),
    )
    for word in _STUB_WORDS:
        assert records[word].p_hat == pytest.approx(exact[word], rel=1e-6)
        assert records[word].home_strata == (policy.labels[-1],)
    sliced = selection.slice_for([0, 2])
    for col, original in enumerate((0, 2)):
        for group in range(n):
            for replicate in range(2):
                np.testing.assert_array_equal(
                    sliced.sel_sample[sliced.cell(col, group, replicate)],
                    selection.sel_sample[selection.cell(original, group, replicate)],
                )


def test_mixed_partitions_are_refused_before_merging_or_estimation(tmp_path):
    selection, indices, partials, paths, kwargs, tok, model = _pipeline(
        tmp_path, _policy()
    )
    for change in (
        dict(stratification_id="another-version"),
        dict(strata=selection.strata[::-1]),
    ):
        with pytest.raises(ValueError, match="stratification"):
            P.merge_index(
                [indices[0], replace(indices[1], **change)], expected_shards=2
            )
        replace(partials[1], **change).save(paths[1])
        with pytest.raises(ValueError, match="stratification"):
            P.merge_sweep(paths, delete_on_consume=True, **kwargs)
        assert all(path.exists() for path in paths)
    with pytest.raises(ValueError, match="disagree on the strata"):
        P.pps_estimate_batch(
            [],
            selection,
            model,
            tok,
            k_eval=256,
            unit_counts={},
            allocation=AllocationSpec(
                strata=selection.strata[::-1], k_eval=256, replicates=2
            ),
        )


def test_legacy_artifacts_need_an_explicit_partition_and_new_ones_validate_it(tmp_path):
    policy = _policy()
    selection, indices, partials, _, _, _, _ = _pipeline(tmp_path, policy)
    for artifact in (indices[0], partials[0], selection):
        path = tmp_path / "artifact.npz"
        artifact.save(path)
        load = type(artifact).load
        with pytest.raises(ValueError, match="does not match"):
            load(path, stratification=replace(policy, name="another-version"))
        with np.load(path, allow_pickle=False) as blob:
            arrays = {
                k: blob[k]
                for k in blob.files
                if k not in {"strata", "stratification_id"}
            }
        if isinstance(artifact, P.PoolSelection):
            arrays["estimator_version"] = np.array("mmc_pps/2")
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match="legacy archive"):
            load(path)
        restored = load(path, stratification=policy)
        assert restored.strata == policy.labels
        assert restored.stratification_id == policy.identity
        for key, values in arrays.items():
            if key == "estimator_version":
                assert restored.estimator_version == P.SELECTION_VERSION != str(values)
            else:
                np.testing.assert_array_equal(getattr(restored, key), values)
        np.savez(path, **arrays, strata=np.array(policy.labels))
        with pytest.raises(ValueError, match="incomplete"):
            load(path, stratification=policy)


def test_classifier_receives_the_declared_prefix_tail_and_rejects_unknown_labels():
    seen = []

    def classify(text):
        seen.append(text)
        return "all"

    policy = Stratification("tail/1", ("all",), 3, classify)
    tok, _ = stub_pair()
    index = P.boundary_index(["a cat sat"], tok, stratification=policy)
    assert seen == ["a", "cat"]
    seen.clear()
    P.boundary_index(
        ["a" * 50 + " cat"], tok, stratification=replace(policy, window_chars=64)
    )
    assert seen == ["a" * 50]
    assert index.stratum.tolist() == [0, 0]
    assert policy.identity != replace(policy, window_chars=4).identity
    with pytest.raises(ValueError, match="unknown stratum"):
        replace(policy, classify=lambda _: "missing").label(["abc"])


@pytest.mark.parametrize("labels", [(), ("a", "a"), ("",), tuple(map(str, range(257)))])
def test_invalid_partition_vocabularies_are_refused(labels):
    with pytest.raises(ValueError):
        Stratification("test/1", labels, 3, lambda _: "a")


def test_allocation_weights_are_explicit_immutable_and_part_of_the_digest():
    weights = {"word": {"a": 1.0, "b": 0.5}}
    spec = AllocationSpec(strata=("a", "b"), k_eval=12, k_home=4, home_weights=weights)
    digest = spec.sha()
    weights["word"]["a"] = 7.0
    assert spec.sha() == digest
    assert spec.home_strata("word") == ((0, 1.0), (1, 0.5))
    assert spec.home_strata("absent") == ()
    assert replace(spec, home_weights=weights).sha() != digest
    assert replace(spec, strata=("b", "a")).sha() != digest
    for row in ({"missing": 1.0}, {"a": -1.0}, {"a": np.nan}, {"a": np.inf}):
        with pytest.raises(ValueError):
            replace(spec, home_weights={"word": row})
    assert replace(spec, home_weights={"word": {"a": 0.0}}).home_strata("word") == ()


def test_invalid_layouts_and_budgets_fail_before_sampling(tmp_path):
    selection, _, _, _, _, _, _ = _pipeline(tmp_path, _policy())
    with pytest.raises(ValueError, match="column mass shape"):
        replace(selection, colsum=selection.colsum[:2])
    with pytest.raises(ValueError, match="capacity shape"):
        replace(selection, k_store=selection.k_store[:, :2])
    spec = AllocationSpec(strata=selection.strata, k_eval=12)
    for capacity in (
        np.array([1, -1, 2]),
        np.array([1.5, 2.0, 3.0]),
        np.ones(2, dtype=int),
    ):
        with pytest.raises(ValueError, match="capacity"):
            spec.budget("word", np.zeros(3), capacity)
    with pytest.raises(ValueError, match="non-negative"):
        spec.budget("word", np.array([1.0, -1.0, 2.0]))
    with pytest.raises(ValueError, match="explicit allocation"):
        SOURCES["mmc_pool"].build(model_name="stub", selection=selection, k_eval=12)


def test_asymmetric_buffers_fund_each_replicate_without_overspending(tmp_path):
    policy = _policy(1)
    selection, _, _, _, _, tok, _ = _pipeline(tmp_path, policy, per_replicate=(128, 4))
    allocation = AllocationSpec(strata=policy.labels, k_eval=16, replicates=2)
    drawn = P._draw_cells(
        selection, allocation, " the", selection.first_col[tok.encode(" the")[0]], 16
    )
    # Both replicas get three draws, leaving a fourth for the priority threshold.
    assert np.bincount(drawn["replicate"], minlength=2).tolist() == [3, 3]
    with pytest.raises(ValueError, match="fund every replicate"):
        allocation.budget(" the", np.ones(1), capacity=np.array([0]))


def test_partitioned_selections_do_not_masquerade_as_the_fixed_language_layout(
    tmp_path,
):
    selection, _, _, _, _, _, _ = _pipeline(tmp_path, _policy())
    assert selection.estimator_version == "mmc_pps_selection/1"
    path = tmp_path / "selection.npz"
    replace(selection, estimator_version="mmc_pps/2").save(path)
    with pytest.raises(ValueError, match="unsupported selection layout"):
        P.PoolSelection.load(path)


@pytest.mark.parametrize("explicit", [False, True])
def test_the_selection_carries_the_prefix_the_pool_was_drawn_from(tmp_path, explicit):
    """The default is the tokenizer's bos; an explicit one-token prefix is
    persisted and every scorer reads it from the selection."""
    policy = _policy(2)
    tok, _ = stub_pair()
    start_ids = [tok.vocab.index("a")] if explicit else None
    shards = (_SAMPLES, _MORE_SAMPLES)
    selection, tok, model = stub_selection(
        shards, stratification=policy, k_store=128, start_ids=start_ids
    )
    head = start_ids[0] if explicit else tok.bos_token_id
    assert selection.start == (head,)
    assert selection.sel_prefix is not None
    tokens, offsets = P._token_store(_SAMPLES + _MORE_SAMPLES, tok, start_ids)
    assert set(tokens[offsets[:-1]].tolist()) == {head}
    path = tmp_path / "selection.npz"
    selection.save(path)
    loaded = P.PoolSelection.load(path)
    assert loaded.start == (head,)
    with np.load(path) as blob:
        assert blob["start"].tolist() == [head]
    samples = _SAMPLES + _MORE_SAMPLES
    allocation = AllocationSpec(
        strata=policy.labels,
        k_eval=256,
        k_floor=16,
        k_home=32,
        replicates=2,
        home_weights={w: {policy.labels[-1]: 1.0} for w in _STUB_WORDS},
    )
    exact = P.exhaustive_estimate_batch(
        _STUB_WORDS, samples, model, tok, 8, start_ids=start_ids
    )
    records = P.pps_estimate_batch(
        _STUB_WORDS,
        loaded,
        model,
        tok,
        allocation=allocation,
        k_eval=256,
        unit_counts=P.unit_counts(_STUB_WORDS, samples),
        take_store=P.take_store_from_shards(
            [samples], tok, loaded.first_ids, start_ids=start_ids
        ),
    )
    for word in _STUB_WORDS:
        assert records[word].p_hat == pytest.approx(exact[word], rel=1e-6)
    with pytest.raises(ValueError, match="exactly one non-negative integer"):
        stub_selection(shards, stratification=policy, k_store=128, start_ids=[1, 2])


def test_an_archive_without_a_recorded_prefix_loads_as_legacy(tmp_path):
    """Only archives written before the prefix was recorded load with ``start`` unset."""
    policy = _policy(2)
    selection, _, _ = stub_selection(
        (_SAMPLES, _MORE_SAMPLES), stratification=policy, k_store=128
    )
    path = tmp_path / "selection.npz"
    selection.save(path)
    with np.load(path) as blob:
        arrays = {k: blob[k] for k in blob.files if k != "start"}
    np.savez(tmp_path / "legacy.npz", **arrays)
    assert P.PoolSelection.load(tmp_path / "legacy.npz").start is None


@pytest.mark.parametrize(
    "start", [[1.5], [-1], [], [1, 2], [True], ["1"], [[1]]], ids=str
)
def test_an_archive_with_an_invalid_start_is_refused(tmp_path, start):
    policy = _policy(2)
    selection, _, _ = stub_selection(
        (_SAMPLES, _MORE_SAMPLES), stratification=policy, k_store=128
    )
    path = tmp_path / "selection.npz"
    selection.save(path)
    with np.load(path) as blob:
        arrays = {k: blob[k] for k in blob.files if k != "start"}
    np.savez(tmp_path / "bad.npz", start=np.array(start), **arrays)
    with pytest.raises(ValueError, match="one non-negative integer id"):
        P.PoolSelection.load(tmp_path / "bad.npz")


@pytest.mark.parametrize("start", [(1.9,), (True,), ("4",), (-1,), (), (1, 2)], ids=str)
def test_a_selection_validates_its_start_when_built_or_replaced(tmp_path, start):
    """The constructor and ``replace`` see the value before any archive does."""
    policy = _policy(2)
    selection, _, _ = stub_selection(
        (_SAMPLES, _MORE_SAMPLES), stratification=policy, k_store=128
    )
    with pytest.raises(ValueError, match="one non-negative integer id"):
        replace(selection, start=start)
    assert replace(selection, start=None).start is None
    assert replace(selection, start=np.array([3], dtype=np.int32)).start == (3,)
