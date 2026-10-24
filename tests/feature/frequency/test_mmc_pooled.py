"""Pooled marginal Monte Carlo estimation: invariants and pinned values.

Enumeration over a five-token vocabulary supplies the exact expected counts.
Stochastic checks use fixed seeds and recorded values.
"""

import ast
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from replica.feature.frequency import allocation as A
from replica.feature.frequency import pooled as P
from replica.feature.frequency import strata as S
from replica.feature.frequency import units as U
from replica.feature.frequency.sampled import (
    _boundary_sweep,
    _word_boundary_marker,
)
from replica.feature.frequency.strata import Stratification
from tests._require import require
from tests.feature.frequency._pooled_stub import (
    _MORE_SAMPLES,
    _SAMPLES,
    _STUB_WORDS,
    _counts,
    _take_store,
    stub_pair,
    stub_selection,
)

_STRATA = ("short", "long", "empty")
_STRATUM_INDEX = {label: i for i, label in enumerate(_STRATA)}
_STRATIFICATION = Stratification(
    "test-prefix-length/1",
    _STRATA,
    40,
    lambda prefix: "short" if len(prefix.split()) <= 2 else "long",
)
_HOME_WEIGHTS = {w: {"short": 1.0} for w in _STUB_WORDS}


needs_transformers = require(
    lambda: importlib.util.find_spec("transformers") is not None,
    "transformers not available",
)


# --- helpers ---------------------------------------------------------------


def _stub_selection(shards, **kwargs):
    return stub_selection(shards, stratification=_STRATIFICATION, **kwargs)


def _allocation(k_eval=12, replicates=2, **kwargs):
    return A.AllocationSpec(
        k_eval=k_eval,
        k_floor=kwargs.pop("k_floor", 2),
        k_home=kwargs.pop("k_home", 4),
        replicates=replicates,
        strata=_STRATA,
        home_weights=_HOME_WEIGHTS,
        **kwargs,
    )


# --- T1, T2: the regression gate and the collapse identities ---------------


def test_the_estimator_is_unbiased_over_every_string_the_lm_can_emit():
    """The estimator's expectation matches exact enumerated occurrence counts."""
    import itertools

    import torch

    from tests.feature.frequency._pooled_stub import TINY_ROWS, tiny_pair

    tok, model = tiny_pair()
    bset = P.boundary_set(tok)
    emit, cap = (1, 2, 3), 4
    step = [torch.softmax(torch.tensor(row).double(), -1) for row in TINY_ROWS]

    def trajectories():
        for length in range(cap + 1):
            for toks in itertools.product(emit, repeat=length):
                p_traj, last = 1.0, tok.bos_token_id
                for t in toks:
                    p_traj *= float(step[last][t])
                    last = t
                tail = float(step[last][tok.eos_token_id]) if length < cap else 1.0
                yield list(toks), p_traj * tail

    words = [" a", " b", " ac", " bc", " acc"]
    mass = 0.0
    predicted = dict.fromkeys(words, 0.0)
    actual = dict.fromkeys(words, 0.0)
    for toks, weight in trajectories():
        mass += weight
        text = tok.decode(toks)
        got = P.exhaustive_estimate_batch(words, [text], model, tok, 8)
        ids = [tok.bos_token_id, *toks]
        denom = 1 + sum(1 for q in range(len(ids)) if bset.opens_unit(ids, q))
        counts = P.unit_counts(words, [text])
        for word in words:
            predicted[word] += weight * got[word] * denom
            actual[word] += weight * counts[word]
    assert mass == pytest.approx(1.0, abs=1e-9)
    for word in words:
        assert predicted[word] == pytest.approx(actual[word], rel=1e-6), word


def test_a_surface_spanning_a_space_is_counted_where_its_units_are_adjacent():
    """Corpus surfaces like ``' 19. Jahrhunderts'`` occupy two units.

    The estimator scores them as one longer token sequence closed by the same
    boundary mass, so the tally has to agree: the units must be consecutive,
    and the same units scattered through a sample are not an occurrence.
    """
    samples = [
        "the dog ran in the 19. Jahrhunderts and the dog",
        "19. Jahrhunderts began then 19. Jahrhunderts",
        "19. and Jahrhunderts apart",
        "a 5. Juni date",
    ]
    words = [" the", " dog", " 19. Jahrhunderts", " 5. Juni", " Jahrhunderts", " gone"]
    got = P.unit_counts(words, samples)
    assert got == {
        " the": 3,
        " dog": 2,
        " 19. Jahrhunderts": 3,
        " 5. Juni": 1,
        " Jahrhunderts": 4,
        " gone": 0,
    }
    assert list(got) == words


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("x 19. Jahrhunderts y", 1),
        ("19. Jahrhunderts y", 1),
        ("x 19. Jahrhundertsxx y", 0),
        ("x 19.\nJahrhunderts y", 0),
        ("x 19.  Jahrhunderts y", 0),
        ("x 19. and Jahrhunderts y", 0),
    ],
)
def test_a_spanning_surface_needs_the_separator_it_was_tokenized_with(sample, expected):
    """The separator is part of the sequence the estimator scores.

    Units held apart by a newline or a double space tokenize differently, so
    they are not an occurrence of the surface however the text splits.
    """
    assert P.unit_counts([" 19. Jahrhunderts"], [sample]) == {
        " 19. Jahrhunderts": expected
    }


def test_a_surface_with_no_unit_is_refused():
    with pytest.raises(ValueError, match="no whitespace-delimited unit"):
        P.unit_counts(["   "], ["the dog"])


def test_the_frame_holds_one_boundary_per_unit_plus_the_terminal_slot():
    """A unit opens after a marker token or after whitespace of any kind.

    Detecting only the marker misses every unit that follows a line break,
    which would leave the frame counting different things from the exact
    occurrence counts the remainder is paired against.
    """
    tok, _ = stub_pair()
    index = P.boundary_index(
        _SAMPLES,
        tok,
        stratification=_STRATIFICATION,
        collapse_bos=False,
        drop_terminal=False,
    )
    units = sum(len(text.split()) for text in _SAMPLES)
    assert index.n_boundaries == units + len(_SAMPLES)

    # A newline token carries no marker but decodes to whitespace, so a unit
    # does open after it.
    marked = P.boundary_set(tok)
    assert tok.encode(" a")[0] in marked.open_ids
    assert tok.encode("a")[0] not in marked.open_ids


# --- T3a, T3b: priority sampling, model-free -------------------------------

#: Realized means of the priority estimator over 2,000 pinned streams.
_PPS_MEAN = {
    1: 8.381937612435777,
    10: 8.392132540388609,
    200: 8.502348797654598,
    2000: 8.492540098749188,
}
#: Realized variance and the mean of the DLT variance estimator, same streams.
_PPS_VARIANCE = {
    200: (0.3949237038669447, 0.4270521268677323),
    2000: (0.0022512466103225917, 0.002208399409447017),
}
_PPS_EXACT = 8.492852046543213


def _pps_population():
    rng = np.random.default_rng(0)
    size = rng.lognormal(mean=-8.0, sigma=2.5, size=10_000)
    value = rng.beta(0.5, 3.0, size=10_000)
    return size, value


def _pps_draws(size, value, k, streams=2000):
    n = size.size
    index = np.arange(n, dtype=np.int64)
    zero = np.zeros(n, dtype=np.int64)
    totals = np.empty(streams)
    variances = np.empty(streams)
    for s in range(streams):
        u = np.asarray(P.boundary_uniform(1000 + s, index, zero), dtype=np.float64)
        q = size / u
        order = P._priority_order(q, index, zero)
        take = order[:k]
        tau = float(q[order[k]]) if k < n else 0.0
        totals[s] = float((np.maximum(size[take], tau) * value[take]).sum())
        variances[s] = P.dlt_variance(size[take], tau, value[take])
    return totals, variances


@pytest.mark.statistical
def test_priority_sampling_reproduces_its_recorded_means():
    """Pinned streams, recorded realized means, and no systematic drift."""
    size, value = _pps_population()
    assert float((size * value).sum()) == pytest.approx(_PPS_EXACT, rel=1e-12)
    for k, recorded in _PPS_MEAN.items():
        totals, _ = _pps_draws(size, value, k)
        mean = float(totals.mean())
        assert mean == pytest.approx(recorded, rel=1e-12)
        se = float(totals.std(ddof=1)) / math.sqrt(totals.size)
        # Neither top-K truncation (low) nor threshold inflation (high).
        assert abs(mean - _PPS_EXACT) < 3.0 * se


@pytest.mark.statistical
def test_the_dlt_variance_estimator_tracks_the_realized_variance():
    """The estimator's mean has to land on the variance it claims to estimate."""
    size, value = _pps_population()
    for k, (recorded_var, recorded_vhat) in _PPS_VARIANCE.items():
        totals, variances = _pps_draws(size, value, k)
        realized = float(totals.var(ddof=1))
        mean_vhat = float(variances.mean())
        assert realized == pytest.approx(recorded_var, rel=1e-12)
        assert mean_vhat == pytest.approx(recorded_vhat, rel=1e-12)
        # The band is tight enough to separate the DLT product from the
        # squared-inflation charge, which falls below it.
        assert 0.9 <= mean_vhat / realized <= 1.2, (k, mean_vhat / realized)


def test_the_dlt_charge_is_the_one_that_makes_the_estimator_unbiased():
    """``E[V_hat] = w*(tau - w)`` for a sub-threshold item, exactly."""
    tau, w, c = 4.0, 1.0, 0.5
    charge = P.dlt_variance(np.array([w]), tau, np.array([c]))
    selection_probability = w / tau
    assert charge * selection_probability == pytest.approx(
        w * (tau - w) * c**2, rel=1e-15
    )
    # A certainty item carries no variance and is charged nothing.
    assert P.dlt_variance(np.array([9.0]), tau, np.array([c])) == 0.0


# --- T4: nesting -----------------------------------------------------------


def test_a_smaller_draw_nests_inside_a_stored_one():
    """``_cell_take`` on a stored cell equals a direct size-k priority sample."""
    selection, _, _ = _stub_selection([_SAMPLES, _MORE_SAMPLES], k_store=64)
    checked = 0
    for mt in range(selection.mt_first_ids.size):
        for stratum in range(len(_STRATA)):
            for replicate in range(selection.replicates):
                sl = selection.cell(mt, stratum, replicate)
                have = sl.stop - sl.start
                if have < 4:
                    continue
                sizes = selection.sel_p0[sl].astype(np.float64) * selection.sel_mult[
                    sl
                ].astype(np.float64)
                u = np.asarray(
                    P.boundary_uniform(
                        selection.rng_seed,
                        selection.sel_sample[sl],
                        selection.sel_pos[sl],
                    ),
                    dtype=np.float64,
                )
                q = sizes / u
                order = P._priority_order(
                    q, selection.sel_sample[sl], selection.sel_pos[sl]
                )
                for want in range(1, have):
                    take, tau = P._cell_take(selection, sl, want)
                    assert np.array_equal(take, order[:want])
                    assert tau == float(q[order[want]])
                checked += 1
    assert checked > 0, "the stub selection stored no cell deep enough to nest"


# --- T5, T6: shard and chunk invariance ------------------------------------


def test_the_selection_does_not_depend_on_the_shard_split():
    """Splitting the pool into shards moves neither the draw nor the mass."""
    pool = _SAMPLES + _MORE_SAMPLES
    layouts = [
        [pool],
        [pool[:4], pool[4:]],
        [pool[:2], pool[2:5], pool[5:]],
        [pool[:1], pool[1:3], pool[3:4], pool[4:6], pool[6:]],
    ]
    base = None
    for shards in layouts:
        selection, _, _ = _stub_selection(shards)
        picked = (
            selection.sel_sample.tolist(),
            selection.sel_pos.tolist(),
            selection.cell_offsets.tolist(),
        )
        if base is None:
            base = (picked, selection.colsum)
            continue
        assert picked == base[0]
        np.testing.assert_allclose(selection.colsum, base[1], rtol=1e-15, atol=0.0)


def test_a_shard_buffer_that_truncates_above_the_threshold_is_refused():
    """A shard cap that could have dropped a qualifying boundary stops the merge."""
    pool = _SAMPLES + _MORE_SAMPLES
    shards = [pool[:2], pool[2:4], pool[4:6], pool[6:]]
    tok, model = stub_pair()
    token_ids = [tok.encode(w) for w in _STUB_WORDS]
    first_ids = np.array(sorted({t[0] for t in token_ids}), dtype=np.int64)
    mt_ids = np.array(sorted({t[0] for t in token_ids if len(t) > 1}), dtype=np.int64)
    store = np.full((mt_ids.size, len(_STRATA), 2), 2, dtype=np.int64)
    with pytest.raises(ValueError, match="above the merged threshold"):
        P.build_pool_selection(
            shards,
            model,
            tok,
            stratification=_STRATIFICATION,
            first_ids=first_ids,
            mt_first_ids=mt_ids,
            k_store=store,
            k_shard=np.ones_like(store),
            rng_seed=7,
        )


def test_batching_and_chunking_do_not_move_a_pooled_estimate():
    pool = [_SAMPLES, _MORE_SAMPLES]
    reference = None
    for batch_size in (1, 2, 3, 8):
        for reduce_chunk in (5, 8192):
            selection, tok, model = _stub_selection(
                pool, batch_size=batch_size, reduce_chunk=reduce_chunk
            )
            allocation = _allocation()
            got = P.pps_estimate_batch(
                _STUB_WORDS,
                selection,
                model,
                tok,
                allocation=allocation,
                k_eval=allocation.k_eval,
                batch_size=batch_size,
                unit_counts=_counts(),
                take_store=_take_store(),
            )
            values = {w: r.p_hat for w, r in got.items()}
            picked = selection.sel_sample.tolist()
            if reference is None:
                reference = (values, picked)
                continue
            assert picked == reference[1]
            for word in _STUB_WORDS:
                assert values[word] == pytest.approx(reference[0][word], rel=1e-13), (
                    word
                )


# --- T7, T8: the index agrees with the shipped sweep -----------------------


@pytest.mark.parametrize("marker", ["Ġ", "▁"])
def test_the_index_counts_the_same_boundaries_as_the_shipped_sweep(marker):
    import torch

    tok, model = stub_pair(marker)
    assert _word_boundary_marker(tok) == marker
    index = P.boundary_index(
        _SAMPLES, tok, stratification=_STRATIFICATION, collapse_bos=False
    )
    prefixes, _, _ = _boundary_sweep(_SAMPLES, model, tok, torch.tensor([1]), marker, 2)
    assert index.n_boundaries == len(prefixes)
    # Termination is forced at the end of a sample, so its terminal boundary
    # counts as a slot but carries no term.
    assert index.frame_size() == len(prefixes) - len(_SAMPLES)


def test_every_zero_position_prefix_is_the_bos_token():
    tok, _ = stub_pair()
    index = P.boundary_index(
        _SAMPLES, tok, stratification=_STRATIFICATION, collapse_bos=False
    )
    zeros = index.position == 0
    assert int(zeros.sum()) == len(_SAMPLES)
    assert set(index.prefix_len[zeros].tolist()) == {1}
    collapsed = P.boundary_index(
        _SAMPLES, tok, stratification=_STRATIFICATION, collapse_bos=True
    )
    assert collapsed.frame_size() == index.frame_size() - len(_SAMPLES)
    assert collapsed.n_boundaries == index.n_boundaries


# --- T9, T10: exactness and the selectivity ladder -------------------------


def test_a_budget_covering_the_frame_reproduces_the_exhaustive_sum():
    """At full coverage tau is zero and the stratified sum is the exact one."""
    pool = [_SAMPLES, _MORE_SAMPLES]
    selection, tok, model = _stub_selection(pool, k_store=64)
    allocation = _allocation(k_eval=400, k_floor=20, k_home=40)
    got = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=400,
        batch_size=3,
        unit_counts=_counts(),
        take_store=_take_store(),
    )
    exact = P.exhaustive_estimate_batch(
        _STUB_WORDS,
        _SAMPLES + _MORE_SAMPLES,
        model,
        tok,
        3,
    )
    for word in _STUB_WORDS:
        # Buffered first-token mass is float32 and the numerator is a
        # difference, so the agreement floor is float32 precision.
        assert got[word].p_hat == pytest.approx(exact[word], rel=1e-6), word


@pytest.mark.parametrize("cap_kind", ["full", "biting"])
def test_the_residual_remainder_agrees_with_the_exhaustive_sum_at_full_coverage(
    cap_kind,
):
    """The nonnegative numerator, priced against the cap, through both paths.

    Unbiasedness of this numerator is settled by enumeration in
    :mod:`tests.feature.frequency.test_mmc_room_aware`; what is checked here is that the
    priority path computes the same thing as the direct sum, cap included.

    The full-length cap leaves the zeroing branch of the closure idle, so a cap
    that truncates runs as well; the pool could not have been drawn under it,
    and both paths must still price it identically.
    """
    from replica.feature.frequency.sampled import _encode_samples

    pool = [_SAMPLES, _MORE_SAMPLES]
    samples = _SAMPLES + _MORE_SAMPLES
    selection, tok, model = _stub_selection(pool, k_store=64)
    allocation = _allocation(k_eval=400, k_floor=20, k_home=40)
    encoded = _encode_samples(tok, samples)
    longest = max(len(ids) - 1 for ids in encoded)
    cap = longest if cap_kind == "full" else 8
    assert cap < longest or cap_kind == "full"
    split = P.unit_counts_split(_STUB_WORDS, encoded, tok)
    got = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=400,
        batch_size=3,
        unit_counts=_counts(),
        take_store=None,
        cap=cap,
        splits={w: split[w] for w in _STUB_WORDS},
    )
    exact = P.exhaustive_estimate_batch(
        _STUB_WORDS,
        samples,
        model,
        tok,
        3,
        cap=cap,
        numerator="expectation",
        encoded=encoded,
    )
    for word in _STUB_WORDS:
        # Buffered first-token mass is float32, as in the sibling test above.
        assert got[word].p_hat == pytest.approx(exact[word], rel=1e-6), word
        assert got[word].p_hat > 0.0, word


def test_the_residual_remainder_is_taken_whole_not_reconstructed():
    """The decoder drops the space before punctuation, so a text count can sit
    below the id-based canonical count. Subtracting one from the other would
    turn negative and refuse the word; the residual count is passed whole."""
    from replica.feature.frequency.sampled import _encode_samples

    pool = [_SAMPLES, _MORE_SAMPLES]
    samples = _SAMPLES + _MORE_SAMPLES
    selection, tok, model = _stub_selection(pool, k_store=64)
    encoded = _encode_samples(tok, samples)
    split = P.unit_counts_split(_STUB_WORDS, encoded, tok)
    # A text count below every canonical count, as the punctuation cases are.
    starved = {w: 0 for w in _STUB_WORDS}
    got = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=_allocation(),
        k_eval=12,
        unit_counts=starved,
        take_store=None,
        cap=max(len(ids) - 1 for ids in encoded),
        splits={w: split[w] for w in _STUB_WORDS},
    )
    for word in _STUB_WORDS:
        assert got[word].p_hat > 0.0, word
        assert not got[word].nonpositive, word
        assert got[word].unit_count == sum(split[word]), word


def test_the_normalizer_can_be_taken_off_the_same_segmentation_as_the_counts():
    """The selection charges every sample a boundary it has no unit for, so its
    slot count is not the pool's word count. Overriding it rescales exactly."""
    from replica.feature.frequency.sampled import _encode_samples

    pool = [_SAMPLES, _MORE_SAMPLES]
    samples = _SAMPLES + _MORE_SAMPLES
    selection, tok, model = _stub_selection(pool, k_store=64)
    encoded = _encode_samples(tok, samples)
    units = P.unit_total(encoded, tok)
    assert units == selection.denom - 2 * len(samples), (units, selection.denom)
    split = P.unit_counts_split(_STUB_WORDS, encoded, tok)
    kw = dict(
        allocation=_allocation(k_eval=400, k_floor=20, k_home=40),
        k_eval=400,
        batch_size=3,
        unit_counts=_counts(),
        take_store=None,
        cap=max(len(ids) - 1 for ids in encoded),
        splits={w: split[w] for w in _STUB_WORDS},
    )
    base = P.pps_estimate_batch(_STUB_WORDS, selection, model, tok, **kw)
    over = P.pps_estimate_batch(_STUB_WORDS, selection, model, tok, denom=units, **kw)
    scale = selection.denom / units
    for w in _STUB_WORDS:
        assert over[w].p_hat == pytest.approx(base[w].p_hat * scale, rel=1e-12), w


def test_a_priced_horizon_without_splits_is_refused():
    """S_hat would be priced against the cap while the take estimate it is
    subtracted from ignores it, which breaks the cancellation."""
    pool = [_SAMPLES, _MORE_SAMPLES]
    selection, tok, model = _stub_selection(pool, k_store=64)
    with pytest.raises(ValueError, match="breaks the cancellation"):
        P.pps_estimate_batch(
            _STUB_WORDS,
            selection,
            model,
            tok,
            allocation=_allocation(),
            k_eval=12,
            unit_counts=_counts(),
            take_store=_take_store(),
            cap=32,
        )


def test_splits_and_a_take_store_together_are_refused():
    pool = [_SAMPLES, _MORE_SAMPLES]
    selection, tok, model = _stub_selection(pool, k_store=64)
    with pytest.raises(ValueError, match="never read"):
        P.pps_estimate_batch(
            _STUB_WORDS,
            selection,
            model,
            tok,
            allocation=_allocation(),
            k_eval=12,
            unit_counts=_counts(),
            take_store=_take_store(),
            cap=32,
            splits=dict.fromkeys(_STUB_WORDS, (0, 0)),
        )


def test_the_residual_remainder_refuses_an_unpriced_horizon():
    pool = [_SAMPLES, _MORE_SAMPLES]
    selection, tok, model = _stub_selection(pool, k_store=64)
    with pytest.raises(ValueError, match="needs cap"):
        P.pps_estimate_batch(
            _STUB_WORDS,
            selection,
            model,
            tok,
            allocation=_allocation(),
            k_eval=12,
            unit_counts=_counts(),
            take_store=_take_store(),
            splits=dict.fromkeys(_STUB_WORDS, (0, 0)),
        )


def test_the_pooled_source_refuses_a_sample_list():
    with pytest.raises(ValueError, match="PoolSelection"):
        P.PooledUnigramSource(
            selection=object(), allocation=_allocation(), k_eval=8, samples=["a"]
        )


def test_the_builder_refuses_a_missing_selection_or_budget():
    from replica.feature.frequency.registry import SOURCES

    with pytest.raises(ValueError, match="PoolSelection"):
        SOURCES["mmc_pool"].build(model_name="gpt2", k_eval=8)
    with pytest.raises(ValueError, match="k_eval"):
        SOURCES["mmc_pool"].build(model_name="gpt2", selection=object())


def test_a_budget_deeper_than_the_stored_buffer_is_capped_at_what_cells_hold():
    """The split is bounded by the stored buffers.

    The leftover above the per-stratum caps is spread by column mass, so a
    stratum that carries most of a pool's mass can be handed far more draws
    than its cells hold. Those draws move to strata with room; where no
    stratum has room the budget falls short of ``k_eval``, which widens the
    interval but keeps the estimate unbiased.
    """
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES], k_store=2)
    allocation = _allocation(k_eval=400, k_floor=20, k_home=40)
    got = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=400,
        unit_counts=_counts(),
        take_store=_take_store(),
    )
    assert set(got) == set(_STUB_WORDS)
    # no cell was asked for more than it stores
    for word in _STUB_WORDS:
        col = selection.first_col[tok.encode(word)[0]]
        mt = selection.mt_col.get(int(selection.first_ids[col]))
        if mt is None:
            continue
        store = np.asarray(selection.k_store[mt], dtype=np.int64)
        capacity = allocation.replicates * np.maximum(store[:, 0] - 1, 0)
        k = allocation.budget(word, selection.colsum[:, col], capacity)
        assert int(k.sum()) <= allocation.k_eval
        assert bool((k <= capacity).all()), word
        assert bool((allocation.replicate_budget(k) < store[:, None]).all()), word


def test_the_estimator_refuses_a_mismatched_budget_or_pool():
    selection, tok, model = _stub_selection([_SAMPLES])
    allocation = _allocation(k_eval=12)
    with pytest.raises(ValueError, match="contradicts"):
        P.pps_estimate_batch(
            _STUB_WORDS,
            selection,
            model,
            tok,
            allocation=allocation,
            k_eval=13,
            unit_counts=_counts(),
            take_store=_take_store(),
        )
    with pytest.raises(ValueError, match="fingerprint"):
        P.pps_estimate_batch(
            _STUB_WORDS,
            selection,
            model,
            tok,
            allocation=allocation,
            k_eval=12,
            pool_fingerprint="other-pool",
            unit_counts=_counts(),
            take_store=_take_store(),
        )


def test_the_index_refuses_to_fold_duplicate_prefixes():
    """Folding hides which boundaries the word's first token was taken at."""
    tok, _ = stub_pair()
    parts = [
        P.boundary_index(_SAMPLES, tok, stratification=_STRATIFICATION, sample_offset=0)
    ]
    with pytest.raises(ValueError, match="dup_collapse_max_len"):
        P.merge_index(parts, expected_shards=1, dup_collapse_max_len=4)


def test_the_merges_refuse_a_missing_shard_or_a_wrong_boundary_count():
    pool = [_SAMPLES, _MORE_SAMPLES]
    tok, _ = stub_pair()
    parts = [
        P.boundary_index(
            chunk, tok, stratification=_STRATIFICATION, sample_offset=i * 4
        )
        for i, chunk in enumerate(pool)
    ]
    with pytest.raises(ValueError, match="index parts"):
        P.merge_index(parts, expected_shards=3)
    with pytest.raises(ValueError, match="tile the pool"):
        P.merge_index([parts[1], parts[1]], expected_shards=2)
    selection, _, _ = _stub_selection(pool)
    args = dict(
        bos_p0=selection.bos_p0,
        k_store=selection.k_store,
        first_ids=selection.first_ids,
        mt_first_ids=selection.mt_first_ids,
        pool_fingerprint="p",
        rng_seed=7,
    )
    with pytest.raises(ValueError, match="sweep partials"):
        P.merge_sweep([], expected_shards=2, n_pool_samples=8, n_boundaries=1, **args)


def test_an_allocation_that_misses_its_budget_is_refused():
    class _Wrong(A.AllocationSpec):
        def budget(self, word, colsum_col, capacity=None):
            out = super().budget(word, colsum_col, capacity)
            out[0] += 1
            return out

    selection, tok, model = _stub_selection([_SAMPLES])
    with pytest.raises(ValueError, match="sums to"):
        P.pps_estimate_batch(
            [" the"],
            selection,
            model,
            tok,
            allocation=_Wrong(
                strata=_STRATA,
                k_eval=12,
                k_floor=2,
                k_home=4,
                replicates=2,
                home_weights=_HOME_WEIGHTS,
            ),
            k_eval=12,
            unit_counts=_counts(),
            take_store=_take_store(),
        )


# --- T20: coverage and the underflow fallback ------------------------------


def test_every_word_gets_a_strictly_positive_estimate():
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    allocation = _allocation(k_eval=8, k_floor=2, k_home=2)
    got = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=8,
        unit_counts=_counts(),
        take_store=_take_store(),
    )
    for word, record in got.items():
        assert record.p_hat > 0.0, word
        assert math.isfinite(record.bits), word


def test_an_underflowing_word_falls_back_to_the_log_domain(monkeypatch):
    """A joint that underflows float64 still yields a positive estimate."""
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    real = P._continuations

    def _tiny(prefixes, toks, *args, **kwargs):
        joint, log_joint, beta, log_beta = real(prefixes, toks, *args, **kwargs)
        return np.zeros_like(joint), log_joint - 800.0, beta, log_beta

    def _no_remainder(*args, **kwargs):
        return 0.0, 0.0

    monkeypatch.setattr(P, "_take_h", _no_remainder)
    monkeypatch.setattr(P, "_continuations", _tiny)
    allocation = _allocation()
    got = P.pps_estimate_batch(
        [" the"],
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=12,
        unit_counts={" the": 0},
        take_store=_take_store(),
    )
    record = got[" the"]
    assert record.underflow_fallback
    assert record.p_hat > 0.0
    assert record.bits > 100.0


# --- T21: numerics ---------------------------------------------------------


def test_the_exhaustive_path_cannot_reach_the_log_domain_fallback():
    """The regression path stays the literal expression, with no rescue."""
    names = {}
    for module in (P, U):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                assert node.name not in names
                names[node.name] = node
    assert "exhaustive_estimate_batch" in names
    seen, reached = set(), set()
    stack = ["exhaustive_estimate_batch"]
    while stack:
        current = stack.pop()
        if current in seen or current not in names:
            continue
        seen.add(current)
        for node in ast.walk(names[current]):
            if isinstance(node, ast.Name):
                reached.add(node.id)
                if isinstance(node.ctx, ast.Load):
                    stack.append(node.id)
    assert {"boundary_set", "unit_counts", "unit_counts_split"} <= seen
    assert "_logsumexp" not in reached
    assert "_build_record" not in reached


@pytest.mark.parametrize("module", [P, A, S, U])
def test_pooled_never_reduces_with_the_builtin_sum(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sum"
    ]
    assert calls == []


def test_weights_and_reductions_are_float64():
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    assert selection.colsum.dtype == np.float64
    assert selection.bos_p0.dtype == np.float64
    assert selection.stratum_boundary_counts.dtype == np.int64
    allocation = _allocation()
    col = selection.first_col[tok.encode(" the")[0]]
    drawn = P._draw_cells(selection, allocation, " the", col, allocation.k_eval)
    assert drawn["weights"].dtype == np.float64
    assert drawn["sizes"].dtype == np.float64
    assert drawn["rows"].dtype == np.int64
    joint, log_joint, beta, log_beta = P._continuations(
        [[0, 1], [0, 2]], tok.encode(" the"), model, tok, 2, True
    )
    for arr in (joint, log_joint, beta, log_beta):
        assert arr.dtype == np.float64
    assert isinstance(selection.denom, int)


# --- T22: zero-size boundaries ---------------------------------------------


def test_a_cell_shallower_than_its_budget_becomes_an_exact_sum():
    """With every eligible item taken, tau is zero and no mass is injected.

    Were tau positive there, ``max(size, tau)`` would inflate each weight and
    the cell would report more mass than the pool holds.
    """
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    assert bool(
        (
            selection.sel_p0.astype(np.float64) * selection.sel_mult.astype(np.float64)
            > 0.0
        ).all()
    )
    cell = None
    for stratum in range(len(_STRATA)):
        candidate = selection.cell(0, stratum, 0)
        if candidate.stop > candidate.start:
            cell = candidate
            break
    assert cell is not None
    have = cell.stop - cell.start
    take, tau = P._cell_take(selection, cell, have + 5)
    assert tau == 0.0
    assert take.size == have
    sizes = selection.sel_p0[cell].astype(np.float64) * selection.sel_mult[cell].astype(
        np.float64
    )
    assert float(np.maximum(sizes, tau).sum()) == float(sizes.sum())
    _, tau_small = P._cell_take(selection, cell, have - 1)
    assert tau_small > 0.0


# --- T23: the counter-based stream -----------------------------------------

#: splitmix64 finalizer outputs for 0..7.
_SPLITMIX_REFERENCE = (
    0,
    6238072747940578789,
    15839785061582574730,
    2185194620014831856,
    13232826040865663252,
    13168350753275463132,
    15093541023163888492,
    1346066267577507604,
)


def test_splitmix64_reproduces_its_reference_vectors():
    got = P.splitmix64(np.arange(8, dtype=np.uint64))
    assert [int(v) for v in got] == list(_SPLITMIX_REFERENCE)
    assert int(P.splitmix64(3)) == _SPLITMIX_REFERENCE[3]


def test_boundary_uniform_is_vectorized_and_bounded():
    rng = np.random.default_rng(3)
    samples = rng.integers(0, 10_000, size=10_000).astype(np.int64)
    positions = rng.integers(0, 60, size=10_000).astype(np.int64)
    batched = np.asarray(P.boundary_uniform(99, samples, positions))
    one_by_one = np.array(
        [
            P.boundary_uniform(99, int(j), int(p))
            for j, p in zip(samples[:200], positions[:200], strict=True)
        ]
    )
    assert np.array_equal(batched[:200], one_by_one)
    assert bool((batched > 0.0).all()) and bool((batched <= 1.0).all())


# --- T24: allocation -------------------------------------------------------


def test_the_budget_sums_to_k_eval_over_pinned_columns():
    allocation = A.AllocationSpec(
        strata=_STRATA,
        k_eval=4096,
        k_floor=64,
        k_home=2048,
        replicates=8,
        home_weights={"w": {"short": 1.0}, "x": {"long": 1.0}, "y": {}},
    )
    rng = np.random.default_rng(11)
    for trial in range(200):
        colsum = rng.random(len(_STRATA)) ** 6
        colsum[rng.random(len(_STRATA)) < 0.3] = 0.0
        word = ("w", "x", "y")[trial % 3]
        budget = allocation.budget(word, colsum)
        if not bool((colsum > 0).any()):
            assert int(budget.sum()) == 0
            continue
        assert int(budget.sum()) == 4096
        assert bool((budget[colsum == 0.0] == 0).all())
        replicated = allocation.replicate_budget(budget)
        assert np.array_equal(replicated.sum(axis=1), budget)


def test_explicit_home_weights_and_floor_order_the_budget():
    allocation = A.AllocationSpec(
        strata=_STRATA,
        k_eval=4096,
        k_floor=64,
        k_home=2048,
        home_weights={"w": {"short": 1.0, "long": 0.5}},
    )
    colsum = np.ones(len(_STRATA))
    budget = allocation.budget("w", colsum)
    home = _STRATUM_INDEX["short"]
    backstop = _STRATUM_INDEX["long"]
    assert budget[home] > budget[backstop] > budget[_STRATUM_INDEX["empty"]]
    assert int(budget.sum()) == 4096
    assert bool((budget >= 1).all())


def test_the_remainder_cap_is_respected():
    allocation = A.AllocationSpec(
        strata=_STRATA,
        k_eval=4096,
        k_floor=64,
        k_home=2048,
        home_weights={"w": {"long": 1.0}},
    )
    colsum = np.zeros(len(_STRATA))
    colsum[_STRATUM_INDEX["short"]] = 1.0
    colsum[_STRATUM_INDEX["long"]] = 1e-12
    budget = allocation.budget("w", colsum)
    assert int(budget.sum()) == 4096
    assert int(budget[_STRATUM_INDEX["empty"]]) == 0


def test_the_largest_remainder_tie_break_favours_the_low_index():
    target = np.zeros(len(_STRATA)) + 0.5
    eligible = np.ones(len(_STRATA), dtype=bool)
    out = A._largest_remainder(target, len(_STRATA) // 2, eligible)
    assert int(out.sum()) == len(_STRATA) // 2
    assert bool((out[: len(_STRATA) // 2] == 1).all())


# --- T25: certificate arithmetic -------------------------------------------


def test_the_design_and_dlt_standard_errors_match_their_definitions():
    totals = np.array([1.0, 2.0, 3.0, 4.0])
    assert P.design_se(totals) == pytest.approx(
        math.sqrt(4) * float(totals.std(ddof=1)), rel=1e-12
    )
    assert P.design_se(np.array([7.0])) == 0.0
    size = np.array([0.25, 4.0])
    value = np.array([0.5, 0.5])
    assert P.dlt_variance(size, 1.0, value) == pytest.approx(
        (1.0 - 0.25) * 1.0 * 0.25, rel=1e-12
    )


def test_certify_reads_every_threshold():
    bars = P.SupportThresholds()
    good = {
        "p_hat": 1.0,
        "cv": 0.05,
        "home_share": 0.95,
        "bos_share": 0.1,
        "top1_share": 0.1,
        "residual_bound": 0.01,
        "n_nonzero_contrib": 100,
    }
    assert P.certify(good, bars)
    for key, bad in (
        ("cv", 0.2),
        ("home_share", 0.5),
        ("bos_share", 0.5),
        ("top1_share", 0.9),
        ("residual_bound", 0.5),
        ("n_nonzero_contrib", 3),
    ):
        assert not P.certify({**good, key: bad}, bars)
    assert not P.certify({**good, "p_hat": 0.0}, bars)


def test_a_word_record_round_trips_through_json():
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    allocation = _allocation()
    got = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=12,
        emit_per_stratum=True,
        unit_counts=_counts(),
        take_store=_take_store(),
    )
    for record in got.values():
        again = P.WordRecord.from_json(json.loads(json.dumps(record.to_json())))
        assert again == record


def test_the_certificate_shares_add_up():
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    allocation = _allocation()
    got = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=12,
        unit_counts=_counts(),
        take_store=_take_store(),
    )
    for word, record in got.items():
        assert 0.0 <= record.bos_share <= 1.0, word
        assert 0.0 <= record.home_share <= 1.0, word
        assert record.p_hat_no_bos <= record.p_hat
        if record.nonpositive:
            # The floor is a placeholder, so ``bits`` is inf.
            assert record.bits == math.inf
        else:
            assert record.bits == pytest.approx(-math.log2(record.p_hat), rel=1e-12)


# --- selection serialization ------------------------------------------------


def test_a_selection_round_trips_through_disk(tmp_path):
    selection, _, _ = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    path = tmp_path / "selection.npz"
    selection.save(path)
    again = P.PoolSelection.load(path)
    assert again.denom == selection.denom
    assert again.pool_fingerprint == selection.pool_fingerprint
    np.testing.assert_array_equal(again.sel_sample, selection.sel_sample)
    np.testing.assert_allclose(again.colsum, selection.colsum, rtol=0, atol=0)
    assert again.prefix(0) == selection.prefix(0)


def test_slicing_a_selection_keeps_its_cells():
    selection, _, _ = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    sub = selection.slice_for([0])
    assert sub.mt_first_ids.tolist() == [int(selection.mt_first_ids[0])]
    whole = selection.cell(0, _STRATUM_INDEX["short"], 0)
    part = sub.cell(0, _STRATUM_INDEX["short"], 0)
    np.testing.assert_array_equal(selection.sel_sample[whole], sub.sel_sample[part])


# --- T26: the refusals that keep the design honest -------------------------


def test_the_allocation_never_leaves_a_replicate_cell_unfunded():
    """A funded stratum reaches every replicate, or the budget is refused."""
    weights = {"w": {"short": 1.0}}
    colsum = np.zeros(len(_STRATA))
    for name in _STRATA:
        colsum[_STRATUM_INDEX[name]] = 1e-3
    colsum[_STRATUM_INDEX["short"]] = 1.0
    for k_eval in (40, 64, 128, 512, 4096):
        allocation = A.AllocationSpec(
            strata=_STRATA,
            k_eval=k_eval,
            k_floor=1,
            k_home=k_eval // 2,
            replicates=8,
            home_weights=weights,
        )
        k = allocation.budget("w", colsum)
        assert int(k.sum()) == k_eval
        funded = k[k > 0]
        assert bool((funded >= allocation.replicates).all()), (k_eval, k.tolist())
        assert bool((allocation.replicate_budget(k)[k > 0] > 0).all())


def test_a_budget_too_small_for_the_replicates_is_refused():
    weights = {"w": {}}
    colsum = np.zeros(len(_STRATA))
    for name in _STRATA:
        colsum[_STRATUM_INDEX[name]] = 1.0
    allocation = A.AllocationSpec(
        strata=_STRATA,
        k_eval=16,
        k_floor=1,
        k_home=1,
        replicates=8,
        home_weights=weights,
    )
    with pytest.raises(ValueError, match="cannot fund"):
        allocation.budget("w", colsum)


def test_replicate_budget_refuses_a_stratum_it_cannot_spread():
    allocation = A.AllocationSpec(strata=_STRATA, k_eval=64, replicates=8)
    k = np.zeros(len(_STRATA), dtype=np.int64)
    k[0] = 3
    with pytest.raises(ValueError, match="across 8 replicates"):
        allocation.replicate_budget(k)


def test_a_non_finite_column_is_refused_rather_than_dropped():
    allocation = A.AllocationSpec(
        strata=_STRATA, k_eval=4096, home_weights={"w": {"short": 1.0}}
    )
    colsum = np.zeros(len(_STRATA))
    colsum[_STRATUM_INDEX["short"]] = 1.0
    for bad in (np.nan, np.inf):
        broken = colsum.copy()
        broken[_STRATUM_INDEX["long"]] = bad
        with pytest.raises(ValueError, match="not finite"):
            allocation.budget("w", broken)


def test_a_selection_carrying_a_non_finite_column_is_refused():
    selection, _, _ = _stub_selection([_SAMPLES])
    colsum = selection.colsum.copy()
    colsum[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        replace(selection, colsum=colsum)


def test_an_exact_word_whose_whole_column_vanishes_is_not_supported():
    """A single-token word never publishes a zero, and never as supported."""
    selection, tok, model = _stub_selection([_SAMPLES])
    col = selection.first_col[tok.encode(" a")[0]]
    colsum = selection.colsum.copy()
    colsum[:, col] = 0.0
    bos = selection.bos_p0.copy()
    bos[col] = 0.0
    blanked = replace(selection, colsum=colsum, bos_p0=bos)
    got = P.pps_estimate_batch(
        [" a"],
        blanked,
        model,
        tok,
        allocation=_allocation(),
        k_eval=12,
        unit_counts={" a": 0},
        take_store=_take_store([_SAMPLES]),
    )
    record = got[" a"]
    assert record.p_hat > 0.0
    assert not record.supported


def test_a_ratio_that_underflows_still_yields_a_positive_estimate(monkeypatch):
    """A numerator above zero that vanishes once divided takes the log domain."""
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    real = P._continuations

    def _small(prefixes, toks, *args, **kwargs):
        joint, log_joint, beta, log_beta = real(prefixes, toks, *args, **kwargs)
        shifted = log_joint - 725.0
        return np.exp(shifted), shifted, beta, log_beta

    def _no_remainder(*args, **kwargs):
        return 0.0, 0.0

    monkeypatch.setattr(P, "_take_h", _no_remainder)
    monkeypatch.setattr(P, "_continuations", _small)
    got = P.pps_estimate_batch(
        [" the"],
        selection,
        model,
        tok,
        allocation=_allocation(),
        k_eval=12,
        unit_counts={" the": 0},
        take_store=_take_store(),
    )
    record = got[" the"]
    assert record.p_hat > 0.0
    assert math.isfinite(record.bits)
    assert record.bits > 100.0


def test_a_multi_token_word_absent_from_the_sweep_is_refused():
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    stripped = replace(
        selection,
        mt_first_ids=np.zeros(0, dtype=np.int64),
        k_store=selection.k_store[:0],
        cell_offsets=np.zeros(1, dtype=np.int64),
    )
    with pytest.raises(KeyError, match="no sweep column"):
        P.pps_estimate_batch(
            [" the"],
            stripped,
            model,
            tok,
            allocation=_allocation(),
            k_eval=12,
            unit_counts=_counts(),
            take_store=_take_store(),
        )


def test_a_cell_stored_out_of_priority_order_is_refused():
    selection, _, _ = _stub_selection([_SAMPLES, _MORE_SAMPLES], k_store=64)
    for mt in range(selection.mt_first_ids.size):
        for stratum in range(len(_STRATA)):
            sl = selection.cell(mt, stratum, 0)
            if sl.stop - sl.start >= 3:
                break
        else:
            continue
        break
    scrambled = selection.sel_p0.copy()
    lo, hi = sl.start, sl.stop
    scrambled[lo:hi] = scrambled[lo:hi][::-1]
    shuffled = replace(selection, sel_p0=scrambled)
    with pytest.raises(ValueError, match="descending priority"):
        P._cell_take(shuffled, sl, 1)


def test_the_duplicate_collapse_key_is_a_full_128_bit_digest():
    """A 64-bit key would fold distinct prefixes together by birthday."""
    tok, _ = stub_pair()
    index = P.boundary_index(_SAMPLES, tok, stratification=_STRATIFICATION)
    assert index.short_hash.ndim == 2
    assert index.short_hash.shape[1] == 2
    assert index.short_hash.dtype == np.uint64
    assert P._prefix_hash([1, 2, 3]) != P._prefix_hash([1, 2, 4])
    assert P._prefix_hash([1, 2, 3]) == P._prefix_hash([1, 2, 3])


@pytest.mark.statistical
def test_the_production_draw_is_unbiased_over_repeated_pool_seeds():
    """``_draw_cells`` weights average onto the exhaustive stratum mass."""
    word, target = " dog", None
    allocation = _allocation(k_eval=12)
    totals = []
    for seed in range(48):
        selection, tok, _ = _stub_selection(
            [_SAMPLES, _MORE_SAMPLES], k_store=64, rng_seed=seed
        )
        col = selection.first_col[tok.encode(word)[0]]
        drawn = P._draw_cells(selection, allocation, word, col, allocation.k_eval)
        totals.append(float(drawn["weights"].sum()))
        if target is None:
            target = float(selection.colsum[:, col].sum())
    mean = float(np.mean(totals))
    se = float(np.std(totals, ddof=1)) / math.sqrt(len(totals))
    assert target is not None and target > 0.0
    assert abs(mean - target) < 3.0 * se, (mean, target, se)


def test_a_bare_surface_is_estimated_through_its_spaced_tokenization():
    """A separator-less key names the same estimand as its spaced form.

    The exact count strips the separator before tallying, so both keys anchor
    to the type-level expected unit count; the sampled channel runs through
    the spaced tokenization because the bare first token has no valid column.
    """
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    allocation = _allocation()
    counts = _counts()
    counts["the"] = counts[" the"]
    got = P.pps_estimate_batch(
        ["the"],
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=allocation.k_eval,
        unit_counts=counts,
        take_store=_take_store(),
    )
    record = got["the"]
    assert record.token_ids == tuple(tok.encode(" the"))
    assert record.p_hat > 0.0
    assert record.word == "the"


def test_a_bare_surface_without_a_column_gets_the_count_estimate():
    """No swept column but a pooled occurrence: publish ``C/denom``.

    Unbiased for the same estimand with no variance reduction; the draw-side
    certificate fields are non-finite, so ``certify`` refuses to vouch for it.
    """
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    allocation = _allocation()
    got = P.pps_estimate_batch(
        ["qq"],
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=allocation.k_eval,
        unit_counts={"qq": 3},
        take_store=_take_store(),
    )
    record = got["qq"]
    assert record.p_hat == pytest.approx(3 / selection.denom)
    assert math.isfinite(record.bits)
    assert record.supported is False
    assert record.nonpositive is False
    assert math.isnan(record.cv)
    assert P.certify(record.to_json(), P.SupportThresholds()) is False


def test_a_bare_surface_with_no_channel_at_all_is_refused_by_certificate():
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    allocation = _allocation()
    got = P.pps_estimate_batch(
        ["qx"],
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=allocation.k_eval,
        unit_counts={"qx": 0},
        take_store=_take_store(),
    )
    record = got["qx"]
    assert record.p_hat == P.TINY
    assert record.nonpositive is True
    assert record.supported is False
    assert P.certify(record.to_json(), P.SupportThresholds()) is False


def test_the_take_store_finds_every_realized_take():
    """The store's per-column counts match a brute-force scan of the frame."""
    tok, _ = stub_pair()
    samples = _SAMPLES + _MORE_SAMPLES
    encoded = P._encode_samples(tok, samples)
    bset = P.boundary_set(tok)
    first_ids = sorted({tok.encode(w)[0] for w in _STUB_WORDS})
    store = _take_store()
    for fid in first_ids:
        expected = 0
        for ids in encoded:
            for pos in range(len(ids)):
                if not bset.opens_unit(ids, pos):
                    continue
                if pos == 0 or pos == len(ids) - 1:
                    continue  # collapsed bos slot and forced-termination slot
                if ids[pos + 1] == fid:
                    expected += 1
        assert store.count(fid) == expected, fid
    total = sum(store.count(f) for f in first_ids)
    assert total == store.sample.size


def test_the_take_store_round_trips_through_npz(tmp_path):
    store = _take_store()
    path = tmp_path / "takes.npz"
    store.save(path)
    again = P.TakeStore.load(path)
    assert again.sample.dtype == np.int32 and again.pos.dtype == np.int16
    np.testing.assert_array_equal(again.first_ids, store.first_ids)
    np.testing.assert_array_equal(again.offsets, store.offsets)
    np.testing.assert_array_equal(again.sample, store.sample)
    np.testing.assert_array_equal(again.pos, store.pos)
    assert again.pool_fingerprint == store.pool_fingerprint


@pytest.mark.statistical
def test_a_partial_take_draw_is_deterministic_bounded_and_unbiased():
    """SRS over the takes: seeded draw, ``n/m`` weights, mean-zero error.

    The summands are probabilities times a closing mass, so every one sits in
    ``[0, 1]`` and the estimate can never exceed ``n`` plus the exact bos
    channel.
    """
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    store = _take_store()
    toks = tok.encode(" the")
    col = selection.first_col[toks[0]]
    fid = int(toks[0])
    n = store.count(fid)
    assert n > 4
    exact_total, exact_se = P._take_h(
        store, toks, col, selection, model, tok, 3, True, n, 0.0
    )
    assert exact_se == 0.0
    samp, pos = store.takes(fid)
    prefixes = store.prefixes(samp, pos)
    cont, _, beta, _ = P._continuations(prefixes, toks, model, tok, 3, True)
    c_vals = np.asarray(cont * beta, dtype=np.float64)
    assert bool((c_vals >= 0.0).all()) and bool((c_vals <= 1.0).all())
    assert exact_total == pytest.approx(float(c_vals.sum()))
    m = 4
    once = P._take_indices(selection.rng_seed, fid, n, m)
    again = P._take_indices(selection.rng_seed, fid, n, m)
    np.testing.assert_array_equal(once, again)
    assert once.size == m and np.unique(once).size == m
    estimates = np.array(
        [n * c_vals[P._take_indices(seed, fid, n, m)].mean() for seed in range(400)]
    )
    se_of_mean = estimates.std(ddof=1) / math.sqrt(estimates.size)
    assert abs(estimates.mean() - c_vals.sum()) < 4.0 * se_of_mean
    assert bool((estimates <= n).all())


def test_a_take_store_from_another_pool_is_refused():
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    foreign = replace(_take_store(), pool_fingerprint="deadbeefdeadbeef")
    with pytest.raises(ValueError, match="take store is from pool"):
        P.pps_estimate_batch(
            _STUB_WORDS,
            selection,
            model,
            tok,
            allocation=_allocation(),
            k_eval=12,
            unit_counts=_counts(),
            take_store=foreign,
            pool_fingerprint=selection.pool_fingerprint,
        )


def test_every_channel_reaches_the_progress_callback():
    """The shard driver persists only what the callback sees.

    A record returned but never emitted is missing from the shard table, so
    every channel must emit through the callback.
    """
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    allocation = _allocation()
    counts = _counts()
    counts.update({"the": counts[" the"], "qq": 3, "qx": 0})
    seen = []
    got = P.pps_estimate_batch(
        ["the", "qq", "qx"],
        selection,
        model,
        tok,
        allocation=allocation,
        k_eval=allocation.k_eval,
        unit_counts=counts,
        take_store=_take_store(),
        progress=lambda r: seen.append(r.word),
    )
    assert sorted(seen) == sorted(got) == ["qq", "qx", "the"]


@pytest.mark.parametrize("numerator", ["pathwise", "expectation"])
def test_a_batch_scans_tokenizer_boundaries_once(monkeypatch, numerator):
    selection, tok, model = _stub_selection([_SAMPLES, _MORE_SAMPLES])
    encoded = P._encode_samples(tok, _SAMPLES + _MORE_SAMPLES)
    inputs = (
        {"take_store": _take_store()}
        if numerator == "pathwise"
        else {"splits": P.unit_counts_split(_STUB_WORDS, encoded, tok), "cap": 8}
    )
    scans = []
    convert = tok.convert_ids_to_tokens

    def record_scan(ids):
        scans.append(tuple(ids))
        return convert(ids)

    monkeypatch.setattr(tok, "convert_ids_to_tokens", record_scan)
    records = P.pps_estimate_batch(
        _STUB_WORDS,
        selection,
        model,
        tok,
        allocation=_allocation(),
        k_eval=12,
        unit_counts=_counts(),
        **inputs,
    )
    assert set(records) == set(_STUB_WORDS)
    assert scans == [tuple(range(len(tok)))]
