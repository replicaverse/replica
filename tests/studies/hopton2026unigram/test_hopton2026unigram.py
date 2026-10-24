import numpy as np
import pandas as pd
import pytest

from replica.external.adapters.context_reading import data_dir
from replica.studies.hopton2026unigram import (
    Hopton2026Unigram,
    PrecomputedFoldSplitter,
)
from replica.studies.hopton2026unigram.dump_variance_inputs import (
    _assemble,
    _check_manifest,
    _completed,
    _token_list_fingerprint,
    _write_manifest,
    _write_shard,
    precompute_mc,
    token_for_row,
)
from replica.studies.hopton2026unigram.variance import (
    VarianceInputs,
    bootstrap_variance,
    load_inputs,
    variance_table,
)


def test_shared_input_location_prefers_checkout_without_fetching(tmp_path, monkeypatch):
    from replica.external import repositories

    def no_fetch(*args, **kwargs):
        raise AssertionError("resolving input paths must not fetch data")

    monkeypatch.setattr(repositories, "fetch_dependency", no_fetch)
    fallback = tmp_path / "datasets/eye-tracking/processed/meco_opedal2024context"
    assert data_dir(tmp_path) == fallback
    checkout = (
        tmp_path
        / repositories.EXTERNAL_REPOS["context-reading-time"][0].path
        / "merged_data_no_zero"
    )
    checkout.mkdir(parents=True)
    assert data_dir(tmp_path) == checkout
    monkeypatch.setenv("PSYCHO_REPL_DATA", str(tmp_path))
    assert data_dir() == checkout


def _splitter(fold_ids, participants=None, items=None):
    n = len(fold_ids)
    participants = np.arange(n) if participants is None else np.asarray(participants)
    items = np.zeros(n, dtype=int) if items is None else np.asarray(items)
    return (
        PrecomputedFoldSplitter(
            np.asarray(fold_ids),
            expected_participants=participants,
            expected_items=items,
        ),
        participants,
        items,
    )


# ---------------------------------------------------------------------------
# PrecomputedFoldSplitter
# ---------------------------------------------------------------------------


def test_split_partitions_every_row_exactly_once():
    fold_ids = [0, 1, 2, 0, 1, 2]
    splitter, participants, items = _splitter(fold_ids)
    y = np.zeros(len(fold_ids))

    folds = list(splitter.split(y, participants, participants, items))

    assert splitter.n_folds == 3
    assert len(folds) == 3
    # Each row is tested once, and train is exactly the complement.
    tested = np.zeros(len(fold_ids), dtype=int)
    for train, test in folds:
        assert np.array_equal(train, ~test)
        tested += test.astype(int)
    assert np.array_equal(tested, np.ones(len(fold_ids), dtype=int))


def test_split_groups_rows_by_their_committed_fold():
    fold_ids = [2, 0, 2, 1]
    splitter, participants, items = _splitter(fold_ids)

    by_fold = {
        int(np.asarray(fold_ids)[test][0]): test
        for _, test in splitter.split(np.zeros(4), participants, participants, items)
    }

    assert np.array_equal(by_fold[2], np.array([True, False, True, False]))
    assert np.array_equal(by_fold[0], np.array([False, True, False, False]))


def test_split_rejects_a_different_row_count():
    """A fold map sized to another row set would silently reshuffle folds."""
    splitter, participants, items = _splitter([0, 1, 0, 1])

    with pytest.raises(ValueError, match="fold ids for"):
        list(splitter.split(np.zeros(3), participants[:3], participants[:3], items[:3]))


def test_split_rejects_reordered_participants():
    fold_ids = [0, 1, 0, 1]
    splitter, participants, items = _splitter(fold_ids)
    shuffled = participants[::-1].copy()

    with pytest.raises(ValueError, match="participants sequence differs"):
        list(splitter.split(np.zeros(4), shuffled, shuffled, items))


def test_split_rejects_reordered_items():
    fold_ids = [0, 1, 0, 1]
    splitter, participants, items = _splitter(fold_ids, items=[0, 0, 1, 1])

    with pytest.raises(ValueError, match="items sequence differs"):
        list(splitter.split(np.zeros(4), participants, participants, items[::-1]))


# ---------------------------------------------------------------------------
# VarianceInputs
# ---------------------------------------------------------------------------


def _inputs(n_tokens=4, n_samples=6, seed=0):
    rng = np.random.default_rng(seed)
    return VarianceInputs(
        tokens=np.array([f"w{i}" for i in range(n_tokens)]),
        rb_prob_sums=rng.random((n_tokens, n_samples)),
        mc_counts=rng.integers(0, 3, (n_tokens, n_samples)).astype(float),
        rb_boundaries=rng.random(n_samples) + 1.0,
        mc_boundaries=rng.random(n_samples) + 1.0,
    )


def test_unit_index_defaults_to_one_row_per_token():
    inputs = _inputs(n_tokens=3)
    assert np.array_equal(inputs.unit_index, np.arange(3))


def test_unit_index_maps_repeated_units_onto_their_form():
    """Corpus units repeat; each must point at its form's row so the mean over
    units weighs a frequent word once per occurrence."""
    inputs = _inputs(n_tokens=3)
    inputs.unit_tokens = np.array(["w2", "w0", "w2", "w2"])

    assert np.array_equal(inputs.unit_index, np.array([2, 0, 2, 2]))


def test_unit_index_rejects_a_unit_with_no_estimate():
    inputs = _inputs(n_tokens=2)
    inputs.unit_tokens = np.array(["w0", "absent"])

    with pytest.raises(ValueError, match="absent from the estimate tables"):
        _ = inputs.unit_index


# ---------------------------------------------------------------------------
# bootstrap_variance
# ---------------------------------------------------------------------------


def test_bootstrap_variance_is_reproducible_from_a_seed():
    inputs = _inputs()
    kwargs = dict(subsample_size=10, n_boot=32)
    a = bootstrap_variance(
        inputs.rb_prob_sums,
        inputs.rb_boundaries,
        rng=np.random.default_rng(7),
        **kwargs,
    )
    b = bootstrap_variance(
        inputs.rb_prob_sums,
        inputs.rb_boundaries,
        rng=np.random.default_rng(7),
        **kwargs,
    )

    np.testing.assert_array_equal(a, b)


def test_shared_and_per_token_draws_agree_in_magnitude():
    """`shared_draws` reuses one resample set across tokens for speed. It has to
    estimate the same quantity, or Table 2 depends on the shortcut."""
    inputs = _inputs(n_tokens=6, n_samples=40, seed=3)
    kwargs = dict(subsample_size=25, n_boot=400)

    shared = bootstrap_variance(
        inputs.rb_prob_sums,
        inputs.rb_boundaries,
        rng=np.random.default_rng(11),
        shared_draws=True,
        **kwargs,
    )
    per_token = bootstrap_variance(
        inputs.rb_prob_sums,
        inputs.rb_boundaries,
        rng=np.random.default_rng(11),
        shared_draws=False,
        **kwargs,
    )

    assert shared.shape == per_token.shape
    np.testing.assert_allclose(np.mean(shared), np.mean(per_token), rtol=0.5)


def test_bootstrap_variance_yields_nan_when_no_resample_is_positive():
    """Non-positive estimates are dropped; a token with none left is NaN, so
    `nanmean` skips it."""
    n_samples = 5
    numerators = np.zeros((2, n_samples))
    out = bootstrap_variance(
        numerators,
        np.ones(n_samples),
        subsample_size=4,
        n_boot=16,
        rng=np.random.default_rng(0),
    )

    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# variance_table and load_inputs
# ---------------------------------------------------------------------------


def test_variance_table_has_one_row_per_subsample_size():
    inputs = _inputs(n_tokens=4, n_samples=20, seed=5)
    table = variance_table(inputs, subsample_sizes=(10, 20), n_boot=64, seed=2)

    assert list(table["J"]) == [10, 20]
    assert set(table.columns) == {"J", "mmc", "nmc", "ratio"}
    finite = table.dropna(subset=["mmc", "nmc"])
    np.testing.assert_allclose(
        finite["ratio"], finite["nmc"] / finite["mmc"], rtol=1e-12
    )


def test_load_inputs_pivots_the_long_sample_tables(tmp_path):
    """Long tables become matrices with aligned token rows and sample columns."""
    long = pd.DataFrame(
        {
            "token": ["b", "a", "b", "a"],
            "sample_idx": [0, 0, 1, 1],
            "rb_prob_sum": [0.5, 0.25, 0.75, 0.125],
            "mc_count": [1.0, 2.0, 3.0, 4.0],
        }
    )
    long.to_parquet(tmp_path / "variance_samples.parquet")
    pd.DataFrame(
        {
            "sample_idx": [1, 0],
            "rb_boundary_count": [3.0, 2.0],
            "mc_boundary_count": [30.0, 20.0],
        }
    ).to_parquet(tmp_path / "variance_boundaries.parquet")

    inputs = load_inputs(tmp_path)

    assert list(inputs.tokens) == ["a", "b"]
    assert (inputs.n_tokens, inputs.n_samples) == (2, 2)
    np.testing.assert_allclose(inputs.rb_prob_sums, [[0.25, 0.125], [0.5, 0.75]])
    np.testing.assert_allclose(inputs.mc_counts, [[2.0, 4.0], [1.0, 3.0]])
    # Boundaries are sorted by sample_idx.
    np.testing.assert_allclose(inputs.rb_boundaries, [2.0, 3.0])


@pytest.mark.parametrize("archived", [False, True])
def test_load_inputs_reads_corpus_units_when_present(tmp_path, monkeypatch, archived):
    from replica.studies.hopton2026unigram import variance

    requested = []

    def resolve(study, name):
        requested.append((study, name))
        return tmp_path / name.removeprefix("data/")

    monkeypatch.setattr(variance, "resource_file", resolve)
    pd.DataFrame(
        {
            "token": ["a", "b"],
            "sample_idx": [0, 0],
            "rb_prob_sum": [0.5, 0.5],
            "mc_count": [1.0, 1.0],
        }
    ).to_parquet(tmp_path / "variance_samples.parquet")
    pd.DataFrame(
        {"sample_idx": [0], "rb_boundary_count": [2.0], "mc_boundary_count": [4.0]}
    ).to_parquet(tmp_path / "variance_boundaries.parquet")
    pd.DataFrame({"token": ["b", "b", "a"]}).to_parquet(
        tmp_path / "variance_units.parquet"
    )

    inputs = load_inputs() if archived else load_inputs(tmp_path)

    assert np.array_equal(inputs.unit_index, np.array([1, 1, 0]))
    assert requested == (
        [
            ("hopton2026unigram", f"data/variance_{name}.parquet")
            for name in ("samples", "boundaries", "units")
        ]
        if archived
        else []
    )


# ---------------------------------------------------------------------------
# Reported values and study surface
# ---------------------------------------------------------------------------


def test_evaluation_spec_rejects_an_unknown_condition():
    from tests.analysis._fixtures import make_toy_dataset

    with pytest.raises(ValueError, match="condition must be one of"):
        Hopton2026Unigram().evaluation_spec(condition="MMC", dataset=make_toy_dataset())


def test_evaluation_spec_rejects_an_unknown_comparison():
    from tests.analysis._fixtures import make_toy_dataset

    with pytest.raises(ValueError, match="comp must be one of"):
        Hopton2026Unigram().evaluation_spec(comp="surp+pmi", dataset=make_toy_dataset())


# ---------------------------------------------------------------------------
# _prepare: the recipe the R reference ports
# ---------------------------------------------------------------------------


def _raw_frame():
    """Two trials of one participant, with a zero frequency to floor."""
    return pd.DataFrame(
        {
            "subid": ["s1"] * 5,
            "model": ["mgpt_lc"] * 5,
            "trialid": [1, 1, 1, 2, 2],
            "ianum": [1, 2, 3, 1, 2],
            "obs_id": [10, 11, 12, 13, 14],
            "freq": [0.5, 0.25, 0.0, 0.5, 0.125],
            "prev_freq": [0.5, 0.5, 0.25, 0.25, 0.5],
            "surp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "prev_surp": [0.5, 1.0, 2.0, 3.0, 4.0],
            "_mmc": [2.0, 4.0, 8.0, 16.0, 32.0],
        }
    )


def test_prepare_converts_frequency_to_surprisal_in_bits():
    out = Hopton2026Unigram._prepare(_raw_frame())

    # The zero-frequency cell uses the smallest positive frequency, 0.125.
    by_obs = out.set_index("obs_id")
    assert by_obs.loc[10, "freq"] == pytest.approx(1.0)
    assert by_obs.loc[11, "freq"] == pytest.approx(2.0)
    assert by_obs.loc[12, "freq"] == pytest.approx(3.0)
    assert np.isfinite(out["freq"]).all()


def test_prepare_drops_rows_without_a_positive_previous_frequency():
    df = _raw_frame()
    df.loc[1, "prev_freq"] = 0.0

    out = Hopton2026Unigram._prepare(df)

    assert 11 not in set(out["obs_id"])
    assert len(out) == len(df) - 1


def test_prepare_lags_the_mmc_estimate_within_a_trial():
    """Spillover MMC is a within-trial lag, so trial 2 must not inherit the
    last value of trial 1."""
    out = Hopton2026Unigram._prepare(_raw_frame()).set_index("obs_id")

    assert np.isnan(out.loc[10, "prev_freq_BW"])
    assert out.loc[11, "prev_freq_BW"] == pytest.approx(2.0)
    assert out.loc[12, "prev_freq_BW"] == pytest.approx(4.0)
    # First unit of the next trial restarts the lag.
    assert np.isnan(out.loc[13, "prev_freq_BW"])
    assert out.loc[14, "prev_freq_BW"] == pytest.approx(16.0)


def test_prepare_derives_pmi_as_frequency_minus_surprisal():
    out = Hopton2026Unigram._prepare(_raw_frame())

    np.testing.assert_allclose(out["pmi"], out["freq"] - out["surp"])
    np.testing.assert_allclose(out["pmi_BW"], out["freq_BW"] - out["surp"])
    np.testing.assert_allclose(out["prev_pmi"], out["prev_freq"] - out["prev_surp"])


def test_prepare_drops_the_second_order_lag_columns():
    df = _raw_frame()
    df["prev2_freq"] = 1.0
    df["prev2_surp"] = 1.0

    out = Hopton2026Unigram._prepare(df)

    assert not [c for c in out.columns if c.startswith("prev2")]


# ---------------------------------------------------------------------------
# figure8
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# dump_variance_inputs: the parts that need no model
# ---------------------------------------------------------------------------


def test_precompute_mc_counts_non_overlapping_occurrences():
    """The NMC numerator. The scan must advance by the token's length, so a
    self-overlapping form is not double counted.

    Only a self-overlapping token separates the two rules: ``aa`` in ``aaaa``
    is 2 advancing by the token and 3 advancing one character at a time.
    """
    counts, boundaries = precompute_mc("aa", ["aaaa", "aaa", "aa a", "xy"])

    np.testing.assert_array_equal(counts, [2, 1, 1, 0])
    # Denominator is whitespace tokens plus one.
    np.testing.assert_array_equal(boundaries, [2, 2, 3, 2])


def test_precompute_mc_treats_a_leading_space_as_part_of_the_match():
    """The space-prefixed form is a different unigram from the bare one; the
    whitespace ablation scores the bare form in place of the spaced one."""
    with_space, _ = precompute_mc(" the", ["the cat and the dog"])
    bare, _ = precompute_mc("the", ["the cat and the dog"])

    assert int(with_space[0]) == 1
    assert int(bare[0]) == 2


def test_token_for_row_prefixes_a_space_after_the_first_unit():
    assert token_for_row("cat", 1, None, 7) == "cat"
    assert token_for_row("cat", 2, None, 7) == " cat"


def test_token_for_row_lets_the_lookup_override_the_position_rule():
    lookup = {(7, 2): False, (7, 3): True}

    assert token_for_row("cat", 2, lookup, 7) == "cat"
    assert token_for_row("cat", 3, lookup, 7) == " cat"
    # Units absent from the lookup fall back to the position rule.
    assert token_for_row("cat", 4, lookup, 7) == " cat"


def test_fingerprint_changes_with_the_token_list():
    """Shards are named by position, so resuming against a changed word list
    would silently pair old shards with new tokens."""
    base = _token_list_fingerprint(["a", "b"])

    assert _token_list_fingerprint(["a", "b"]) == base
    assert _token_list_fingerprint(["b", "a"]) != base
    assert _token_list_fingerprint(["ab"]) != _token_list_fingerprint(["a", "b"])


def test_check_manifest_rejects_shards_from_another_run(tmp_path):
    class _Args:
        model_name = "mgpt"
        batch_size = 8
        samples = "s.json"
        words = "w.csv"

    _write_manifest(tmp_path, ["a", "b"], samples=4, args=_Args())

    _check_manifest(tmp_path, ["a", "b"])  # same list: no complaint
    with pytest.raises(ValueError, match="different token list"):
        _check_manifest(tmp_path, ["a", "c"])


def test_assemble_refuses_to_finalize_with_a_shard_missing(tmp_path):
    _write_shard(tmp_path, 0, "a", np.zeros(2), np.zeros(2))

    with pytest.raises(ValueError, match="shards are missing"):
        _assemble(tmp_path, tmp_path / "out", ["a", "b"])


def test_assemble_concatenates_shards_in_token_order(tmp_path):
    _write_shard(tmp_path, 1, "b", np.array([0.5, 0.5]), np.array([1.0, 1.0]))
    _write_shard(tmp_path, 0, "a", np.array([0.25, 0.25]), np.array([2.0, 2.0]))

    long = _assemble(tmp_path, tmp_path / "out", ["a", "b"])

    assert list(long["token"]) == ["a", "a", "b", "b"]
    assert (tmp_path / "out" / "variance_samples.parquet").exists()


def test_completed_drops_a_torn_shard(tmp_path):
    _write_shard(tmp_path, 0, "a", np.zeros(2), np.zeros(2))
    (tmp_path / "token_000001.parquet").write_bytes(b"not parquet")

    done = _completed(tmp_path, n_tokens=2)

    assert done == {0}
    # The unreadable shard is removed so the resume recomputes it.
    assert not (tmp_path / "token_000001.parquet").exists()


def test_figure8_draws_one_panel_per_estimator():
    from replica.studies.hopton2026unigram.variance import figure8

    fig = figure8(
        _inputs(n_tokens=4, n_samples=12, seed=9),
        subsample_sizes=(10, 20),
        n_boot=16,
        seed=0,
    )

    titles = [ax.get_title() for ax in fig.axes]
    assert any("MMC" in t for t in titles)
    assert any("NMC" in t for t in titles)
    assert all(ax.get_yscale() == "log" for ax in fig.axes)


@pytest.mark.needs_model
def test_precompute_rb_reproduces_the_shared_estimator():
    """The variance dump must agree with ``mmc_estimate``.

    ``precompute_rb`` decomposes the estimate per sample so the bootstrap can
    resample it, but the total it implies is the same quantity the shared
    estimator returns. It keeps its own copy of the boundary-continuation loop,
    so this test pins the two together.

    Multi-token words are the discriminating case: for a single-token word the
    continuation loop never runs and the two paths agree trivially.
    """
    import importlib.util

    import pytest

    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not available")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from replica.feature.frequency.sampled import mmc_estimate
    from replica.studies.hopton2026unigram.dump_variance_inputs import precompute_rb

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    samples = [
        "the dog ran across the field",
        "a cat sat on the mat",
        "hello world this is a test",
    ]

    for word in (" antidisestablishment", " Beekeeping", " the"):
        for batch_size in (2, 8):
            sums, counts = precompute_rb(
                word, samples, model, tokenizer, batch_size, marker="Ġ"
            )
            implied = sums.sum() / (len(samples) + counts.sum())
            expected = mmc_estimate(word, samples, model, tokenizer, batch_size)
            assert implied == pytest.approx(expected, rel=1e-4, abs=1e-15), (
                f"{word!r} at batch_size={batch_size}: dump {implied} vs "
                f"estimator {expected}"
            )
