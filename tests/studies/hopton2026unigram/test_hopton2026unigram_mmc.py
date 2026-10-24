"""Refusals and determinism of the MMC pipeline stages.

No model is loaded: every case here exercises the JSON and manifest handling
that decides whether a table is safe to merge and publish.
"""

import argparse
import json
import math

import numpy as np
import pytest

from replica.feature.frequency import pooled as P
from replica.feature.frequency.sampled import _encode_samples
from replica.studies.hopton2026unigram.mmc import (
    cli,
    common,
    estimate,
    index,
    materialize,
    plan,
    pool,
    sweep,
    table,
    tally,
)
from tests.feature.frequency._pooled_stub import stub_pair

_WORDS = [" the", " dog"]


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _record(word, p_hat, **overrides):
    base = {
        "word": word,
        "token_ids": [1, 2],
        "home_strata": ["lat_en"],
        "bound_share": [],
        "n_tokens": 2,
        "unit_count": 0,
        "h_hat": 0.0,
        "nonpositive": False,
        "p_hat": p_hat,
        "p_hat_no_bos": p_hat,
        "bits": 1.0,
        "k_eval": 8,
        "se_replicate": 0.0,
        "se_dlt": 0.0,
        "cv": 0.0,
        "home_share": 1.0,
        "bos_share": 0.0,
        "top1_share": 0.1,
        "residual_bound": 0.0,
        "n_nonzero_contrib": 8,
        "n_strata_contributing": 1,
        "underflow_fallback": False,
        "supported": True,
    }
    base.update(overrides)
    return base


def _shard(words_and_values, fingerprint="pool-a"):
    return {
        "pool_fingerprint": fingerprint,
        "n_pool_samples": 8,
        "denom": 60,
        "k_eval": 8,
        "allocation_sha": "abc",
        "estimator_version": P.ESTIMATOR_VERSION,
        "normalizer": 60,
        "cap": None,
        "selection_version": P.SELECTION_VERSION,
        "k_take": 2048,
        "code_git_sha": "",
        "records": {w: _record(w, v) for w, v in words_and_values.items()},
    }


def test_the_merge_refuses_a_shard_from_another_estimator(tmp_path):
    _write(tmp_path / "words.json", _WORDS)
    stale = _shard({" the": 0.1, " dog": 0.2, " cat": 0.3}, "pool-a")
    stale["estimator_version"] = "mmc_pps/2"
    _write(tmp_path / "shard_0.json", stale)
    with pytest.raises(SystemExit, match="estimated by"):
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--pool-fingerprint",
                "pool-a",
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_resume_replays_only_records_from_this_estimator_and_normalizer(tmp_path):
    """A sidecar line from another numerator, or divided by another count, is
    recomputed; the version alone cannot tell two normalizers apart."""
    path = tmp_path / "table_0.jsonl"
    current = dict(
        _record(" the", 0.1), estimator_version=P.ESTIMATOR_VERSION, normalizer=60
    )
    other_numerator = dict(
        _record(" dog", 0.2), estimator_version=P.EE_ESTIMATOR_VERSION, normalizer=60
    )
    other_normalizer = dict(
        _record(" cat", 0.3), estimator_version=P.ESTIMATOR_VERSION, normalizer=59
    )
    bare = _record(" fox", 0.4)
    path.write_text(
        "\n".join(
            json.dumps(r) for r in (current, other_numerator, other_normalizer, bare)
        )
        + "\n",
        encoding="utf-8",
    )
    resumed = estimate._resume(
        path, k_eval=8, version=P.ESTIMATOR_VERSION, normalizer=60
    )
    assert sorted(resumed) == [" the"]
    resumed = estimate._resume(
        path, k_eval=8, version=P.EE_ESTIMATOR_VERSION, normalizer=60
    )
    assert sorted(resumed) == [" dog"]


def test_the_merge_publishes_each_numerator_under_its_own_source(tmp_path):
    """A table stamped by one numerator merges only under the source that
    names it, since the two families must never share a cell id."""
    _write(tmp_path / "words.json", _WORDS)
    shard = _shard({" the": 0.1, " dog": 0.2})
    shard["estimator_version"] = P.EE_ESTIMATOR_VERSION
    _write(tmp_path / "shard_0.json", shard)
    argv = [
        "merge",
        "--shards",
        str(tmp_path / "shard_0.json"),
        "--words",
        str(tmp_path / "words.json"),
        "--out",
        str(tmp_path / "table.json"),
    ]
    with pytest.raises(SystemExit, match="mmc_pool merges"):
        cli.main([*argv, "--source", "mmc_pool"])
    assert cli.main([*argv, "--source", "mmc_pool_ee"]) == 0
    meta = json.loads((tmp_path / "table.meta.json").read_text(encoding="utf-8"))
    assert meta["estimator_version"] == P.EE_ESTIMATOR_VERSION


def test_the_merge_refuses_shards_divided_by_different_normalizers_or_caps(tmp_path):
    _write(tmp_path / "words.json", _WORDS)
    first = _shard({" the": 0.1})
    second = _shard({" dog": 0.2})
    second["normalizer"] = 59
    _write(tmp_path / "shard_0.json", first)
    _write(tmp_path / "shard_1.json", second)
    argv = [
        "merge",
        "--source",
        "mmc_pool",
        "--shards",
        str(tmp_path / "shard_0.json"),
        str(tmp_path / "shard_1.json"),
        "--words",
        str(tmp_path / "words.json"),
        "--out",
        str(tmp_path / "table.json"),
    ]
    with pytest.raises(SystemExit, match="normalizer"):
        cli.main(argv)
    second["normalizer"] = 60
    second["cap"] = 50
    _write(tmp_path / "shard_1.json", second)
    with pytest.raises(SystemExit, match="cap"):
        cli.main(argv)


class _Inputs:
    """The shard/audit arguments that pick the numerator."""

    def __init__(self, source, **given):
        self.source = source
        self.counts = None
        self.take_store = None
        self.pool_dir = None
        self.split_counts = None
        self.cap = None
        for key, value in given.items():
            setattr(self, key, value)


def test_each_pooled_source_takes_only_its_own_numerator_inputs():
    """Given both kinds of input the run would be one estimator recorded under
    the other's name, so each source refuses the other's flags."""
    pathwise = dict(counts="c.json", take_store="t.npz", pool_dir="pool")
    expectation = dict(split_counts="s.json", cap=50)
    assert (
        estimate._pooled_inputs(_Inputs("mmc_pool", **pathwise), "shard") == "mmc_pool"
    )
    assert (
        estimate._pooled_inputs(_Inputs("mmc_pool_ee", **expectation), "shard")
        == "mmc_pool_ee"
    )
    with pytest.raises(SystemExit, match="needs --counts"):
        estimate._pooled_inputs(
            _Inputs("mmc_pool", take_store="t.npz", pool_dir="p"), "shard"
        )
    with pytest.raises(SystemExit, match="needs --take-store"):
        estimate._pooled_inputs(_Inputs("mmc_pool", counts="c.json"), "shard")
    with pytest.raises(SystemExit, match="belongs to --source mmc_pool_ee"):
        estimate._pooled_inputs(_Inputs("mmc_pool", **pathwise, cap=50), "shard")
    with pytest.raises(SystemExit, match="needs --split-counts"):
        estimate._pooled_inputs(_Inputs("mmc_pool_ee", cap=50), "audit")
    with pytest.raises(SystemExit, match="needs --cap"):
        estimate._pooled_inputs(_Inputs("mmc_pool_ee", split_counts="s.json"), "audit")
    with pytest.raises(SystemExit, match="drop --take-store"):
        estimate._pooled_inputs(
            _Inputs("mmc_pool_ee", **expectation, take_store="t.npz"), "shard"
        )
    with pytest.raises(SystemExit, match="drop --counts"):
        estimate._pooled_inputs(
            _Inputs("mmc_pool_ee", **expectation, counts="c.json"), "shard"
        )
    with pytest.raises(SystemExit, match="not a pooled source"):
        estimate._pooled_inputs(_Inputs("mmc"), "shard")


def test_the_splits_stages_partition_the_token_store_and_total_its_units(
    tmp_path, monkeypatch
):
    """Both parts of every count and the unit total come off the sampled ids,
    classified against the swept columns, and the merge adds them."""
    tok, _ = stub_pair()
    monkeypatch.setattr(common, "_tokenizer", lambda name: tok)
    shards = [["the dog", "a cat"], ["dog the dog"]]
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    encoded = [_encode_samples(tok, shard) for shard in shards]
    flat, offsets = [], [0]
    for shard in encoded:
        for ids in shard:
            flat.append(np.asarray(ids, dtype=np.int32))
            offsets.append(offsets[-1] + len(ids))
    np.concatenate(flat).tofile(pool_dir / "tokens.i32")
    np.array(offsets, dtype=np.int64).tofile(pool_dir / "tokens.off.i64")
    _write(
        pool_dir / "manifest.json",
        {
            "model_name": "stub",
            "shard_sizes": [len(s) for s in shards],
            "pool_fingerprint": "pool-a",
        },
    )
    words = [" the", " dog", "the"]
    _write(tmp_path / "words.json", words)
    columns = np.array(
        sorted({tok.encode(w)[0] for w in (" the", " dog")}), dtype=np.int64
    )
    np.savez(tmp_path / "alloc.npz", first_ids=columns)
    for shard in range(len(shards)):
        assert (
            cli.main(
                [
                    "splits",
                    "--source",
                    "mmc_pool_ee",
                    "--pool-manifest",
                    str(pool_dir / "manifest.json"),
                    "--pool-dir",
                    str(pool_dir),
                    "--shard-index",
                    str(shard),
                    "--words",
                    str(tmp_path / "words.json"),
                    "--alloc",
                    str(tmp_path / "alloc.npz"),
                    "--out",
                    str(tmp_path / f"splits_{shard}.json"),
                ]
            )
            == 0
        )
    merged = tmp_path / "splits.json"
    assert (
        cli.main(
            [
                "splits-merge",
                "--source",
                "mmc_pool_ee",
                "--parts",
                str(tmp_path / "splits_0.json"),
                str(tmp_path / "splits_1.json"),
                "--expected-shards",
                "2",
                "--words",
                str(tmp_path / "words.json"),
                "--out",
                str(merged),
            ]
        )
        == 0
    )
    got = json.loads(merged.read_text(encoding="utf-8"))
    rows = [ids for shard in encoded for ids in shard]
    want = P.unit_counts_split(words, rows, tok, columns=set(columns.tolist()))
    assert got["counts"] == {w: list(want[w]) for w in words}
    assert got["n_units"] == P.unit_total(rows, tok) == 7
    # the sample-initial spellings carry no marker, so they are residual
    assert got["counts"][" the"] == [1, 1]
    assert got["counts"][" dog"] == [2, 1]
    assert got["counts"]["the"] == got["counts"][" the"]
    splits, units = estimate._load_splits(merged, "pool-a")
    assert units == 7 and splits[" dog"] == (2, 1)
    with pytest.raises(SystemExit, match="taken over pool"):
        estimate._load_splits(merged, "pool-b")


def test_the_plan_gives_a_separator_less_surface_its_spaced_column(
    tmp_path, monkeypatch
):
    """The bare first token cannot open a unit, so a column on it is never
    scored; the spaced first token is the one the channel runs through."""
    tok, _ = stub_pair()
    monkeypatch.setattr(common, "_tokenizer", lambda name: tok)
    words = ["the", " dog"]
    _write(tmp_path / "words.json", words)
    _write(tmp_path / "homes.json", {w: ["meco_en"] for w in words})
    assert (
        cli.main(
            [
                "plan",
                "--source",
                "mmc_pool_ee",
                "--model-name",
                "stub",
                "--words",
                str(tmp_path / "words.json"),
                "--homes",
                str(tmp_path / "homes.json"),
                "--shards",
                "1",
                "--k-eval",
                "40",
                "--k-floor",
                "1",
                "--k-home",
                "8",
                "--replicates",
                "4",
                "--out",
                str(tmp_path / "plan.tsv"),
                "--alloc-out",
                str(tmp_path / "alloc.npz"),
            ]
        )
        == 0
    )
    with np.load(tmp_path / "alloc.npz", allow_pickle=False) as blob:
        first_ids = set(blob["first_ids"].tolist())
    bset = P.boundary_set(tok)
    assert tok.encode("the")[0] not in bset.open_ids
    assert tok.encode("the")[0] not in first_ids
    assert tok.encode(" the")[0] in first_ids
    assert tok.encode(" dog")[0] in first_ids


def test_merge_refuses_a_mismatched_pool_fingerprint(tmp_path):
    _write(tmp_path / "words.json", _WORDS)
    _write(tmp_path / "shard_0.json", _shard({" the": 0.1, " dog": 0.2}, "pool-b"))
    with pytest.raises(SystemExit, match="pool"):
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--pool-fingerprint",
                "pool-a",
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_merge_refuses_shards_from_two_pools(tmp_path):
    _write(tmp_path / "words.json", _WORDS)
    _write(tmp_path / "shard_0.json", _shard({" the": 0.1}, "pool-a"))
    _write(tmp_path / "shard_1.json", _shard({" dog": 0.2}, "pool-b"))
    with pytest.raises(SystemExit, match="disagrees with the earlier shards"):
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                str(tmp_path / "shard_1.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_merge_refuses_a_bare_table_from_the_pooled_source(tmp_path):
    _write(tmp_path / "words.json", _WORDS)
    _write(tmp_path / "shard_0.json", {" the": 0.1, " dog": 0.2})
    with pytest.raises(SystemExit, match="no records block"):
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_merge_requires_the_word_union(tmp_path):
    _write(tmp_path / "shard_0.json", {" the": 0.1})
    with pytest.raises(SystemExit):
        cli.main(
            [
                "merge",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_merge_refuses_an_incomplete_or_oversized_table(tmp_path):
    _write(tmp_path / "words.json", _WORDS)
    _write(tmp_path / "shard_0.json", {" the": 0.1})
    argv = [
        "merge",
        "--shards",
        str(tmp_path / "shard_0.json"),
        "--words",
        str(tmp_path / "words.json"),
        "--out",
        str(tmp_path / "table.json"),
    ]
    with pytest.raises(SystemExit, match="absent"):
        cli.main(argv)
    _write(tmp_path / "shard_0.json", {" the": 0.1, " dog": 0.2, " cat": 0.3})
    with pytest.raises(SystemExit, match="not in the union"):
        cli.main(argv)


def test_merge_rejects_a_non_finite_record_with_a_clean_exit(tmp_path):
    """A NaN inside ``records`` must not surface as a TypeError."""
    _write(tmp_path / "words.json", _WORDS)
    shard = _shard({" the": 0.1, " dog": 0.2})
    shard["records"][" dog"]["p_hat"] = float("nan")
    _write(tmp_path / "shard_0.json", shard)
    with pytest.raises(SystemExit, match="non-finite"):
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_merge_writes_the_certificate_sidecars(tmp_path):
    _write(tmp_path / "words.json", _WORDS)
    shard = _shard({" the": 0.1, " dog": 0.2})
    shard["records"][" dog"]["supported"] = False
    _write(tmp_path / "shard_0.json", shard)
    assert (
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--out",
                str(tmp_path / "table.json"),
            ]
        )
        == 0
    )
    meta = json.loads((tmp_path / "table.meta.json").read_text(encoding="utf-8"))
    assert meta["n_words"] == 2
    assert meta["n_supported"] == 1
    assert meta["n_audited"] == 0
    assert meta["pool_fingerprint"] == "pool-a"
    lines = (tmp_path / "table.records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert P.WordRecord.from_json(json.loads(lines[0])).word == " dog"


def test_the_pool_stage_refuses_a_missing_sample_shard(tmp_path, monkeypatch):
    def unavailable(_name):
        raise AssertionError("tokenizer loaded before validating sample shards")

    monkeypatch.setattr(common, "_tokenizer", unavailable)
    (tmp_path / "shards").mkdir()
    _write(tmp_path / "shards" / "shard_0000.json", ["a b c"])
    with pytest.raises(SystemExit, match="sample shards absent"):
        cli.main(
            [
                "pool",
                "--source",
                "mmc_pool",
                "--model-name",
                "gpt2",
                "--shards-dir",
                str(tmp_path / "shards"),
                "--ids-dir",
                str(tmp_path / "shards"),
                "--expected-shards",
                "2",
                "--out",
                str(tmp_path / "pool"),
            ]
        )


def test_plan_is_deterministic_and_freezes_the_allocation(tmp_path, monkeypatch):
    tok, _ = stub_pair()
    monkeypatch.setattr(common, "_tokenizer", lambda name: tok)
    words = [" the", " dog", " cat", " a"]
    _write(tmp_path / "words.json", words)
    _write(tmp_path / "homes.json", {w: ["meco_en"] for w in words})
    argv = [
        "plan",
        "--source",
        "mmc_pool",
        "--model-name",
        "stub",
        "--words",
        str(tmp_path / "words.json"),
        "--homes",
        str(tmp_path / "homes.json"),
        "--shards",
        "2",
        "--k-eval",
        "40",
        "--k-floor",
        "1",
        "--k-home",
        "8",
        "--replicates",
        "4",
        "--out",
        str(tmp_path / "plan.tsv"),
        "--alloc-out",
        str(tmp_path / "alloc.npz"),
    ]
    assert cli.main(argv) == 0
    first = (tmp_path / "plan.tsv").read_text(encoding="utf-8")
    with np.load(tmp_path / "alloc.npz", allow_pickle=False) as blob:
        sha = json.loads(str(blob["params"]))["alloc_sha"]
        k_store = blob["k_store"].copy()
    (tmp_path / "plan.tsv").unlink()
    (tmp_path / "alloc.npz").unlink()
    assert cli.main(argv) == 0
    assert (tmp_path / "plan.tsv").read_text(encoding="utf-8") == first
    with np.load(tmp_path / "alloc.npz", allow_pickle=False) as blob:
        assert json.loads(str(blob["params"]))["alloc_sha"] == sha
        assert np.array_equal(blob["k_store"], k_store)


def test_the_allocation_digest_catches_a_changed_word_set(tmp_path, monkeypatch):
    tok, _ = stub_pair()
    monkeypatch.setattr(common, "_tokenizer", lambda name: tok)
    words = [" the", " dog"]
    _write(tmp_path / "words.json", words)
    _write(tmp_path / "homes.json", {w: ["meco_en"] for w in words})
    assert (
        cli.main(
            [
                "plan",
                "--source",
                "mmc_pool",
                "--model-name",
                "stub",
                "--words",
                str(tmp_path / "words.json"),
                "--homes",
                str(tmp_path / "homes.json"),
                "--shards",
                "1",
                "--out",
                str(tmp_path / "plan.tsv"),
                "--alloc-out",
                str(tmp_path / "alloc.npz"),
            ]
        )
        == 0
    )
    _write(tmp_path / "homes.json", {" the": ["meco_tr"], " dog": ["meco_en"]})
    with pytest.raises(SystemExit, match="recorded digest"):
        plan._load_allocation(
            tmp_path / "alloc.npz", tmp_path / "homes.json", [" the", " dog"]
        )


def test_a_stage_skips_an_output_that_already_exists(tmp_path):
    out = tmp_path / "table.json"
    _write(out, {" the": 0.1})
    assert common._done(out, "merge") is True
    assert common._done(tmp_path / "missing.json", "merge") is False


def test_materialize_refuses_a_table_from_another_pool(tmp_path):
    _write(tmp_path / "table.json", {" the": 0.1})
    _write(tmp_path / "table.meta.json", {"pool_fingerprint": "pool-b"})
    with pytest.raises(SystemExit, match="names pool"):
        cli.main(
            [
                "materialize",
                "--source",
                "mmc_pool",
                "--model-name",
                "ai-forever/mGPT",
                "--table",
                str(tmp_path / "table.json"),
                "--datasets",
                "meco_en",
                "--out-dir",
                str(tmp_path / "pickles"),
                "--expect-fingerprint",
                "pool-a",
            ]
        )


def test_the_pooled_pickle_prefix_carries_the_pool_and_the_budget():
    class _Args:
        source = "mmc_pool"
        model_name = "ai-forever/mGPT"
        n_samples = 40_000_000
        k_eval = 4096
        seed = 100_000

    assert (
        materialize._pickle_prefix(_Args())
        == "ai-forever_mGPT_mmc_pps_unigram_surprisal_N40000000_k4096_s100000"
    )
    _Args.source = "mmc_pool_ee"
    assert (
        materialize._pickle_prefix(_Args())
        == "ai-forever_mGPT_mmc_pps_ee_unigram_surprisal_N40000000_k4096_s100000"
    )
    _Args.source = "mmc"
    assert (
        materialize._pickle_prefix(_Args()) == "ai-forever_mGPT_mmc_unigram_surprisal"
    )


@pytest.mark.parametrize(
    ("stamped", "source"),
    [(P.EE_ESTIMATOR_VERSION, "mmc_pool"), (P.ESTIMATOR_VERSION, "mmc_pool_ee")],
)
def test_materialize_publishes_a_table_only_under_its_own_family(
    tmp_path, stamped, source
):
    """The tag names the family, so a table stamped by the other numerator
    would land under a cell id that means something else."""
    _write(tmp_path / "table.json", {" the": 0.1})
    _write(
        tmp_path / "table.meta.json",
        {
            "pool_fingerprint": "pool-a",
            "n_pool_samples": 8,
            "k_eval": 8,
            "estimator_version": stamped,
        },
    )
    with pytest.raises(SystemExit, match="do not share a cell id"):
        cli.main(
            [
                "materialize",
                "--source",
                source,
                "--model-name",
                "ai-forever/mGPT",
                "--table",
                str(tmp_path / "table.json"),
                "--datasets",
                "meco_en",
                "--out-dir",
                str(tmp_path / "pickles"),
                "--expect-fingerprint",
                "pool-a",
                "--n-samples",
                "8",
                "--k-eval",
                "8",
            ]
        )


def test_merge_refuses_an_estimate_that_is_not_strictly_positive(tmp_path):
    """A zero publishes as NaN downstream, so it stops at the merge."""
    _write(tmp_path / "words.json", _WORDS)
    shard = _shard({" the": 0.1, " dog": 0.2})
    shard["records"][" dog"]["p_hat"] = 0.0
    _write(tmp_path / "shard_0.json", shard)
    with pytest.raises(SystemExit, match="not strictly positive"):
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_merge_refuses_shards_drawn_at_different_budgets(tmp_path):
    """One published cell carries one budget and one allocation."""
    _write(tmp_path / "words.json", _WORDS)
    first = _shard({" the": 0.1})
    second = _shard({" dog": 0.2})
    second["k_eval"] = 16
    _write(tmp_path / "shard_0.json", first)
    _write(tmp_path / "shard_1.json", second)
    argv = [
        "merge",
        "--source",
        "mmc_pool",
        "--shards",
        str(tmp_path / "shard_0.json"),
        str(tmp_path / "shard_1.json"),
        "--words",
        str(tmp_path / "words.json"),
        "--out",
        str(tmp_path / "table.json"),
    ]
    with pytest.raises(SystemExit, match="k_eval"):
        cli.main(argv)
    second["k_eval"] = 8
    second["allocation_sha"] = "other"
    _write(tmp_path / "shard_1.json", second)
    with pytest.raises(SystemExit, match="allocation_sha"):
        cli.main(argv)


def _audit_records(values):
    tok, _ = stub_pair()
    return {
        word: P.WordRecord.from_json(
            _record(word, value, token_ids=list(tok.encode(word)))
        )
        for word, value in values.items()
    }


def test_the_audit_compares_against_the_merged_table(tmp_path):
    """A re-run under another layout has to reproduce the published value."""
    _write(tmp_path / "table.json", {" the": 0.1, " dog": 0.2})
    table._compare_audit(_audit_records({" the": 0.1}), tmp_path / "table.json")
    with pytest.raises(SystemExit, match="differ from the merged table"):
        table._compare_audit(
            _audit_records({" dog": 0.2000000001}), tmp_path / "table.json"
        )
    with pytest.raises(SystemExit, match="absent from the merged table"):
        table._compare_audit(_audit_records({" cat": 0.3}), tmp_path / "table.json")


def test_materialize_refuses_a_tag_that_misstates_the_run(tmp_path):
    """N and k in the published tag come from the table."""
    _write(tmp_path / "table.json", {" the": 0.1})
    _write(
        tmp_path / "table.meta.json",
        {"pool_fingerprint": "pool-a", "n_pool_samples": 500, "k_eval": 8},
    )

    class _Args:
        source = "mmc_pool"
        table = tmp_path / "table.json"
        expect_fingerprint = "pool-a"
        n_samples = 500
        k_eval = 8

    materialize._check_expected_fingerprint(_Args())
    _Args.n_samples = 40_000_000
    with pytest.raises(SystemExit, match="n_pool_samples"):
        materialize._check_expected_fingerprint(_Args())
    _Args.n_samples = 500
    _Args.k_eval = 4096
    with pytest.raises(SystemExit, match="k_eval"):
        materialize._check_expected_fingerprint(_Args())


def test_merge_refuses_a_value_clamped_to_the_float64_floor(tmp_path):
    """A floored probability understates the surprisal the record carries."""
    _write(tmp_path / "words.json", _WORDS)
    shard = _shard({" the": 0.1, " dog": 0.2})
    shard["records"][" dog"]["p_hat"] = P.TINY
    shard["records"][" dog"]["underflow_fallback"] = True
    shard["records"][" dog"]["bits"] = 2390.0
    _write(tmp_path / "shard_0.json", shard)
    with pytest.raises(SystemExit, match="clamped to the float64 floor"):
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--out",
                str(tmp_path / "table.json"),
            ]
        )


def test_every_stage_defines_the_arguments_its_handler_reads():
    """A handler must not read an ``args`` attribute its subparser omits.

    The test walks the AST of every ``cmd_*`` handler in the MMC stage modules.
    """
    import argparse
    import ast
    import inspect

    handlers = {}
    for module in (cli, index, materialize, plan, pool, sweep, table, tally, estimate):
        tree = ast.parse(inspect.getsource(module))
        handlers.update(
            {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name.startswith("cmd_")
            }
        )
    # what each handler reads off args, following the helpers it hands args to
    reads: dict[str, set[str]] = {}
    for name, node in handlers.items():
        got: set[str] = set()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "args"
            ):
                got.add(sub.attr)
        reads[name] = got

    parser = cli.build_parser()
    actions = {}
    for action in parser._subparsers._group_actions:
        if isinstance(action, argparse._SubParsersAction):
            for cmd, sub in action.choices.items():
                dests = {a.dest for a in sub._actions}
                func = sub.get_default("func")
                if func is not None:
                    actions[func.__name__] = (cmd, dests)

    missing = []
    for handler, (cmd, dests) in sorted(actions.items()):
        for attr in sorted(reads.get(handler, set())):
            if attr not in dests and attr != "func":
                missing.append(f"{cmd}: {handler} reads args.{attr}, parser omits it")
    assert not missing, "\n".join(missing)


def test_the_plan_stores_boundaries_for_single_token_words_too(tmp_path, monkeypatch):
    """Every word's first token needs a stored buffer, single-token words included.

    The estimator charges each boundary the closing mass beta, a forward pass on
    that boundary's own prefix that cannot be folded into the column sum, so a
    single-token word whose first id has no buffer reaches ``_draw_cells`` with
    a budget and no cell to draw from.
    """
    tok, _ = stub_pair()
    monkeypatch.setattr(common, "_tokenizer", lambda name: tok)
    # " a" is one token; " the" and " cat" are several.
    words = [" a", " the", " cat"]
    _write(tmp_path / "words.json", words)
    _write(tmp_path / "homes.json", {w: ["meco_en"] for w in words})
    assert (
        cli.main(
            [
                "plan",
                "--source",
                "mmc_pool",
                "--model-name",
                "stub",
                "--words",
                str(tmp_path / "words.json"),
                "--homes",
                str(tmp_path / "homes.json"),
                "--shards",
                "2",
                "--k-eval",
                "40",
                "--k-floor",
                "1",
                "--k-home",
                "8",
                "--replicates",
                "4",
                "--out",
                str(tmp_path / "plan.tsv"),
                "--alloc-out",
                str(tmp_path / "alloc.npz"),
            ]
        )
        == 0
    )
    with np.load(tmp_path / "alloc.npz", allow_pickle=False) as blob:
        first_ids = set(blob["first_ids"].tolist())
        mt_first_ids = set(blob["mt_first_ids"].tolist())
        k_store = blob["k_store"]
    single = [w for w in words if len(tok.encode(w)) == 1]
    assert single, "the case under test needs a single-token word"
    for word in words:
        head = tok.encode(word)[0]
        assert head in first_ids, word
        assert head in mt_first_ids, f"{word!r} has no stored boundary buffer"
    assert k_store.shape[0] == len(mt_first_ids)


def test_merge_publishes_the_spaced_twin_under_the_bare_key(tmp_path):
    """A bare key is aliased to its spaced twin's record, so one type has one value.

    The exact count strips the separator before tallying, so the two keys name
    the same estimand; estimated independently they would carry two slightly
    different numbers for one quantity.
    """
    _write(tmp_path / "words.json", [" the", "the", " dog"])
    shard = _shard({" the": 0.1, " dog": 0.2, "the": 0.05})
    _write(tmp_path / "shard_0.json", shard)
    assert (
        cli.main(
            [
                "merge",
                "--source",
                "mmc_pool",
                "--shards",
                str(tmp_path / "shard_0.json"),
                "--words",
                str(tmp_path / "words.json"),
                "--pool-fingerprint",
                "pool-a",
                "--out",
                str(tmp_path / "table.json"),
            ]
        )
        == 0
    )
    table = json.loads((tmp_path / "table.json").read_text())
    assert table["the"] == table[" the"]
    lines = (tmp_path / "table.records.jsonl").read_text().splitlines()
    records = {json.loads(ln)["word"]: json.loads(ln) for ln in lines if ln}
    spaced, bare = dict(records[" the"]), dict(records["the"])
    spaced.pop("word"), bare.pop("word")
    assert spaced == bare


def test_the_audit_resolves_an_aliased_bare_key_to_its_spaced_twin():
    """The audit checks the spaced twin, which a re-run reproduces; the copy
    covers the bare key."""
    words = [" the", "the", " dog", "Birøkt"]
    subset = table._audit_subset(words, [0, 1, 3])
    assert subset == sorted({" the", "Birøkt"})


def test_materialize_serves_a_refused_word_as_missing(tmp_path):
    """A floor probability must not become a ~1074-bit covariate."""
    table = {" the": 0.1, "qx": P.TINY}
    rec = _record("qx", P.TINY)
    rec["nonpositive"] = True
    (tmp_path / "table.records.jsonl").write_text(json.dumps(rec) + "\n")
    materialize._null_refused(table, tmp_path / "table.json")
    assert math.isnan(table["qx"])
    assert table[" the"] == 0.1
    assert math.isnan(materialize._TableSource(table).probability("qx"))


def test_the_start_ids_argument_is_one_token_id():
    assert common.start_ids_arg("5") == [5]
    assert common.start_ids_arg(" 5 ") == [5]
    with pytest.raises(
        argparse.ArgumentTypeError, match="exactly one non-negative token id"
    ):
        common.start_ids_arg("5,6")
    with pytest.raises(
        argparse.ArgumentTypeError, match="exactly one non-negative token id"
    ):
        common.start_ids_arg("")


def test_the_sweep_merge_reads_the_pool_prefix_from_the_manifest_and_checks_the_store():
    """A recorded encoding names the prefix; an encoding-less archive is read
    under the tokenizer's convention and its row heads are verified."""
    tok, _ = stub_pair()
    assert sweep._pool_start({"encoding": {"start": [5]}}, tok) == [5]
    assert sweep._pool_start({"model_name": "x"}, tok) == [tok.bos_token_id]
    assert sweep._pool_start({"encoding": {"format": "x"}}, tok) == [tok.bos_token_id]
    tokens = np.array([5, 1, 2, 5, 3], dtype=np.int32)
    offsets = np.array([0, 3, 5], dtype=np.int64)
    sweep._check_store_start(tokens, offsets, [5])
    with pytest.raises(SystemExit, match="recorded prefix"):
        sweep._check_store_start(tokens, offsets, [1])
    with pytest.raises(SystemExit, match="non-empty rows"):
        sweep._check_store_start(tokens, np.array([0, 5, 5], dtype=np.int64), [5])
    sweep._check_store_start(tokens[:0], np.array([0], dtype=np.int64), [5])
