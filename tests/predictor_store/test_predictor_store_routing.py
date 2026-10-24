"""Predictor readers resolve per-cell files through the store index.

Network tests skip when the store is unavailable.
"""

import numpy as np
import pyarrow as pa
import pytest

from replica import predictors


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch, request):
    if request.node.get_closest_marker("needs_network") is None:
        # Local routing fixtures do not declare method metadata.
        monkeypatch.setattr(predictors, "_index_method", lambda _: None)
    cached = (
        predictors._tag_index,
        predictors._indexed_table,
        predictors._cell_subtable,
    )
    for function in cached:
        function.cache_clear()
    yield
    for function in cached:
        function.cache_clear()


def _index_or_skip() -> dict[str, tuple[str, ...]]:
    index = predictors._tag_index()
    if not index:
        pytest.skip("predictor store index unreachable")
    return index


@pytest.mark.needs_network
def test_index_records_subfolder_and_multi_file_paths():
    """The index records a cell's subfolder and every file it spans.

    A ``.shard<a>_<b>`` suffix never reaches the store: it names a stimulus
    slice written during compute, ``merge_shards.py`` stitches those back into
    the canonical pickle before publication, and the reader strips the suffix
    from a query tag. What the router does need is the subfolder a cell sits
    in, and every file a cell spanning more than one is published across.
    """
    index = _index_or_skip()
    nested = [files for files in index.values() if any("/" in name for name in files)]
    assert nested, "no subfolder cell in the index"
    multi_file = [files for files in index.values() if len(files) > 1]
    assert multi_file, "a cell published across several files must keep every path"
    assert all(len(set(files)) == len(files) for files in index.values())


@pytest.mark.needs_network
def test_subfolder_scalar_cell_reads_back():
    index = _index_or_skip()
    tag = "gpt2_responsivity_N512_s0_provo"
    if tag not in index:
        pytest.skip(f"{tag} is not published")
    outputs = predictors.load_outputs_for_tag(tag)
    assert outputs, "subfolder cell came back empty"
    keys = {k for units in outputs.values() for unit in units for k in unit.values}
    assert keys == {
        "decontextualised_distance",
        "expected_decontextualised_distance",
    }


@pytest.mark.needs_network
def test_array_cell_is_not_served_as_scalars():
    """An array family has no ``predictor``/``value`` columns, so the scalar
    route declines it and the caller still gets the documented error."""
    index = _index_or_skip()
    tag = next(
        (
            t
            for t, files in index.items()
            if any(name.startswith("next_log_probs/") for name in files)
        ),
        None,
    )
    if tag is None:
        pytest.skip("no next_log_probs cell published")
    assert predictors._indexed_scalar_cell(tag) is None


@pytest.mark.needs_network
def test_missing_cell_still_raises_file_not_found():
    _index_or_skip()
    with pytest.raises(FileNotFoundError):
        predictors.load_outputs_for_tag("gpt2_responsivity_N512_s0_nosuchcorpus")


def test_scalar_columns_decide_the_shape(monkeypatch):
    """The table schema decides the route."""
    scalar = pa.table(
        {
            "tag": ["t"],
            "stimulus_id": ["s"],
            "unit_index": [0],
            "predictor": ["p"],
            "value": [1.0],
        }
    )
    arrays = pa.table({"stimulus_id": ["s"], "unit_index": [0], "samples": [[1.0]]})
    monkeypatch.setattr(predictors, "_tag_index", lambda: {"t": ("sub/t.parquet",)})
    for table, expected in ((scalar, 1), (arrays, None)):
        monkeypatch.setattr(
            predictors, "_indexed_table", lambda _f, _c=None, _t=table: _t
        )
        got = predictors._indexed_scalar_cell("t")
        assert (None if got is None else got.num_rows) == expected


def test_index_routed_array_cell_concatenates_shards(monkeypatch):
    """Any subfolder file with one data column reads back as arrays, shards
    concatenated, with no entry in the marker table."""
    shards = {
        "lens/t.shard00000_00001.parquet": pa.table(
            {"stimulus_id": ["s1"], "unit_index": [0], "layers": [[1.0, 2.0]]}
        ),
        "lens/t.shard00001_00002.parquet": pa.table(
            {"stimulus_id": ["s2"], "unit_index": [0], "layers": [[3.0, 4.0]]}
        ),
    }
    monkeypatch.setattr(predictors, "_tag_index", lambda: {"t": tuple(sorted(shards))})
    monkeypatch.setattr(predictors, "_indexed_table", lambda f, _c=None: shards[f])
    out = predictors.load_samples_for_tag("t")
    assert sorted(out) == ["s1", "s2"]
    assert out["s1"][0].tolist() == [1.0, 2.0]
    assert out["s2"][0].tolist() == [3.0, 4.0]


def test_multi_array_cell_needs_an_explicit_column(monkeypatch):
    bad = pa.table(
        {"stimulus_id": ["s"], "unit_index": [0], "a": [[1.0]], "b": [[2.0]]}
    )
    monkeypatch.setattr(predictors, "_tag_index", lambda: {"t": ("sub/t.parquet",)})
    monkeypatch.setattr(predictors, "_indexed_table", lambda f, _c=None: bad)
    with pytest.raises(ValueError, match="pass\\s+column="):
        predictors.load_samples_for_tag("t")
    out = predictors.load_samples_for_tag("t", column="b")
    assert out["s"][0].tolist() == [2.0]


def test_marker_fallback_survives_an_empty_index(monkeypatch):
    """Offline (empty index), the original three families still resolve."""
    table = pa.table({"stimulus_id": ["s"], "unit_index": [0], "logprobs": [[0.5]]})
    monkeypatch.setattr(predictors, "_tag_index", lambda: {})
    monkeypatch.setattr(predictors, "_samples_table", lambda sub, tag, col: table)
    out = predictors.load_samples_for_tag("gpt2_ft_entropy_provo")
    assert out["s"][0].tolist() == [0.5]


@pytest.mark.needs_network
def test_published_array_cell_reads_via_the_index():
    index = _index_or_skip()
    tag = next(
        (
            t
            for t, files in index.items()
            if any(name.startswith("next_log_probs/") for name in files)
        ),
        None,
    )
    if tag is None:
        pytest.skip("no next_log_probs cell published")
    routed = predictors._indexed_array_cell(tag)
    assert routed is not None
    table, column = routed
    assert column == "logprobs"
    assert table.num_rows


def test_indexed_scalar_file_filters_to_the_requested_tag(monkeypatch):
    table = pa.table(
        {
            "tag": ["wanted", "other"],
            "stimulus_id": ["s", "s"],
            "unit_index": [0, 0],
            "predictor": ["p", "p"],
            "value": [1.0, 99.0],
        }
    )
    monkeypatch.setattr(
        predictors,
        "_tag_index",
        lambda: {"wanted": ("sub/shared.parquet",), "other": ("sub/shared.parquet",)},
    )
    monkeypatch.setattr(predictors, "_indexed_table", lambda *args: table)
    result = predictors._reconstruct(predictors._indexed_scalar_cell("wanted"))
    assert result["s"][0].values["p"] == 1.0


def test_duplicate_scalar_and_array_rows_are_rejected():
    scalar = pa.table(
        {
            "stimulus_id": ["s", "s"],
            "unit_index": [0, 0],
            "predictor": ["p", "p"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate predictor row"):
        predictors._reconstruct(scalar)
    arrays = pa.table(
        {"stimulus_id": ["s", "s"], "unit_index": [0, 0], "samples": [[1.0], [2.0]]}
    )
    with pytest.raises(ValueError, match="Duplicate predictor array"):
        predictors._reconstruct_arrays(arrays, "samples")


@pytest.mark.parametrize(("second_unit", "second_predictor"), [(0, "b"), (1, "a")])
def test_scalar_reader_keeps_distinct_stimulus_ids(
    monkeypatch, second_unit, second_predictor
):
    table = pa.table(
        {
            "tag": ["t", "t"],
            "stimulus_id": ["trial_1", "trial_1.0"],
            "unit_index": [0, second_unit],
            "predictor": ["a", second_predictor],
            "value": [1.0, 2.0],
            "text_unit": ["cat", "dog"],
        }
    )
    monkeypatch.setattr(predictors, "_tag_index", lambda: {"t": ("sub/t.parquet",)})
    monkeypatch.setattr(predictors, "_indexed_table", lambda *_: table)
    out = predictors.load_outputs_for_tag("t")
    assert set(out) == {"trial_1", "trial_1.0"}
    assert out["trial_1"][0].values == {"a": 1.0}
    assert out["trial_1"][0].metadata == {"text_unit": "cat"}
    assert out["trial_1.0"][second_unit].values == {second_predictor: 2.0}
    assert out["trial_1.0"][second_unit].metadata == {"text_unit": "dog"}
    if second_unit:
        assert out["trial_1.0"][0].values == {}


def test_array_reader_keeps_distinct_stimulus_ids(monkeypatch):
    table = pa.table(
        {
            "stimulus_id": ["trial_1", "trial_1.0"],
            "unit_index": [0, 1],
            "samples": pa.array([[2**63 + 1], [2**63 + 2]], pa.list_(pa.uint64())),
        }
    )
    monkeypatch.setattr(predictors, "_tag_index", lambda: {"t": ("sub/t.parquet",)})
    monkeypatch.setattr(predictors, "_indexed_table", lambda *_: table)
    out = predictors.load_samples_for_tag("t")
    assert set(out) == {"trial_1", "trial_1.0"}
    assert out["trial_1"][0].tolist() == [2**63 + 1]
    assert out["trial_1.0"][0] is None
    assert out["trial_1.0"][1].tolist() == [2**63 + 2]


@pytest.mark.parametrize(
    "texts",
    [("cat", "dog"), ("dog", "cat"), ("cat", "cat"), ("cat", None), (None, "cat")],
)
def test_scalar_predictors_require_consistent_unit_text(texts):
    table = pa.table(
        {
            "stimulus_id": ["s", "s"],
            "unit_index": [0, 0],
            "predictor": ["a", "b"],
            "value": [1.0, 2.0],
            "text_unit": texts,
        }
    )
    if len(set(texts) - {None}) > 1:
        with pytest.raises(ValueError, match="Conflicting predictor text_unit"):
            predictors._reconstruct(table)
    else:
        (row,) = predictors._reconstruct(table)["s"]
        assert row.values == {"a": 1.0, "b": 2.0}
        assert row.metadata == {"text_unit": "cat"}


@pytest.mark.parametrize(
    ("values", "kind", "dtype"),
    [
        ([[[0.2, 0.4], [0.6, 0.8]], []], pa.list_(pa.list_(pa.float64())), np.float64),
        ([[1, 2**63 + 1, 2**64 - 1]], pa.list_(pa.uint64()), np.uint64),
    ],
)
def test_array_reader_preserves_matrices_and_integer_identities(
    monkeypatch, values, kind, dtype
):
    table = pa.table(
        {
            "tag": ["t"] * len(values),
            "stimulus_id": ["s"] * len(values),
            "unit_index": list(range(len(values))),
            "samples": pa.array(values, type=kind),
        }
    )
    monkeypatch.setattr(predictors, "_tag_index", lambda: {"t": ("arrays/t.parquet",)})
    monkeypatch.setattr(predictors, "_indexed_table", lambda _: table)
    actual = predictors.load_samples_for_tag("t")["s"]
    assert actual[0].dtype == dtype
    np.testing.assert_array_equal(actual[0], np.asarray(values[0], dtype=dtype))
    if len(values) == 2:
        assert actual[1].shape == (0, 0)


def test_array_reader_rejects_ragged_matrices():
    table = pa.table(
        {
            "stimulus_id": ["s"],
            "unit_index": [0],
            "samples": [[[1.0], [2.0, 3.0, 4.0]]],
        }
    )
    with pytest.raises(ValueError, match="rectangular"):
        predictors._reconstruct_arrays(table, "samples")


@pytest.mark.parametrize("row", [[[1.0, None]], [None]])
def test_array_reader_rejects_nested_nulls(row):
    table = pa.table(
        {
            "stimulus_id": ["s"],
            "unit_index": [0],
            "samples": pa.array([row], type=pa.list_(pa.list_(pa.float64()))),
        }
    )
    with pytest.raises(ValueError, match="nested nulls"):
        predictors._reconstruct_arrays(table, "samples")


def test_array_reader_preserves_empty_leading_dimensions_and_missing_cells():
    table = pa.table(
        {
            "stimulus_id": ["s", "s"],
            "unit_index": [0, 1],
            "samples": pa.array(
                [[[], []], None], type=pa.list_(pa.list_(pa.list_(pa.float64())))
            ),
        }
    )
    rows = predictors._reconstruct_arrays(table, "samples")["s"]
    assert rows[0].shape == (2, 0, 0)
    assert rows[1] is None


def test_tagless_shared_array_file_cannot_claim_two_cells(monkeypatch):
    arrays = pa.table({"stimulus_id": ["s"], "unit_index": [0], "samples": [[1.0]]})
    monkeypatch.setattr(
        predictors,
        "_tag_index",
        lambda: {"a": ("sub/shared.parquet",), "b": ("sub/shared.parquet",)},
    )
    monkeypatch.setattr(predictors, "_indexed_table", lambda *args: arrays)
    with pytest.raises(ValueError, match="ambiguous ownership"):
        predictors._indexed_array_cell("a")


@pytest.mark.parametrize("index", [-1, None, 0.5])
def test_invalid_unit_indices_are_rejected(index):
    scalar = pa.table(
        {
            "stimulus_id": ["s"],
            "unit_index": [index],
            "predictor": ["p"],
            "value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="Invalid predictor row"):
        predictors._reconstruct(scalar)


def test_store_downloads_use_one_resolved_revision(monkeypatch):
    from types import SimpleNamespace

    from replica.external import hub as assets
    from replica.feature.frequency.registry import default_owt_counts

    assets._store_revision.cache_clear()
    calls = []
    resolved = []
    head = "a" * 40

    class Hub:
        def repo_info(self, *args, **kwargs):
            resolved.append(kwargs)
            return SimpleNamespace(sha=head)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("PSYCHO_OWT_COUNTS", raising=False)
    monkeypatch.setattr(assets, "HF_REVISION", "main")
    monkeypatch.setattr(assets, "HfApi", Hub)
    monkeypatch.setattr(
        assets,
        "hf_hub_download",
        lambda repo, filename, **kwargs: calls.append((filename, kwargs)) or filename,
    )
    try:
        predictors._hub_file("tags.parquet")
        report = predictors.predictor_store_identity()
        report["revision"] = "b" * 40
        head = "c" * 40
        assert default_owt_counts() == "owt/unigram_counts.tsv"
        predictors._hub_file("provo.parquet")
        assert len(resolved) == 1
        assert [filename for filename, _ in calls] == [
            "tags.parquet",
            "owt/unigram_counts.tsv",
            "provo.parquet",
        ]
        assert [call["revision"] for _, call in calls] == ["a" * 40] * 3
        assert predictors.predictor_store_identity()["revision"] == "a" * 40
    finally:
        assets._store_revision.cache_clear()


def test_a_token_grain_sample_cell_is_not_blocked_by_its_scalar_table(monkeypatch):
    """A cell indexed at both its array file and the token table still loads.

    A token-grain sample cell writes its scalars into
    ``tokens/<dataset>.parquet`` and its per-token sample pool into its own
    family file, so the index names both. The token table carries no list
    column, so the reader loads the pool from the family file.
    """
    samples = pa.table(
        {
            "tag": pa.array(["t", "t"]),
            "stimulus_id": pa.array(["s", "s"]),
            "unit_index": pa.array([0, 1], pa.int32()),
            "samples": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64())),
        }
    )
    scalars = pa.table(
        {
            "tag": pa.array(["t"]),
            "predictor": pa.array(["continuation_entropy"]),
            "stimulus_id": pa.array(["s"]),
            "token_index": pa.array([0], pa.int32()),
            "value": pa.array([1.5]),
        }
    )
    files = {
        "cont_entropy_samples/t.parquet": samples,
        "tokens/toyset.parquet": scalars,
    }
    monkeypatch.setattr(predictors, "_tag_index", lambda: {"t": tuple(sorted(files))})
    monkeypatch.setattr(predictors, "_indexed_table", lambda f, _c=None: files[f])
    loaded = predictors.load_samples_for_tag("t")
    assert [list(a) for a in loaded["s"]] == [[1.0, 2.0], [3.0, 4.0]]
    assert predictors.load_samples_for_tag("t", column="samples")["s"][1][1] == 4.0


def test_a_cell_indexed_only_at_scalar_files_still_serves_no_arrays(monkeypatch):
    scalars = pa.table(
        {
            "tag": pa.array(["t"]),
            "predictor": pa.array(["surprisal"]),
            "stimulus_id": pa.array(["s"]),
            "unit_index": pa.array([0], pa.int32()),
            "value": pa.array([1.5]),
        }
    )
    monkeypatch.setattr(
        predictors, "_tag_index", lambda: {"t": ("responsivity/t.parquet",)}
    )
    monkeypatch.setattr(predictors, "_indexed_table", lambda f, _c=None: scalars)
    assert predictors._indexed_array_cell("t") is None


_OLD, _NEW = "gpt2_next_word_entropy_provo", "gpt2_restricted_first_token_entropy_provo"


def _table_under_old_name():
    return pa.table(
        {
            "tag": [_OLD, "other"],
            "stimulus_id": ["s", "s"],
            "unit_index": [0, 0],
            "text_unit": ["w", "w"],
            "predictor": ["next_word_entropy", "p"],
            "value": [1.0, 99.0],
        }
    )


def test_the_index_keys_cells_by_their_current_name(monkeypatch, tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "tags.parquet"
    pq.write_table(
        pa.table({"tag": [_OLD, _OLD], "file": ["provo.parquet", "sub/x.parquet"]}),
        path,
    )
    monkeypatch.setattr(predictors, "_hub_file", lambda name: str(path))
    assert predictors._tag_index() == {_NEW: ("provo.parquet", "sub/x.parquet")}


def test_an_indexed_cell_under_an_earlier_name_reads_under_the_current_one(
    monkeypatch,
):
    monkeypatch.setattr(
        predictors,
        "_tag_index",
        lambda: {_NEW: ("sub/shared.parquet",), "other": ("sub/shared.parquet",)},
    )
    monkeypatch.setattr(
        predictors, "_indexed_table", lambda *args: _table_under_old_name()
    )
    result = predictors._reconstruct(predictors._indexed_scalar_cell(_NEW))
    assert result["s"][0].values == {"restricted_first_token_entropy": 1.0}


def test_the_per_corpus_table_serves_a_cell_under_an_earlier_name(monkeypatch):
    monkeypatch.setattr(predictors, "_tag_index", lambda: {})
    monkeypatch.setattr(
        predictors, "_dataset_table", lambda dataset: _table_under_old_name()
    )
    result = predictors.load_outputs_for_tag(_NEW)
    assert result["s"][0].values == {"restricted_first_token_entropy": 1.0}
