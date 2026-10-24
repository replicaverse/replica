"""Store readers refuse a declared reproduction-only cell without consent."""

import json
import pickle
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import replica.dataset.base as dataset_base
from replica import predictors
from replica.artifacts import manifest_path, write_manifest
from replica.core import AnnotationLayer, Stimulus
from replica.dataset.catalog.provenance import corpus_fingerprint
from replica.feature import FeatureModelOutput
from replica.feature.core.reproduction import ReproductionOnlyWarning

BUGGY = "gpt2_renyi05_buggy_provo"
CORRECTED = "gpt2_surprisal_provo"


def test_a_prior_gate_does_not_replace_explicit_consent():
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.check_reproduction_consent(
            BUGGY, allow_reproduction=False, gated="renyi_entropy_buggy"
        )


def _rows(tag):
    return pa.table(
        {
            "tag": [tag, tag],
            "stimulus_id": ["s", "s"],
            "unit_index": [0, 1],
            "predictor": ["x", "x"],
            "value": [1.0, 2.0],
            "text_unit": ["a", "b"],
        }
    )


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setattr(predictors, "_tag_index", lambda: {})
    monkeypatch.setattr(predictors, "_index_method", lambda tag: None)
    monkeypatch.setattr(predictors, "_cell_subtable", lambda dataset, tag: _rows(tag))
    monkeypatch.setattr(predictors, "_indexed_scalar_cell", lambda tag: None)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(predictors, "_tag_index", lambda: {})
    monkeypatch.setattr(predictors, "_index_method", lambda tag: None)

    def unreachable(dataset, tag):
        raise OSError("hub unreachable")

    monkeypatch.setattr(predictors, "_cell_subtable", unreachable)
    monkeypatch.setattr(predictors, "_indexed_scalar_cell", lambda tag: None)


def _local_cell(tmp_path, monkeypatch, name, method):
    units = ["a", "b"]
    stimulus = Stimulus(text="a b", stimulus_id="s")
    stimulus.add_layer(
        AnnotationLayer("words", dataset_base.build_word_spans(units), units)
    )
    monkeypatch.setattr(predictors, "_corpus_stimuli", lambda dataset: [stimulus])
    path = tmp_path / f"{name}.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {"s": [FeatureModelOutput(values={"x": 1.0}) for _ in units]}, stream
        )
    identity = {
        "schema_version": 1,
        "method": method,
        "options": {"unit": "words", "context_scope": "stimulus"},
        "corpus_sha256": corpus_fingerprint("provo", [stimulus], "words", "stimulus"),
    }
    write_manifest(path, identity, status="completed")
    return path


def test_hub_cell_needs_consent_and_then_warns(hub):
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_outputs_for_tag(BUGGY)
    with pytest.warns(ReproductionOnlyWarning):
        data = predictors.load_outputs_for_tag(BUGGY, allow_reproduction=True)
    assert len(data["s"]) == 2


def test_corrected_hub_cell_loads_without_consent_or_warning(hub):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(predictors.load_outputs_for_tag(CORRECTED)["s"]) == 2


def test_local_cell_is_gated_by_its_name(offline, tmp_path, monkeypatch):
    path = _local_cell(tmp_path, monkeypatch, BUGGY, "renyi_entropy_buggy")
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_outputs_for_tag(BUGGY, local_path=path)
    with pytest.warns(ReproductionOnlyWarning) as caught:
        data = predictors.load_outputs_for_tag(
            BUGGY, local_path=path, allow_reproduction=True
        )
    assert len(data["s"]) == 2
    assert len([w for w in caught if w.category is ReproductionOnlyWarning]) == 1


def test_unrecognized_name_is_gated_by_its_manifest(offline, tmp_path, monkeypatch):
    path = _local_cell(
        tmp_path, monkeypatch, "custom_cell_provo", "renyi_entropy_buggy"
    )
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_outputs_for_tag("custom_cell_provo", local_path=path)
    with pytest.warns(ReproductionOnlyWarning):
        predictors.load_outputs_for_tag(
            "custom_cell_provo", local_path=path, allow_reproduction=True
        )


@pytest.mark.parametrize("name", [CORRECTED, "gpt2_ft_entropy_buggy_provo"])
def test_a_name_and_manifest_that_disagree_never_load(
    offline, tmp_path, monkeypatch, name
):
    path = _local_cell(tmp_path, monkeypatch, name, "renyi_entropy_buggy")
    for consent in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ReproductionOnlyWarning)
            with pytest.raises(ValueError, match="disagree|allow_reproduction"):
                predictors.load_outputs_for_tag(
                    name, local_path=path, allow_reproduction=consent
                )
        if consent:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ReproductionOnlyWarning)
                with pytest.raises(ValueError, match="disagree"):
                    predictors.load_outputs_for_tag(
                        name, local_path=path, allow_reproduction=True
                    )


@pytest.mark.parametrize(
    "method", ["", "  ", 5, None, " renyi_entropy_buggy", "renyi_entropy_buggy\n"]
)
def test_a_manifest_with_an_invalid_method_is_refused(method):
    with pytest.raises(ValueError, match="no valid method"):
        predictors.declared_method(CORRECTED, {"method": method})
    assert predictors.declared_method(CORRECTED, {}) == "surprisal"


def test_a_malformed_manifest_is_still_refused(offline, tmp_path, monkeypatch):
    path = _local_cell(tmp_path, monkeypatch, CORRECTED, "surprisal")
    manifest_path(path).write_text(json.dumps({"status": "completed"}))
    with pytest.raises(ValueError, match="invalid artifact manifest"):
        predictors.load_outputs_for_tag(CORRECTED, local_path=path)


def test_sample_cells_are_gated_before_any_read(monkeypatch):
    tag = "gpt2_cont_entropy_buggy_N512_L5_s0_devarda2023"
    reached = []

    def record(sub, tag, column):
        reached.append(tag)
        raise RuntimeError("stop")

    monkeypatch.setattr(predictors, "_tag_index", lambda: {})
    monkeypatch.setattr(predictors, "_samples_table", record)
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_samples_for_tag(tag)
    assert reached == []
    with pytest.warns(ReproductionOnlyWarning), pytest.raises(RuntimeError):
        predictors.load_samples_for_tag(tag, allow_reproduction=True)
    assert reached == [tag]


LOOKAHEAD = "gpt2_fwd_word_lookahead_entropy_N64_M5_K3_s0_provo"


def _array_cell(monkeypatch, tag, hash_type):
    monkeypatch.setattr(predictors, "_index_method", lambda tag: None)
    table = pa.table(
        {
            "stimulus_id": ["s", "s"],
            "unit_index": [0, 1],
            "id_hash": pa.array([[1, 2], [3]], type=pa.list_(hash_type)),
            "samples": pa.array([[0.5, 0.25], [1.0]], type=pa.list_(pa.float64())),
        }
    )
    monkeypatch.setattr(predictors, "_tag_index", lambda: {tag: ("look/t.parquet",)})
    monkeypatch.setattr(predictors, "_indexed_table", lambda f, _c=None: table)


def test_truncated_lookahead_hashes_warn_on_the_public_sample_path(monkeypatch):
    _array_cell(monkeypatch, LOOKAHEAD, pa.uint32())
    with pytest.warns(UserWarning, match="32-bit truncated word hashes"):
        out = predictors.load_samples_for_tag(LOOKAHEAD, column="id_hash")
    assert out["s"][0].tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("tag", "hash_type", "column"),
    [
        (LOOKAHEAD, pa.uint64(), "id_hash"),
        (LOOKAHEAD, pa.uint32(), "samples"),
        ("gpt2_responsivity_N64_s0_provo", pa.uint32(), "id_hash"),
    ],
)
def test_full_hashes_other_columns_and_other_cells_do_not_warn(
    monkeypatch, tag, hash_type, column
):
    _array_cell(monkeypatch, tag, hash_type)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "s" in predictors.load_samples_for_tag(tag, column=column)


def _caches():
    return (
        predictors._tag_index,
        predictors._index_methods,
        predictors._indexed_table,
        predictors._cell_subtable,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A local store: index rows ``(tag, file, method)`` and the files they name."""
    root = tmp_path / "store"
    (root / "cells").mkdir(parents=True)

    def build(rows, files, *, index_methods=True):
        columns = {"tag": [r[0] for r in rows], "file": [r[1] for r in rows]}
        if index_methods:
            columns["method"] = [r[2] for r in rows]
        pq.write_table(pa.table(columns), root / "tags.parquet")
        for name, table in files.items():
            pq.write_table(table, root / name)
        monkeypatch.setattr(predictors, "_hub_file", lambda name: str(root / name))
        for cache in _caches():
            cache.cache_clear()
        return root

    yield build
    for cache in _caches():
        cache.cache_clear()


def _scalar(tag):
    return pa.table(
        {
            "tag": [tag],
            "stimulus_id": ["s"],
            "unit_index": [0],
            "predictor": ["x"],
            "value": [1.0],
        }
    )


def _array(tag):
    return pa.table(
        {"tag": [tag], "stimulus_id": ["s"], "unit_index": [0], "samples": [[1.0, 2.0]]}
    )


def _consented(load, *args, **kwargs):
    with pytest.warns(ReproductionOnlyWarning) as caught:
        out = load(*args, allow_reproduction=True, **kwargs)
    assert len([w for w in caught if w.category is ReproductionOnlyWarning]) == 1
    return out


def test_index_declared_method_gates_custom_scalar_and_array_tags(store):
    store(
        [
            ("custom_scalar", "cells/scalar.parquet", "renyi_entropy_buggy"),
            ("custom_array", "cells/array.parquet", "continuation_entropy_buggy"),
        ],
        {
            "cells/scalar.parquet": _scalar("custom_scalar"),
            "cells/array.parquet": _array("custom_array"),
        },
    )
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_outputs_for_tag("custom_scalar")
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_samples_for_tag("custom_array")
    assert "s" in _consented(predictors.load_outputs_for_tag, "custom_scalar")
    assert "s" in _consented(predictors.load_samples_for_tag, "custom_array")


def test_indexed_filename_declares_the_method_when_the_index_does_not(store):
    store(
        [("custom_scalar", "cells/gpt2_renyi05_buggy_provo.parquet", None)],
        {"cells/gpt2_renyi05_buggy_provo.parquet": _scalar("custom_scalar")},
        index_methods=False,
    )
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_outputs_for_tag("custom_scalar")
    assert "s" in _consented(predictors.load_outputs_for_tag, "custom_scalar")


def test_an_undeclared_custom_cell_in_a_legacy_index_still_loads(store):
    store(
        [("custom_scalar", "cells/custom.parquet", None)],
        {"cells/custom.parquet": _scalar("custom_scalar")},
        index_methods=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "s" in predictors.load_outputs_for_tag("custom_scalar")


def test_local_filename_declares_the_method_behind_an_alias(store, tmp_path):
    root = store([], {})
    path = root / "gpt2_renyi05_buggy_provo.pkl"
    path.write_bytes(pickle.dumps({"s": []}))
    write_manifest(
        path, {"schema_version": 1, "corpus_sha256": "legacy"}, status="completed"
    )
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_outputs_for_tag("custom_alias", local_path=path)
    assert _consented(
        predictors.load_outputs_for_tag, "custom_alias", local_path=path
    ) == {"s": []}


def test_index_and_name_that_disagree_never_load(store):
    store(
        [("gpt2_renyi05_buggy_provo", "cells/scalar.parquet", "surprisal")],
        {"cells/scalar.parquet": _scalar("gpt2_renyi05_buggy_provo")},
    )
    with pytest.raises(ValueError, match="allow_reproduction=True"):
        predictors.load_outputs_for_tag("gpt2_renyi05_buggy_provo")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReproductionOnlyWarning)
        with pytest.raises(ValueError, match="disagree"):
            predictors.load_outputs_for_tag(
                "gpt2_renyi05_buggy_provo", allow_reproduction=True
            )


def test_an_index_method_with_surrounding_whitespace_is_refused(store):
    store(
        [("custom_scalar", "cells/scalar.parquet", " renyi_entropy_buggy")],
        {"cells/scalar.parquet": _scalar("custom_scalar")},
    )
    with pytest.raises(ValueError, match="no valid method"):
        predictors.load_outputs_for_tag("custom_scalar", allow_reproduction=True)
