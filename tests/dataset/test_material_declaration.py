"""Dataset metadata defines material identities independently of file location."""

import numpy as np
import pandas as pd
import pytest

from replica.analysis.cv.materials import material_id
from replica.dataset.base import Dataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.eye_tracking.copco import CopCo
from replica.dataset.eye_tracking.emtec import EMTeC
from replica.dataset.eye_tracking.meco.waves import MECOL1W2, MECOL2W1, MECOL2W2
from replica.dataset.eye_tracking.onestop import OneStop
from replica.dataset.eye_tracking.sbsat import SBSAT
from replica.dataset.metadata.io import (
    ChecksumMismatchError,
    from_dict,
    load_yaml,
    save_yaml,
    to_dict,
)


class _Loader(Dataset):
    def __init__(self, info=None):
        super().__init__()
        self.info = (
            info
            if info is not None
            else DatasetInfo(
                name="books",
                material_namespace="book",
                material_fields=["title", "edition"],
            )
        )

    def load(self, path):
        raise NotImplementedError


def _built(stimulus_id, metadata, *, info=None):
    return _Loader(info)._make_stimulus(["a", "b"], stimulus_id, metadata)


def test_declared_fields_derive_the_material_identity():
    one = _built("p1", {"title": "Emma", "edition": 2, "page": 1})
    two = _built("p2", {"title": "Emma", "edition": 2, "page": 2})
    other = _built("p3", {"title": "Emma", "edition": 3, "page": 1})
    assert material_id(one) == material_id(two) == "book:Emma:2"
    assert material_id(other) == "book:Emma:3"
    assert one.metadata["page"] == 1


def test_explicit_key_wins_without_deriving():
    explicit = _built("p", {"cv_group": "x"})
    assert material_id(explicit) == "x"


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"title": "Emma"}, "material fields \\['edition'\\]"),
        ({"title": "", "edition": 1}, "'title' is blank"),
        ({"title": None, "edition": 1}, "'title' is missing"),
        ({"title": "Emma", "edition": float("nan")}, "'edition' is missing"),
        ({"title": "Emma", "edition": np.float32("nan")}, "'edition' is missing"),
        ({"title": pd.NA, "edition": 1}, "'title' is missing"),
        ({"title": pd.NaT, "edition": 1}, "'title' is missing"),
        ({"title": ["Emma"], "edition": 1}, "'title' is not a scalar"),
        ({"title": {"name": "Emma"}, "edition": 1}, "'title' is not a scalar"),
    ],
)
def test_absent_blank_or_missing_values_raise(metadata, message):
    with pytest.raises(ValueError, match=message):
        _built("p", metadata)


def test_separator_in_a_value_cannot_collide():
    left = _built("p1", {"title": "a:b", "edition": "c"})
    right = _built("p2", {"title": "a", "edition": "b:c"})
    assert material_id(left) != material_id(right)
    assert material_id(left) == "book:a\\:b:c"
    slash = _built("p3", {"title": "a\\", "edition": ":c"})
    assert material_id(slash) == "book:a\\\\:\\:c"


@pytest.mark.parametrize("namespace", ["", " ", "a:b", "a\\b", None, 7])
def test_blank_or_reserved_namespaces_are_rejected(namespace):
    info = DatasetInfo(
        name="books", material_namespace=namespace, material_fields=["title"]
    )
    with pytest.raises(ValueError, match="material_namespace"):
        _built("p", {"title": "Emma"}, info=info)


@pytest.mark.parametrize("fields", ["title", "", None, {}, [""], [1]])
def test_malformed_material_fields_are_not_ignored(fields):
    info = from_dict({"name": "books", "material_fields": fields})
    with pytest.raises(ValueError, match="material_fields"):
        _built("p", {"title": "Emma"}, info=info)
    assert material_id(_built("p", {"cv_group": "explicit"}, info=info)) == "explicit"


def test_material_metadata_roundtrips_without_requiring_yaml(tmp_path):
    info = _Loader().info
    path = tmp_path / "custom.yaml"
    save_yaml(info, path)
    restored = load_yaml(path)
    assert restored == info
    assert (
        material_id(_built("p", {"title": "Emma", "edition": 2}, info=restored))
        == "book:Emma:2"
    )
    default = DatasetInfo(name="ungrouped")
    assert to_dict(default) == {"name": "ungrouped"}
    assert material_id(_built("p", {}, info=default)) == "p"
    with pytest.raises(ValueError, match="material_fields"):
        _built("p", {}, info=DatasetInfo(name="books", material_namespace="book"))


@pytest.mark.parametrize(
    "loader", [SBSAT, OneStop, CopCo, EMTeC, MECOL1W2, MECOL2W1, MECOL2W2]
)
def test_bundled_materials_require_their_packaged_metadata(
    tmp_path, monkeypatch, loader
):
    from replica.dataset.metadata import io

    monkeypatch.setattr(io, "__file__", str(tmp_path / "metadata" / "io.py"))
    dataset = loader(site="ru_mo") if loader is MECOL1W2 else loader()
    with pytest.raises(FileNotFoundError, match="Required dataset metadata"):
        dataset.load(str(tmp_path))
    assert not dataset.stimuli()


@pytest.mark.parametrize("family", ["sbsat", "onestop", "copco"])
def test_packaged_materials_survive_a_custom_data_directory(
    tmp_path, monkeypatch, family
):
    from tests.dataset.test_loader_audit_regressions import _sbsat_row
    from tests.dataset.test_loader_input_contracts import (
        _copco_rows,
        _eyelink_rows,
        _write_rows,
    )

    if family == "sbsat":
        rows = [_sbsat_row("dickens", page, 1, "one") for page in (1, 2)]
        _write_rows(tmp_path / "18sat_fixfinal.csv", rows)
        dataset, expected = SBSAT(), "sbsat:dickens"
    elif family == "onestop":
        rows = [
            {**row, "article_id": 4, "paragraph_id": page}
            for page in (1, 2)
            for row in _eyelink_rows()
        ]
        _write_rows(tmp_path / "ia_Paragraph.csv", rows)
        dataset, expected = OneStop(), "onestop:1:4"
    else:
        rows = [
            {**row, "speechId": 7, "paragraphId": page}
            for page in (1, 2)
            for row in _copco_rows()
        ]
        _write_rows(tmp_path / "ExtractedFeatures/P01.csv", rows)
        dataset, expected = CopCo(), "copco:7"
    dataset.load(str(tmp_path))
    assert dataset.info.material_namespace == family
    assert len(dataset.stimuli()) == 2
    assert [material_id(stimulus) for stimulus in dataset.stimuli()] == [
        expected,
        expected,
    ]
    monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
    error = FileNotFoundError if family == "copco" else ChecksumMismatchError
    with pytest.raises(error):
        type(dataset)().load(str(tmp_path))


@pytest.mark.parametrize("loader", [MECOL1W2, MECOL2W1, MECOL2W2])
def test_meco_load_metadata_groups_tokenization_variants(tmp_path, monkeypatch, loader):
    from replica.dataset.eye_tracking.meco import waves
    from tests.dataset.eye_tracking.meco.test_meco_waves import _frame

    l1 = loader is MECOL1W2
    dataset = loader(site="ru_mo") if l1 else loader()
    frame = _frame(l1=l1, tokenization_variant=True)
    (tmp_path / dataset.release.filename).touch()
    monkeypatch.setattr(waves, "read_r_dataframe", lambda _: frame)
    dataset.load(str(tmp_path))
    assert dataset.info.material_namespace == "meco"
    assert dataset.info.material_fields == [
        "reader_group",
        "wave",
        "site",
        "source_item_id",
    ]
    first = [
        stimulus for stimulus in dataset.stimuli() if stimulus.metadata["text_id"] == 1
    ]
    expected = f"meco:{dataset.release.reader_group}:{dataset.release.wave}:{'ru_mo' if l1 else 'multiple'}:{2 if l1 else 1}"
    assert len(first) == 2
    assert [material_id(stimulus) for stimulus in first] == [expected, expected]
    assert len({material_id(stimulus) for stimulus in dataset.stimuli()}) == 12
