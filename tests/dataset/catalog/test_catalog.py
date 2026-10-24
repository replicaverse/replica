"""Data-free tests for dataset discovery and convenience loading."""

from pathlib import Path

import pytest

from replica.dataset import (
    GECO,
    PROVO,
    ZuCo2,
    available_datasets,
    describe_dataset,
    load_dataset,
)
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.eye_tracking.meco.character import MECO_char
from replica.dataset.measures import FIRST_PASS_RUN


def _parameter(description, name):
    return next(
        parameter for parameter in description.parameters if parameter.name == name
    )


def test_available_datasets_contains_family_and_concrete_registry_keys():
    available = available_datasets()
    assert available == sorted(available)
    assert "meco" in available
    assert set(REGISTRY) <= set(available)
    assert "meco_l1_w1_all" not in available
    assert "meco_l1_w1_all" in available_datasets(include_groups=True)


def test_describe_meco_exposes_conditional_variants_and_choices():
    description = describe_dataset("meco")
    assert description.kind == "family"
    assert _parameter(description, "release").choices == (
        "l1_w1",
        "l1_w2",
        "l2_w1",
        "l2_w2",
    )
    variants = {variant.name: variant for variant in description.variants}
    assert set(variants) == {"l1_w1", "l1_w2", "l2_w1", "l2_w2"}
    l1w1 = {parameter.name: parameter for parameter in variants["l1_w1"].parameters}
    assert "en" in l1w1["language"].choices
    assert l1w1["representation"].choices == ("word", "character")
    site = variants["l1_w2"].parameters[0]
    assert site.required
    assert ("en_uk", "en") in site.choice_descriptions


def test_describe_regular_dataset_exposes_init_and_load_options():
    geco = describe_dataset("geco")
    assert _parameter(geco, "group").choices == GECO.GROUPS
    assert {"first_run", "first_pass"} <= set(geco.measures)
    assert geco.default_measure == FIRST_PASS_RUN
    assert geco.download_keys == ("geco",)
    assert geco.fixations.startswith("unavailable")

    zuco = describe_dataset("zuco2")
    assert _parameter(zuco, "task").choices == ZuCo2.TASKS
    assert _parameter(zuco, "load_eeg").stage == "load"


def test_describe_concrete_variant_shows_pinned_options():
    description = describe_dataset("meco_l1_w2_en_uk")
    assert description.loader == "MECOL1W2"
    assert description.selected_options == (("site", "en_uk"),)
    assert description.download_keys == (
        "meco_l1_w2",
        "meco_l1_w2_fixations",
    )


def test_describe_group_lists_members():
    description = describe_dataset("meco_l2_all")
    assert description.kind == "group"
    assert description.members == ("meco_l2_w1", "meco_l2_w2")


def test_describe_unknown_key_reports_available_names():
    with pytest.raises(KeyError, match="available datasets"):
        describe_dataset("not_a_dataset")


def test_load_dataset_splits_constructor_and_load_options(monkeypatch, tmp_path):
    seen = {}

    def fake_load(self, path, *, max_subjects=None, load_eeg=True):
        seen.update(path=path, max_subjects=max_subjects, load_eeg=load_eeg)

    monkeypatch.setattr(ZuCo2, "load", fake_load)
    dataset = load_dataset(
        "zuco2",
        path=tmp_path,
        task="TSR",
        max_subjects=2,
        load_eeg=False,
    )
    assert isinstance(dataset, ZuCo2)
    assert dataset.task == "TSR"
    assert seen == {
        "path": str(tmp_path),
        "max_subjects": 2,
        "load_eeg": False,
    }


def test_load_meco_family_selects_character_data_path(monkeypatch, tmp_path):
    seen = {}

    def fake_load(self, path):
        seen["path"] = path

    monkeypatch.setattr(MECO_char, "load", fake_load)
    dataset = load_dataset(
        "meco",
        repo_root=tmp_path,
        release="l1_w1",
        language="de",
        representation="character",
    )
    assert isinstance(dataset, MECO_char)
    assert seen["path"] == str(tmp_path / "datasets/eye-tracking/processed/meco_char")


def test_load_dataset_uses_explicit_path(monkeypatch, tmp_path):
    seen = {}

    def fake_load(self, path):
        seen["path"] = path

    monkeypatch.setattr(PROVO, "load", fake_load)
    assert isinstance(load_dataset("provo", path=tmp_path), PROVO)
    assert seen["path"] == str(tmp_path)


def test_load_dataset_rejects_unknown_or_conflicting_options(tmp_path):
    with pytest.raises(TypeError, match="Unknown option"):
        load_dataset("provo", path=tmp_path, language="en")
    with pytest.raises(ValueError, match="fixes language"):
        load_dataset("meco_en", path=tmp_path, language="de")
    with pytest.raises(ValueError, match="either path"):
        load_dataset("provo", path=tmp_path, repo_root=Path("repo"))
    with pytest.raises(ValueError, match="dataset group"):
        load_dataset("meco_l2_all", path=tmp_path)


def test_catalog_descriptions_have_valid_loaders_and_unique_parameters():
    failures = []
    for name in available_datasets():
        description = describe_dataset(name)
        parameter_names = [parameter.name for parameter in description.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            failures.append(f"{name}: duplicate parameters {parameter_names}")
        if description.kind == "dataset" and not description.loader:
            failures.append(f"{name}: missing loader")
        if name not in str(description):
            failures.append(f"{name}: formatted description omits its key")
    assert not failures, "invalid catalog descriptions:\n" + "\n".join(failures)


def test_meco_measures_are_declared_per_representation():
    variant = describe_dataset("meco").variants[0]
    measures = dict(variant.measures_by_representation)
    assert set(measures["word"]) == set(describe_dataset("meco_en").measures)
    assert set(measures["character"]) == set(describe_dataset("meco_char_en").measures)
    assert "right_bounded" not in measures["word"]


def test_shain_dundee_needs_the_author_distributed_release():
    """The loader aligns dundee_X.csv against Dundee V.1 items and MA1 events."""
    from replica.dataset.catalog import facts

    assert describe_dataset("dundee").download_keys == ("dundee", "dundee_official")
    record = facts.dataset_facts(["dundee"], use_snapshot=False)["datasets"]["dundee"]
    assert record["gated"] is True
    assert "V.1" in record["access_instructions"]


def test_every_option_is_described_and_every_description_is_an_option():
    from replica.dataset.catalog.description import _DESCRIPTIONS, _parameters

    loaders = {spec.loader_class: spec.loader_cls() for spec in REGISTRY.values()}
    accepted = {
        (loader, parameter.name)
        for loader, cls in loaders.items()
        for parameter in _parameters(cls)
    }
    generic = sorted(
        (loader, name)
        for loader, name in accepted
        if (loader, name) not in _DESCRIPTIONS and name != "drop_skips"
    )
    assert not generic, f"options without a description: {generic}"
    assert not set(_DESCRIPTIONS) - accepted


def test_option_choices_come_from_the_loaders():
    from replica.dataset.eye_tracking.zuco.zuco1 import ZuCo1
    from replica.dataset.maze.natural_stories_maze import _POST_ERROR_POLICIES

    assert _parameter(describe_dataset("zuco1"), "task").choices == ZuCo1.TASKS
    maze = describe_dataset("natural_stories_maze")
    assert _parameter(maze, "post_error").choices == _POST_ERROR_POLICIES
