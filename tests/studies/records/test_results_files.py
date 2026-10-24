"""The results resolver and the provenance stamp, on a synthetic study tree.

A study package with one imported library module, one run in the checkout and
one file listed on the hub. The hub is a directory the download hook reads,
so no network is involved.
"""

import hashlib
import json
import shutil

import pytest

from replica.studies.records import cli as C
from replica.studies.records import files as R
from replica.studies.records.layout import render_readme

STUDY = "fakestudy"
RUN = "reproduction/repetition/reading_time/r0123456789ab"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    src = tmp_path / "src"
    package = src / "replica" / "studies" / STUDY
    (src / "replica" / "analysis").mkdir(parents=True)
    (src / "replica" / "studies").mkdir()
    (src / "replica" / "__init__.py").write_text("")
    (src / "replica" / "analysis" / "__init__.py").write_text("")
    (src / "replica" / "analysis" / "evaluate.py").write_text("X = 1\n")
    (src / "replica" / "analysis" / "unused.py").write_text("Y = 1\n")
    (src / "replica" / "studies" / "__init__.py").write_text("")
    package.mkdir()
    (package / "__init__.py").write_text("from ...analysis.evaluate import X\n")
    (package / "recipe.R").write_text("x <- 1\n")
    (package / "README.md").write_text("prose\n")
    (package / "figures").mkdir()
    (package / "figures" / "f.png").write_bytes(b"png")
    run = package / "results" / RUN
    run.mkdir(parents=True)
    (run / "local.tsv").write_text("a\n1\n")
    hub = tmp_path / "hub" / "studies" / STUDY / "results" / RUN
    hub.mkdir(parents=True)
    (hub / "bulk.tsv").write_text("b\n2\n")
    index = {
        "schema": R.SCHEMA,
        "resources": {
            "figures/f.png": {
                "role": "figure",
                "description": "Reading-time figure",
                "sha256": _sha(b"png"),
            }
        },
        "runs": {
            RUN: {
                "role": "evidence",
                "description": "Reading-time fit",
                "files": {"local.tsv": _sha(b"a\n1\n"), "bulk.tsv": _sha(b"b\n2\n")},
                "produced_by": None,
                "hub": {"bulk.tsv": _sha(b"b\n2\n")},
            }
        },
    }
    (package / "results" / "index.json").write_text(json.dumps(index, indent=2))
    monkeypatch.setattr(R, "STUDIES_DIR", src / "replica" / "studies")
    monkeypatch.setattr(R, "SRC_DIR", src)
    monkeypatch.setattr(
        R, "_hub_download", lambda remote, **kwargs: str(tmp_path / "hub" / remote)
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv(R.STORE_ENV, raising=False)
    R._verified.clear()
    return tmp_path


@pytest.mark.parametrize(
    "relative",
    [
        "author_results/values.txt",
        "figures/summary.csv",
        "repetition/cells.tsv",
        "data/estimates.parquet",
        "repetition/archive.json",
        "nested/requirements.txt",
        "requirements-dev.txt",
    ],
)
def test_package_check_rejects_unindexed_artifacts(tree, relative):
    package = R.STUDIES_DIR / STUDY
    path = package / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text("value\n1\n")
    assert any(relative in problem for problem in R.validate_package(STUDY))


def test_dependency_declaration_is_metadata_but_affects_source_identity(tree):
    path = R.STUDIES_DIR / STUDY / "requirements.txt"
    path.write_text("numpy==2.0.0\n")
    assert R.validate_package(STUDY) == []
    before = R.code_footprint(STUDY)
    path.write_text("numpy==2.1.0\n")
    assert R.code_footprint(STUDY) != before


def test_package_check_verifies_explicit_input_identity(tree):
    package = R.STUDIES_DIR / STUDY
    path = package / "inputs.tsv"
    path.write_bytes(b"word\na\n")
    document = R.read_index(STUDY)
    document["resources"]["inputs.tsv"] = {
        "role": "input",
        "description": "Released trial words",
        "sha256": _sha(path.read_bytes()),
    }
    R.write_index(STUDY, document)
    assert R.validate_package(STUDY) == []
    path.write_bytes(b"word\nb\n")
    assert any(
        "inputs.tsv: resource differs" in problem
        for problem in R.validate_package(STUDY)
    )


def test_tables_cannot_be_declared_as_figures(tree):
    package = R.STUDIES_DIR / STUDY
    path = package / "figures/values.csv"
    path.write_bytes(b"1\n")
    document = R.read_index(STUDY)
    document["resources"]["figures/values.csv"] = {
        "role": "figure",
        "description": "Values",
        "sha256": _sha(path.read_bytes()),
    }
    R.write_index(STUDY, document)
    assert any(
        "figure resource must be an image" in problem
        for problem in R.validate_package(STUDY)
    )


def test_archive_bindings_require_an_immutable_revision(tree):
    document = R.read_index(STUDY)
    digest = "a" * 64
    document["archives"] = {
        digest: {
            "description": "Reading-time fit",
            "reference": {
                "repo_id": "example/results",
                "revision": "main",
                "path": f"bundles/{STUDY}/{digest}.tar.gz",
                "sha256": digest,
                "bytes": 1,
            },
        }
    }
    R.write_index(STUDY, document)
    assert any(
        "invalid immutable archive reference" in p for p in R.validate_package(STUDY)
    )


@pytest.mark.parametrize("flag", [True, False, "false", 1, None])
def test_archive_verification_flag_requires_a_boolean(tree, flag):
    document = R.read_index(STUDY)
    digest = "a" * 64
    document["archives"] = {
        digest: {
            "description": "Released source used to verify the figure",
            "verification_required": flag,
            "reference": {
                "repo_id": "example/results",
                "revision": "b" * 40,
                "path": f"bundles/{STUDY}/{digest}.tar.gz",
                "sha256": digest,
                "bytes": 1,
            },
        }
    }
    R.write_index(STUDY, document)
    problems = R.validate_package(STUDY)
    if isinstance(flag, bool):
        assert problems == []
    else:
        assert problems == [f"{digest}: verification_required must be a boolean"]


def test_footprint_covers_the_package_and_its_imports_only(tree):
    files = R.footprint_files(STUDY)
    assert set(files) == {
        "replica/__init__.py",
        "replica/analysis/__init__.py",
        "replica/analysis/evaluate.py",
        "replica/studies/__init__.py",
        f"replica/studies/{STUDY}/__init__.py",
        f"replica/studies/{STUDY}/recipe.R",
    }


def test_footprint_moves_with_code_and_stays_with_prose(tree):
    src = tree / "src" / "replica"
    before = R.code_footprint(STUDY)
    (src / "studies" / STUDY / "README.md").write_text("more prose\n")
    (src / "studies" / STUDY / "architecture.toml").write_text("rules = []\n")
    (src / "studies" / STUDY / "results" / RUN / "local.tsv").write_text("a\n2\n")
    (src / "analysis" / "unused.py").write_text("Y = 2\n")
    assert R.code_footprint(STUDY) == before
    assert R.validate_package(STUDY) == []
    (src / "analysis" / "evaluate.py").write_text("X = 2\n")
    assert R.code_footprint(STUDY) != before


def test_dataset_yaml_is_tracked_when_metadata_io_is_imported(tree):
    metadata = tree / "src/replica/dataset/metadata"
    yaml_path = metadata / "yaml/eye-tracking/example.yaml"
    yaml_path.parent.mkdir(parents=True)
    (metadata.parent / "__init__.py").write_text("")
    (metadata / "__init__.py").write_text("")
    (metadata / "io.py").write_text("def load_yaml(path): pass\n")
    yaml_path.write_text("material_fields: [article]\n")
    before = R.code_footprint(STUDY)
    yaml_path.write_text("material_fields: [article, page]\n")
    assert R.code_footprint(STUDY) == before

    (R.STUDIES_DIR / STUDY / "__init__.py").write_text(
        "from ...dataset.metadata.io import load_yaml\n"
    )
    relative = "dataset/metadata/yaml/eye-tracking/example.yaml"
    assert R.footprint_files(STUDY)[f"replica/{relative}"] == yaml_path
    R.stamp(STUDY, RUN, R.produced_by_now(STUDY))
    assert R.stale_runs(STUDY) == []
    explicit = R.code_footprint(STUDY, depends_on=[relative])
    unrelated = R.code_footprint(STUDY, depends_on=[f"studies/{STUDY}/recipe.R"])
    yaml_path.write_text("material_fields: [article]\n")
    assert [run for run, _ in R.stale_runs(STUDY)] == [RUN]
    assert R.code_footprint(STUDY, depends_on=[relative]) != explicit
    assert (
        R.code_footprint(STUDY, depends_on=[f"studies/{STUDY}/recipe.R"]) == unrelated
    )


@pytest.fixture
def patch_registry(tree, monkeypatch):
    from replica.external import DataDependency, repositories

    package = tree / "src/replica"
    external = package / "external"
    (external / "adapters").mkdir(parents=True)
    (external / "__init__.py").write_text("")
    (external / "repositories.py").write_text("EXTERNAL_REPOS = {}\n")
    patch = external / "adapters/compat.patch"
    patch.write_text("original patch\n")
    dependency = DataDependency(
        path="example",
        source="git:example/upstream",
        patch="external/adapters/compat.patch",
    )
    monkeypatch.setattr(repositories, "EXTERNAL_REPOS", {"example": (dependency,)})
    return patch, repositories


def test_footprint_tracks_declared_patches_only_when_registry_is_imported(
    tree, patch_registry
):
    patch, _ = patch_registry
    before = R.code_footprint(STUDY)
    patch.write_text("unreferenced change\n")
    assert R.code_footprint(STUDY) == before

    (R.STUDIES_DIR / STUDY / "__init__.py").write_text(
        "from ...external.repositories import EXTERNAL_REPOS\n"
    )
    assert R.footprint_files(STUDY)["replica/external/adapters/compat.patch"] == patch
    before = R.code_footprint(STUDY)
    (patch.parent / "unregistered.patch").write_text("unrelated resource\n")
    assert R.code_footprint(STUDY) == before
    patch.write_text("changed dependency\n")
    assert R.code_footprint(STUDY) != before


@pytest.mark.parametrize(
    "relative", ["/outside.patch", "../outside.patch", "missing.patch"]
)
def test_footprint_rejects_missing_or_escaping_registry_patches(
    tree, patch_registry, relative
):
    from dataclasses import replace

    _, repositories = patch_registry
    repositories.EXTERNAL_REPOS["example"] = (
        replace(repositories.EXTERNAL_REPOS["example"][0], patch=relative),
    )
    (R.STUDIES_DIR / STUDY / "__init__.py").write_text(
        "from ...external.repositories import EXTERNAL_REPOS\n"
    )
    with pytest.raises((ValueError, FileNotFoundError), match="example:.*patch"):
        R.footprint_files(STUDY)


def test_results_file_prefers_the_checkout(tree):
    path = R.results_file(STUDY, RUN, "local.tsv")
    assert (
        path
        == tree / "src" / "replica" / "studies" / STUDY / "results" / RUN / "local.tsv"
    )


def test_results_file_falls_back_to_the_store_then_the_hub(tree, monkeypatch):
    store = tree / "store" / "run_20260913" / "results"
    store.mkdir(parents=True)
    (store / "bulk.tsv").write_text("store copy\n")
    index = R.read_index(STUDY)
    index["runs"][RUN]["store"] = "run_20260913"
    R.write_index(STUDY, index)
    monkeypatch.setenv(R.STORE_ENV, str(tree / "store"))
    with pytest.raises(ValueError, match="manifest digest"):
        R.results_file(STUDY, RUN, "bulk.tsv")
    (store / "bulk.tsv").write_text("b\n2\n")
    assert R.results_file(STUDY, RUN, "bulk.tsv") == store / "bulk.tsv"
    monkeypatch.delenv(R.STORE_ENV)
    assert (
        R.results_file(STUDY, RUN, "bulk.tsv")
        == tree / "hub" / "studies" / STUDY / "results" / RUN / "bulk.tsv"
    )


def test_hub_copy_must_match_the_recorded_digest(tree):
    (tree / "hub" / "studies" / STUDY / "results" / RUN / "bulk.tsv").write_text(
        "tampered\n"
    )
    with pytest.raises(ValueError, match="digest"):
        R.results_file(STUDY, RUN, "bulk.tsv")


def test_unlisted_file_is_reported_with_where_it_was_sought(tree):
    with pytest.raises(FileNotFoundError, match="not listed under hub"):
        R.results_file(STUDY, RUN, "missing.tsv")
    with pytest.raises(FileNotFoundError, match="no run"):
        R.results_file(STUDY, "robustness/none", "x.tsv")


def test_run_dir_joins_checkout_and_hub_files(tree):
    view = R.run_dir(STUDY, RUN)
    assert (view / "local.tsv").read_text() == "a\n1\n"
    assert (view / "bulk.tsv").read_text() == "b\n2\n"
    index = R.read_index(STUDY)
    index["runs"][RUN].pop("hub")
    R.write_index(STUDY, index)
    assert (
        R.run_dir(STUDY, RUN)
        == tree / "src" / "replica" / "studies" / STUDY / "results" / RUN
    )


def test_locate_and_hub_url_split_the_run_from_the_name(tree):
    assert R.locate(STUDY, f"{RUN}/local.tsv").name == "local.tsv"
    assert R.hub_url(STUDY, f"{RUN}/bulk.tsv") == (
        f"{R.HUB_URL}/blob/main/studies/{STUDY}/results/{RUN}/bulk.tsv"
    )
    assert R.hub_url(STUDY, f"{RUN}/local.tsv") is None


def test_validation_preserves_the_producer_and_tracks_later_code_changes(tree):
    assert R.unstamped_runs(STUDY) == [RUN]
    producer = R.produced_by_now(STUDY, {"corpus": "0" * 64})
    producer["restamped"] = {"reason": "original producer annotation"}
    R.stamp(STUDY, RUN, producer)
    original = R.read_index(STUDY)["runs"][RUN]
    assert R.unstamped_runs(STUDY) == []
    assert R.stale_runs(STUDY) == []
    (tree / "src" / "replica" / "analysis" / "evaluate.py").write_text("X = 3\n")
    ((stale_run, message),) = R.stale_runs(STUDY)
    assert stale_run == RUN and message.startswith("produced at")
    validation = R.produced_by_now(STUDY, {"assessment": "1" * 64})
    R.stamp(STUDY, RUN, validation, validate=True)
    assert R.stale_runs(STUDY) == []
    recorded = R.read_index(STUDY)["runs"][RUN]
    assert recorded == {**original, "validated_by": validation}
    assert recorded["produced_by"] == producer
    assert R.validate_index(STUDY) == []
    (tree / "src" / "replica" / "analysis" / "evaluate.py").write_text("X = 4\n")
    ((_, message),) = R.stale_runs(STUDY)
    commit = (validation["commit"] or "unknown commit")[:12]
    assert message == f"validated at {commit} on {validation['produced_at']}"


def test_a_stamp_may_name_the_files_it_depends_on(tree):
    src = tree / "src" / "replica"
    depends_on = [f"studies/{STUDY}/recipe.R"]
    stamp = R.produced_by_now(STUDY, depends_on=depends_on)
    assert stamp["depends_on"] == depends_on
    R.stamp(STUDY, RUN, stamp)
    (src / "analysis" / "evaluate.py").write_text("X = 9\n")
    (src / "studies" / STUDY / "__init__.py").write_text("Y = 1\n")
    assert R.stale_runs(STUDY) == []
    (src / "studies" / STUDY / "recipe.R").write_text("x <- 2\n")
    assert [run for run, _ in R.stale_runs(STUDY)] == [RUN]
    (src / "studies" / STUDY / "recipe.R").unlink()
    assert [run for run, _ in R.stale_runs(STUDY)] == [RUN]
    assert (
        C.main(["stamp", STUDY, RUN, "--depends-on", depends_on[0], "--validate"]) == 0
    )
    assert R.stale_runs(STUDY) == []
    index = R.read_index(STUDY)
    index["runs"][RUN]["produced_by"]["depends_on"] = [""]
    R.write_index(STUDY, index)
    assert any("depends_on must be" in p for p in R.validate_index(STUDY))


def test_a_run_no_package_file_produced_never_goes_stale(tree):
    src = tree / "src" / "replica"
    stamp = R.produced_by_now(STUDY, depends_on=[])
    assert stamp["depends_on"] == []
    R.stamp(STUDY, RUN, stamp)
    assert R.validate_index(STUDY) == []
    (src / "analysis" / "evaluate.py").write_text("X = 9\n")
    (src / "studies" / STUDY / "__init__.py").write_text("Y = 1\n")
    assert R.stale_runs(STUDY) == []


def test_depends_on_paths_are_package_relative(tree):
    src = tree / "src" / "replica"
    recipe = f"studies/{STUDY}/recipe.R"
    for wrong in (f"src/replica/{recipe}", f"replica/{recipe}", str(src / recipe)):
        with pytest.raises(ValueError, match="not relative to the replica package"):
            R.stamp(STUDY, RUN, R.produced_by_now(STUDY, depends_on=[wrong]))
        with pytest.raises(SystemExit, match="not relative to the replica package"):
            C.main(["stamp", STUDY, RUN, "--depends-on", wrong])
    assert R.unstamped_runs(STUDY) == [RUN]
    wrong = f"src/replica/{recipe}"
    record = {
        **R.produced_by_now(STUDY),
        "depends_on": [wrong],
        "footprint": R.code_footprint(STUDY, depends_on=[wrong]),
    }
    index = R.read_index(STUDY)
    index["runs"][RUN]["produced_by"] = record
    R.write_index(STUDY, index)
    (src / "studies" / STUDY / "recipe.R").write_text("x <- 2\n")
    assert R.stale_runs(STUDY) == []
    assert C.main(["stamp", STUDY, RUN, "--depends-on", recipe, "--validate"]) == 0
    assert R.validate_index(STUDY) == []
    assert R.read_index(STUDY)["runs"][RUN]["produced_by"] == record
    (src / "studies" / STUDY / "recipe.R").write_text("x <- 3\n")
    assert [run for run, _ in R.stale_runs(STUDY)] == [RUN]


def test_validate_index_names_each_defect(tree):
    assert R.validate_index(STUDY) == []
    index = R.read_index(STUDY)
    entry = index["runs"][RUN]
    entry["role"] = "guess"
    entry["hub"]["bulk.tsv"] = "xyz"
    entry["produced_by"] = {"commit": "c"}
    entry["extra"] = 1
    index["runs"]["robustness/ghost"] = {"role": "check"}
    R.write_index(STUDY, index)
    problems = R.validate_index(STUDY)
    assert any("role 'guess'" in p for p in problems)
    assert any("not sha256" in p for p in problems)
    assert any("lacks footprint" in p for p in problems)
    assert any("unknown key extra" in p for p in problems)
    assert any("ghost: only in index.json" in p for p in problems)
    assert any("ghost: no produced_by" in p for p in problems)
    shutil.copytree(
        tree / "src" / "replica" / "studies" / STUDY / "results" / RUN,
        tree
        / "src"
        / "replica"
        / "studies"
        / STUDY
        / "results"
        / "robustness"
        / "ghost",
    )
    (
        tree / "src" / "replica" / "studies" / STUDY / "results" / RUN / "bulk.tsv"
    ).write_text("other\n")
    problems = R.validate_index(STUDY)
    assert any("differs between checkout and hub" in p for p in problems)


def test_cli_reports_and_exits_on_stale_runs(tree, capsys):
    assert C.main(["stamp", STUDY, RUN]) == 0
    assert C.main(["stale", STUDY]) == 0
    (tree / "src" / "replica" / "analysis" / "evaluate.py").write_text("X = 4\n")
    assert C.main(["stale", STUDY]) == 1
    assert f"stale: {RUN} (produced at" in capsys.readouterr().out
    assert C.main(["stamp", STUDY, RUN, "--validate"]) == 0
    assert C.main(["stale", STUDY]) == 0
    assert C.main(["validate", STUDY]) == 0
    manifest = tree / "MANIFEST.json"
    manifest.write_text(json.dumps({"produced_by": R.produced_by_now(STUDY)}))
    assert (
        C.main(["stamp", STUDY, RUN, "--manifest", str(manifest), "--replace-producer"])
        == 0
    )
    entry = R.read_index(STUDY)["runs"][RUN]
    assert entry["produced_by"] == json.loads(manifest.read_text())["produced_by"]
    assert "validated_by" not in entry
    with pytest.raises(SystemExit, match="LABEL=SHA256"):
        C.main(["stamp", STUDY, RUN, "--input", "corpus"])


def test_cli_stamp_names_one_run_and_keeps_recorded_producers(tree, capsys):
    other = "robustness/reading_time_evaluation/rfedcba987654"
    assert C.main(["stamp", STUDY, RUN]) == 0
    document = R.read_index(STUDY)
    document["runs"][other] = {**document["runs"][RUN], "produced_by": None}
    R.write_index(STUDY, document)
    before = R.index_path(STUDY).read_bytes()
    (tree / "src" / "replica" / "analysis" / "evaluate.py").write_text("X = 5\n")
    for argv in (
        ["stamp", STUDY],
        ["stamp", STUDY, "--validate"],
        ["stamp", STUDY, RUN],
        ["stamp", STUDY, RUN, "--validate", "--replace-producer"],
        ["stamp", STUDY, other, "--validate"],
        ["stamp", STUDY, "robustness/reading_time_evaluation/r000000000000"],
    ):
        with pytest.raises(SystemExit):
            C.main(argv)
        assert R.index_path(STUDY).read_bytes() == before, argv
    for run, replace in ((RUN, True), (other, False)):
        with pytest.raises(ValueError):
            R.stamp(
                STUDY, run, R.produced_by_now(STUDY), validate=True, replace=replace
            )
        assert R.index_path(STUDY).read_bytes() == before
    with pytest.raises(SystemExit, match="--replace-producer"):
        C.main(["stamp", STUDY, RUN])
    with pytest.raises(SystemExit, match="requires an existing producer"):
        C.main(["stamp", STUDY, other, "--validate"])
    with pytest.raises(SystemExit, match="no run"):
        C.main(["stamp", STUDY, "robustness/reading_time_evaluation/r000000000000"])
    assert C.main(["stamp", STUDY, other]) == 0
    assert C.main(["stamp", STUDY, RUN, "--replace-producer"]) == 0
    runs = R.read_index(STUDY)["runs"]
    assert runs[RUN]["produced_by"]["footprint"] == R.code_footprint(STUDY)
    assert runs[other]["produced_by"]["footprint"] == R.code_footprint(STUDY)


@pytest.mark.parametrize(
    "defect",
    [
        "unindexed_file",
        "changed_file",
        "missing_file",
        "missing_manifest",
        "missing_description",
        "reference_role",
        "validation_type",
        "validation_reason",
        "validation_without_producer",
    ],
)
def test_record_manifest_rejects_incomplete_or_misclassified_results(tree, defect):
    index = R.read_index(STUDY)
    entry = index["runs"][RUN]
    local = R.STUDIES_DIR / STUDY / "results" / RUN / "local.tsv"
    if defect == "unindexed_file":
        local.with_name("loose.tsv").write_text("x\n")
    elif defect == "changed_file":
        local.write_text("a\n99\n")
    elif defect == "missing_file":
        local.unlink()
    elif defect == "missing_manifest":
        del entry["files"]
    elif defect == "missing_description":
        del entry["description"]
    elif defect.startswith("validation_"):
        entry["produced_by"] = R.produced_by_now(STUDY)
        entry["validated_by"] = dict(entry["produced_by"])
        if defect == "validation_type":
            entry["validated_by"] = []
        elif defect == "validation_reason":
            entry["validated_by"]["reason"] = "unstructured text"
        else:
            entry["produced_by"] = None
    else:
        entry["role"] = "published"
    R.write_index(STUDY, index)
    assert R.validate_index(STUDY)


@pytest.mark.parametrize("description", ["Reading-time fit", "  READING-TIME   fit "])
@pytest.mark.parametrize("axis", ["reproduction", "replication"])
def test_result_descriptions_distinguish_records_within_each_stage(
    tree, description, axis
):
    document = R.read_index(STUDY)
    other = f"{axis}/repetition/reading_time/rabcdef012345"
    entry = dict(document["runs"][RUN])
    entry.update(description=description, hub=dict(entry["files"]))
    document["runs"][other] = entry
    R.write_index(STUDY, document)
    problems = R.validate_index(STUDY)
    if axis == "reproduction":
        assert problems == [f"{other}: description also labels {RUN}"]
        with pytest.raises(ValueError, match="description also labels"):
            render_readme(STUDY, R.STUDIES_DIR / STUDY / "results", R.HUB_URL)
    else:
        assert problems == []


def test_concurrent_run_materialization_replaces_links_atomically(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from threading import Barrier

    source = tmp_path / "source.tsv"
    source.write_text("value\n1\n")
    target = tmp_path / "view" / "cells.tsv"
    rendezvous = Barrier(4)
    symlink_to = Path.symlink_to

    def synchronized_link(path, other):
        rendezvous.wait(timeout=10)
        symlink_to(path, other)

    monkeypatch.setattr(Path, "symlink_to", synchronized_link)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: R._link(target, source), range(4)))
    assert target.resolve() == source
    assert target.read_bytes() == source.read_bytes()
    assert list(target.parent.iterdir()) == [target]


def test_cli_resolves_one_file_or_a_complete_run_without_checkout_payloads(
    tree, capsys
):
    assert C.main(["path", STUDY, RUN, "bulk.tsv"]) == 0
    assert capsys.readouterr().out.strip() == str(
        R.results_file(STUDY, RUN, "bulk.tsv")
    )
    assert C.main(["path", STUDY, RUN]) == 0
    assert capsys.readouterr().out.strip() == str(R.run_dir(STUDY, RUN))


def test_hub_links_include_indexed_record_directories(tree):
    assert (
        R.hub_url(STUDY, RUN) == f"{R.HUB_URL}/tree/main/studies/{STUDY}/results/{RUN}"
    )
    assert R.hub_url(STUDY, RUN + "/") == R.hub_url(STUDY, RUN)


def test_footprint_resolves_aliased_dynamic_and_typing_imports(tree):
    src = tree / "src/replica"
    (src / "analysis/typed.py").write_text("VALUE = 1\n")
    (src / "studies" / STUDY / "__init__.py").write_text(
        "from importlib import import_module as load\n"
        "from typing import TYPE_CHECKING\n"
        "load('.evaluate', package='replica.analysis')\n"
        "if TYPE_CHECKING:\n    from ...analysis import typed\n"
        "client.import_module('replica.analysis.unused')\n"
    )
    files = R.footprint_files(STUDY)
    assert "replica/analysis/evaluate.py" in files
    assert "replica/analysis/typed.py" in files
    assert "replica/analysis/unused.py" not in files


def test_indexed_repository_routes_downloads_and_links(tree, monkeypatch):
    document = R.read_index(STUDY)
    document["hub_repo"] = "example/study-bundles"
    R.write_index(STUDY, document)
    calls = []
    download = R._hub_download

    def record(remote, **kwargs):
        calls.append(kwargs["repo_id"])
        return download(remote, **kwargs)

    monkeypatch.setattr(R, "_hub_download", record)
    monkeypatch.delenv("REPLICA_RESULTS_HF_REPO", raising=False)
    assert R.results_file(STUDY, RUN, "bulk.tsv").read_text() == "b\n2\n"
    assert calls == ["example/study-bundles"]
    assert R.hub_url(STUDY, RUN).startswith(
        "https://huggingface.co/datasets/example/study-bundles/"
    )
    monkeypatch.setenv("REPLICA_RESULTS_HF_REPO", "example/alternate")
    R.results_file(STUDY, RUN, "bulk.tsv")
    assert calls[-1] == "example/alternate"
