"""Static dependency policies detect violations without importing backends."""

import ast
import json
import tomllib
from importlib.util import resolve_name
from pathlib import Path
from textwrap import indent

import pytest

import replica
from replica.utils.imports import import_names, import_references
from scripts.checks.source.import_boundaries import _check, _impact, load_policy, main

_ROOT = Path(__file__).resolve().parents[2]


def test_repository_obeys_architecture_policy():
    assert _check(_ROOT, load_policy(_ROOT))["errors"] == []


@pytest.mark.parametrize("indirect", [False, True], ids=["direct", "via-helper"])
@pytest.mark.parametrize(
    "source,target,rule",
    [
        ("results_files", "results_store", "result-index-is-independent-of-store"),
        ("results_files", "results_cli", "result-services-have-no-cli-dependencies"),
        ("results_store", "results_cli", "result-services-have-no-cli-dependencies"),
    ],
)
def test_results_boundaries_reject_reverse_dependencies(
    tmp_path, source, target, rule, indirect
):
    with (_ROOT / "scripts/checks/source/architecture.toml").open("rb") as stream:
        rules = tomllib.load(stream)["rules"]
    policy = {"rules": [r for r in rules if r["name"] == rule]}
    package = tmp_path / "src/replica/studies"
    package.mkdir(parents=True)
    for name in ("results_files", "results_store", "results_cli"):
        (package / f"{name}.py").write_text("")
    imported = "helper" if indirect else target
    (package / f"{source}.py").write_text(f"from . import {imported}\n")
    if indirect:
        (package / "helper.py").write_text(f"from . import {target}\n")
    assert any(rule in error for error in _check(tmp_path, policy)["errors"])


def test_static_public_exports_match_runtime_names_and_owners():
    """An extension must be visible to both runtime callers and type checkers."""
    expected = {
        name: f"{owner}.{name}" for name, owner in replica._EXPORT_MODULE.items()
    }
    expected.update({name: f"replica.{name}" for name in replica._SUBMODULES})
    actual = {}
    for node in ast.parse((_ROOT / "src/replica/__init__.pyi").read_text()).body:
        if isinstance(node, ast.ImportFrom):
            owner = resolve_name("." * node.level + (node.module or ""), "replica")
            for alias in node.names:
                name = alias.asname or alias.name
                assert name not in actual, f"Duplicate static export: {name}"
                actual[name] = f"{owner}.{alias.name}"
    assert actual == expected
    assert sum(map(len, replica._EXPORT_GROUPS.values())) == len(replica.__all__)


def test_analysis_exports_the_extension_contracts():
    """A backend, filter or transform author imports its contract from the facade."""
    from replica import analysis

    for name in (
        "GroupedModel",
        "GroupedFrame",
        "GroupedFit",
        "FoldScore",
        "StatsPredictor",
        "OutlierFilter",
        "FoldTransform",
        "Residualize",
    ):
        assert getattr(analysis, name) is not None
        assert name in analysis.__all__


@pytest.mark.parametrize(
    "source",
    [
        "import replica.studies.some_study",
        "from ...studies import some_study",
        "def f():\n    from replica import studies",
        "import importlib as loader\nloader.import_module('replica.studies.some_study')",
        "from importlib import import_module as load\nload('.studies', 'replica')",
        "__import__('replica.studies')",
        "__import__('replica', fromlist=['studies'])",
        "import importlib\nimportlib.import_module(name='.studies', package='replica')",
    ],
)
def test_import_resolver_detects_boundary_bypasses(source):
    assert any(
        name.startswith("replica.studies")
        for name in import_names(source, "replica.analysis.models")
    )


def test_import_resolver_reports_computed_imports_and_ignores_unrelated_methods():
    refs = import_references(
        "from importlib import import_module as load\n"
        "load(variable)\nclient.import_module('replica.studies')\n",
        "replica.analysis",
    )
    assert [(r.module, r.line) for r in refs] == [("importlib", 1), (None, 2)]


@pytest.mark.parametrize(
    "header, condition",
    [
        ("from typing import TYPE_CHECKING as TC", "TC"),
        ("import typing as t", "t.TYPE_CHECKING"),
    ],
)
def test_import_resolver_keeps_typing_and_runtime_contexts_separate(header, condition):
    refs = import_references(
        f"{header}\nif {condition}:\n    import replica.a\n"
        "else:\n    import replica.b\nimport replica.c\n",
        "replica",
    )
    assert [(r.module, r.type_only) for r in refs[1:]] == [
        ("replica.a", True),
        ("replica.b", False),
        ("replica.c", False),
    ]


def _tree(root, **modules):
    package = root / "src/replica"
    package.mkdir(parents=True, exist_ok=True)
    for name, source in modules.items():
        (package / f"{name}.py").write_text(source)
    return root


def test_boundary_rule_reports_exact_source_and_rejects_dead_selectors(tmp_path):
    root = _tree(tmp_path, a="def f():\n    from replica import b\n", b="")
    rule = dict(name="a-no-b", sources=["replica.a"], forbidden=["replica.b"])
    report = _check(root, {"rules": [rule]})
    assert report["errors"] == ["src/replica/a.py:2: replica.a -> replica.b [a-no-b]"]
    rule["sources"].append("replica.gone")
    assert any(
        "matches no modules" in error
        for error in _check(root, {"rules": [rule]})["errors"]
    )


def test_cycle_baseline_rejects_equal_count_replacements_and_stale_edges(tmp_path):
    root = _tree(tmp_path, a="from . import b", b="from . import a", c="")
    policy = {
        "cycle_exceptions": [
            dict(
                reason="Separate cell configuration from execution.",
                milestone="split-cell-config",
                edges=[["replica.a", "replica.b"], ["replica.b", "replica.a"]],
            )
        ]
    }
    assert _check(root, policy)["errors"] == []
    _tree(root, a="from . import c", b="", c="from . import a")
    errors = _check(root, policy)["errors"]
    assert len(errors) == 4
    assert sum("Unreviewed cycle" in error for error in errors) == 2
    assert sum("Stale cycle" in error for error in errors) == 2


def test_package_cycle_rejects_crossed_acyclic_modules_including_type_imports(tmp_path):
    for package, other in (("left", "right"), ("right", "left")):
        directory = tmp_path / "src/replica" / package
        directory.mkdir(parents=True)
        (directory / "leaf.py").write_text("")
        source = f"import replica.{other}.leaf\n"
        if package == "right":
            source = (
                "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    " + source
            )
        (directory / "consumer.py").write_text(source)
    report = _check(tmp_path, {})
    assert report["cycle_edges"] == []
    assert report["package_cycle_edges"] == [
        (2, "replica.left", "replica.right"),
        (2, "replica.right", "replica.left"),
    ]
    assert len(report["errors"]) == 2
    (tmp_path / "src/replica/right/consumer.py").write_text("")
    assert _check(tmp_path, {})["errors"] == []


def _study(root, name, policy, **modules):
    package = root / "src/replica/studies" / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "README.md").write_text("# study\n")
    for module, source in modules.items():
        (package / f"{module}.py").write_text(source)
    (package / "architecture.toml").write_text(policy)
    shared = root / "policy.toml"
    shared.write_text(
        '[[rules]]\nname = "a-no-b"\nsources = ["replica.a"]\nforbidden = ["replica.b"]\n'
    )
    return shared


def test_study_policy_files_merge_into_the_shared_policy(tmp_path):
    root = _tree(tmp_path, a="", b="")
    shared = _study(
        root,
        "some2020x",
        "[[rules]]\n"
        'name = "some2020x-frames-are-independent"\n'
        'sources = ["replica.studies.some2020x.frames"]\n'
        'forbidden = ["replica.b"]\n\n'
        "[[cycle_exceptions]]\n"
        'reason = "Frames and predictors share helpers."\n'
        'milestone = "extract-some2020x"\n'
        "edges = [\n"
        '  ["replica.studies.some2020x.frames", "replica.studies.some2020x.predictors"],\n'
        '  ["replica.studies.some2020x.predictors", "replica.studies.some2020x.frames"],\n'
        "]\n",
        frames="from . import predictors",
        predictors="from . import frames",
    )
    with shared.open("rb") as stream:
        shared_only = tomllib.load(stream)
    assert (
        sum("Unreviewed cycle" in e for e in _check(root, shared_only)["errors"]) == 2
    )
    policy = load_policy(root, shared)
    assert [rule["name"] for rule in policy["rules"]] == [
        "a-no-b",
        "some2020x-frames-are-independent",
    ]
    assert _check(root, policy)["errors"] == []
    (root / "src/replica/studies/some2020x/frames.py").write_text(
        "from replica import b"
    )
    assert any(
        "[some2020x-frames-are-independent]" in e
        for e in _check(root, policy)["errors"]
    )
    (root / "src/replica/studies/some2020x").rename(root / "removed-study")
    assert load_policy(root, shared)["rules"] == shared_only["rules"]
    assert _check(root, load_policy(root, shared))["errors"] == []


@pytest.mark.parametrize("suffix", ["", ".frames"])
def test_shared_policy_refuses_a_rule_scoped_to_one_study_package(tmp_path, suffix):
    root = _tree(tmp_path, a="", b="")
    shared = _study(root, "some2020x", "", frames="")
    shared.write_text(
        '[[rules]]\nname = "scoped"\n'
        f'sources = ["replica.studies.some2020x{suffix}"]\n'
        'forbidden = ["replica.b"]\n'
    )
    with pytest.raises(ValueError, match="belongs in that package"):
        load_policy(root, shared)
    shared.write_text(
        '[[rules]]\nname = "spanning"\n'
        'sources = ["replica.studies.*.frames", "replica.a"]\nforbidden = ["replica.b"]\n'
        '[[rules]]\nname = "shared-package"\n'
        'sources = ["replica.studies.shared.frames"]\nforbidden = ["replica.b"]\n'
    )
    shared_package = root / "src/replica/studies/shared"
    shared_package.mkdir()
    (shared_package / "__init__.py").write_text("")
    (shared_package / "frames.py").write_text("")
    names = [rule["name"] for rule in load_policy(root, shared)["rules"]]
    assert names == ["spanning", "shared-package"]


def test_shared_cycle_exception_belongs_with_its_study(tmp_path):
    root = _tree(tmp_path, a="", b="")
    shared = _study(root, "some2020x", "", frames="", predictors="")
    shared.write_text(
        '[[cycle_exceptions]]\nreason = "Shared helpers."\n'
        'milestone = "extract-some2020x"\n'
        'edges = [["replica.studies.some2020x.frames", '
        '"replica.studies.some2020x.predictors"]]\n'
    )
    with pytest.raises(ValueError, match="belongs in that package"):
        load_policy(root, shared)


@pytest.mark.parametrize(
    "policy",
    [
        '[[rules]]\nname = "foreign"\n'
        'sources = ["replica.studies.other2020x.data.*"]\n'
        'forbidden = ["replica.b"]\n',
        '[[cycle_exceptions]]\nreason = "Shared helpers."\n'
        'milestone = "extract-some2020x"\n'
        'edges = [["replica.studies.some2020x.frames", '
        '"replica.studies.other2020x.data.*"]]\n',
    ],
)
def test_study_policy_file_cannot_name_another_package(tmp_path, capsys, policy):
    root = _tree(tmp_path, a="", b="")
    shared = _study(
        root,
        "some2020x",
        policy,
    )
    with pytest.raises(ValueError, match="outside replica.studies.some2020x"):
        load_policy(root, shared)
    assert main(["--root", str(root), "--policy", str(shared)]) == 1
    assert "replica.studies.other2020x.data.*" in capsys.readouterr().err


def test_cycle_graph_includes_typing_without_implicit_parent_edges(tmp_path):
    root = _tree(
        tmp_path,
        __init__="from . import a",
        a="from . import b",
        b="from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from . import a",
    )
    report = _check(root, {})
    assert report["cycle_edges"] == [
        ("replica.a", "replica.b"),
        ("replica.b", "replica.a"),
    ]
    assert len(report["errors"]) == 2
    assert any(edge["type_only"] for edge in report["edges"])
    _tree(root, b="def f():\n    from . import a")
    assert len(_check(root, {})["cycle_edges"]) == 2


def test_cycle_exceptions_require_reasons_and_milestones(tmp_path):
    root = _tree(tmp_path, a="from . import b", b="from . import a")
    policy = {"cycle_exceptions": [dict(edges=[["replica.a", "replica.b"]])]}
    assert any(
        "reason and milestone" in error for error in _check(root, policy)["errors"]
    )


def test_transitive_boundary_finds_helper_and_public_facade_bypasses(tmp_path):
    root = _tree(
        tmp_path,
        __init__="",
        reference="from . import helper",
        helper="from replica import evaluate",
    )
    rule = dict(
        name="independent",
        sources=["replica.reference"],
        forbidden=["replica"],
        transitive=True,
    )
    report = _check(root, {"rules": [rule]})
    assert report["errors"] == [
        "src/replica/helper.py:1: replica.reference -> replica.helper -> replica.evaluate [independent]"
    ]


@pytest.mark.parametrize("type_only", [False, True], ids=["runtime", "typing"])
@pytest.mark.parametrize(
    "source",
    [
        "import replica as public",
        "from replica import *",
        "import importlib as loader\npublic = loader.import_module('replica')",
        "public = __import__('replica', fromlist=['*'])",
    ],
    ids=["alias", "star", "import-module", "fromlist-star"],
)
def test_package_imports_cannot_hide_root_export_owners(tmp_path, source, type_only):
    if type_only:
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n" + indent(
            source, "    "
        )
    root = _tree(tmp_path, __init__="", contracts=source, core="")
    (root / "src/replica/__init__.pyi").write_text(
        (_ROOT / "src/replica/__init__.pyi").read_text()
    )
    name = "package-imports-name-root-exports"
    policy = {"rules": [r for r in load_policy(_ROOT)["rules"] if r["name"] == name]}
    report = _check(root, policy)
    assert len(report["errors"]) == 1
    assert report["errors"][0].endswith(f"replica.contracts -> replica [{name}]")
    edge = next(e for e in report["edges"] if e["target"] == "replica")
    assert edge["type_only"] is type_only

    _tree(root, contracts="from replica import Stimulus\n")
    assert _check(root, policy)["errors"] == []


@pytest.mark.parametrize("type_only", [False, True], ids=["runtime", "typing"])
def test_root_facade_keeps_selected_export_owner_for_boundaries_and_cycles(
    tmp_path, type_only
):
    source = "from replica import ModelFeature\n"
    if type_only:
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    " + source
    root = _tree(
        tmp_path,
        __init__="",
        contracts=source,
        feature="from . import contracts\n",
        core="",
    )
    (root / "src/replica/__init__.pyi").write_text(
        (_ROOT / "src/replica/__init__.pyi").read_text()
    )
    rule = dict(
        name="contracts-no-feature",
        sources=["replica.contracts"],
        forbidden=["replica.feature", "replica.feature.*"],
        transitive=True,
    )
    report = _check(root, {"rules": [rule]})
    line = 3 if type_only else 1
    assert report["errors"] == [
        "Unreviewed cycle edge: replica.contracts -> replica.feature",
        "Unreviewed cycle edge: replica.feature -> replica.contracts",
        f"src/replica/contracts.py:{line}: replica.contracts -> replica.feature.ModelFeature [contracts-no-feature]",
    ]
    edge = next(e for e in report["edges"] if e["owner"] == "replica.feature")
    assert edge["type_only"] is type_only
    assert _impact(report, "replica.feature")[0]["chain"] == [
        "replica.contracts",
        "replica.feature",
    ]
    # The stub describes possible exports; it must not import all of them.
    _tree(root, contracts="from replica import Stimulus\n")
    assert _check(root, {"rules": [rule]})["errors"] == []


def test_impact_explains_shortest_paths_with_aliases_types_and_cycles(tmp_path):
    root = _tree(
        tmp_path,
        changed="from . import helper_a",
        helper_a="from .changed import value as alias",
        helper_b="from . import changed",
        app="from . import helper_b, helper_a",
        typed="from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from . import changed",
        dynamic="import importlib as loader\nloader.import_module('replica.changed')",
        unrelated="",
    )
    report = _check(root, {})
    impact = {item["module"]: item for item in _impact(report, "replica.changed")}
    assert set(impact) == {
        f"replica.{name}"
        for name in ("helper_a", "helper_b", "app", "typed", "dynamic")
    }
    assert impact["replica.app"]["chain"] == [
        "replica.app",
        "replica.helper_a",
        "replica.changed",
    ]
    assert [(e["path"], e["line"]) for e in impact["replica.app"]["edges"]] == [
        ("src/replica/app.py", 1),
        ("src/replica/helper_a.py", 1),
    ]
    assert impact["replica.typed"]["edges"][0]["type_only"]
    assert impact["replica.dynamic"]["edges"][0]["line"] == 2
    assert _impact(report, "replica.unrelated") == []
    with pytest.raises(ValueError, match="Unknown source module"):
        _impact(report, "replica.typo")


def test_rescan_observes_edits_renames_deletions_and_stale_rules(tmp_path):
    root = _tree(tmp_path, a="from . import b", b="")
    policy = {
        "rules": [dict(name="b-no-a", sources=["replica.b"], forbidden=["replica.a"])]
    }
    before = _check(root, policy)
    assert _check(root, policy) == before
    package = root / "src/replica"
    (package / "b.py").rename(package / "c.py")
    (package / "a.py").write_text("from . import c\n")
    renamed = _check(root, policy)
    assert "replica.b" not in renamed["module_paths"]
    assert renamed["errors"] == ["b-no-a: source selector replica.b matches no modules"]
    assert _impact(renamed, "replica.c")[0]["chain"] == ["replica.a", "replica.c"]
    policy["rules"][0]["sources"] = ["replica.c"]
    assert _check(root, policy)["errors"] == []
    (package / "c.py").write_text("from . import a\n")
    assert any("[b-no-a]" in error for error in _check(root, policy)["errors"])
    (package / "c.py").unlink()
    deleted = _check(root, policy)
    assert "replica.c" not in deleted["module_paths"]
    assert deleted["cycle_edges"] == []
    assert deleted["edges"][0]["owner"] is None
    assert deleted["errors"] == ["b-no-a: source selector replica.c matches no modules"]


def test_impact_cli_reloads_policy_and_keeps_check_exit_status(tmp_path, capsys):
    root = _tree(tmp_path, a="from . import b", b="")
    policy = tmp_path / "policy.toml"
    policy.write_text(
        '[[rules]]\nname="a-no-b"\nsources=["replica.a"]\nforbidden=["replica.b"]\n'
    )
    args = [
        "--root",
        str(root),
        "--policy",
        str(policy),
        "--impact",
        "replica.b",
        "--json",
    ]
    assert main(args) == 1
    first = json.loads(capsys.readouterr().out)
    assert first["impact"]["replica.b"][0]["chain"] == ["replica.a", "replica.b"]
    policy.write_text("")
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["errors"] == []
    assert second["impact"] == first["impact"]
    with pytest.raises(SystemExit) as exc:
        main([*args, "--impact", "replica.typo"])
    assert exc.value.code == 2
    assert "Unknown source module" in capsys.readouterr().err
