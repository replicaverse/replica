#!/usr/bin/env python3
"""Enforce import boundaries and reviewed cycle edges without importing backends."""

import argparse
import ast
import json
import sys
import tomllib
from collections import deque
from fnmatch import fnmatchcase
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: The policy file a study package may carry beside its code.
STUDY_POLICY = "architecture.toml"
sys.path.insert(0, str(ROOT / "src"))

from replica.utils.imports import import_references


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _module(path: Path, src: Path) -> str:
    parts = path.relative_to(src).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _owner(name: str, modules: dict[str, Path]) -> str | None:
    while name:
        if name in modules:
            return name
        name = name.rpartition(".")[0]
    return None


def _public_exports(src: Path) -> dict[str, str]:
    """Resolve the lazy root facade from its checked static export declarations."""
    stub = src / "replica/__init__.pyi"
    if not stub.is_file():
        return {}
    exports = {}
    for node in ast.parse(stub.read_text()).body:
        if isinstance(node, ast.ImportFrom):
            module = resolve_name("." * node.level + (node.module or ""), "replica")
            for alias in node.names:
                if alias.name != "*":
                    exports[f"replica.{alias.asname or alias.name}"] = (
                        f"{module}.{alias.name}"
                    )
    return exports


def _scoped(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _study_package(pattern: str, root: Path) -> str | None:
    """The study package a selector names; None for a shared package or a glob.

    A study package is a subpackage of the studies package with a README.
    """
    parts = pattern.split(".")
    if (
        parts[:2] != ["replica", "studies"]
        or len(parts) < 3
        or any(char in parts[2] for char in "*?[")
    ):
        return None
    if not (root / "src/replica/studies" / parts[2] / "README.md").is_file():
        return None
    return ".".join(parts[:3])


def load_policy(root: Path, policy_path: Path | None = None) -> dict:
    """The shared policy plus the policy file of every study package in the tree.

    A study file selects and excepts modules of its own package only, so a
    tree without that package carries none of its entries.

    Args:
        root: Repository root holding ``src`` and the shared policy.
        policy_path: The shared policy; defaults to the audited file.

    Raises:
        ValueError: A study file names a module outside its package, or a
            shared rule or cycle exception selects one study package only.
    """
    policy_path = policy_path or root / "scripts/checks/source/architecture.toml"
    with policy_path.open("rb") as stream:
        policy = tomllib.load(stream)
    policy.setdefault("rules", [])
    policy.setdefault("cycle_exceptions", [])
    for rule in policy["rules"]:
        owners = {_study_package(pattern, root) for pattern in rule["sources"]}
        if len(owners) == 1 and None not in owners:
            raise ValueError(
                f"{policy_path.relative_to(root)}: rule {rule['name']} selects only "
                f"{owners.pop()}; it belongs in that package's {STUDY_POLICY}"
            )
    for entry in policy["cycle_exceptions"]:
        owners = {
            _study_package(name, root) for edge in entry["edges"] for name in edge
        }
        if len(owners) == 1 and None not in owners:
            raise ValueError(
                f"{policy_path.relative_to(root)}: cycle exception for {owners.pop()} "
                f"belongs in that package's {STUDY_POLICY}"
            )
    for path in sorted((root / "src/replica/studies").glob(f"*/{STUDY_POLICY}")):
        package = f"replica.studies.{path.parent.name}"
        with path.open("rb") as stream:
            study = tomllib.load(stream)
        rules = study.get("rules", [])
        exceptions = study.get("cycle_exceptions", [])
        names = [pattern for rule in rules for pattern in rule["sources"]]
        names += [
            name for entry in exceptions for edge in entry["edges"] for name in edge
        ]
        foreign = sorted(name for name in names if not _scoped(name, package))
        if foreign:
            raise ValueError(
                f"{path.relative_to(root)} names modules outside {package}: {foreign}"
            )
        policy["rules"].extend(rules)
        policy["cycle_exceptions"].extend(exceptions)
    return policy


def _cyclic_edges(graph: dict[str, set[str]]) -> set[tuple[str, str]]:
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    active: set[str] = set()
    stack: list[str] = []
    cycles: set[tuple[str, str]] = set()

    def visit(node: str) -> None:
        indices[node] = low[node] = len(indices)
        stack.append(node)
        active.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] != indices[node]:
            return
        component = set()
        while True:
            member = stack.pop()
            active.remove(member)
            component.add(member)
            if member == node:
                break
        if len(component) > 1:
            cycles.update(
                (source, target)
                for source in component
                for target in graph[source] & component
            )

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return cycles


def _scan(root: Path) -> tuple[dict[str, Path], list[dict], list[str]]:
    src = root / "src"
    modules = {_module(p, src): p for p in sorted((src / "replica").rglob("*.py"))}
    exports = _public_exports(src)
    edges = []
    unresolved = []
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for ref in import_references(path.read_text(), package):
            if ref.module is None:
                unresolved.append(f"{path.relative_to(root)}:{ref.line}")
                continue
            for target in ref.members or (ref.module,):
                target = exports.get(target, target)
                edges.append(
                    dict(
                        source=name,
                        target=target,
                        owner=_owner(target, modules),
                        path=path.relative_to(root).as_posix(),
                        line=ref.line,
                        type_only=ref.type_only,
                    )
                )
    return modules, edges, unresolved


def _package_cycles(graph: dict[str, set[str]]) -> list[tuple[int, str, str]]:
    """Find cycles introduced by grouping modules at each package depth."""
    cycles = []
    maximum = max((len(name.split(".")) for name in graph), default=0)
    for depth in range(2, maximum):
        groups = {name: ".".join(name.split(".")[:depth]) for name in graph}
        grouped: dict[str, set[str]] = {name: set() for name in groups.values()}
        for source, targets in graph.items():
            grouped[groups[source]].update(
                groups[target] for target in targets if groups[target] != groups[source]
            )
        cycles.extend(
            (depth, source, target) for source, target in sorted(_cyclic_edges(grouped))
        )
    return cycles


def _check(root: Path, policy: dict) -> dict:
    modules, edges, unresolved = _scan(root)
    errors = []
    rules = policy.get("rules", [])
    names = [rule["name"] for rule in rules]
    if len(set(names)) != len(names):
        errors.append("Duplicate architecture rule name")
    for rule in rules:
        for pattern in rule["sources"]:
            if not any(fnmatchcase(module, pattern) for module in modules):
                errors.append(
                    f"{rule['name']}: source selector {pattern} matches no modules"
                )
        errors.extend(_rule_findings(rule, modules, edges))
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for edge in edges:
        if edge["owner"]:
            graph[edge["source"]].add(edge["owner"])
    cycles = _cyclic_edges(graph)
    exceptions = policy.get("cycle_exceptions", [])
    declared = [tuple(edge) for entry in exceptions for edge in entry["edges"]]
    if len(set(declared)) != len(declared):
        errors.append("Duplicate cycle exception")
    allowed = set(declared)
    for entry in exceptions:
        if (
            not entry.get("reason", "").strip()
            or not entry.get("milestone", "").strip()
        ):
            errors.append(f"Cycle exception needs a reason and milestone: {entry}")
    for source, target in sorted(cycles - allowed):
        errors.append(f"Unreviewed cycle edge: {source} -> {target}")
    for source, target in sorted(allowed - cycles):
        errors.append(f"Stale cycle exception: {source} -> {target}")
    package_cycles = _package_cycles(graph)
    for depth, source, target in package_cycles:
        errors.append(f"Package cycle at depth {depth}: {source} -> {target}")
    return dict(
        errors=sorted(errors),
        unresolved_dynamic_imports=sorted(unresolved),
        modules=len(modules),
        module_paths={
            name: path.relative_to(root).as_posix() for name, path in modules.items()
        },
        edges=edges,
        cycle_edges=sorted(cycles),
        package_cycle_edges=package_cycles,
    )


def _impact(report: dict, changed: str) -> list[dict]:
    if changed not in report["module_paths"]:
        raise ValueError(f"Unknown source module: {changed}")
    incoming: dict[str, list[dict]] = {name: [] for name in report["module_paths"]}
    for edge in report["edges"]:
        if edge["owner"]:
            incoming[edge["owner"]].append(edge)
    for edges in incoming.values():
        edges.sort(
            key=lambda edge: (
                edge["source"],
                edge["path"],
                edge["line"],
                edge["target"],
            )
        )
    routes: dict[str, list[dict]] = {changed: []}
    pending = deque([changed])
    while pending:
        target = pending.popleft()
        for edge in incoming[target]:
            source = edge["source"]
            if source not in routes:
                routes[source] = [edge, *routes[target]]
                pending.append(source)
    return [
        dict(
            module=source,
            chain=[edge["source"] for edge in routes[source]] + [changed],
            edges=routes[source],
        )
        for source in sorted(routes)
        if source != changed
    ]


def _rule_findings(rule: dict, modules: dict, edges: list[dict]) -> list[str]:
    outgoing: dict[str, list[dict]] = {name: [] for name in modules}
    for edge in edges:
        outgoing[edge["source"]].append(edge)
    errors = []
    for origin in modules:
        if not _matches(origin, rule["sources"]):
            continue
        pending = [(origin, ())]
        visited = set()
        while pending:
            source, chain = pending.pop()
            if source in visited:
                continue
            visited.add(source)
            for edge in outgoing[source]:
                if any(
                    name and _matches(name, rule["forbidden"])
                    for name in (edge["target"], edge["owner"])
                ):
                    route = " -> ".join((*chain, source, edge["target"]))
                    errors.append(
                        f"{edge['path']}:{edge['line']}: {route} [{rule['name']}]"
                    )
                elif rule.get("transitive") and edge["owner"]:
                    pending.append((edge["owner"], (*chain, source)))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Check the source graph against the policy and optionally emit its facts.

    Args:
        argv: Command arguments; None uses the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--impact",
        action="append",
        default=[],
        metavar="MODULE",
        help="explain transitive importers of a source module; repeat for multiple modules",
    )
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.root, args.policy)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    report = _check(args.root, policy)
    if args.impact:
        try:
            report["impact"] = {
                name: _impact(report, name) for name in sorted(set(args.impact))
            }
        except ValueError as exc:
            parser.error(str(exc))
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(error)
        print(
            f"Architecture: {report['modules']} modules, "
            f"{len(report['errors'])} errors, "
            f"{len(report['unresolved_dynamic_imports'])} computed imports to review."
        )
        for changed, affected in report.get("impact", {}).items():
            print(
                f"Impact of {changed}: {len(affected)} transitive importers (including typing imports)"
            )
            for item in affected:
                locations = ", ".join(
                    f"{edge['path']}:{edge['line']}" for edge in item["edges"]
                )
                print(f"  {' -> '.join(item['chain'])} [{locations}]")
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
