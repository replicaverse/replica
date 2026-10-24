#!/usr/bin/env python3
"""Check public-symbol docstrings, summary lines, and parameter descriptions.

Google-style Args sections must match callable signatures. Unsupported section
names and undocumented public symbols are reported. Tests and designated
upstream sources are excluded.
"""

import argparse
import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOTS = ("src/replica", "scripts")

# The camera-ready paper cites this path, so its prose is fixed. The study's
# MMC producer package is library code and stays in scope.
EXCLUDE = ("src/replica/studies/hopton2026unigram",)
IN_SCOPE = ("src/replica/studies/hopton2026unigram/mmc/",)

# Dunders that carry an API contract worth documenting; the rest are protocol
# plumbing whose meaning is fixed by the data model.
DOCUMENTED_DUNDERS = frozenset({"__init__", "__call__"})

SECTION_HEAD = re.compile(r"^[ \t]*(?:Args|Arguments|Parameters)[ \t]*:[ \t]*$")
# quartodoc's markdown renderer raises NotImplementedError on these two griffe
# sections, which fails the docs build rather than degrading the page. Describe
# a generator's output under ``Returns:`` and a warning in prose.
UNRENDERABLE_HEAD = re.compile(r"^[ \t]*(Yields|Warns)[ \t]*:[ \t]*$")
OTHER_HEAD = re.compile(
    r"^[ \t]*(?:Returns|Yields|Raises|Attributes|Examples?|Notes?|References|"
    r"Warns|Warnings|See Also|Todo)[ \t]*:[ \t]*$"
)
# "name:", "name (int):", "*args:", "**kwargs:" at the start of an entry. The
# colon binds to the name, so a continuation line carrying a Sphinx role
# (``see :data:`WINDOWS```) stays part of the description.
ENTRY = re.compile(r"^([ \t]*)(\*{0,2}[A-Za-z_]\w*)[ \t]*(?:\([^)]*\))?:")


def documented_params(docstring: str) -> set[str]:
    """Parameter names an ``Args:`` section describes.

    Args:
        docstring: The raw docstring to read, section headings included.

    Returns:
        The names, stripped of any leading asterisks. Empty when the docstring
        has no ``Args:`` section.
    """
    names: set[str] = set()
    inside = False
    entry_indent: int | None = None
    for line in docstring.splitlines():
        if SECTION_HEAD.match(line):
            inside, entry_indent = True, None
            continue
        if not inside:
            continue
        if OTHER_HEAD.match(line) or (line.strip() and not line[:1].isspace()):
            inside = False
            continue
        match = ENTRY.match(line)
        if not match:
            continue
        indent = len(match.group(1).expandtabs())
        if entry_indent is None:
            entry_indent = indent
        # A deeper line continues the entry above it, colon or not.
        if indent <= entry_indent:
            names.add(match.group(2).lstrip("*"))
    return names


def required_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameter names the docstring has to describe.

    Args:
        node: The function definition to read the signature from.

    Returns:
        Every parameter in declaration order, ``*args`` and ``**kwargs``
        included, without ``self``/``cls`` and without the underscore names a
        signature uses to declare a parameter unused.
    """
    return [p for p in signature_params(node) if not p.startswith("_")]


def signature_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Every parameter name a call can bind, ``self``/``cls`` aside.

    Args:
        node: The function definition to read the signature from.

    Returns:
        Named parameters in declaration order, then ``*args`` and ``**kwargs``.
        Describing any of these is allowed; :func:`required_params` says which
        ones a docstring has to describe.
    """
    args = node.args
    named = (*args.posonlyargs, *args.args, *args.kwonlyargs)
    variadic = (p for p in (args.vararg, args.kwarg) if p is not None)
    return [p.arg for p in (*named, *variadic) if p.arg not in ("self", "cls")]


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Decorator names on *node*, dotted decorators reduced to their attribute."""
    out = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            out.add(target.id)
        elif isinstance(target, ast.Attribute):
            out.add(target.attr)
    return out


def _is_public(name: str) -> bool:
    """True for a name the API reference renders."""
    return not name.startswith("_") or name in DOCUMENTED_DUNDERS


def _walk_api(
    body: list[ast.stmt], prefix: str = "", class_doc: str | None = None
) -> Iterator[tuple[ast.AST, str, str | None]]:
    """Public definitions at API level, paired with their enclosing class docstring.

    Definitions inside a function body are closures, so recursion follows class
    bodies and stops at function bodies.

    Args:
        body: Statements of the module or class body to walk.
        prefix: Dotted name of the enclosing class, empty at module level.
        class_doc: Docstring of the enclosing class, ``None`` at module level.

    Returns:
        One ``(node, qualified_name, enclosing class docstring)`` triple per
        public definition.
    """
    for node in body:
        if isinstance(node, ast.ClassDef):
            if not _is_public(node.name):
                continue
            qualified = f"{prefix}{node.name}"
            yield node, qualified, class_doc
            yield from _walk_api(node.body, f"{qualified}.", ast.get_docstring(node))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _is_public(node.name) and "overload" not in _decorators(node):
                yield node, f"{prefix}{node.name}", class_doc


def violations(path: Path) -> list[tuple[int, str, str]]:
    """Every rule violation in one file, as ``(line, rule, detail)``.

    Args:
        path: The Python file to check.

    Returns:
        One entry per violation; ``rule`` is ``module``, ``summary``,
        ``section``, ``missing``, ``params``, or ``phantom``. Empty when the
        file is clean or cannot be parsed.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    out: list[tuple[int, str, str]] = []
    module_doc = ast.get_docstring(tree)
    if not module_doc:
        out.append((1, "module", "module has no docstring"))
    else:
        lines = module_doc.splitlines()
        if len(lines) > 1 and lines[1].strip():
            out.append((1, "summary", "summary line runs into the next line"))
    for holder in ast.walk(tree):
        if not isinstance(
            holder, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        doc = ast.get_docstring(holder, clean=False)
        if not doc:
            continue
        for offset, line in enumerate(doc.splitlines()):
            found = UNRENDERABLE_HEAD.match(line)
            if found:
                lineno = getattr(holder, "lineno", 1) + offset
                out.append((lineno, "section", f"{found.group(1)}: does not render"))
    for node, name, class_doc in _walk_api(tree.body):
        doc = ast.get_docstring(node)
        if not doc:
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            out.append((node.lineno, "missing", f"{kind} {name}"))
            continue
        if isinstance(node, ast.ClassDef):
            continue
        wanted = required_params(node)
        described = documented_params(doc)
        if node.name == "__init__" and class_doc:
            # A class may describe its constructor in the class docstring.
            described |= documented_params(class_doc)
        undescribed = [p for p in wanted if p not in described]
        if undescribed:
            out.append((node.lineno, "params", f"{name}: {', '.join(undescribed)}"))
        phantom = sorted(described - set(signature_params(node)))
        if phantom and described:
            out.append((node.lineno, "phantom", f"{name}: {', '.join(phantom)}"))
    return out


def targets(roots: list[str]) -> list[Path]:
    """Python files in scope, sorted.

    Args:
        roots: Repo-relative directories or files to scan.

    Returns:
        Files under *roots*, minus caches, private modules, and the excluded
        study package outside ``IN_SCOPE``. Privacy is read from the path relative to the
        repository, so only directories inside it can exclude a file.
    """
    found: list[Path] = []
    for root in roots:
        base = REPO / root
        candidates = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in candidates:
            rel = path.relative_to(REPO)
            posix = rel.as_posix()
            if "__pycache__" in rel.parts or (
                posix.startswith(EXCLUDE) and not posix.startswith(IN_SCOPE)
            ):
                continue
            if any(
                part.startswith("_") and part != "__init__.py" for part in rel.parts
            ):
                continue
            found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    """Check every file in scope and print the violations.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        1 when a path is missing, no file is in scope, or any rule is
        violated, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Directories to check (default: all)")
    parser.add_argument(
        "--report", action="store_true", help="Group the violations by rule"
    )
    args = parser.parse_args(argv)

    roots = args.paths or list(ROOTS)
    missing = [root for root in roots if not (REPO / root).exists()]
    if missing:
        print(f"no such path in the repository: {', '.join(missing)}", file=sys.stderr)
        return 1
    scope = targets(roots)
    if not scope:
        print(f"no Python files in scope under {', '.join(roots)}", file=sys.stderr)
        return 1
    found: list[tuple[Path, int, str, str]] = []
    for path in scope:
        found.extend((path, *v) for v in violations(path))

    if not found:
        print(f"  ok  docstrings: {len(scope)} files, no violations")
        return 0

    counts: dict[str, int] = {}
    for _, _, rule, _ in found:
        counts[rule] = counts.get(rule, 0) + 1
    if args.report:
        for rule in sorted(counts):
            print(f"\n{rule} ({counts[rule]}):")
            for path, line, kind, detail in found:
                if kind == rule:
                    print(f"  {path.relative_to(REPO)}:{line} {detail}")
    else:
        for path, line, rule, detail in found[:40]:
            print(f"  {path.relative_to(REPO)}:{line}: {rule} {detail}")
        if len(found) > 40:
            print(f"  ... {len(found) - 40} more; rerun with --report")

    summary = ", ".join(f"{rule} {n}" for rule, n in sorted(counts.items()))
    print(f"\n{len(found)} docstring violation(s): {summary}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
