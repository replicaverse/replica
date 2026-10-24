"""Static Python import references for architecture and provenance checks."""

import ast
from dataclasses import dataclass
from importlib.util import resolve_name


@dataclass(frozen=True)
class ImportReference:
    """A source import, including imported members and its execution context.

    Args:
        module: Resolved module name, or None for a computed dynamic import.
        members: Fully qualified names imported from the module.
        line: Source line containing the import.
        type_only: Whether the import is guarded by TYPE_CHECKING.
    """

    module: str | None
    members: tuple[str, ...]
    line: int
    type_only: bool


class _Imports(ast.NodeVisitor):
    def __init__(self, tree: ast.Module, package: str):
        self.package = package
        self.references: list[ImportReference] = []
        self.type_only = False
        self.aliases = {"__import__": "builtins.__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom) and not node.level:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = (
                        f"{node.module}.{alias.name}"
                    )

    def _qualified(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self._qualified(node.value)}.{node.attr}"
        return ""

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        previous = self.type_only
        self.type_only |= self._qualified(node.test) == "typing.TYPE_CHECKING"
        for child in node.body:
            self.visit(child)
        self.type_only = previous
        for child in node.orelse:
            self.visit(child)

    def visit_Import(self, node: ast.Import) -> None:
        self.references.extend(
            ImportReference(alias.name, (), node.lineno, self.type_only)
            for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        name = "." * node.level + (node.module or "")
        module = resolve_name(name, self.package) if node.level else name
        members = tuple(
            f"{module}.{alias.name}" for alias in node.names if alias.name != "*"
        )
        self.references.append(
            ImportReference(module, members, node.lineno, self.type_only)
        )

    def visit_Call(self, node: ast.Call) -> None:
        function = self._qualified(node.func)
        if function in {"importlib.import_module", "builtins.__import__"}:
            args = {k.arg: k.value for k in node.keywords}
            name = node.args[0] if node.args else args.get("name")
            module = name.value if isinstance(name, ast.Constant) else None
            if not isinstance(module, str):
                module = None
            elif module.startswith("."):
                package = node.args[1] if len(node.args) > 1 else args.get("package")
                context = (
                    package.value
                    if isinstance(package, ast.Constant)
                    and isinstance(package.value, str)
                    else self.package
                )
                module = resolve_name(module, context)
            self.references.append(
                ImportReference(
                    module,
                    self._fromlist(node, args, module)
                    if function == "builtins.__import__"
                    else (),
                    node.lineno,
                    self.type_only,
                )
            )
        self.generic_visit(node)

    @staticmethod
    def _fromlist(node: ast.Call, args: dict, module: str | None) -> tuple[str, ...]:
        value = node.args[3] if len(node.args) > 3 else args.get("fromlist")
        if module is None or not isinstance(value, (ast.List, ast.Tuple)):
            return ()
        return tuple(
            f"{module}.{item.value}"
            for item in value.elts
            if isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and item.value != "*"
        )


def import_references(source: str, package: str) -> tuple[ImportReference, ...]:
    """Find static and recognized dynamic imports without executing source.

    Aliases of importlib and typing imports are recognized. Computed dynamic
    imports are retained with no resolved module; arbitrary methods named
    ``import_module`` are not treated as imports. Alias resolution is syntactic:
    rebinding and control flow are not followed.

    Args:
        source: Python source text.
        package: Package containing the module, or the package itself for an init.
    """
    tree = ast.parse(source)
    visitor = _Imports(tree, package)
    visitor.visit(tree)
    return tuple(visitor.references)


def import_names(source: str, package: str) -> set[str]:
    """Return modules and imported members, including typing-only references.

    Args:
        source: Python source text.
        package: Package against which relative imports resolve.
    """
    return {
        name
        for reference in import_references(source, package)
        for name in (reference.module, *reference.members)
        if name is not None
    }
