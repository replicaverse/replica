"""Check repository paths in Python prose and R string literals."""

import ast
import io
import re
import subprocess
import tokenize
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_PATH = re.compile(
    r"(?<![\w/\-])(?:src/replica|tests|scripts|docs)/"
    r"[\w./\-]+\.(?:py|R|md|ipynb|sh|tsv|yaml|toml|Rmd|csv|json)\b"
)
_R_TEXT = re.compile(r"""\#[^\n]*|(?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")


def _prose(source: str) -> Iterator[tuple[int, str]]:
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                yield node.body[0].lineno, doc


def check(root: Path) -> list[str]:
    """Report references to files absent from the tracked tree.

    Args:
        root: Git checkout containing the source, scripts and tests.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
    )
    tracked = set(filter(None, listing.stdout.split("\0")))
    errors = []
    for name in sorted(tracked):
        path = root / name
        if (
            not name.startswith(("src/", "tests/", "scripts/"))
            or path.suffix not in {".py", ".R"}
            or not path.is_file()
        ):
            continue
        source = path.read_text(encoding="utf8")
        sections = (
            _prose(source)
            if path.suffix == ".py"
            else (
                (source.count("\n", 0, match.start()) + 1, match.group())
                for match in _R_TEXT.finditer(source)
                if match.group("string") is not None
            )
        )
        for line, prose in sections:
            for match in _PATH.finditer(prose):
                target = match.group(0)
                if target not in tracked or not (root / target).is_file():
                    number = line + prose.count("\n", 0, match.start())
                    errors.append(f"{name}:{number}: missing repository file {target}")
    return errors


if __name__ == "__main__":
    errors = check(ROOT)
    if errors:
        print("\n".join(errors))
    raise SystemExit(bool(errors))
