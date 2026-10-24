"""Repository references resolve independently of local ignored files."""

import subprocess

import pytest

from scripts.checks.source.references import check


@pytest.mark.parametrize(
    "suffix,source",
    [
        (".py", "# See docs/guide.md.\n"),
        (".py", '"""See docs/guide.md."""\n'),
        (".py", 'def f():\n    """See docs/guide.md."""\n'),
        (".R", 'stop("Missing package; see docs/guide.md")\n'),
    ],
)
def test_prose_references_require_tracked_files(tmp_path, suffix, source):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / f"src/example{suffix}").write_text(source)
    subprocess.run(["git", "add", "src"], cwd=tmp_path, check=True)
    (tmp_path / ".git/info/exclude").write_text("/docs/guide.md\n")
    assert len(check(tmp_path)) == 1
    (tmp_path / "docs").mkdir()
    guide = tmp_path / "docs/guide.md"
    guide.write_text("# Guide\n")
    assert len(check(tmp_path)) == 1
    subprocess.run(["git", "add", "-f", "docs/guide.md"], cwd=tmp_path, check=True)
    assert check(tmp_path) == []
    guide.unlink()
    assert len(check(tmp_path)) == 1


def test_data_strings_and_external_paths_are_outside_scope(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text(
        'path = "docs/generated.md"\n'
        "# See upstream/scripts/build.py.\n"
        "# Historical frequencies from 2024-01-01 use the ``NOT IN`` predicate.\n"
    )
    subprocess.run(["git", "add", "src"], cwd=tmp_path, check=True)
    assert check(tmp_path) == []
