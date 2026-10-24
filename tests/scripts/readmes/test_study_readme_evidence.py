"""Study renderers stay independent of analysis imports and preserve README prose."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.readmes.regen_study_readmes import STUDIES_DIR, with_evidence

BEGIN = "<!-- BEGIN example -->"
END = "<!-- END example -->"


@pytest.fixture
def renderer(tmp_path):
    (tmp_path / "readme_evidence.py").write_text(
        f"BEGIN = {BEGIN!r}\nEND = {END!r}\n"
        "def render():\n    return BEGIN + '\\nnew evidence\\n' + END\n"
    )
    return tmp_path


def test_optional_evidence_preserves_surrounding_prose(renderer, tmp_path):
    text = f"# Study\n\nBefore\n{BEGIN}\nold evidence\n{END}\nAfter\n"
    assert with_evidence(text, renderer) == text.replace("old evidence", "new evidence")
    assert with_evidence(text, tmp_path / "no_renderer") == text


@pytest.mark.parametrize(
    "text",
    [f"{BEGIN}\nmissing end", f"{BEGIN}\n{BEGIN}\n{END}", f"{END}\n{BEGIN}"],
)
def test_invalid_markers_cannot_overwrite_prose(renderer, text):
    with pytest.raises(ValueError, match="evidence"):
        with_evidence(text, renderer)


def test_incomplete_rendered_block_is_rejected(renderer):
    (renderer / "readme_evidence.py").write_text(
        f"BEGIN = {BEGIN!r}\nEND = {END!r}\ndef render():\n    return 'incomplete'\n"
    )
    with pytest.raises(ValueError, match="delimited block"):
        with_evidence(f"{BEGIN}\n{END}", renderer)


def test_renderers_load_without_importing_analysis_backends():
    script = """
import importlib.abc, runpy, sys
class NoAnalysis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('numpy', 'pandas', 'scipy', 'statsmodels', 'torch', 'rpy2',
                   'replica.analysis', 'replica.dataset', 'replica.feature', sys.argv[2])
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in blocked):
            raise AssertionError('renderer imported analysis code: ' + fullname)
sys.meta_path.insert(0, NoAnalysis())
module = runpy.run_path(sys.argv[1])
assert callable(module['render'])
assert module['BEGIN'] and module['END']
"""
    for path in sorted(STUDIES_DIR.glob("*/readme_evidence.py")):
        study = path.parent.name
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(path), f"replica.studies.{study}"],
            cwd=Path(__file__).parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (study, result.stderr)
