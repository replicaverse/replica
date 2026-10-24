"""check_resources on synthetic wheels."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.checks.distribution.wheel_contents import check_resources


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "src" / "replica"
    (root / "studies" / "one").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "studies" / "one" / "anchor.R").write_text("x <- 1\n")
    (root / "studies" / "one" / "fix.patch").write_text("--- a\n+++ b\n")
    (root / "studies" / "one" / "table.tsv").write_text("a\n1\n")
    (root.parent.parent / "pyproject.toml").write_text(
        '[tool.setuptools.package-data]\n"replica" = ["**/*.R"]\n'
    )
    return root


def _wheel(path: Path, files: dict[str, str]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def test_matching_wheel_passes(source, tmp_path):
    wheel = _wheel(
        tmp_path / "ok.whl",
        {
            "replica/__init__.py": "",
            "replica/studies/one/anchor.R": "x <- 1\n",
            "replica/studies/one/fix.patch": "--- a\n+++ b\n",
            "replica-0.0.1.dist-info/METADATA": "Name: replica\n",
        },
    )
    assert check_resources(wheel, source) == (1, 1)


def test_file_absent_from_source_fails(source, tmp_path):
    wheel = _wheel(
        tmp_path / "stale.whl",
        {
            "replica/studies/one/anchor.R": "x <- 1\n",
            "replica/external/adapters/old.patch": "stale\n",
        },
    )
    with pytest.raises(
        ValueError,
        match="not in the source tree=\\['replica/external/adapters/old.patch'\\]",
    ):
        check_resources(wheel, source)


def test_missing_r_file_or_changed_file_fails(source, tmp_path):
    with pytest.raises(
        ValueError, match="missing required files=\\['replica/studies/one/anchor.R'\\]"
    ):
        check_resources(
            _wheel(tmp_path / "no_r.whl", {"replica/__init__.py": ""}), source
        )
    wheel = _wheel(
        tmp_path / "changed.whl",
        {
            "replica/studies/one/anchor.R": "x <- 1\n",
            "replica/studies/one/table.tsv": "a\n2\n",
        },
    )
    with pytest.raises(
        ValueError, match="changed=\\['replica/studies/one/table.tsv'\\]"
    ):
        check_resources(wheel, source)


def test_source_file_absent_from_wheel_passes(source, tmp_path):
    wheel = _wheel(
        tmp_path / "lean.whl",
        {"replica/__init__.py": "", "replica/studies/one/anchor.R": "x <- 1\n"},
    )
    assert check_resources(wheel, source) == (1, 0)


@pytest.mark.parametrize(
    "name", ["__init__.py", "__init__.pyi", "py.typed", "dataset/catalog.yaml"]
)
def test_missing_code_typing_or_declared_data_fails(source, tmp_path, name):
    path = source / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("")
    (source.parent.parent / "pyproject.toml").write_text(
        '[tool.setuptools.package-data]\n"replica" = ["**/*.R", "dataset/*.yaml"]\n'
    )
    contents = {"replica/studies/one/anchor.R": "x <- 1\n"}
    if name != "__init__.py":
        contents["replica/__init__.py"] = ""
    with pytest.raises(ValueError, match="missing required files") as exc:
        check_resources(_wheel(tmp_path / "missing.whl", contents), source)
    assert f"replica/{name}" in str(exc.value)


@pytest.mark.parametrize("stale", [True, False])
def test_stale_or_changed_python_file_fails(source, tmp_path, stale):
    contents = {
        "replica/__init__.py": "",
        "replica/studies/one/anchor.R": "x <- 1\n",
    }
    contents["replica/deleted.py" if stale else "replica/__init__.py"] = "old code"
    with pytest.raises(
        ValueError, match="not in the source tree" if stale else "changed="
    ):
        check_resources(_wheel(tmp_path / "old-code.whl", contents), source)


def test_released_inputs_require_packaged_files_or_archive_manifests(source, tmp_path):
    import hashlib
    import json

    study = source / "studies" / "one"
    (study / "results").mkdir()
    (study / "data").mkdir()
    (study / "data" / "table.parquet").write_bytes(b"rows")
    digest = hashlib.sha256(b"rows").hexdigest()
    (study / "results" / "index.json").write_text(
        json.dumps(
            {"resources": {"data/table.parquet": {"role": "input", "sha256": digest}}}
        )
    )
    files = {"replica/__init__.py": "", "replica/studies/one/anchor.R": "x <- 1\n"}
    with pytest.raises(
        ValueError,
        match="missing required files=\\['replica/studies/one/data/table.parquet'\\]",
    ):
        check_resources(_wheel(tmp_path / "public.whl", files), source)
    rules = source.parent.parent / "release"
    rules.mkdir()
    (rules / "public.toml").write_text('[studies]\none = "released"\n')
    with pytest.raises(ValueError, match="missing required files=.*table.parquet"):
        check_resources(_wheel(tmp_path / "released.whl", files), source)
    (rules / "public.toml").write_text('[studies]\nother = "released"\n')
    assert check_resources(_wheel(tmp_path / "integration.whl", files), source) == (
        1,
        0,
    )
    (rules / "public.toml").write_text('[studies]\none = "released"\n')
    index = study / "results" / "index.json"
    document = json.loads(index.read_text())
    document["resources"]["data/table.parquet"]["archive"] = None
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="missing required files=.*table.parquet"):
        check_resources(_wheel(tmp_path / "local-input.whl", files), source)
    document["resources"]["data/table.parquet"]["archive"] = "a" * 64
    index.write_text(json.dumps(document))
    (study / "data" / "table.parquet").unlink()
    with pytest.raises(ValueError, match="missing required files=.*results/index.json"):
        check_resources(_wheel(tmp_path / "missing-manifest.whl", files), source)
    files["replica/studies/one/results/index.json"] = index.read_text()
    assert check_resources(_wheel(tmp_path / "archived.whl", files), source) == (1, 1)
    files["replica/studies/one/results/index.json"] = "{}"
    with pytest.raises(ValueError, match="changed=.*results/index.json"):
        check_resources(_wheel(tmp_path / "stale-manifest.whl", files), source)
