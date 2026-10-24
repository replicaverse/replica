"""Check wheel code and declared resources against the source package."""

import argparse
import json
import tomllib
from pathlib import Path
from zipfile import ZipFile

_PYTHON_SUFFIXES = {".py", ".pyc", ".pyi"}


def _is_data(name: str) -> bool:
    path = Path(name)
    return (
        path.suffix.lower() not in _PYTHON_SUFFIXES
        and "__pycache__" not in path.parts
        and not path.parts[0].endswith(".dist-info")
    )


def released_inputs(source: Path) -> set[str]:
    """Wheel paths of released study inputs and their archive manifests.

    ``release/public.toml`` beside the source names the released studies; a
    tree without it is the public tree, where every indexed study is released.

    Args:
        source: Source package directory, normally src/replica.
    """
    rules = source.parent.parent / "release" / "public.toml"
    released = None
    if rules.is_file():
        tiers = tomllib.loads(rules.read_text()).get("studies", {})
        released = {study for study, tier in tiers.items() if tier == "released"}
    paths = set()
    for index in sorted((source / "studies").glob("*/results/index.json")):
        study = index.parents[1].name
        if released is not None and study not in released:
            continue
        resources = json.loads(index.read_text()).get("resources", {})
        for name, resource in resources.items():
            if resource.get("role") == "input":
                path = (
                    "results/index.json"
                    if resource.get("archive") is not None
                    else name
                )
                paths.add(f"{source.name}/studies/{study}/{path}")
    return paths


def check_resources(wheel: Path, source: Path) -> tuple[int, int]:
    """Require source modules, typing files, R files, declared package data and released study inputs.

    Args:
        wheel: Built wheel archive to inspect.
        source: Source package directory, normally src/replica, below the
            project whose pyproject.toml declares the package data.

    Returns:
        The number of R resources and of other data files the wheel carries.

    Raises:
        ValueError: A required file is missing, or a packaged file is absent
            from the source tree or differs from it.
    """
    expected = {
        f"{source.name}/{path.relative_to(source).as_posix()}": path
        for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    r_expected = {name for name in expected if Path(name).suffix.lower() == ".r"}
    if not r_expected:
        raise ValueError(f"No source R resources found under {source}")
    required = r_expected | {
        name
        for name in expected
        if Path(name).suffix in {".py", ".pyi"} or Path(name).name == "py.typed"
    }
    project = tomllib.loads((source.parent.parent / "pyproject.toml").read_text())
    for package, patterns in project["tool"]["setuptools"]["package-data"].items():
        directory = source.parent.joinpath(*package.split("."))
        for pattern in patterns:
            required.update(
                path.relative_to(source.parent).as_posix()
                for path in directory.glob(pattern)
                if path.is_file()
            )
    required.update(released_inputs(source))
    with ZipFile(wheel) as archive:
        actual = {
            entry.filename: archive.read(entry)
            for entry in archive.infolist()
            if not entry.is_dir()
            and not Path(entry.filename).parts[0].endswith(".dist-info")
        }
    missing = sorted(required - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(
        name
        for name in expected.keys() & actual.keys()
        if expected[name].read_bytes() != actual[name]
    )
    if missing or unexpected or changed:
        raise ValueError(
            f"Wheel files differ from source: missing required files={missing}, "
            f"not in the source tree={unexpected}, changed={changed}"
        )
    r_count = sum(1 for name in actual if name in r_expected)
    return r_count, sum(_is_data(name) for name in actual) - r_count


def main() -> None:
    """Validate wheel contents against source modules and declared resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "src" / "replica",
    )
    args = parser.parse_args()
    r_count, other = check_resources(args.wheel, args.source)
    print(
        f"Wheel code and resources match source: {r_count} R files and "
        f"{other} other data files"
    )


if __name__ == "__main__":
    main()
