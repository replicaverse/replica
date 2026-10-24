"""The studies package exports every study package and resolves every study class."""

from pathlib import Path

from replica import studies

_STUDIES_DIR = Path(studies.__file__).parent


def test_study_ids_are_the_subpackages_with_a_readme():
    expected = sorted(
        path.name
        for path in _STUDIES_DIR.iterdir()
        if (path / "__init__.py").is_file() and (path / "README.md").is_file()
    )
    assert list(studies.STUDY_IDS) == expected


def test_all_lists_every_study_package():
    assert set(studies.STUDY_IDS) <= set(studies.__all__)
    assert {"ALL_STUDIES", "STUDY_IDS", "DataDependency", "Study"} <= set(
        studies.__all__
    )


_SHARED_PACKAGES = {"claims", "grouped", "records", "shain"}


def test_shared_packages_are_not_studies():
    assert not _SHARED_PACKAGES & set(studies.STUDY_IDS)


def test_every_other_subpackage_is_a_discovered_study():
    """A study package without a README would otherwise vanish from discovery."""
    subpackages = {
        path.name
        for path in _STUDIES_DIR.iterdir()
        if (path / "__init__.py").is_file() and path.name != "__pycache__"
    }
    assert subpackages - _SHARED_PACKAGES == set(studies.STUDY_IDS)
