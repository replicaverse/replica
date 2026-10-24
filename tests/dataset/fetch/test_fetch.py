"""Consistency tests between the dataset-fetch registry and the loaders.

Validates that the download catalog
(``replica.dataset.catalog.sources.DATASETS``)
is consistent with the actual dataset loader classes and dataset.yaml metadata
files, without performing any network calls.

Run with: pytest tests/dataset/fetch/test_fetch.py -v
"""

import fnmatch
from pathlib import Path

import pytest

from replica.dataset import describe_dataset, find_dataset_yaml, load_yaml
from replica.dataset import fetch as fetch_mod
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.catalog.registry import data_dir as _data_dir
from replica.dataset.catalog.sources import (
    DATASETS,
    dataset_directory,
    provenance_paths_for,
)
from replica.dataset.eye_tracking.meco.releases import (
    FIXATION_RELEASES,
    WORD_RELEASES,
)
from replica.dataset.fetch import list_datasets
from replica.dataset.fetch.sources import link_study_release


def test_fetch_default_verifies_download_not_regenerates(monkeypatch, tmp_path):
    """The default fetch path verifies the download against the shipped
    checksums. Regenerating the baseline from a corrupt/truncated download
    would baseline the corruption as 'correct' and defeat load-time
    verification, so regeneration stays an opt-in maintainer path."""
    calls: list[str] = []
    monkeypatch.setattr(fetch_mod, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(fetch_mod, "ensure_ssl_cert_chain", lambda: None)
    monkeypatch.setattr(fetch_mod, "_fetch_dataset_source", lambda *a, **k: None)
    monkeypatch.setattr(
        fetch_mod, "_verify_download", lambda *a, **k: calls.append("verify")
    )
    monkeypatch.setattr(
        fetch_mod, "_regenerate_yaml", lambda *a, **k: calls.append("regenerate")
    )

    fetch_mod.fetch_dataset("provo")  # end-user default
    fetch_mod.fetch_dataset("provo", regenerate_yaml=True)  # maintainer opt-in
    assert calls == ["verify", "regenerate"]


def test_fetch_all_excludes_large_optional_fixation_bundles(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fetch_mod, "fetch_dataset", lambda name, **_kwargs: calls.append(name)
    )
    fetch_mod.run_fetch(names=[], fetch_all=True)
    assert calls
    assert not any(name.endswith("_fixations") for name in calls)


def test_meco_fetch_configs_are_derived_from_release_manifest():
    """Paths, object IDs, filenames, and checksums must share one authority."""
    for key, release in WORD_RELEASES.items():
        config = DATASETS[key]
        assert config["source"] == "osf"
        assert config["files"] == [(release.osf_id, release.filename)]
        assert dataset_directory(key) == release.data_subdir

    for key, release in FIXATION_RELEASES.items():
        config = DATASETS[f"{key}_fixations"]
        assert config["source"] == "osf"
        assert config["dir"] == release.data_subdir
        assert config["files"] == [(release.osf_id, release.rda_filename)]
        assert config["sha256"] == {release.rda_filename: release.source_sha256}
        assert config["include_in_all"] is False


# ---------------------------------------------------------------------------
# Valid source types recognized by replica.dataset.fetch
# ---------------------------------------------------------------------------

VALID_SOURCE_TYPES = {
    "osf",
    "osf_zip",
    "github_raw",
    "zip",
    "direct_url",
    "manual",
    "study_release",
}

# Mapping from fetch key -> expected directory (relative to repo root). Dirs are
# resolved from the registry via ``dataset_directory``; this pins the convention
# (datasets/<paradigm>/{original,processed}/<name>) for the datasets named here.
EXPECTED_DIR_PATTERNS = {
    "provo": "eye-tracking/original/provo",
    "meco": "eye-tracking/original/meco",
    "natural_stories": "self-paced-reading/original/natural_stories",
    "ucl_spr": "self-paced-reading/original/ucl",
    "ucl_et": "eye-tracking/original/ucl",
    "geco": "eye-tracking/original/geco",
    "celer": "eye-tracking/original/celer",
    "onestop": "eye-tracking/original/onestop",
    "copco": "eye-tracking/original/copco",
    "potec": "eye-tracking/original/potec",
    "emtec": "eye-tracking/original/emtec",
    "sbsat": "eye-tracking/original/sbsat",
    "bsc": "eye-tracking/original/bsc",
    # Shain et al. (2024) preprocessed corpora, public release of Brown SPR /
    # Dundee whose raw data is licence-restricted. Lives under processed/.
    "dundee": "eye-tracking/processed/dundee_shain2024logarithmic",
    "brown_spr": "self-paced-reading/processed/brown_spr_shain2024logarithmic",
    "brothers_kuperberg": "self-paced-reading/original/brothers_kuperberg",
    "zuco2": "eye-tracking/original/zuco2",
    "narratives": "fmri/original/narratives",
}


# ---------------------------------------------------------------------------
# 1. Every dataset has the correct directory path
# ---------------------------------------------------------------------------


class TestDirectoryPaths:
    """Verify directory paths in DATASETS follow the expected convention."""

    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_dir_layout(self, name: str):
        """Every entry's dir lives under ``datasets/{original,processed}/``
        and, where pinned, matches the documented expected path."""
        d = dataset_directory(name)
        assert not d.startswith("/") and not d.startswith("datasets/"), (
            f"{name}: dir={d} must be relative to data_dir() "
            f"(no leading slash, no 'datasets/' prefix)"
        )
        assert "/original/" in d or "/processed/" in d or d.startswith("frequency/"), (
            f"{name}: dir={d} does not contain '/original/' or '/processed/' "
            f"(frequency resources excepted)"
        )
        if name in EXPECTED_DIR_PATTERNS:
            assert d == EXPECTED_DIR_PATTERNS[name], (
                f"{name}: dir={d} != expected {EXPECTED_DIR_PATTERNS[name]}"
            )


# ---------------------------------------------------------------------------
# 2. Dataset YAML files are consistent
# ---------------------------------------------------------------------------


class TestDatasetYaml:
    """Verify dataset.yaml files in DATASETS directories.

    Scoped to *paradigm* datasets (eye-tracking / SPR). Frequency resources
    such as ``subtlex_en`` (dir ``frequency/<lang>``) are fetched the same way
    but carry no paradigm ``dataset.yaml`` (the YAML registry is
    ``yaml/{paradigm}/{name}.yaml``), so they are not in this test's domain.
    """

    @pytest.mark.parametrize(
        "name",
        [n for n in DATASETS if not dataset_directory(n).startswith("frequency/")],
    )
    def test_yaml_loadable_with_required_fields(self, name: str):
        """dataset.yaml must exist, load without errors, expose a non-empty
        string ``name``, and provide at least one of ``url`` / ``download_url``."""
        directory = _data_dir() / dataset_directory(name)
        yaml_path = find_dataset_yaml(str(directory))
        assert yaml_path is not None, f"{name}: no dataset.yaml found"
        info = load_yaml(yaml_path)
        assert isinstance(info.name, str) and info.name.strip(), (
            f"{name}: dataset.yaml name is empty or not a string"
        )
        assert bool(info.url) or bool(info.download_url), (
            f"{name}: dataset.yaml has neither 'url' nor 'download_url'"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "provo",
            "meco",
            "natural_stories",
            "ucl_spr",
            "ucl_et",
            "geco",
            "celer",
            "onestop",
            "emtec",
            "sbsat",
            "devarda2023",
            "brothers_kuperberg",
            "zuco1",
            "zuco2",
            "frank2015_erp",
            *(n for n in DATASETS if DATASETS[n]["source"] == "study_release"),
        ],
    )
    def test_file_provenance_matches_fetch_registry(self, name: str):
        """Acquisition locators and shipped per-file provenance cannot drift."""
        directory = _data_dir() / dataset_directory(name)
        yaml_path = find_dataset_yaml(str(directory))
        assert yaml_path is not None
        actual = {
            entry.path: entry.original_path for entry in load_yaml(yaml_path).files
        }
        expected = provenance_paths_for(name)
        assert set(expected) <= set(actual), (
            f"{name}: fetch registry files absent from dataset YAML: "
            f"{sorted(set(expected) - set(actual))}"
        )
        assert {path: actual[path] for path in expected} == expected

    @pytest.mark.parametrize("key", sorted(REGISTRY))
    def test_a_fetchable_corpus_names_its_source_url(self, key: str):
        """``download_url`` is the data source, a URL when ``replica fetch`` gets it all."""
        keys = describe_dataset(key).download_keys
        if not keys or any(DATASETS[k]["source"] == "manual" for k in keys):
            return
        yaml_path = find_dataset_yaml(str(Path(REGISTRY[key].default_subdir)))
        assert yaml_path is not None
        assert load_yaml(yaml_path).download_url.startswith("https://"), key

    @pytest.mark.parametrize(
        "name",
        [
            n
            for n, config in DATASETS.items()
            if config["source"] != "manual" and config.get("verify_yaml", True)
        ],
    )
    def test_every_pinned_file_is_fetched(self, name: str):
        """A fetch verifies every pinned checksum, so it must obtain each file."""
        yaml_path = find_dataset_yaml(str(_data_dir() / dataset_directory(name)))
        assert yaml_path is not None
        config = DATASETS[name]
        archives = (
            {f"https://osf.io/download/{file_id}/" for file_id, _ in config["files"]}
            if config["source"] == "osf_zip"
            else set()
        )
        fetched = provenance_paths_for(name)
        missing = [
            entry.path
            for entry in load_yaml(yaml_path).files
            if entry.checksum and not _obtained(entry, fetched, archives)
        ]
        assert not missing, f"{name}: pinned files the fetch never obtains: {missing}"


def _obtained(entry, fetched: dict[str, str], archives: set[str]) -> bool:
    """Whether a download yields a pinned file, or every file of a pinned glob.

    Extracted archive members are not listed offline, so a file whose
    provenance is empty or names one of the downloaded archives counts.
    """
    if entry.n_files:
        if len(fnmatch.filter(fetched, entry.path)) == entry.n_files:
            return True
    elif entry.path in fetched:
        return True
    return bool(archives) and entry.original_path.split("#")[0] in archives | {""}


# ---------------------------------------------------------------------------
# 3. list_datasets() runs without error
# ---------------------------------------------------------------------------


class TestListDatasets:
    def test_status_follows_the_declared_files(self, tmp_path, monkeypatch, capsys):
        """A MATLAB release counts as downloaded once every pinned file is there."""
        monkeypatch.setattr(fetch_mod, "_data_dir", lambda: tmp_path)
        directory = tmp_path / dataset_directory("zuco1")
        directory.mkdir(parents=True)
        pinned = [
            entry.path for entry in load_yaml(find_dataset_yaml(str(directory))).files
        ]

        def status() -> str:
            list_datasets()
            line = next(
                line
                for line in capsys.readouterr().out.splitlines()
                if line.split()[:1] == ["zuco1"]
            )
            return line.split()[1]

        (directory / pinned[0]).touch()
        assert status() == "[partial]"
        for name in pinned:
            (directory / name).touch()
        assert status() == "[downloaded]"

    def test_list_datasets_shows_all(self, capsys):
        """``list_datasets()`` must execute and print every dataset in the
        registry. Catches both crashes and silent omission of an entry."""
        list_datasets()
        captured = capsys.readouterr()
        assert "Datasets" in captured.out
        for name in DATASETS:
            assert name in captured.out, f"list_datasets() does not mention '{name}'"


# ---------------------------------------------------------------------------
# 4. All fetchable datasets have valid source types
# ---------------------------------------------------------------------------


class TestSourceTypes:
    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_source_is_valid(self, name: str):
        """Every dataset entry must have a recognized source type."""
        source = DATASETS[name].get("source")
        assert source in VALID_SOURCE_TYPES, (
            f"{name}: source={source!r} not in {VALID_SOURCE_TYPES}"
        )

    @pytest.mark.parametrize(
        "name",
        [n for n in DATASETS if DATASETS[n]["source"] == "manual"],
    )
    def test_manual_has_instructions(self, name: str):
        """Manual datasets must provide download instructions."""
        assert "instructions" in DATASETS[name], (
            f"{name}: manual dataset missing 'instructions' key"
        )
        assert len(DATASETS[name]["instructions"]) > 0, (
            f"{name}: manual dataset has empty instructions"
        )

    @pytest.mark.parametrize(
        "name",
        [n for n in DATASETS if DATASETS[n]["source"] == "manual"],
    )
    def test_manual_steps_verify_rather_than_regenerate_checksums(self, name: str):
        """Regenerating rewrites the shipped checksum baseline from a user's copy."""
        instructions = DATASETS[name]["instructions"]
        assert "yaml_gen" not in instructions
        assert "regenerate-yaml" not in instructions

    @pytest.mark.parametrize(
        "name",
        [n for n in DATASETS if DATASETS[n]["source"] not in ("manual",)],
    )
    def test_downloadable_has_locator(self, name: str):
        """Non-manual datasets must declare what to download via ``files``."""
        cfg = DATASETS[name]
        assert "files" in cfg, f"{name}: downloadable dataset missing 'files' key"
        files = cfg["files"]
        if isinstance(files, list):
            assert len(files) > 0, f"{name}: files list is empty"
        elif isinstance(files, dict):
            assert len(files) > 0, f"{name}: files dict is empty"
        else:
            pytest.fail(f"{name}: files has unexpected type {type(files)}")

    @pytest.mark.parametrize(
        "name",
        [n for n in DATASETS if DATASETS[n]["source"] == "github_raw"],
    )
    def test_github_raw_has_repo(self, name: str):
        """github_raw sources must specify repo and branch."""
        assert "repo" in DATASETS[name], f"{name}: github_raw missing 'repo'"
        assert "branch" in DATASETS[name], f"{name}: github_raw missing 'branch'"

    @pytest.mark.parametrize(
        "name",
        [n for n in DATASETS if DATASETS[n]["source"] == "zip"],
    )
    def test_zip_has_url(self, name: str):
        """zip sources must specify a url."""
        assert "url" in DATASETS[name], f"{name}: zip source missing 'url'"


class TestStudyReleaseSources:
    """Corpora that are extracts from a study's released data."""

    @pytest.mark.parametrize(
        "name",
        [n for n in DATASETS if DATASETS[n]["source"] == "study_release"],
    )
    def test_study_release_names_a_repo_and_its_members(self, name: str):
        config = DATASETS[name]
        assert config["repo"], f"{name}: study_release missing 'repo'"
        assert config["files"], f"{name}: study_release lists no files"
        for member, filename in config["files"]:
            assert "/" in member, f"{name}: {member!r} is not a path in the release"
            assert "/" not in filename, f"{name}: {filename!r} is not a plain name"

    def test_the_release_is_taken_not_downloaded_again(self, tmp_path, monkeypatch):
        release = tmp_path / "release"
        (release / "data" / "ZuCO").mkdir(parents=True)
        member = release / "data" / "ZuCO" / "all.txt.averaged_rt.annotation"
        member.write_text("annotation")

        asked = []

        def fake_repo_path(name, *_, fetch=True):
            asked.append((name, fetch))
            return release

        monkeypatch.setattr("replica.dataset.fetch.sources.repo_path", fake_repo_path)
        out = tmp_path / "corpus" / "all.txt.averaged_rt.annotation"
        link_study_release(
            "surprisal_internal_layers",
            "data/ZuCO/all.txt.averaged_rt.annotation",
            out,
        )
        assert out.read_text() == "annotation"
        assert out.resolve() == member.resolve()
        # The release is fetched through the study dependency, once.
        assert asked == [("surprisal_internal_layers", True)]

    def test_a_member_missing_from_the_release_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "replica.dataset.fetch.sources.repo_path",
            lambda name, data_root=None, *, fetch=True: tmp_path,
        )
        with pytest.raises(FileNotFoundError, match="is not in the"):
            link_study_release("some_repo", "data/gone.tsv", tmp_path / "out.tsv")


def test_checksum_reporting_counts_only_pinned_entries(tmp_path, monkeypatch, capsys):
    from replica.dataset.contracts import DatasetInfo, FileEntry
    from replica.dataset.fetch import _verify_download
    from replica.dataset.metadata import io as metadata

    monkeypatch.setattr(
        metadata, "find_dataset_yaml", lambda _: tmp_path / "dataset.yaml"
    )
    info = DatasetInfo(name="test", files=[FileEntry(path="unverified.csv")])
    monkeypatch.setattr(metadata, "load_yaml", lambda _: info)
    verified = []
    monkeypatch.setattr(
        metadata,
        "verify_checksums",
        lambda directory, entries: verified.extend(entries),
    )
    _verify_download(tmp_path)
    assert verified == []
    assert "no shipped checksums" in capsys.readouterr().out
    info.files.append(FileEntry(path="verified.csv", checksum="sha256:abc"))
    _verify_download(tmp_path)
    assert [entry.path for entry in verified] == ["verified.csv"]
    output = capsys.readouterr().out
    assert "verified 1 manifest" in output
    assert "1 manifest entry/entries have no checksum" in output


def test_zip_companion_download_is_independent_of_archive_marker(monkeypatch, tmp_path):
    """An archive marker must not suppress missing companion files."""
    config = DATASETS["potec"]
    for _, filename in config["files"]:
        (tmp_path / f".{filename}.extracted").touch()
    calls = []

    def download(url, out):
        calls.append(url)
        out.write_text("{}")

    monkeypatch.setattr(fetch_mod, "download_direct_url", download)
    fetch_mod._fetch_osf_zip_files(config, tmp_path, force=False)
    assert (tmp_path / "word_limits.json").is_file()
    assert calls == [provenance_paths_for("potec")["word_limits.json"]]
    fetch_mod._fetch_osf_zip_files(config, tmp_path, force=False)
    assert len(calls) == 1
