"""Tests for dataset metadata: dataclasses, checksums, YAML serialization, integration."""

import pytest

from replica.dataset import Dataset
from replica.dataset.contracts import DatasetInfo, FileEntry, IndexMapping
from replica.dataset.metadata.io import (
    ChecksumMismatchError,
    compute_aggregate_sha256,
    compute_checksums,
    compute_sha256,
    find_dataset_yaml,
    from_dict,
    load_yaml,
    resolve_data_dir,
    save_yaml,
    to_dict,
    verify_checksums,
)

# ── 1. FileEntry / IndexMapping construction ──────────────────────────────


class TestFileEntry:
    def test_defaults(self):
        fe = FileEntry(path="data.csv")
        assert fe.role == ""
        assert fe.checksum == ""
        assert fe.original_path == ""
        assert fe.provenance_unavailable == ""


class TestIndexMapping:
    def test_defaults(self):
        im = IndexMapping()
        assert im.stimulus_col == ""
        assert im.unit_col == ""
        assert im.participant_col == ""
        assert im.stimulus_text_file == ""


# ── 2. DatasetInfo construction ─────────────────────────────────


class TestDatasetInfo:
    def test_minimal_construction(self):
        info = DatasetInfo(name="test")
        assert info.name == "test"
        assert info.citation == ""
        assert info.original is True
        assert info.measures == []
        assert info.files == []
        assert info.index_mapping is None

    def test_provenance_fields(self):
        info = DatasetInfo(
            name="test",
            version="1.0",
            source="https://example.com",
            original=False,
            derived_from="original_dataset",
            processing="tokenized with spaCy",
            measures=["first_run", "first_fixation"],
            files=[FileEntry(path="data.csv", role="measurements")],
            index_mapping=IndexMapping(stimulus_col="Text_ID"),
        )
        assert info.version == "1.0"
        assert info.original is False
        assert info.derived_from == "original_dataset"
        assert len(info.files) == 1
        assert info.index_mapping.stimulus_col == "Text_ID"


# ── 3. SHA-256 computation ────────────────────────────────────────────────


class TestComputeSha256:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_sha256(tmp_path / "nonexistent.txt")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        result = compute_sha256(f)
        assert result.startswith("sha256:")
        # SHA-256 of empty content
        assert (
            result
            == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("deterministic content")
        r1 = compute_sha256(f)
        r2 = compute_sha256(f)
        assert r1 == r2


# ── 4. Checksum verification ─────────────────────────────────────────────


class TestVerifyChecksums:
    def test_match(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\na,b\n")
        checksum = compute_sha256(f)
        files = [FileEntry(path="data.csv", role="measurements", checksum=checksum)]
        # Should not raise
        verify_checksums(tmp_path, files)

    def test_mismatch(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\na,b\n")
        files = [
            FileEntry(
                path="data.csv",
                role="measurements",
                checksum="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            )
        ]
        with pytest.raises(ChecksumMismatchError) as exc_info:
            verify_checksums(tmp_path, files)
        assert exc_info.value.path == "data.csv"
        assert "0000" in exc_info.value.expected

    def test_missing_file(self, tmp_path):
        files = [FileEntry(path="missing.csv", role="data", checksum="sha256:abc")]
        with pytest.raises(FileNotFoundError):
            verify_checksums(tmp_path, files)

    def test_empty_checksum_skipped(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("some data")
        files = [FileEntry(path="data.csv", role="data", checksum="")]
        # Should not raise, empty checksum means skip verification
        verify_checksums(tmp_path, files)


class TestGlobAggregateChecksums:
    """Glob entries (n_files > 0) with aggregate SHA-256."""

    @staticmethod
    def _make_group(tmp_path, subdir="reports", n=6):
        d = tmp_path / subdir
        d.mkdir()
        for i in range(n):
            (d / f"r{i:02d}.txt").write_text(f"contents-{i}")
        rels = sorted(f"{subdir}/r{i:02d}.txt" for i in range(n))
        return rels

    def test_aggregate_is_deterministic(self, tmp_path):
        rels = self._make_group(tmp_path)
        a = compute_aggregate_sha256(tmp_path, rels)
        b = compute_aggregate_sha256(tmp_path, list(reversed(rels)))
        assert a == b
        assert a.startswith("sha256:") and len(a) == 7 + 64

    def test_aggregate_changes_when_file_changes(self, tmp_path):
        rels = self._make_group(tmp_path)
        before = compute_aggregate_sha256(tmp_path, rels)
        (tmp_path / rels[0]).write_text("changed")
        after = compute_aggregate_sha256(tmp_path, rels)
        assert before != after

    def test_verify_glob_match(self, tmp_path):
        rels = self._make_group(tmp_path)
        agg = compute_aggregate_sha256(tmp_path, rels)
        files = [FileEntry(path="reports/r*.txt", checksum=agg, n_files=len(rels))]
        verify_checksums(tmp_path, files)  # no raise

    def test_verify_glob_mismatch_raises(self, tmp_path):
        rels = self._make_group(tmp_path)
        agg = compute_aggregate_sha256(tmp_path, rels)
        # Tamper with one file
        (tmp_path / rels[2]).write_text("tampered")
        files = [FileEntry(path="reports/r*.txt", checksum=agg, n_files=len(rels))]
        with pytest.raises(ChecksumMismatchError):
            verify_checksums(tmp_path, files)

    def test_verify_glob_count_mismatch_raises(self, tmp_path):
        rels = self._make_group(tmp_path, n=6)
        agg = compute_aggregate_sha256(tmp_path, rels)
        files = [FileEntry(path="reports/r*.txt", checksum=agg, n_files=6)]
        # Add an extra matching file → count drift
        (tmp_path / "reports/r99.txt").write_text("extra")
        with pytest.raises(FileNotFoundError, match="expected 6"):
            verify_checksums(tmp_path, files)

    def test_glob_yaml_round_trip(self, tmp_path):
        rels = self._make_group(tmp_path)
        agg = compute_aggregate_sha256(tmp_path, rels)
        info = DatasetInfo(
            name="test",
            files=[FileEntry(path="reports/r*.txt", checksum=agg, n_files=len(rels))],
        )
        path = tmp_path / "dataset.yaml"
        save_yaml(info, path)
        loaded = load_yaml(path)
        assert len(loaded.files) == 1
        assert loaded.files[0].path == "reports/r*.txt"
        assert loaded.files[0].n_files == len(rels)
        assert loaded.files[0].checksum == agg
        verify_checksums(tmp_path, loaded.files)


# ── 5. YAML round-trip ───────────────────────────────────────────────────


class TestYamlRoundTrip:
    def test_minimal(self, tmp_path):
        info = DatasetInfo(name="test")
        path = tmp_path / "dataset.yaml"
        save_yaml(info, path)
        loaded = load_yaml(path)
        assert loaded.name == "test"
        assert loaded.citation == ""

    def test_full_info(self, tmp_path):
        info = DatasetInfo(
            name="PROVO",
            citation="Luke & Christianson (2018).",
            url="https://osf.io/sjefs/",
            paradigm="eye_tracking",
            language="en",
            description="Eye-tracking corpus.",
            modalities=["eye-tracking"],
            source="https://osf.io/sjefs/download",
            version="1.0",
            release_date="2016-01-01",
            access_date="2026-03-01",
            license="CC-BY-4.0",
            measures=["first_run", "first_fixation"],
            files=[
                FileEntry(
                    path="et_data.csv", role="measurements", checksum="sha256:abcd"
                ),
                FileEntry(path="cloze.csv", role="cloze_norms"),
            ],
            index_mapping=IndexMapping(
                stimulus_col="Text_ID",
                unit_col="Word_Number",
                participant_col="PID",
            ),
        )
        path = tmp_path / "dataset.yaml"
        save_yaml(info, path)
        loaded = load_yaml(path)
        assert loaded.name == "PROVO"
        assert loaded.version == "1.0"
        assert loaded.license == "CC-BY-4.0"
        assert loaded.modalities == ["eye-tracking"]
        assert loaded.measures == ["first_run", "first_fixation"]
        assert len(loaded.files) == 2
        assert loaded.files[0].role == "measurements"
        assert loaded.files[0].checksum == "sha256:abcd"
        assert loaded.files[1].checksum == ""
        assert loaded.index_mapping is not None
        assert loaded.index_mapping.stimulus_col == "Text_ID"

    def test_postprocessed(self, tmp_path):
        info = DatasetInfo(
            name="PROVO_processed",
            original=False,
            derived_from="PROVO",
            processing="Tokenized with spaCy, filtered outliers",
        )
        path = tmp_path / "dataset.yaml"
        save_yaml(info, path)
        loaded = load_yaml(path)
        assert loaded.original is False
        assert loaded.derived_from == "PROVO"
        assert loaded.processing == "Tokenized with spaCy, filtered outliers"

    def test_defaults_omitted(self, tmp_path):
        info = DatasetInfo(name="minimal")
        path = tmp_path / "dataset.yaml"
        save_yaml(info, path)
        content = path.read_text()
        assert "original" not in content  # True is default, should be omitted
        assert "version" not in content
        assert "files" not in content
        assert "index_mapping" not in content
        assert "name: minimal" in content


# ── 6. to_dict / from_dict edge cases ────────────────────────────────────


class TestToDictFromDict:
    def test_empty_fields_omitted(self):
        info = DatasetInfo(name="test")
        d = to_dict(info)
        assert "name" in d
        assert "citation" not in d
        assert "version" not in d
        assert "files" not in d

    def test_unknown_keys_ignored(self):
        data = {"name": "test", "future_field": "value", "another": 42}
        info = from_dict(data)
        assert info.name == "test"
        assert not hasattr(info, "future_field")

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            from_dict({"citation": "some citation"})

    def test_files_roundtrip(self):
        info = DatasetInfo(
            name="test",
            files=[
                FileEntry(
                    path="a.csv",
                    role="data",
                    checksum="sha256:abc",
                    provenance_unavailable="Restricted source; contact authors.",
                )
            ],
        )
        d = to_dict(info)
        assert len(d["files"]) == 1
        assert d["files"][0]["path"] == "a.csv"
        restored = from_dict(d)
        assert len(restored.files) == 1
        assert restored.files[0].path == "a.csv"
        assert restored.files[0].checksum == "sha256:abc"
        assert (
            restored.files[0].provenance_unavailable
            == "Restricted source; contact authors."
        )

    def test_index_mapping_roundtrip(self):
        info = DatasetInfo(
            name="test",
            index_mapping=IndexMapping(stimulus_col="tid", unit_col="wid"),
        )
        d = to_dict(info)
        assert d["index_mapping"]["stimulus_col"] == "tid"
        restored = from_dict(d)
        assert restored.index_mapping is not None
        assert restored.index_mapping.stimulus_col == "tid"
        assert restored.index_mapping.unit_col == "wid"

    def test_original_false_included(self):
        info = DatasetInfo(name="test", original=False)
        d = to_dict(info)
        assert d["original"] is False


# ── 7. find_dataset_yaml ─────────────────────────────────────────────────


class TestFindDatasetYaml:
    def test_finds_paradigm_scoped_package_metadata(self, tmp_path):
        directory = tmp_path / "eye-tracking" / "original" / "provo"
        result = find_dataset_yaml(directory)
        assert result is not None
        assert result.name == "provo.yaml"
        assert result.parent.name == "eye-tracking"

    def test_registered_paths_select_metadata_explicitly(self, tmp_path):
        from replica.dataset.catalog.registry import REGISTRY

        for spec in REGISTRY.values():
            dataset = spec.loader_cls()(**spec.init_kwargs)
            initial = dataset.info
            noncanonical = tmp_path / "custom-location"
            dataset._load_metadata(
                str(noncanonical),
                verify=False,
                metadata_path=find_dataset_yaml(noncanonical),
            )
            assert dataset.info is initial
            canonical = tmp_path / spec.default_subdir
            metadata = find_dataset_yaml(canonical)
            dataset._load_metadata(
                str(canonical),
                verify=False,
                metadata_path=metadata,
            )
            expected = initial if metadata is None else load_yaml(metadata)
            assert dataset.info == expected, spec.name

    def test_not_found(self, tmp_path):
        result = find_dataset_yaml(tmp_path)
        assert result is None


def test_data_redirection_requires_explicit_metadata(tmp_path):
    directory = tmp_path / "eeg/original/zuco2"
    directory.mkdir(parents=True)
    target = tmp_path / "eye-tracking/original/zuco2"
    target.mkdir(parents=True)
    metadata = find_dataset_yaml(directory)

    assert resolve_data_dir(directory) == directory
    assert resolve_data_dir(directory, metadata_path=metadata) == target
    target.rmdir()
    assert resolve_data_dir(directory, metadata_path=metadata) == directory


# ── 8. _load_metadata integration ────────────────────────────────────────


class _ConcreteDataset(Dataset):
    """A minimal concrete Dataset for testing _load_metadata."""

    info = DatasetInfo(
        name="TestDS",
        citation="Original Citation",
        paradigm="eye_tracking",
        language="en",
    )

    def load(self, path: str) -> None:
        self._load_metadata(path)


@pytest.fixture
def metadata_path(tmp_path):
    return tmp_path / "metadata.yaml"


class TestLoadMetadataIntegration:
    @pytest.mark.parametrize(
        "subpath", ["eeg/original/zuco2", "eye-tracking/original/provo"]
    )
    def test_custom_loader_keeps_its_info_under_a_bundled_name(
        self, tmp_path, monkeypatch, subpath
    ):
        monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
        directory = tmp_path / subpath
        directory.mkdir(parents=True)
        ds = _ConcreteDataset()
        ds.info = DatasetInfo(name="custom_french", language="fr")
        original = ds.info
        ds.load(str(directory))
        assert ds.info is original

    def test_yaml_replaces_class_info(self, tmp_path, metadata_path):
        yaml_content = (
            "name: ExplicitCorpus\nversion: '2.0'\nlicense: CC-BY-4.0\n"
            "citation: YAML Citation\nparadigm: spr\n"
        )
        metadata_path.write_text(yaml_content)

        ds = _ConcreteDataset()
        ds._load_metadata(str(tmp_path), verify=False, metadata_path=metadata_path)

        # YAML is the single source of truth
        assert ds.info.name == "ExplicitCorpus"
        assert ds.info.version == "2.0"
        assert ds.info.license == "CC-BY-4.0"
        assert ds.info.citation == "YAML Citation"
        assert ds.info.paradigm == "spr"

    def test_no_yaml_keeps_class_info(self, tmp_path):
        ds = _ConcreteDataset()
        ds._load_metadata(str(tmp_path), verify=False)

        # No YAML → class-level info kept as-is
        assert ds.info.name == "TestDS"
        assert ds.info.citation == "Original Citation"

    def test_class_attribute_not_mutated(self, tmp_path, metadata_path):
        yaml_content = "name: TestDS\nversion: '3.0'\n"
        metadata_path.write_text(yaml_content)

        ds1 = _ConcreteDataset()
        ds1._load_metadata(str(tmp_path), verify=False, metadata_path=metadata_path)

        ds2 = _ConcreteDataset()
        # ds2 should still have the original class info
        assert _ConcreteDataset.info.version == ""
        assert ds1.info.version == "3.0"
        assert ds2.info.version == ""  # Not yet loaded

    def test_checksum_verification_on_load(self, tmp_path, metadata_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("col1,col2\na,b\n")
        checksum = compute_sha256(data_file)

        yaml_content = (
            f"name: TestDS\n"
            f"files:\n"
            f"  - path: data.csv\n"
            f"    role: measurements\n"
            f"    checksum: {checksum}\n"
        )
        metadata_path.write_text(yaml_content)

        ds = _ConcreteDataset()
        # Should not raise
        ds._load_metadata(str(tmp_path), verify=True, metadata_path=metadata_path)
        assert len(ds.info.files) == 1

    def test_checksum_mismatch_raises(self, tmp_path, monkeypatch, metadata_path):
        # The suite sets PSYCHO_SKIP_CHECKSUMS=1 for speed; this test exercises
        # the verification path itself, so force it back on.
        monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
        data_file = tmp_path / "data.csv"
        data_file.write_text("col1,col2\na,b\n")

        yaml_content = (
            "name: TestDS\n"
            "files:\n"
            "  - path: data.csv\n"
            "    role: measurements\n"
            "    checksum: sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        )
        metadata_path.write_text(yaml_content)

        ds = _ConcreteDataset()
        original = ds.info
        with pytest.raises(ChecksumMismatchError):
            ds._load_metadata(str(tmp_path), verify=True, metadata_path=metadata_path)
        assert ds.info is original

    def test_missing_declared_file_names_itself(
        self, tmp_path, monkeypatch, metadata_path
    ):
        """A declared file that is absent is named in the error."""
        monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
        metadata_path.write_text(
            "name: TestDS\n"
            "files:\n"
            "  - path: absent.csv\n"
            "    role: measurements\n"
            "    checksum: sha256:00\n"
        )

        ds = _ConcreteDataset()
        with pytest.raises(FileNotFoundError, match="absent.csv.*declared"):
            ds._load_metadata(str(tmp_path), verify=True, metadata_path=metadata_path)

    def test_files_argument_restricts_verification(
        self, tmp_path, monkeypatch, metadata_path
    ):
        """A loader reading one of several declared alternatives verifies only
        that one, so the others need not be on disk."""
        monkeypatch.delenv("PSYCHO_SKIP_CHECKSUMS", raising=False)
        present = tmp_path / "present.csv"
        present.write_text("col1\na\n")
        digest = compute_sha256(present)
        metadata_path.write_text(
            "name: TestDS\n"
            "files:\n"
            f"  - path: present.csv\n    role: measurements\n    checksum: {digest}\n"
            "  - path: absent.csv\n    role: measurements\n    checksum: sha256:00\n"
        )

        ds = _ConcreteDataset()
        ds._load_metadata(
            str(tmp_path),
            verify=True,
            files=["present.csv"],
            metadata_path=metadata_path,
        )
        # Both entries stay on info; only the named one is checked.
        assert [entry.path for entry in ds.info.files] == ["present.csv", "absent.csv"]
        with pytest.raises(FileNotFoundError):
            ds._load_metadata(str(tmp_path), verify=True, metadata_path=metadata_path)

    def test_verify_false_skips(self, tmp_path, metadata_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("col1,col2\na,b\n")

        yaml_content = (
            "name: TestDS\n"
            "files:\n"
            "  - path: data.csv\n"
            "    role: measurements\n"
            "    checksum: sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        )
        metadata_path.write_text(yaml_content)

        ds = _ConcreteDataset()
        # verify=False must skip the checksum check and still load the YAML
        # , a bug that aborted before populating `info` would otherwise look
        # the same as the success case.
        ds._load_metadata(str(tmp_path), verify=False, metadata_path=metadata_path)
        assert ds.info.name == "TestDS"
        assert len(ds.info.files) == 1


# ── 9. compute_checksums utility ──────────────────────────────────────────


class TestComputeChecksums:
    def test_populates_checksums(self, tmp_path):
        (tmp_path / "a.csv").write_text("data_a")
        (tmp_path / "b.csv").write_text("data_b")

        files = [
            FileEntry(path="a.csv", role="data"),
            FileEntry(path="b.csv", role="data"),
        ]
        result = compute_checksums(tmp_path, files)

        assert len(result) == 2
        assert result[0].checksum.startswith("sha256:")
        assert result[1].checksum.startswith("sha256:")
        assert result[0].role == "data"

    def test_returns_new_list(self, tmp_path):
        (tmp_path / "a.csv").write_text("data_a")
        files = [FileEntry(path="a.csv", role="data")]
        result = compute_checksums(tmp_path, files)
        assert result is not files
        assert files[0].checksum == ""  # Original not mutated

    def test_missing_file_raises(self, tmp_path):
        files = [FileEntry(path="missing.csv", role="data")]
        with pytest.raises(FileNotFoundError):
            compute_checksums(tmp_path, files)


# ── 10. Regenerating the shipped metadata ─────────────────────────────────


class TestGenerateYaml:
    def test_rewrites_only_the_files_section(self, tmp_path, monkeypatch):
        import yaml

        from replica.dataset.metadata import generate as yaml_gen

        directory = tmp_path / "data" / "eye-tracking" / "original" / "demo"
        (directory / "english" / "items").mkdir(parents=True)
        for rel in ("english/sa01ma1p.dat", "english/sb01ma1p.dat"):
            (directory / rel).write_text(rel)
        (directory / "english" / "items" / "text00.itm").write_text("items")
        yaml_root = tmp_path / "yaml"
        declared = yaml_root / "eye-tracking" / "demo.yaml"
        declared.parent.mkdir(parents=True)
        provenance = "author-distributed archive: english/*ma1p.dat"
        declared.write_text(
            yaml.dump(
                {
                    "name": "Demo",
                    "language": "en",
                    "data_path": "eye-tracking/original/shared",
                    "material_namespace": "books",
                    "material_fields": ["title", "edition"],
                    "files": [
                        {
                            "path": "english/*ma1p.dat",
                            "role": "measurements",
                            "checksum": "sha256:" + "0" * 64,
                            "original_path": provenance,
                            "n_files": 1,
                        }
                    ],
                },
                sort_keys=False,
            )
        )
        monkeypatch.setattr(yaml_gen, "_PKG_YAML_ROOT", yaml_root)
        monkeypatch.setattr(yaml_gen, "find_dataset_yaml", lambda _: declared)

        yaml_gen.generate_yaml(directory)

        written = yaml.safe_load(declared.read_text())
        assert written["language"] == "en"
        assert written["data_path"] == "eye-tracking/original/shared"
        assert written["material_namespace"] == "books"
        assert written["material_fields"] == ["title", "edition"]
        files = {entry["path"]: entry for entry in written["files"]}
        assert set(files) == {"english/*ma1p.dat", "english/items/text00.itm"}
        glob = files["english/*ma1p.dat"]
        assert glob["original_path"] == provenance
        assert glob["n_files"] == 2
        assert glob["checksum"] == compute_aggregate_sha256(
            directory, ["english/sa01ma1p.dat", "english/sb01ma1p.dat"]
        )
