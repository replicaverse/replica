"""Unit tests for strict per-file provenance auditing."""

import sys

import pytest

from scripts.checks.distribution.dataset_metadata import _file_entry_findings_one


def _findings(entry: dict, tmp_path):
    return _file_entry_findings_one(entry, tmp_path, is_original=True)


def test_original_file_requires_provenance(tmp_path):
    errors, warnings = _findings({"path": "data.csv"}, tmp_path)
    assert errors == [
        "file 'data.csv' missing original_path or provenance_unavailable reason"
    ]
    assert warnings == ["file 'data.csv' has no checksum; content is not verified"]


def test_original_path_satisfies_provenance_requirement(tmp_path):
    errors, warnings = _findings(
        {"path": "data.csv", "original_path": "https://example.test/data.csv"},
        tmp_path,
    )
    assert errors == []
    assert warnings == ["file 'data.csv' has no checksum; content is not verified"]


def test_documented_unavailable_provenance_is_allowed(tmp_path):
    errors, warnings = _findings(
        {
            "path": "data.csv",
            "provenance_unavailable": "Restricted deposit; contact the authors.",
        },
        tmp_path,
    )
    assert errors == []
    assert warnings == ["file 'data.csv' has no checksum; content is not verified"]


def test_provenance_fields_are_mutually_exclusive(tmp_path):
    errors, _ = _findings(
        {
            "path": "data.csv",
            "original_path": "https://example.test/data.csv",
            "provenance_unavailable": "Unknown",
        },
        tmp_path,
    )
    assert errors == [
        "file 'data.csv' sets both original_path and provenance_unavailable"
    ]


def test_unavailable_provenance_requires_text_reason(tmp_path):
    errors, _ = _findings(
        {"path": "data.csv", "provenance_unavailable": True}, tmp_path
    )
    assert errors == [
        "file 'data.csv' provenance_unavailable must be a non-empty reason"
    ]


def test_packaged_checksum_checks_data_root_and_redirect(tmp_path, monkeypatch):
    import yaml

    from replica.dataset.metadata.io import compute_sha256
    from scripts.checks.distribution import dataset_metadata as audit

    package = tmp_path / "package"
    metadata = package / "eeg" / "shared.yaml"
    metadata.parent.mkdir(parents=True)
    root = tmp_path / "downloads"
    release = root / "eye-tracking" / "original" / "shared"
    release.mkdir(parents=True)
    source = release / "data.csv"
    source.write_text("original\n")
    metadata.write_text(
        yaml.safe_dump(
            {
                "name": "shared",
                "citation": "test",
                "bibtex": "@misc{test, author={A}, title={T}, year={2026}}",
                "description": "Test corpus.",
                "paradigm": "eeg",
                "language": "en",
                "download_url": "https://example.test",
                "data_path": "eye-tracking/original/shared",
                "files": [
                    {
                        "path": "data.csv",
                        "original_path": "data.csv",
                        "checksum": compute_sha256(source),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(audit, "PACKAGE_YAML_DIR", package)
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(root))
    assert audit.verify_one(metadata) == ([], [])
    source.write_text("changed\n")
    errors, warnings = audit.verify_one(metadata)
    assert len(errors) == 1 and "checksum mismatch" in errors[0]
    assert warnings == []
    source.unlink()
    errors, warnings = audit.verify_one(metadata)
    assert errors == []
    assert len(warnings) == 1 and "checksum not verified" in warnings[0]


def test_packaged_processed_and_local_paths(tmp_path, monkeypatch):
    from scripts.checks.distribution import dataset_metadata as audit

    package = tmp_path / "package"
    monkeypatch.setattr(audit, "PACKAGE_YAML_DIR", package)
    root = tmp_path / "data"
    assert (
        audit._dataset_directory(
            package / "eye-tracking" / "test.yaml", {"original": False}, root
        )
        == root / "eye-tracking/processed/test"
    )
    assert (
        audit._dataset_directory(tmp_path / "custom" / "dataset.yaml", {}, root)
        == tmp_path / "custom"
    )


def test_empty_manifest_reports_no_verification(tmp_path):
    from scripts.checks.distribution.dataset_metadata import _file_entry_findings

    errors, warnings = _file_entry_findings([], tmp_path, is_original=True)
    assert errors == []
    assert warnings == ["no files declared; no checksums can be verified"]


def test_an_empty_metadata_tree_fails(tmp_path, monkeypatch, capsys):
    from scripts.checks.distribution import dataset_metadata as audit

    monkeypatch.setattr(audit, "PACKAGE_YAML_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_dataset_yaml.py"])
    with pytest.raises(SystemExit) as exit_info:
        audit.main()
    assert exit_info.value.code == 1
    assert "no dataset YAML found" in capsys.readouterr().err


def test_a_missing_yaml_path_fails(tmp_path, monkeypatch, capsys):
    from scripts.checks.distribution import dataset_metadata as audit

    absent = tmp_path / "absent.yaml"
    monkeypatch.setattr(sys, "argv", ["verify_dataset_yaml.py", str(absent)])
    with pytest.raises(SystemExit) as exit_info:
        audit.main()
    assert exit_info.value.code == 1
    assert f"no such file: {absent}" in capsys.readouterr().err


def test_stimulus_type_must_come_from_the_vocabulary():
    from scripts.checks.distribution.dataset_metadata import _stimulus_type_errors

    assert _stimulus_type_errors({"stimulus_type": "sentence"}) == []
    assert _stimulus_type_errors({}) == []
    assert _stimulus_type_errors({"stimulus_type": "sentences"}) == [
        "unknown stimulus_type: 'sentences'"
    ]
