"""The per-study claims files assemble into the registries the readers expect."""

import yaml

from replica.studies import STUDY_IDS
from replica.studies.claims import predictor_needs, registry, replication_results


def test_registry_assembles_every_study_with_a_claims_file():
    assembled = registry.load_registry()
    with_claims = [s for s in STUDY_IDS if registry.claims_path(s).is_file()]
    assert assembled["scope"]["active_studies"] == with_claims
    assert assembled["scope"]["deferred"] == [
        s for s in STUDY_IDS if s not in with_claims
    ]
    assert list(assembled["study_sources"]) == with_claims
    assert [c["study"] for c in assembled["claims"]] == sorted(
        (c["study"] for c in assembled["claims"]), key=with_claims.index
    )
    for claim in assembled["claims"]:
        assert claim["id"].startswith(claim["study"] + "_")


def test_needs_follow_the_claims_files():
    needs = predictor_needs.load_needs()
    assert list(needs["studies"]) == registry.load_registry()["scope"]["active_studies"]


def test_results_files_hold_only_their_study_claims():
    for study in STUDY_IDS:
        path = replication_results.results_path(study)
        if not path.is_file():
            continue
        doc = yaml.safe_load(path.read_text())
        assert doc["schema"] == replication_results.SCHEMA
        assert all(cid.startswith(study + "_") for cid in doc["claims"]), study
        assert doc["sources"] == sorted(set(doc["sources"])), study


def test_saving_the_assembled_results_rewrites_the_same_files(tmp_path, monkeypatch):
    before = {}
    for study in STUDY_IDS:
        path = replication_results.results_path(study)
        if path.is_file():
            (tmp_path / study).mkdir()
            (tmp_path / study / path.name).write_text(path.read_text())
            before[study] = path.read_text()
    monkeypatch.setattr(replication_results, "STUDIES_DIR", tmp_path)
    replication_results.save_results(replication_results.load_raw_results())
    after = {s: (tmp_path / s / "replication_results.yaml").read_text() for s in before}
    assert after == before
