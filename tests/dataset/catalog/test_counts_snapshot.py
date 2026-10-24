"""The count snapshot covers registered corpora and explains missing counts."""

from pathlib import Path

from replica.dataset.catalog import facts, registry
from replica.dataset.metadata.io import find_dataset_yaml, load_yaml


def _snapshot() -> dict:
    return facts.load_snapshot() or {}


def test_snapshot_covers_registry_with_counts_or_a_reason():
    snapshot = _snapshot()
    assert facts.snapshot_drift(snapshot) == {"missing": [], "stale": []}
    assert not set(snapshot.get("counts", {})) & set(snapshot.get("unmeasured", {}))
    for key, reason in snapshot.get("unmeasured", {}).items():
        assert isinstance(reason, str) and reason.strip(), key


def test_counted_corpora_report_the_fields_the_page_prints():
    for key, counts in _snapshot().get("counts", {}).items():
        for field in (
            "participants",
            "participants_averaged",
            "stimuli",
            "units",
            "characters",
        ):
            assert field in counts, f"{key} counts lack {field}"


def test_fresh_counts_detect_values_and_metadata_drift():
    snapshot = {"counts": {"devarda2023": {"participants": 8}}}
    fresh = {"devarda2023": {"participants": 1, "participants_averaged": True}}
    assert facts.snapshot_drift(snapshot, measured_counts=fresh)["changed"] == [
        "devarda2023"
    ]
    assert (
        facts.snapshot_drift({"counts": fresh}, measured_counts=fresh)["changed"] == []
    )


def test_count_updates_preserve_the_other_entries_provenance(tmp_path, monkeypatch):
    import json

    path = tmp_path / "counts.json"
    measured = {
        "measured_at": "2026-10-23T10:00:00Z",
        "producer": "replica.dataset.catalog.facts.counts_for",
        "producer_sha256": "count-function-digest",
        "load_for_counting_sha256": "load-function-digest",
        "configuration": {
            "path": "dataset/metadata/yaml/eye-tracking/provo.yaml",
            "sha256": "configuration-digest",
        },
        "inputs": {"declared_checksums": "matched"},
    }
    path.write_text(
        json.dumps(
            {
                "counts": {"old": {"participants": 2}},
                "count_provenance": {"old": measured},
            }
        )
    )
    current = {"generated": "2026-10-24T10:00:00Z", "input_checksums": "not-verified"}
    monkeypatch.setattr(facts, "_provenance", lambda: current)
    facts.write_snapshot(
        {
            "datasets": {
                "new": {
                    "counts_source": "measured",
                    "counts": {"participants": 1},
                    "count_provenance": {"measured_at": "earlier"},
                },
                "fresh": {"counts_source": "measured", "counts": {"participants": 3}},
            }
        },
        path,
        merge=True,
    )
    saved = json.loads(path.read_text())
    assert saved["count_provenance"] == {
        "old": measured,
        "new": {"measured_at": "earlier"},
        "fresh": current,
    }


#: Units per stimulus that a stimulus type allows, from the vocabulary in
#: ``DatasetInfo``: one unit for a single-unit stimulus, at most ten for a phrase.
_UNIT_BOUNDS = {"single-unit": (1, 1), "phrase": (1, 10)}


def test_stimulus_types_agree_with_the_counted_units():
    wrong = []
    for key, counts in _snapshot().get("counts", {}).items():
        yaml_path = find_dataset_yaml(Path(registry.get(key).default_subdir))
        if yaml_path is None or not counts["stimuli"]:
            continue
        kind = load_yaml(yaml_path).stimulus_type
        if kind not in _UNIT_BOUNDS:
            continue
        low, high = _UNIT_BOUNDS[kind]
        mean = counts["units"] / counts["stimuli"]
        if not low <= mean <= high:
            wrong.append(f"{key}: {kind} with {mean:.1f} units per stimulus")
    assert not wrong, wrong
