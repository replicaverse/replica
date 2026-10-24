"""Data-free tests for the dataset facts layer.

All of it must pass with no corpora on disk, as the docs build does.
"""

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from replica.core import AnnotationLayer, Span, Stimulus
from replica.dataset.catalog import facts
from replica.dataset.catalog.registry import REGISTRY


def _stimulus(text, n_units, layer="words"):
    stimulus = Stimulus(text=text)
    stimulus.add_layer(
        AnnotationLayer(
            name=layer,
            spans=[Span(i, i + 1) for i in range(n_units)],
            values=[None] * n_units,
        )
    )
    return stimulus


class _FakeDataset:
    def __init__(self, stimuli, participants=()):
        self._stimuli = list(stimuli)
        self.participants = list(participants)

    def stimuli(self):
        return self._stimuli


def test_data_free_facts_cover_every_registered_corpus():
    collected = facts.dataset_facts(use_snapshot=False)["datasets"]
    assert set(collected) == set(REGISTRY)
    for key, record in collected.items():
        assert record["key"] == key
        assert record["paradigm"]
        assert record["loader"]
        assert record["loader_module"].startswith("replica.dataset.")
        assert record["counts"] is None
        assert record["counts_source"] == ""
        assert "describe_error" not in record


def test_metadata_keeps_default_valued_fields():
    record = facts.dataset_facts(["provo"], use_snapshot=False)["datasets"]["provo"]
    metadata = record["metadata"]
    # ``metadata.to_dict`` drops fields equal to their default, which would
    # silently hide the language of every English corpus.
    assert metadata["language"] == "en"
    assert metadata["citation"] == "provo-2018"
    assert metadata["url"].startswith("https://doi.org/")
    assert [entry["path"] for entry in metadata["files"]]
    assert record["yaml"] == "eye-tracking/provo.yaml"
    assert {"first_run", "first_pass"} <= set(record["measures"])
    assert record["gated"] is False


def test_yaml_resolves_without_the_corpus_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(tmp_path))
    record = facts.dataset_facts(["provo"], use_snapshot=False)["datasets"]["provo"]
    assert record["path_exists"] is False
    assert record["yaml"] == "eye-tracking/provo.yaml"
    assert record["metadata"]["name"] == "PROVO"


def test_gated_corpus_carries_the_catalog_access_instructions():
    record = facts.dataset_facts(["celer"], use_snapshot=False)["datasets"]["celer"]
    assert record["gated"] is True
    assert "LDC" in record["access_instructions"]


def test_absent_corpus_is_not_probed(tmp_path, monkeypatch):
    monkeypatch.setenv("PSYCHO_DATA_DIR", str(tmp_path))

    def fail(*args, **kwargs):
        raise AssertionError("must not spawn a probe for absent data")

    monkeypatch.setattr(subprocess, "run", fail)
    record = facts.dataset_facts(["provo"], measure=True, use_snapshot=False)[
        "datasets"
    ]["provo"]
    assert record["probe_status"] == "data-absent"
    assert record["counts"] is None


def test_counts_dedupe_stimuli_and_sum_layers():
    counts = facts.counts_for(
        _FakeDataset(
            [
                _stimulus("the same text", 3),
                _stimulus("the same text", 3),
                _stimulus("another one", 2),
            ],
            participants=["a", "b"],
        )
    )
    assert counts["participants"] == 2
    assert counts["stimuli"] == 2
    assert counts["units"] == 5
    assert counts["units_by_layer"] == {"words": 5}
    assert counts["characters"] == len("the same text") + len("another one")


def test_counts_use_the_only_layer_when_it_is_not_words():
    counts = facts.counts_for(_FakeDataset([_stimulus("abc", 3, layer="characters")]))
    assert counts["units"] == 3
    assert counts["units_by_layer"] == {"characters": 3}


def test_counts_take_fmri_participants_from_the_loaded_cohort(tmp_path):
    import numpy as np

    from replica.dataset.fmri.narratives import Narratives

    (tmp_path / "story").mkdir()
    (tmp_path / "story_align.csv").write_text("one,one,0,0.5\n")
    for subject in ("sub_1", "sub_2"):
        np.save(tmp_path / "story" / f"{subject}.npy", np.ones((3, 4)))
    dataset = Narratives(story="story", trim_trs=(0, 0))
    dataset.load(str(tmp_path), align_path=str(tmp_path), participants=["sub_1"])
    assert facts.counts_for(dataset)["participants"] == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("FileNotFoundError: /data/provo missing", "data-absent"),
        ("KeyError: 'first_run'", "error"),
        ("ChecksumMismatchError: source checksum differs", "error"),
        ("ValueError: Language 'en' not found in .rda", "error"),
        ("EmptyDataError: No columns to parse from file", "error"),
    ],
)
def test_probe_failures_are_classified(monkeypatch, error, expected):
    def fake_run(*args, **kwargs):
        payload = facts._MARKER + json.dumps({"error": error})
        return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    counts, status, detail = facts._probe_counts("provo", timeout=5)
    assert counts is None
    assert status == expected
    assert detail == error


def test_probe_timeout_is_reported_as_an_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    counts, status, detail = facts._probe_counts("provo", timeout=5)
    assert counts is None
    assert status == "error"
    assert "timed out" in detail


def test_snapshot_drift_flags_missing_and_stale_keys():
    key = next(iter(REGISTRY))
    drift = facts.snapshot_drift(
        {"counts": {key: {}}, "unmeasured": {"retired_corpus": "gone"}}
    )
    assert drift["stale"] == ["retired_corpus"]
    assert key not in drift["missing"]
    assert set(drift["missing"]) == set(REGISTRY) - {key}


def test_write_snapshot_records_counts_unmeasured_and_provenance(tmp_path):
    collected = {
        "datasets": {
            "provo": {
                "counts": {"participants": 84},
                "counts_source": "measured",
            },
            "celer": {
                "counts": None,
                "counts_source": "",
                "probe_status": "data-absent",
                "probe_detail": "requires LDC access",
            },
        }
    }
    path = facts.write_snapshot(collected, tmp_path / "counts.json")
    written = json.loads(path.read_text())
    assert written["counts"] == {"provo": {"participants": 84}}
    assert written["unmeasured"] == {"celer": "data-absent: requires LDC access"}
    assert written["provenance"]["generated"].endswith("Z")
    assert written["provenance"]["data_root"]
    assert written["provenance"]["producer"] == (
        "replica.dataset.catalog.facts.counts_for"
    )
    assert (
        written["provenance"]["producer_sha256"]
        == hashlib.sha256(
            inspect.getsource(facts.counts_for).encode("utf-8")
        ).hexdigest()
    )
    assert written["provenance"]["input_checksums"] == "not-verified"


def test_a_file_presence_pass_does_not_discard_measured_counts(tmp_path):
    path = tmp_path / "counts.json"
    facts.write_snapshot(
        {
            "datasets": {
                "provo": {
                    "counts": {"participants": 84},
                    "counts_source": "measured",
                    "probe_status": "ok",
                }
            }
        },
        path,
    )
    # A pass that only stats files reports the snapshot's counts back, never a
    # measurement, so it must leave them alone.
    facts.write_snapshot(
        {
            "datasets": {
                "provo": {
                    "counts": {"participants": 84},
                    "counts_source": "snapshot",
                    "files_on_disk": {"declared": 3, "present": 3},
                }
            }
        },
        path,
        merge=True,
    )
    written = json.loads(path.read_text())
    assert written["counts"] == {"provo": {"participants": 84}}
    assert written["files"] == {"provo": {"declared": 3, "present": 3}}
    assert written["unmeasured"] == {}


def test_only_a_full_sweep_may_rebuild_the_snapshot(monkeypatch):
    """Naming a corpus adds to the snapshot; only a full sweep rebuilds it."""
    seen: dict[str, bool] = {}

    def capture(collected, path=None, *, merge=False):
        seen["merge"] = merge
        return Path("counts.json")

    monkeypatch.setattr(facts, "dataset_facts", lambda *a, **k: {"datasets": {}})
    monkeypatch.setattr(facts, "write_snapshot", capture)

    facts.main(["dundee", "--measure", "--write-snapshot"])
    assert seen["merge"] is True

    facts.main(["--measure", "--write-snapshot"])
    assert seen["merge"] is False


def test_merged_write_keeps_counts_the_new_facts_do_not_cover(tmp_path):
    path = tmp_path / "counts.json"
    facts.write_snapshot(
        {
            "datasets": {
                "provo": {"counts": {"participants": 84}, "counts_source": "measured"}
            }
        },
        path,
    )
    facts.write_snapshot(
        {
            "datasets": {
                "dundee": {"counts": {"participants": 10}, "counts_source": "measured"}
            }
        },
        path,
        merge=True,
    )
    written = json.loads(path.read_text())
    assert set(written["counts"]) == {"provo", "dundee"}


def test_merged_write_clears_a_stale_unmeasured_entry(tmp_path):
    path = tmp_path / "counts.json"
    facts.write_snapshot(
        {
            "datasets": {
                "celer": {
                    "counts": None,
                    "counts_source": "",
                    "probe_status": "data-absent",
                    "probe_detail": "not on disk",
                }
            }
        },
        path,
    )
    facts.write_snapshot(
        {
            "datasets": {
                "celer": {"counts": {"participants": 365}, "counts_source": "measured"}
            }
        },
        path,
        merge=True,
    )
    written = json.loads(path.read_text())
    assert written["counts"]["celer"] == {"participants": 365}
    assert "celer" not in written["unmeasured"]


def test_snapshot_counts_fill_the_data_free_tier(tmp_path, monkeypatch):
    snapshot = tmp_path / "counts.json"
    snapshot.write_text(
        json.dumps({"counts": {"provo": {"participants": 84}}, "unmeasured": {}})
    )
    monkeypatch.setattr(facts, "SNAPSHOT_PATH", snapshot)
    record = facts.dataset_facts(["provo"])["datasets"]["provo"]
    assert record["counts"] == {"participants": 84}
    assert record["counts_source"] == "snapshot"


def test_a_rebuild_refuses_records_that_attempted_no_load(tmp_path):
    path = tmp_path / "counts.json"
    facts.write_snapshot(
        {
            "datasets": {
                "provo": {"counts": {"participants": 84}, "counts_source": "measured"}
            }
        },
        path,
    )
    before = path.read_text()
    unmeasured = {
        "datasets": {
            "provo": {"counts": {"participants": 84}, "counts_source": "snapshot"}
        }
    }
    with pytest.raises(ValueError, match="attempted no load"):
        facts.write_snapshot(unmeasured, path)
    assert path.read_text() == before
