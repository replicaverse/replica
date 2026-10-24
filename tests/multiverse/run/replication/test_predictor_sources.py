"""Predictor source resolution and local-file authority."""

import numpy as np
import pytest

from replica.multiverse.run.replication import sources


def test_lagged_nan_fills_the_leading_boundary():
    out = sources.lagged(np.array([1.0, 2.0, 3.0]), 2)
    assert np.isnan(out[:2]).all() and out[2] == 1.0


@pytest.mark.parametrize(
    "lm,aliases", [("lab/custom:v2", None), ("small", {"small": "lab/custom:v2"})]
)
def test_resolve_predictor_source_uses_same_identity_locally_and_in_store(
    tmp_path, monkeypatch, lm, aliases
):
    monkeypatch.setattr(
        "replica.predictors.published_tags",
        lambda: {"lab_custom_v2_surprisal_provo"},
    )
    got = sources.resolve_predictor_source(
        tmp_path, lm, "surprisal", "provo", aliases=aliases
    )
    assert got == "lab_custom_v2_surprisal_provo"
    local = tmp_path / f"{got}.pkl"
    local.write_bytes(b"x" * 101)
    assert (
        sources.resolve_predictor_source(
            tmp_path, lm, "surprisal", "provo", aliases=aliases
        )
        == local
    )
    assert (
        sources.resolve_predictor_source(
            tmp_path, lm, "surprisal", "geco", aliases=aliases
        )
        is None
    )


def test_load_predictor_source_keeps_a_dotted_tag_intact(monkeypatch):
    seen = {}

    def fake_load(tag, local_path=None, allow_reproduction=False):
        seen["tag"], seen["local_path"] = tag, local_path
        return {}

    monkeypatch.setattr("replica.predictors.load_outputs_for_tag", fake_load)
    sources.load_predictor_source("gpt2_fte_renyi_a0.5_geco")
    assert seen["tag"] == "gpt2_fte_renyi_a0.5_geco" and seen["local_path"] is None


def test_explicit_predictor_file_cannot_be_shadowed_by_published_tag(
    tmp_path, monkeypatch
):
    import pickle

    from replica.artifacts import write_manifest

    path = tmp_path / "gpt2_surprisal_geco.pkl"
    expected = {"geco_p3_t50": [1.0, 2.0, 3.0]}
    path.write_bytes(pickle.dumps(expected))
    write_manifest(
        path,
        {"schema_version": 1, "fixture": "synthetic local-authority payload"},
        status="completed",
    )

    def unexpected_hub_read(*args, **kwargs):
        raise AssertionError("An explicit local input must not read a published tag")

    monkeypatch.setattr("replica.predictors.load_outputs_for_tag", unexpected_hub_read)
    assert sources.load_predictor_source(path) == expected
    assert (
        sources.load_store_cell(path.stem, allow_compute_dir=True, cdir=tmp_path)
        == expected
    )
    with pytest.raises(FileNotFoundError):
        sources.load_predictor_source(tmp_path / "missing.pkl")


@pytest.mark.parametrize(
    "manifest_state", ["missing", "partial", "quarantined", "tampered"]
)
def test_local_predictor_readers_require_provenance_before_unpickling(
    tmp_path, monkeypatch, manifest_state
):
    from replica.artifacts import write_manifest

    path = tmp_path / "custom_cell.pkl"
    path.write_bytes(b"unreadable predictor payload")
    if manifest_state != "missing":
        write_manifest(
            path,
            {"schema_version": 1, "fixture": "invalid local artifact"},
            status="completed" if manifest_state == "tampered" else manifest_state,
        )
    if manifest_state == "tampered":
        path.write_bytes(b"changed after manifest creation")

    def unexpected_read(*args, **kwargs):
        raise AssertionError(
            "Invalid provenance must fail before unpickling or Hub access"
        )

    monkeypatch.setattr(sources.pickle, "load", unexpected_read)
    monkeypatch.setattr("replica.predictors.load_outputs_for_tag", unexpected_read)
    message = (
        "no provenance manifest"
        if manifest_state == "missing"
        else "invalid artifact manifest"
    )
    with pytest.raises(ValueError, match=message):
        sources.load_predictor_source(path, allow_reproduction=True)
    with pytest.raises(ValueError, match=message):
        sources.load_store_cell(
            path.stem, allow_compute_dir=True, cdir=tmp_path, allow_reproduction=True
        )


@pytest.mark.parametrize("source_only", [False, True])
def test_local_model_identity_mismatch_is_not_replaced_by_a_store_tag(
    tmp_path, monkeypatch, source_only
):
    from replica.artifacts import write_manifest

    path = tmp_path / "lab_custom_surprisal_provo.pkl"
    path.write_bytes(b"x" * 101)
    aliases = {"small": "lab/custom"}
    for model_id in ("lab/custom", "lab_custom"):
        identity = {
            "schema_version": 1,
            "model": {"source": model_id if source_only else "resolved/canonical"},
        }
        if not source_only:
            identity["options"] = {"model_id": model_id}
        write_manifest(path, identity, status="completed")
        if model_id == "lab/custom":
            assert (
                sources.resolve_predictor_source(
                    tmp_path, "small", "surprisal", "provo", aliases=aliases
                )
                == path
            )
        else:
            monkeypatch.setattr(
                "replica.predictors.published_tags", lambda: {path.stem}
            )
            with pytest.raises(ValueError, match="stored model .* differs"):
                sources.resolve_predictor_source(
                    tmp_path, "small", "surprisal", "provo", aliases=aliases
                )


def test_local_file_identity_gates_a_reproduction_only_cell(tmp_path):
    import pickle

    from replica.artifacts import write_manifest
    from replica.feature.core.reproduction import ReproductionOnlyWarning

    named = tmp_path / "gpt2_renyi05_buggy_provo.pkl"
    suspect = tmp_path / "gpt2_renyi05_buggy_provo.suspect.pkl"
    unnamed = tmp_path / "custom_cell.pkl"
    for path in (named, suspect, unnamed):
        path.write_bytes(pickle.dumps({"s": []}))
        identity = {"schema_version": 1, "corpus_sha256": "legacy"}
        if path is unnamed:
            identity["method"] = "renyi_entropy_buggy"
        write_manifest(path, identity, status="completed")
        with pytest.raises(ValueError, match="allow_reproduction=True"):
            sources.load_predictor_source(path)
        with pytest.warns(ReproductionOnlyWarning):
            assert sources.load_predictor_source(path, allow_reproduction=True) == {
                "s": []
            }
    with pytest.raises(ValueError, match="disagree"):
        sources.load_predictor_source(
            _renamed(unnamed, "gpt2_ft_entropy_buggy_provo.pkl"),
            allow_reproduction=True,
        )


def _renamed(path, name):
    """The same payload and manifest under another cell name."""
    target = path.with_name(name)
    target.write_bytes(path.read_bytes())
    from replica.artifacts import manifest_path

    manifest_path(target).write_bytes(manifest_path(path).read_bytes())
    return target


def test_store_cell_threads_consent_to_the_reader(monkeypatch):
    seen = {}

    def fake_load(tag, local_path=None, allow_reproduction=False):
        seen["allow"] = allow_reproduction
        return {}

    monkeypatch.setattr("replica.predictors.load_outputs_for_tag", fake_load)
    sources.load_store_cell("gpt2_surprisal_provo", allow_reproduction=True)
    assert seen["allow"] is True
