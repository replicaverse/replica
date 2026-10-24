"""Discovering cells across output directories."""

import json

import pytest

from replica.artifacts import manifest_path
from replica.predictor_store.cells import discover
from tests.predictor_store._cells import (
    DATASET,
    word_payload,
    write_cell,
    write_cell_under_old_name,
)

OLD = f"gpt2_next_word_entropy_{DATASET}"
NEW = f"gpt2_restricted_first_token_entropy_{DATASET}"


def _write_current(out, scale=-1.0):
    return write_cell(
        out,
        NEW,
        word_payload("restricted_first_token_entropy", scale),
        method="restricted_first_token_entropy",
    )


def test_shards_under_an_earlier_name_belong_to_the_current_cell(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / f"{OLD}.shard00000_00001.pkl").write_bytes(b"")
    (out / f"{OLD}.shard00001_00002.pkl").write_bytes(b"")
    found = discover([("w", out)], [DATASET])
    assert found.shard_sets == {NEW: 2} and found.completed == []
    _write_current(out)
    found = discover([("w", out)], [DATASET])
    assert found.shard_sets == {} and found.tags == {NEW}


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_different_payloads_under_alias_names_are_ambiguous(tmp_path, order):
    write_cell_under_old_name(tmp_path / "a", OLD, "next_word_entropy")
    _write_current(tmp_path / "b")
    with pytest.raises(ValueError, match="different payloads"):
        discover([(name, tmp_path / name) for name in order], [DATASET])


def test_two_payloads_under_one_name_are_refused(tmp_path):
    _write_current(tmp_path / "a")
    _write_current(tmp_path / "b", scale=3.0)
    with pytest.raises(ValueError, match="completed in both"):
        discover([("a", tmp_path / "a"), ("b", tmp_path / "b")], [DATASET])


@pytest.mark.parametrize(
    ("requested", "source", "expected"),
    [
        ("gpt2", "openai-community/gpt2", "gpt2"),
        (None, "openai-community/gpt2", "openai-community/gpt2"),
        (None, None, None),
    ],
)
def test_manifest_requested_and_resolved_identities_are_distinct(
    tmp_path, requested, source, expected
):
    path = _write_current(tmp_path)
    sidecar = manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    manifest["identity"]["options"]["model_id"] = requested
    manifest["identity"]["model"]["source"] = source
    sidecar.write_text(json.dumps(manifest))
    before = sidecar.read_bytes()
    found = discover([("w", tmp_path)], [DATASET])
    assert not found.problems
    assert found.completed[0].info.model == expected
    assert sidecar.read_bytes() == before


def test_requested_manifest_identity_must_match_the_filename(tmp_path):
    write_cell(
        tmp_path,
        NEW,
        word_payload(),
        method="restricted_first_token_entropy",
        model_id="other",
    )
    found = discover([("w", tmp_path)], [DATASET])
    assert found.completed == []
    assert "model_id conflicts with the filename" in found.problems[0]


@pytest.mark.parametrize(
    ("field", "value", "valid"),
    [
        ("identity", [], False),
        ("options", "bad", False),
        ("options", None, False),
        ("model", "bad", False),
        ("model", None, True),
    ],
)
def test_discovery_checks_manifest_metadata_shapes(tmp_path, field, value, valid):
    path = _write_current(tmp_path)
    sidecar = manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    target = manifest if field == "identity" else manifest["identity"]
    target[field] = value
    sidecar.write_text(json.dumps(manifest))
    found = discover([("w", tmp_path)], [DATASET])
    assert bool(found.completed) is valid
    if valid:
        assert found.problems == []
    else:
        assert "must be an object" in found.problems[0]


@pytest.mark.parametrize(
    ("method", "add_bos", "start", "problem"),
    [
        ("surprisal", False, {"ids": [5]}, "contradicts add_bos=False"),
        ("next_word_entropy", False, {"ids": [5]}, "contradicts add_bos=False"),
        ("surprisal", False, {"ids": []}, None),
        ("surprisal", False, {}, None),
        ("surprisal", False, None, None),
        ("surprisal", True, {"ids": [5]}, None),
        ("surprisal", None, {"ids": [5]}, None),
        ("mmc_unigram_surprisal", False, {"ids": [5]}, None),
        ("surprisal", False, [5], "start must be an object"),
        ("surprisal", False, {"ids": 5}, "start.ids must be an array"),
    ],
)
def test_discovery_rejects_contradictory_disabled_prefixes(
    tmp_path, method, add_bos, start, problem
):
    from replica.compute.cache import method_cache_tag
    from replica.utils.naming import canonical_method

    canonical = canonical_method(method)
    tag = f"gpt2_{method_cache_tag(canonical, add_bos=add_bos is not False)}_{DATASET}"
    path = write_cell(tmp_path, tag, word_payload(), method=canonical)
    sidecar = manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    identity = manifest["identity"]
    identity["method"] = method
    if add_bos is not None:
        identity["options"]["add_bos"] = add_bos
    if start is not None:
        identity["model"]["start"] = start
    sidecar.write_text(json.dumps(manifest))
    before = sidecar.read_bytes(), path.read_bytes()
    found = discover([("w", tmp_path)], [DATASET])
    if problem is None:
        assert len(found.completed) == 1 and not found.problems
    else:
        assert not found.completed and problem in found.problems[0]
    assert (sidecar.read_bytes(), path.read_bytes()) == before


@pytest.mark.parametrize("reverse", [False, True])
def test_payload_alias_collisions_are_rejected_before_normalization(tmp_path, reverse):
    from replica.predictor_store.cells import load_payload

    pairs = [("next_word_entropy", 1.0), ("restricted_first_token_entropy", 9.0)]
    payload = word_payload()
    payload["toy_1"][0].values = dict(reversed(pairs) if reverse else pairs)
    write_cell(tmp_path, NEW, payload, method="restricted_first_token_entropy")
    (cell,) = discover([("w", tmp_path)], [DATASET]).completed
    with pytest.raises(ValueError, match="both name"):
        load_payload(cell)


@pytest.mark.parametrize(
    ("units", "problem"),
    [
        (
            {"next_word_entropy": "bits", "restricted_first_token_entropy": "bits"},
            "both name",
        ),
        (
            {"restricted_first_token_entropy": "nats"},
            "conflicts with the unit registry",
        ),
    ],
)
def test_original_unit_declarations_must_be_unambiguous(tmp_path, units, problem):
    write_cell(
        tmp_path,
        NEW,
        word_payload(),
        method="restricted_first_token_entropy",
        units=units,
    )
    found = discover([("w", tmp_path)], [DATASET])
    assert not found.completed
    assert problem in found.problems[0]


@pytest.mark.parametrize("alpha", [None, 0.5])
def test_encoded_options_distinguish_missing_and_conflicting_values(tmp_path, alpha):
    tag = f"gpt2_token_renyi_entropy_a1_{DATASET}"
    path = write_cell(
        tmp_path,
        tag,
        word_payload("token_renyi_entropy_alpha1.0"),
        method="token_renyi_entropy",
    )
    sidecar = manifest_path(path)
    manifest = json.loads(sidecar.read_text())
    if alpha is None:
        del manifest["identity"]["options"]["alpha"]
    else:
        manifest["identity"]["options"]["alpha"] = alpha
    sidecar.write_text(json.dumps(manifest))
    found = discover([("w", tmp_path)], [DATASET])
    assert not found.completed
    assert (
        "missing encoded settings" if alpha is None else "conflict with filename"
    ) in found.problems[0]


@pytest.mark.parametrize("change", ["revision", "precise_alpha", "environment"])
def test_duplicate_bytes_require_matching_computational_identity(tmp_path, change):
    tag = f"gpt2_token_renyi_entropy_a0.123457_{DATASET}"
    for name in ["a", "b"]:
        path = write_cell(
            tmp_path / name,
            tag,
            word_payload("token_renyi_entropy_alpha0.123456789"),
            method="token_renyi_entropy",
        )
        sidecar = manifest_path(path)
        manifest = json.loads(sidecar.read_text())
        manifest["identity"]["options"]["alpha"] = 0.123456789
        if name == "b":
            if change == "revision":
                manifest["identity"]["model"]["revision"] = "another-revision"
            elif change == "precise_alpha":
                manifest["identity"]["options"]["alpha"] = 0.123456788
            else:
                manifest.pop("environment", None)
        sidecar.write_text(json.dumps(manifest))
    assert (tmp_path / "a" / f"{tag}.pkl").read_bytes() == (
        tmp_path / "b" / f"{tag}.pkl"
    ).read_bytes()
    for names in [("a", "b"), ("b", "a")]:
        sources = [(name, tmp_path / name) for name in names]
        if change == "environment":
            found = discover(sources, [DATASET])
            assert len(found.completed) == 1 and not found.problems
        else:
            with pytest.raises(
                ValueError, match="conflicting computational identities"
            ):
                discover(sources, [DATASET])
