"""Reusable predictor-source loading and alignment contracts."""

import pytest

import replica.predictors
from replica.multiverse.run import features as pp
from replica.multiverse.run.features import pickle_template_for, resolve_existing_pickle


def test_pickle_template_accepts_canonical_identifiers_and_explicit_aliases():
    assert pickle_template_for("gpt2", "surprisal") == "gpt2_surprisal_{ds}.pkl"
    expected = "lab_new-model_v2_surprisal_{ds}.pkl"
    assert pickle_template_for("lab/new-model:v2", "surprisal") == expected
    assert (
        pickle_template_for("small", "surprisal", aliases={"small": "lab/new-model:v2"})
        == expected
    )
    assert pickle_template_for("mgpt", "surprisal") == "mgpt_surprisal_{ds}.pkl"


def test_resolve_existing_pickle_requires_exact_tag_and_size(tmp_path):
    assert resolve_existing_pickle(tmp_path, "gpt2", "surprisal", "provo") is None
    pkl = tmp_path / "gpt2_surprisal_provo.pkl"
    # At or below the truncation guard the pickle counts as absent.
    pkl.write_bytes(b"x" * pp.MIN_VALID_PICKLE_BYTES)
    assert resolve_existing_pickle(tmp_path, "gpt2", "surprisal", "provo") is None
    pkl.write_bytes(b"x" * (pp.MIN_VALID_PICKLE_BYTES + 1))
    assert resolve_existing_pickle(tmp_path, "gpt2", "surprisal", "provo") == pkl
    assert (
        resolve_existing_pickle(
            tmp_path, "small", "surprisal", "provo", aliases={"small": "gpt2"}
        )
        == pkl
    )
    assert resolve_existing_pickle(tmp_path, "gpt2", "surprisal_s0", "provo") is None


def test_load_pickle_outputs_rejects_non_dict(monkeypatch, tmp_path):
    # The default loader is the predictor-store boundary; this is the one
    # test that patches it, pinning the store's type contract.
    monkeypatch.setattr(
        replica.predictors,
        "load_outputs_for_tag",
        lambda tag, local_path=None, allow_reproduction=False: [1],
    )
    with pytest.raises(TypeError, match="expected dict"):
        pp.load_pickle_outputs(tmp_path / "bad_tag.pkl")


@pytest.mark.parametrize(
    "model_id", ["", "../model", "/model", "lab/../model", "lab\\model", "{ds}"]
)
def test_pickle_template_rejects_invalid_canonical_ids_and_alias_targets(model_id):
    with pytest.raises(ValueError, match="invalid canonical model"):
        pickle_template_for(model_id, "surprisal")
    with pytest.raises(ValueError, match="invalid canonical model"):
        pickle_template_for("small", "surprisal", aliases={"small": model_id})


def test_pickle_template_rejects_known_normalization_collisions():
    with pytest.raises(ValueError, match="collide in cache filenames"):
        pickle_template_for(
            "first", "surprisal", aliases={"first": "lab/model", "second": "lab_model"}
        )
    with pytest.raises(ValueError, match="collide in cache filenames"):
        pickle_template_for("lab_model", "surprisal", aliases={"first": "lab/model"})
