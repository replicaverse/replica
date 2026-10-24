"""Completeness guard for the value-key unit registry.

Every key a compute method extracts has to declare its base in
:mod:`replica.feature.core.outputs.units`, so a new predictor cannot reach a published
predictor table without one. The base literals themselves are pinned against
closed forms in ``test_unit_conventions.py``; this file checks coverage and
that lookup fails loudly.
"""

from dataclasses import replace

import pytest

from replica.compute.methods import METHOD_SPECS, ComputeContext
from replica.feature.core.outputs.units import (
    _PREFIX_UNITS,
    VALUE_UNITS,
    UnknownValueKey,
    unit_for,
)

# Information value's length distance is an absolute token-count difference.
_UNITS = {"bits", "nats", "dimensionless", "tokens", "identifier", "text"}

_CONTEXT = ComputeContext(
    model_id="gpt2",
    beam_width=5,
    n_samples=64,
    max_tokens=10,
    alpha=0.5,
    seed=0,
    add_bos=True,
    dataset_language="en",
    language_override=None,
)


@pytest.mark.parametrize("method", METHOD_SPECS)
def test_every_extracted_key_has_a_unit(method, tmp_path):
    # Per-layer keys need only a layer count.
    (tmp_path / "config.json").write_text('{"model_type": "gpt2", "n_layer": 12}')
    context = replace(_CONTEXT, model_id=str(tmp_path))
    for key in METHOD_SPECS[method].extract_keys(context):
        assert unit_for(key) in _UNITS


def test_every_registered_unit_is_known():
    registered = set(VALUE_UNITS.values()) | {unit for _, unit in _PREFIX_UNITS}
    assert registered <= _UNITS


@pytest.mark.parametrize(
    "key, expected_unit",
    [
        ("token_renyi_entropy_alpha0.5", "bits"),
        ("restricted_first_token_renyi_entropy_alpha1.0", "bits"),
        ("logit_lens_log_p_L6", "nats"),
        ("tuned_lens_log_p_L0", "nats"),
    ],
)
def test_parameterised_keys_resolve(key, expected_unit):
    assert unit_for(key) == expected_unit


@pytest.mark.parametrize(
    "key",
    [
        "",
        "entropy",
        "token_renyi_entropy_alphaX",
        "logit_lens_log_p_Lx",
        "logit_lens_log_p_L1e1",
    ],
)
def test_unknown_key_raises(key):
    with pytest.raises(UnknownValueKey):
        unit_for(key)


@pytest.mark.parametrize("alpha", [1.0, 1e-5, 1e20])
@pytest.mark.parametrize(
    "method", ["token_renyi_entropy", "restricted_first_token_renyi_entropy"]
)
def test_produced_renyi_keys_have_units(alpha, method):
    context = replace(_CONTEXT, alpha=alpha)
    (key,) = METHOD_SPECS[method].extract_keys(context)
    assert key.endswith(str(alpha))
    assert unit_for(key) == "bits"


@pytest.mark.parametrize("suffix", ["nan", "inf", "1e999", "1e", "1e+", "1e-", "1.0\n"])
def test_nonfinite_or_malformed_key_parameters_are_rejected(suffix):
    with pytest.raises(UnknownValueKey):
        unit_for(f"token_renyi_entropy_alpha{suffix}")
