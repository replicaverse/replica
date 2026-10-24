"""Explicit component configuration for analysis provenance."""

import hashlib
import json

import numpy as np


def component_options(component) -> dict | None:
    """Copy declared options; None means configuration was not supplied.

    Components implement ``configuration()`` on their concrete class. Subclasses
    must declare their own method so inherited options cannot hide new settings.
    Fitted state is excluded by the declaration. Opaque nested values retain
    their qualified type and explicitly unknown options.

    Args:
        component: Model, splitter, or other configurable analysis component.
    """
    if "configuration" not in type(component).__dict__:
        return None
    options = component.configuration()
    if not isinstance(options, dict):
        raise TypeError("configuration() must return a JSON-compatible dictionary")
    return json.loads(
        json.dumps(options, default=_configuration_value, allow_nan=False)
    )


def _configuration_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "options": component_options(value),
    }


def array_identity(values: np.ndarray | None) -> dict | None:
    """Bind numerical input content without embedding observation arrays.

    Args:
        values: Numerical input array; None denotes absent input.
    """
    if values is None:
        return None
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
