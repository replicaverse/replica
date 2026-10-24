"""Check resource markers on collected tests and their fixture closures."""

import ast
import inspect
import textwrap
from functools import cache
from pathlib import Path

from scripts.checks.source.test_policy import (
    _DATA_DIRS,
    _MODEL_MARKERS,
    _data_access,
    _model_use,
)


@cache
def _requirements(function):
    try:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(function)))
    except (OSError, TypeError):
        return False, False
    tree = ast.parse(source)
    loaders, _ = _model_use(tree)
    parameters = set(inspect.signature(function).parameters)
    data = any(
        isinstance(node, ast.Call)
        and _data_access(node) is not None
        and not (isinstance(node.func, ast.Name) and node.func.id in parameters)
        for node in ast.walk(tree)
    )
    return bool(loaders), data


def resource_errors(item):
    """Report missing markers using the item's inherited and parameter markers.

    Args:
        item: Collected pytest function with its resolved fixtures.
    """
    if not hasattr(item, "obj") or not hasattr(item, "fixturenames"):
        return []
    markers = {marker.name for marker in item.iter_markers()}
    fixtures = set(item.fixturenames)
    model, data = _requirements(item.obj)
    # pytest resolves fixture overrides and their dependencies at collection.
    for name, definitions in item._fixtureinfo.name2fixturedefs.items():
        for definition in reversed(definitions or ()):
            fixture_model, fixture_data = _requirements(definition.func)
            model |= fixture_model
            data |= fixture_data
            if name not in definition.argnames:
                break
    model |= "lm_cache" in fixtures
    data |= "data_root" in fixtures
    errors = []
    exemption = item.get_closest_marker("resource_free")
    if exemption is not None:
        if (
            not exemption.args
            or not isinstance(exemption.args[0], str)
            or not exemption.args[0].strip()
        ):
            errors.append(f"{item.nodeid}: resource_free requires a reason")
        if not (model or data):
            errors.append(f"{item.nodeid}: stale resource_free marker")
        if fixtures.intersection({"lm_cache", "data_root"}):
            errors.append(
                f"{item.nodeid}: resource_free cannot exempt resource fixtures"
            )
        model, data = False, False
    if model and not markers.intersection(_MODEL_MARKERS):
        errors.append(
            f"{item.nodeid}: model loading requires needs_model or needs_data"
        )
    relative = Path(item.path).relative_to(item.config.rootpath)
    corpus_test = len(relative.parts) > 2 and relative.parts[1] in _DATA_DIRS
    if (
        "data_root" in fixtures or (corpus_test and data)
    ) and "needs_data" not in markers:
        errors.append(f"{item.nodeid}: on-disk data access requires needs_data")
    return errors
