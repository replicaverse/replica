"""Every corpus declares where its measures come from, and the loader agrees.

The declaration is the check's reference point: without it a measure name is
only a label, and two corpora can ship the same name for different quantities.
These run without a corpus, since they compare the declaration to what the
loader and the catalog say about themselves.
"""

import pytest

from replica.core import CONDITIONS, DERIVED_DVS
from replica.dataset.catalog.description import describe_dataset
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.measures import (
    CANONICAL_MEASURES,
    PASSTHROUGH_MEASURES,
    with_derived,
)
from replica.dataset.measures.sources import (
    MEASURE_SOURCES,
    QUANTITIES,
    RULES,
    UNDECLARED,
)


def _loader_classes() -> dict[str, list[str]]:
    """Loader class name to the registry entries that use it."""
    classes: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        classes.setdefault(spec.loader_class, []).append(name)
    return classes


LOADER_CLASSES = _loader_classes()


def test_every_loader_declares_its_measures_or_names_the_gap():
    undeclared = set(LOADER_CLASSES) - set(MEASURE_SOURCES) - set(UNDECLARED)
    assert not undeclared, (
        f"loaders with neither a declaration nor a recorded reason: "
        f"{sorted(undeclared)}"
    )


def test_a_recorded_gap_names_a_real_loader_and_its_reason():
    for cls, reason in UNDECLARED.items():
        assert cls in LOADER_CLASSES, f"{cls} is not a registered loader"
        assert cls not in MEASURE_SOURCES, f"{cls} is both declared and exempted"
        assert reason.strip(), f"{cls} is exempted without a reason"


@pytest.mark.parametrize("loader_class", sorted(MEASURE_SOURCES))
def test_a_declaration_is_well_formed(loader_class: str):
    vocabulary = (
        set(CANONICAL_MEASURES)
        | set(PASSTHROUGH_MEASURES)
        | set(CONDITIONS)
        | set(DERIVED_DVS)
    )
    declared = MEASURE_SOURCES[loader_class]
    assert declared, f"{loader_class} declares no measure"
    for measure, source in declared.items():
        where = f"{loader_class}.{measure}"
        assert measure in vocabulary, f"{where} is outside the measure vocabulary"
        assert source.quantity in QUANTITIES, f"{where}: unknown quantity"
        assert source.rule in RULES, f"{where}: unknown rule"
        if source.rule in {"released", "skip_flag", "conditional_column"}:
            assert source.source, f"{where}: a released measure names no field"
            assert source.basis is None, f"{where}: a released measure is not derived"
        if source.rule == "skip_flag":
            assert measure == "first_pass" and source.quantity == "in_first_pass", (
                f"{where}: a skip flag yields the first-pass flag and nothing else"
            )
        if source.rule == "conditional_column":
            assert measure in {"first_run", "first_pass", "go_past"}, (
                f"{where}: a conditional column yields a supported base window or flag"
            )
        if source.rule == "fixation_stream":
            assert source.source is None, (
                f"{where}: a stream-derived measure reads no field"
            )
        if source.basis is not None:
            assert source.basis in declared, (
                f"{where}: basis {source.basis!r} is not declared for this loader"
            )
        if measure == "first_pass":
            assert source.quantity == "in_first_pass", f"{where}: the flag is a flag"
        else:
            assert source.quantity != "in_first_pass", (
                f"{where}: only the flag is a flag"
            )


@pytest.mark.parametrize("loader_class", sorted(MEASURE_SOURCES))
def test_the_declaration_matches_what_the_loader_advertises(loader_class: str):
    for name in LOADER_CLASSES[loader_class]:
        advertised = set(describe_dataset(name).measures)
        expected = with_derived(MEASURE_SOURCES[loader_class])
        assert advertised == expected, (
            f"{name}: the catalog advertises {sorted(advertised)} but the "
            f"declaration composes to {sorted(expected)}"
        )


def test_the_first_pass_flag_is_only_claimed_where_a_release_supports_it():
    """The flag needs a skip column, a conditional column or a stream."""
    for loader_class, declared in MEASURE_SOURCES.items():
        source = declared.get("first_pass")
        if source is None:
            continue
        assert source.rule in {"skip_flag", "conditional_column", "fixation_stream"}, (
            f"{loader_class}: first_pass is never read as a plain released value"
        )


def test_a_conditional_column_yields_the_run_and_the_flag_together():
    """A release that reports the first run in first pass only gives both or neither."""
    for loader_class, declared in MEASURE_SOURCES.items():
        conditional = {
            name
            for name, source in declared.items()
            if source.rule == "conditional_column" and name != "go_past"
        }
        if conditional:
            assert conditional <= {"first_run", "first_pass"}, loader_class
            sources = {declared[name].source for name in conditional}
            assert len(sources) == 1, f"{loader_class}: two conditional columns"
            assert (
                "first_pass" in conditional
                or declared["first_pass"].rule == "skip_flag"
            ), loader_class


def test_loader_subclasses_keep_source_selection_and_provenance():
    import numpy as np

    from replica.analysis import OLS, AnalysisConfig, prepare_observations
    from replica.analysis.specification.evaluation import EvaluationSpec
    from replica.core import Datum
    from replica.dataset.eye_tracking.provo import PROVO
    from replica.feature import UnitLength

    class CustomPROVO(PROVO):
        pass

    prepared = []
    for loader in (PROVO, CustomPROVO):
        dataset = loader()
        stimulus = dataset._make_stimulus(["A", "long", "word"], "example")
        dataset.add_datum(
            Datum(stimulus, "first_fixation", np.array([200.0, 250.0, 300.0]), "reader")
        )
        response = dataset.feature("IA_FIRST_FIXATION_DURATION", source="released")
        prepared.append(
            prepare_observations(
                dataset,
                EvaluationSpec(
                    predictors=[UnitLength()],
                    targets=[response],
                ),
                AnalysisConfig(model=OLS(), design=None),
            )
        )
    original, inherited = prepared
    np.testing.assert_array_equal(inherited.y, original.y)
    assert inherited.response["origins"] == original.response["origins"]
    assert inherited.response["origins"][0]["source_declaration"]["field"] == (
        "IA_FIRST_FIXATION_DURATION"
    )


def test_subclass_source_declaration_replaces_inherited_mapping(monkeypatch):
    from replica.dataset.eye_tracking.provo import PROVO
    from replica.dataset.measures.sources import MeasureSource

    class CustomPROVO(PROVO):
        pass

    monkeypatch.setitem(
        MEASURE_SOURCES,
        "CustomPROVO",
        {"first_fixation": MeasureSource("first_fixation", source="custom_duration")},
    )
    assert (
        CustomPROVO().feature("custom_duration", source="released").name
        == "first_fixation"
    )
    with pytest.raises(ValueError, match="matching declarations: \\[\\]"):
        CustomPROVO().feature("IA_FIRST_FIXATION_DURATION", source="released")
    monkeypatch.setitem(MEASURE_SOURCES, "CustomPROVO", {})
    with pytest.raises(ValueError, match="matching declarations: \\[\\]"):
        CustomPROVO().feature("IA_FIRST_FIXATION_DURATION", source="released")
