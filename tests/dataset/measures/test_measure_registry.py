"""The measure vocabulary covers every name the loaders declare.

Constructing a loader is enough to read its declared measures and its
presentation, so these run without a corpus on disk.
"""

import pytest

from replica.core import CONDITIONS, DERIVED_DVS, Presentation, resolve_dv
from replica.dataset.catalog.description import describe_dataset
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.measures import (
    CANONICAL_MEASURES,
    CANONICAL_RT_MEASURES,
    FIRST_PASS_RUN,
    FIRST_RUN,
    MEASURE_SPELLINGS,
    PASSTHROUGH_MEASURES,
    ResponseFamily,
    canonical_measure,
)

_DESCRIPTIONS = {name: describe_dataset(name) for name in sorted(REGISTRY)}


def test_every_declared_measure_is_in_the_vocabulary():
    known = (
        set(CANONICAL_MEASURES)
        | set(PASSTHROUGH_MEASURES)
        | set(CONDITIONS)
        | set(DERIVED_DVS)
    )
    unknown = {
        name: sorted(set(described.measures) - known)
        for name, described in _DESCRIPTIONS.items()
        if set(described.measures) - known
    }
    assert not unknown, f"measures outside the vocabulary: {unknown}"


def test_a_passthrough_column_is_not_documented_as_a_response():
    """Boundary flags and saccade lengths are controls the release carries."""
    assert not set(PASSTHROUGH_MEASURES) & set(CANONICAL_MEASURES)


def test_a_condition_is_neither_a_measure_nor_a_control():
    """The first-pass flag is a condition that restricts a measure."""
    assert not set(CONDITIONS) & (set(CANONICAL_MEASURES) | set(PASSTHROUGH_MEASURES))


def test_a_derived_variable_names_a_measure_and_a_condition_it_is_built_from():
    for name, (measure, condition) in DERIVED_DVS.items():
        assert measure in CANONICAL_MEASURES, name
        assert condition in CONDITIONS, name
        assert name not in CANONICAL_MEASURES and name not in CONDITIONS, name
        assert resolve_dv(name) == (measure, condition)
    assert resolve_dv("total_time") == ("total_time", None)
    assert resolve_dv(FIRST_PASS_RUN) == ("first_run", "first_pass")
    assert resolve_dv(FIRST_RUN) == ("first_run", None)


def test_the_reading_time_subset_is_the_eye_tracking_family():
    assert CANONICAL_RT_MEASURES == {
        name
        for name, spec in CANONICAL_MEASURES.items()
        if spec.family is ResponseFamily.EYE_TRACKING
    } | set(DERIVED_DVS)


@pytest.mark.parametrize("family", list(ResponseFamily))
def test_every_family_carries_a_measure(family):
    assert any(spec.family is family for spec in CANONICAL_MEASURES.values())


def test_every_quantity_is_written_as_a_sentence():
    """The site prints these verbatim, as the definition of the measure."""
    for name, spec in CANONICAL_MEASURES.items():
        assert spec.description[:1].isupper(), f"{name} does not open a sentence"
        assert spec.description.endswith("."), f"{name} does not close a sentence"


def test_a_reading_duration_is_defined_over_units_not_words():
    """Loaders aggregate fixations to characters and spans as well as to words,
    so a name shared across them cannot define itself over one of them."""
    for name, spec in CANONICAL_MEASURES.items():
        if spec.family is not ResponseFamily.EYE_TRACKING:
            continue
        assert "word" not in spec.description.lower(), f"{name} names words"


def test_durations_and_amplitudes_state_their_unit():
    for name, spec in CANONICAL_MEASURES.items():
        if spec.family is ResponseFamily.BOLD or spec.family is (
            ResponseFamily.BAND_POWER
        ):
            continue
        assert spec.unit, f"{name} states no unit"


def test_presentation_values_come_from_the_enum():
    allowed = {str(p) for p in Presentation}
    for name, described in _DESCRIPTIONS.items():
        unknown = set(described.presentation) - allowed
        assert not unknown, f"{name} declares unknown presentation {unknown}"


def test_a_corpus_is_not_assumed_to_be_read():
    """Narratives was listened to, and nothing marks it as reading."""
    assert describe_dataset("narratives").presentation == (str(Presentation.LISTENING),)


def test_every_spelling_resolves_to_a_canonical_label():
    """A release column name lands on a measure, conditioned where the column is."""
    for spelling, target in MEASURE_SPELLINGS.items():
        assert target in CANONICAL_MEASURES or target in DERIVED_DVS, spelling
        assert canonical_measure(spelling) == target


def test_a_column_reporting_the_run_in_first_pass_only_spells_the_conditional():
    for spelling in ("fdurFP", "RTfirstpass", "FPRT", "first_pass_reading_time"):
        assert canonical_measure(spelling) == FIRST_PASS_RUN, spelling
    for spelling in ("IA_FIRST_RUN_DWELL_TIME", "FRT"):
        assert canonical_measure(spelling) == FIRST_RUN, spelling
    with pytest.raises(KeyError, match="requires source context"):
        canonical_measure("firstrun.dur")


@pytest.mark.parametrize("name", ["gaze_rt", "GazeDuration", "GAZE_RT", "gazeduration"])
def test_study_gaze_columns_require_source_context(name):
    with pytest.raises(KeyError, match="requires source context"):
        canonical_measure(name)


def test_canonical_names_and_derived_variables_resolve_to_themselves():
    for name in (*CANONICAL_MEASURES, *DERIVED_DVS):
        assert canonical_measure(name) == name
    assert canonical_measure("N400") == canonical_measure("n400") == "n400"


def test_an_unknown_spelling_is_refused():
    """The field's own term is refused, since it names two quantities."""
    for name in ("reading_time", "gaze_duration", "first_pass", "first_pass_run"):
        with pytest.raises(KeyError):
            canonical_measure(name)


def test_self_paced_corpora_agree_on_the_measure_name():
    """The same quantity carries one name across the SPR corpora."""
    for name in ("natural_stories", "ucl_spr", "brown_spr"):
        assert "spr_rt" in describe_dataset(name).measures, name
