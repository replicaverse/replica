"""Source fields, quantities, and derivation rules for each dataset loader.

``MEASURE_SOURCES`` declares each measure in the default loader configuration. ``UNDECLARED`` lists
loaders without declarations and the reason each is excluded.
"""

from dataclasses import dataclass

#: Measurement quantities shared across corpora.
QUANTITIES: frozenset[str] = frozenset(
    {
        "first_fixation",  # the first fixation on a unit
        "first_fixation_in_first_pass",
        "first_run_in_first_pass",
        "first_run",  # the first run of fixations, whenever it occurs
        "in_first_pass",  # whether the unit was reached before the eye passed it
        "regression_path",  # arrival until the eye passes the unit
        "regression_path_in_first_pass",  # the same window, first-pass entry only
        "right_bounded",  # fixations on the unit before the eye passes it
        "total_dwell",  # every fixation on the unit
        "reading_time",  # one self-paced or maze response
        "amplitude",  # an averaged ERP component
        "band_power",  # EEG power in one frequency band
        "bold",  # a haemodynamic time course
        "control",  # a per-observation covariate the release ships
    }
)

#: How a loader obtains a measure.
RULES: frozenset[str] = frozenset(
    {
        "released",  # read from a field of the release
        "skip_flag",  # the first-pass flag, inverted from a release skip flag
        "conditional_column",  # unconditional values recoverable from a conditional field
        "fixation_stream",  # aggregated from the release's fixation events
    }
)


@dataclass(frozen=True)
class MeasureSource:
    """Source and derivation rule for a loader measurement.

    Args:
        quantity: A term from :data:`QUANTITIES`.
        source: Release field the value is read from, for a released, skip-flag
            or conditional-column measure.
        basis: Released measure a stream-derived measure is checked against
            before it is exposed.
        rule: A term from :data:`RULES`.
        released_source: Alternative release field retained under the same measure.
    """

    quantity: str
    source: str | None = None
    basis: str | None = None
    rule: str = "released"
    released_source: str | None = None


def _released(quantity: str, source: str) -> MeasureSource:
    """Declare a measure read from a release field."""
    return MeasureSource(quantity=quantity, source=source)


def _skip_flag(source: str) -> MeasureSource:
    """The first-pass flag, from a release flag set where the unit was skipped."""
    return MeasureSource(quantity="in_first_pass", source=source, rule="skip_flag")


def _conditional(source: str) -> dict[str, MeasureSource]:
    """First run and first-pass flag from a column reporting the run in first pass only."""
    return {
        "first_run": MeasureSource(
            quantity="first_run", source=source, rule="conditional_column"
        ),
        "first_pass": MeasureSource(
            quantity="in_first_pass", source=source, rule="conditional_column"
        ),
    }


def _stream(quantity: str, basis: str | None = None) -> MeasureSource:
    """A measure aggregated from the release's fixation events."""
    return MeasureSource(quantity=quantity, basis=basis, rule="fixation_stream")


def _eyelink() -> dict[str, MeasureSource]:
    """Declare the shared EyeLink interest-area fields used by these loaders."""
    return {
        "first_fixation": _released("first_fixation", "IA_FIRST_FIXATION_DURATION"),
        "first_run": _released("first_run", "IA_FIRST_RUN_DWELL_TIME"),
        "first_pass": _skip_flag("IA_SKIP"),
        "go_past": _released("regression_path", "IA_REGRESSION_PATH_DURATION"),
        "total_time": _released("total_dwell", "IA_DWELL_TIME"),
    }


def _meco_frame() -> dict[str, MeasureSource]:
    """The reading-measure frame shared by every MECO word-level release."""
    return {
        "first_fixation": _released("first_fixation", "firstfix.dur"),
        "first_run": MeasureSource(
            "first_run", rule="fixation_stream", released_source="firstrun.dur"
        ),
        "first_pass": _skip_flag("firstrun.skip"),
        **_conditional_go_past("firstrun.gopast"),
        "total_time": _released("total_dwell", "dur"),
    }


def _conditional_go_past(source: str) -> dict[str, MeasureSource]:
    """Declare first-pass go-past and the recoverable arrival-based values.

    Args:
        source: Release field holding the regression path in first pass only.
    """
    return {
        "go_past_in_first_pass": _released("regression_path_in_first_pass", source),
        "go_past": MeasureSource(
            quantity="regression_path", source=source, rule="conditional_column"
        ),
    }


def _pymovements() -> dict[str, MeasureSource]:
    """The reading-measure table pymovements writes, whose FPRT is conditional."""
    return {
        "first_fixation": _released("first_fixation", "FD"),
        "first_fixation_in_first_pass": _released(
            "first_fixation_in_first_pass", "FFD"
        ),
        "first_run_in_first_pass": _released("first_run_in_first_pass", "FPRT"),
        **_conditional("FPRT"),
        **_conditional_go_past("RPD_inc"),
        "total_time": _released("total_dwell", "TFT"),
    }


def _frank_release() -> dict[str, MeasureSource]:
    """Frank et al.'s per-word table, whose first-pass column is conditional."""
    return {
        "first_fixation_in_first_pass": _released(
            "first_fixation_in_first_pass", "RTfirstfix"
        ),
        **_conditional("RTfirstpass"),
        **_conditional_go_past("RTgopast"),
        "right_bounded": _released("right_bounded", "RTrightbound"),
    }


def _zuco_bands() -> dict[str, MeasureSource]:
    """Per-word band power over the total reading time, eight bands."""
    return {
        f"eeg_{band}": _released("band_power", field)
        for field, band in {
            "TRT_t1": "theta_1",
            "TRT_t2": "theta_2",
            "TRT_a1": "alpha_1",
            "TRT_a2": "alpha_2",
            "TRT_b1": "beta_1",
            "TRT_b2": "beta_2",
            "TRT_g1": "gamma_1",
            "TRT_g2": "gamma_2",
        }.items()
    }


def _zuco_mat() -> dict[str, MeasureSource]:
    """ZuCo's per-word MATLAB fields; GD is populated for every fixated word."""
    return {
        "first_fixation": _released("first_fixation", "FFD"),
        "first_run": _released("first_run", "GD"),
        "go_past": _released("regression_path", "GPT"),
        "total_time": _released("total_dwell", "TRT"),
    }


#: Loader class name -> measure name -> source in the default configuration.
MEASURE_SOURCES: dict[str, dict[str, MeasureSource]] = {
    # EyeLink interest-area reports.
    "PROVO": _eyelink(),
    "OneStop": _eyelink(),
    "CELER": _eyelink(),
    "RSC": _eyelink(),
    "RastrOS": _eyelink(),
    # MECO's word-level frames.
    "MECO_L1": _meco_frame(),
    "MECOL1W2": _meco_frame(),
    "MECOL2W1": _meco_frame(),
    "MECOL2W2": _meco_frame(),
    "MECO_char": {
        "total_time": _stream("total_dwell"),
        "right_bounded": _stream("right_bounded"),
    },
    # Complete measures from the author fixation exports.
    "EMTeC": {
        name: _stream(source.quantity) for name, source in _pymovements().items()
    },
    "PoTeC": {
        name: _stream(source.quantity) for name, source in _pymovements().items()
    },
    # Frank et al.'s releases. UCL completes the first run from its fixation
    # file, which must reproduce the released column on every first-pass word.
    "UCL_ET": {
        **_frank_release(),
        "first_fixation": _stream(
            "first_fixation", basis="first_fixation_in_first_pass"
        ),
        "first_run": _stream("first_run", basis="first_pass"),
        "total_time": _stream("total_dwell"),
    },
    # Frank et al. (2015): the six component amplitudes and their baselines,
    # indexed on the release's component axis.
    "Frank2015ERP": {
        **{
            name: _released("amplitude", f"ERP[:, :, {index}]")
            for index, name in enumerate(("elan", "lan", "n400", "epnp", "p600", "pnp"))
        },
        **{
            f"{name}_baseline": _released("control", f"ERPbase[:, :, {index}]")
            for index, name in enumerate(("elan", "lan", "n400", "epnp", "p600", "pnp"))
        },
    },
    "DeVarda2023": {
        **{
            name: source
            for name, source in _frank_release().items()
            if name not in {"go_past", "first_run", "first_pass"}
        },
        "first_run_in_first_pass": _released("first_run_in_first_pass", "RTfirstpass"),
        "spr_rt": _released("reading_time", "self_paced_reading_time"),
        "elan": _released("amplitude", "ELAN"),
        "lan": _released("amplitude", "LAN"),
        "n400": _released("amplitude", "N400"),
        "epnp": _released("amplitude", "EPNP"),
        "p600": _released("amplitude", "P600"),
        "pnp": _released("amplitude", "PNP"),
    },
    # GECO's own spreadsheet columns.
    "GECO": {
        "first_fixation": _released("first_fixation", "WORD_FIRST_FIXATION_DURATION"),
        "first_run": _released("first_run", "WORD_GAZE_DURATION"),
        "first_pass": _skip_flag("WORD_SKIP"),
        "go_past": _released("regression_path", "WORD_GO_PAST_TIME"),
        "total_time": _released("total_dwell", "WORD_TOTAL_READING_TIME"),
    },
    # CopCo's extracted features, plus the flag the release omits, rebuilt
    # from the fixation reports where they reproduce the released first run.
    "CopCo": {
        "first_fixation": _released("first_fixation", "word_first_fix_dur"),
        "first_run": _released("first_run", "word_first_pass_dur"),
        "first_pass": _stream("in_first_pass", basis="first_run"),
        "go_past": MeasureSource(
            "regression_path",
            basis="first_run",
            rule="fixation_stream",
            released_source="word_go_past_time",
        ),
        "go_past_in_first_pass": _stream(
            "regression_path_in_first_pass", basis="first_run"
        ),
        "total_time": _released("total_dwell", "word_total_fix_dur"),
    },
    # Dundee, as Shain et al. released it, with the release's own controls.
    "Dundee": {
        "first_fixation": _released("first_fixation", "fdurSP"),
        **_conditional("fdurFP"),
        "first_pass": _skip_flag("inregression"),
        **_conditional_go_past("fdurGP"),
        "total_time": _released("total_dwell", "fdurTT"),
        "wdelta": _released("control", "wdelta"),
        "prevwasfix": _released("control", "prevwasfix"),
        "startofsentence": _released("control", "startofsentence"),
        "endofsentence": _released("control", "endofsentence"),
        "startoffile": _released("control", "startoffile"),
        "endoffile": _released("control", "endoffile"),
    },
    # Corpora whose measures are computed from an event stream.
    "DundeeOfficial": {
        "first_fixation": _stream("first_fixation"),
        "first_run": _stream("first_run"),
        "first_pass": _stream("in_first_pass"),
        "go_past": _stream("regression_path"),
        "go_past_in_first_pass": _stream("regression_path_in_first_pass"),
        "total_time": _stream("total_dwell"),
    },
    "SBSAT": {
        "first_fixation": _stream("first_fixation"),
        "first_run": _stream("first_run"),
        "first_pass": _stream("in_first_pass"),
        "total_time": _stream("total_dwell"),
    },
    "BSC": {
        "first_fixation": _stream("first_fixation"),
        "first_run": _stream("first_run"),
        "first_pass": _stream("in_first_pass"),
        "total_time": _stream("total_dwell"),
    },
    # ZuCo's word measures, which carry no first-pass skip flag, and the
    # released band power of both views.
    "ZuCo1": {**_zuco_mat(), **_zuco_bands()},
    "ZuCo2": {**_zuco_mat(), **_zuco_bands()},
    "ZuCo1EEG": _zuco_bands(),
    "ZuCo2EEG": _zuco_bands(),
    "ZuCoN400": {
        "n400": _released("amplitude", "N400"),
        "p600": _released("amplitude", "P600"),
        "ant": _released("amplitude", "ANT"),
    },
    # Self-paced reading and maze.
    "NaturalStories": {"spr_rt": _released("reading_time", "RT")},
    "UCL_SPR": {"spr_rt": _released("reading_time", "RT")},
    "BrownSPR": {"spr_rt": _released("reading_time", "fdur")},
    "BrothersKuperberg": {
        "sum_3rt": _released("reading_time", "SUM_3RT"),
        "sum_3rt_trimmed": _released("reading_time", "SUM_3RT_trimmed"),
    },
    "NaturalStoriesMaze": {"maze_rt": _released("reading_time", "rt")},
    # Functional imaging.
    "Narratives": {"bold": _released("bold", "BOLD run surface files")},
}


def measure_sources(dataset: object) -> dict[str, MeasureSource]:
    """Resolve the nearest declared loader in the dataset's method resolution order.

    Args:
        dataset: Loader instance whose source declarations are requested.

    A subclass entry, including an empty mapping, replaces its inherited entry.
    """
    for cls in type(dataset).__mro__:
        if cls.__name__ in MEASURE_SOURCES:
            return MEASURE_SOURCES[cls.__name__]
    return {}


#: Loaders this table does not cover yet, each with the reason.
UNDECLARED: dict[str, str] = {
    "MichaelovN400": "the source column depends on the selected grain",
    "SzewczykN400": "the source column depends on the selected grain",
}
