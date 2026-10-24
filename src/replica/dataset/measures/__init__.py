"""Shared response vocabulary and released reading-measure column adapters."""

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum

from replica.core import DERIVED_DVS
from replica.dataset.loading.utils import (
    conditional_first_run,
    first_pass_flag,
    parse_reading_measure,
)


class ResponseFamily(StrEnum):
    """Response categories for durations, ERP amplitudes, EEG power, and BOLD."""

    EYE_TRACKING = "eye-tracking duration"
    SELF_PACED_READING = "self-paced reading time"
    MAZE = "maze response time"
    ERP = "ERP amplitude"
    BAND_POWER = "EEG band power"
    BOLD = "BOLD response"


@dataclass(frozen=True)
class MeasureSpec:
    """Description, response family, and unit of a measurement."""

    description: str
    family: ResponseFamily
    unit: str = ""


def _erp(component: str, window: str, character: str) -> MeasureSpec:
    """Describe an ERP component using Frank et al. (2015)'s measurement window.

    Electrode selections are component-specific and defined by the release.
    """
    return MeasureSpec(
        f"Mean amplitude from {window} after unit onset, {character} ({component}).",
        ResponseFamily.ERP,
        "µV",
    )


def _erp_baseline(component: str) -> MeasureSpec:
    """Describe the pre-onset baseline a release pairs with an ERP component."""
    return MeasureSpec(
        f"Baseline amplitude the release pairs with the {component}, used as a control.",
        ResponseFamily.ERP,
        "µV",
    )


def _band(band: str, span: str) -> MeasureSpec:
    """Describe EEG band power measured over a unit's total reading time.

    Bands and their edges are those of ZuCo 2.0 (Hollenstein et al., 2020),
    whose per-unit features are computed over the total reading time.
    """
    return MeasureSpec(
        f"Mean power in the {band} band, {span}, over the total time a unit "
        "was read, averaged across the scalp electrodes.",
        ResponseFamily.BAND_POWER,
    )


#: Every response measure the loaders expose, with the quantity it holds. A
#: name outside this table is a corpus-specific extra, defined by the corpus
#: that carries it.
CANONICAL_MEASURES: dict[str, MeasureSpec] = {
    # Eye-tracking durations. A unit is whatever the loader aggregates
    # fixations to, so the same name holds for a word, a character or a span.
    "first_fixation": MeasureSpec(
        "Duration of the first fixation on a unit.",
        ResponseFamily.EYE_TRACKING,
        "ms",
    ),
    "first_run": MeasureSpec(
        "Total fixation time in the first run of fixations on a unit, from the "
        "eye's first arrival until it leaves, whenever that arrival occurs. "
        "Restricted to first pass (first_run_in_first_pass) it is the classical "
        "gaze duration, zero on a unit first entered by a regression.",
        ResponseFamily.EYE_TRACKING,
        "ms",
    ),
    "go_past": MeasureSpec(
        "Total fixation time from the first fixation on a unit up to the first "
        "fixation on any unit to its right, so time spent back on earlier "
        "units counts, whenever the first arrival occurs. Restricting this "
        "window to first pass gives go_past_in_first_pass.",
        ResponseFamily.EYE_TRACKING,
        "ms",
    ),
    "right_bounded": MeasureSpec(
        "Total fixation time on a unit before the first fixation on any unit "
        "to its right, so a return to the unit counts and time on earlier "
        "units does not.",
        ResponseFamily.EYE_TRACKING,
        "ms",
    ),
    "total_time": MeasureSpec(
        "Total fixation time on a unit over the whole trial, re-readings included.",
        ResponseFamily.EYE_TRACKING,
        "ms",
    ),
    "total_fixation_duration": MeasureSpec(
        "Total fixation time on a unit over the whole trial, as the release "
        "reports it.",
        ResponseFamily.EYE_TRACKING,
        "ms",
    ),
    # Self-paced reading. sum_3rt is defined over the three words its release
    # fixes, so it names words rather than units.
    "spr_rt": MeasureSpec(
        "Time from the button press that revealed a unit to the press that "
        "moved past it.",
        ResponseFamily.SELF_PACED_READING,
        "ms",
    ),
    "sum_3rt": MeasureSpec(
        "Self-paced reading time summed over the three-word critical region, "
        "as released in SUM_3RT (at most 4500 ms).",
        ResponseFamily.SELF_PACED_READING,
        "ms",
    ),
    "sum_3rt_trimmed": MeasureSpec(
        "The sum_3rt value with the paper's cutoffs applied: one above 2.5 "
        "standard deviations over the participant's condition mean, or below "
        "300 ms, is replaced by that cutoff.",
        ResponseFamily.SELF_PACED_READING,
        "ms",
    ),
    # Maze.
    "maze_rt": MeasureSpec(
        "Time to choose the unit that continues the sentence over a "
        "distractor, in a two-alternative forced choice at every unit.",
        ResponseFamily.MAZE,
        "ms",
    ),
    # ERP component amplitudes.
    "elan": _erp("ELAN", "125 to 175 ms", "an early left anterior negativity"),
    "elan_baseline": _erp_baseline("ELAN"),
    "lan": _erp("LAN", "300 to 400 ms", "a left anterior negativity"),
    "lan_baseline": _erp_baseline("LAN"),
    "n400": _erp("N400", "300 to 500 ms", "a centro-parietal negativity"),
    "n400_baseline": _erp_baseline("N400"),
    "epnp": _erp("EPNP", "400 to 600 ms", "an early post-N400 positivity"),
    "epnp_baseline": _erp_baseline("EPNP"),
    "p600": MeasureSpec(
        "P600 late-positive amplitude: 500 to 700 ms in Frank et al.'s release; "
        "600 to 700 ms over posterior channels in the ZuCo ERP release.",
        ResponseFamily.ERP,
        "µV",
    ),
    "p600_baseline": _erp_baseline("P600"),
    "pnp": _erp("PNP", "600 to 700 ms", "a post-N400 positivity"),
    "pnp_baseline": _erp_baseline("PNP"),
    "ant": MeasureSpec(
        "Mean anterior amplitude from 200 to 400 ms after word onset.",
        ResponseFamily.ERP,
        "µV",
    ),
    # EEG band power, as ZuCo 2.0 releases it.
    "eeg_theta_1": _band("theta_1", "4 to 6 Hz"),
    "eeg_theta_2": _band("theta_2", "6.5 to 8 Hz"),
    "eeg_alpha_1": _band("alpha_1", "8.5 to 10 Hz"),
    "eeg_alpha_2": _band("alpha_2", "10.5 to 13 Hz"),
    "eeg_beta_1": _band("beta_1", "13.5 to 18 Hz"),
    "eeg_beta_2": _band("beta_2", "18.5 to 30 Hz"),
    "eeg_gamma_1": _band("gamma_1", "30.5 to 40 Hz"),
    "eeg_gamma_2": _band("gamma_2", "40 to 49.5 Hz"),
    # fMRI.
    "bold": MeasureSpec(
        "Surface-projected BOLD time series, one series per vertex.",
        ResponseFamily.BOLD,
    ),
}

#: Observation-level covariates exposed through the measure API.
#: These can vary by participant and are not response variables.
PASSTHROUGH_MEASURES: dict[str, str] = {
    "wdelta": "saccade length in word positions, against the previous fixated word",
    "prevwasfix": "whether the previous word was the source of the saccade",
    "startofsentence": "first word of a sentence",
    "endofsentence": "last word of a sentence",
    "startoffile": "first word of a file",
    "endoffile": "last word of a file",
}

#: The two quantities the field calls gaze duration, as dependent variables.
FIRST_RUN = "first_run"
FIRST_PASS_RUN = "first_run_in_first_pass"

#: Regression-path duration restricted to a first-pass entry.
FIRST_PASS_GO_PAST = "go_past_in_first_pass"

#: Release-column aliases for canonical measures and derived variables.
#: Keys preserve source capitalization. First-pass columns map to the
#: corresponding conditional variable.
MEASURE_SPELLINGS: dict[str, str] = {
    # EyeLink interest-area reports.
    "IA_FIRST_FIXATION_DURATION": "first_fixation",
    "IA_FIRST_RUN_DWELL_TIME": FIRST_RUN,
    "IA_REGRESSION_PATH_DURATION": "go_past",
    "IA_DWELL_TIME": "total_time",
    # pymovements reading measures (Krakowczyk et al., 2023), in both the
    # abbreviated and the spelled-out form.
    "FRT": FIRST_RUN,
    "FPRT": FIRST_PASS_RUN,
    "RPD_inc": FIRST_PASS_GO_PAST,
    "TFT": "total_time",
    "first_reading_time": FIRST_RUN,
    "first_pass_reading_time": FIRST_PASS_RUN,
    "regression_path_duration": FIRST_PASS_GO_PAST,
    "total_reading_time": "total_time",
    "firstfix_rt": "first_fixation",
    "total_rt": "total_time",
    # ModelBlocks per-word durations (Clark et al., 2025; Shain et al.).
    "fdurSP": "first_fixation",
    "fdurFP": FIRST_PASS_RUN,
    "fdurGP": FIRST_PASS_GO_PAST,
    "fdurTT": "total_time",
    # de Varda et al. (2023) release, as Giulianelli et al. (2024a) reads it.
    "RTfirstfix": "first_fixation",
    "RTfirstpass": FIRST_PASS_RUN,
    "RTgopast": FIRST_PASS_GO_PAST,
    "firstrun.gopast": FIRST_PASS_GO_PAST,
    "RTrightbound": "right_bounded",
    "self_paced_reading_time": "spr_rt",
    "ELAN": "elan",
    "LAN": "lan",
    "N400": "n400",
    "EPNP": "epnp",
    "P600": "p600",
    "PNP": "pnp",
    # Giulianelli et al. (2024b): UCL's RTfirstpass becomes FirstPassDuration,
    # and CELER's IA_FIRST_RUN_DWELL_TIME becomes FirstPassDwellTime.
    "FirstFixationDuration": "first_fixation",
    "FirstPassDuration": FIRST_PASS_RUN,
    "FirstPassDwellTime": FIRST_RUN,
    "RightBoundedDuration": "right_bounded",
    "DwellTime": "total_time",
}


def with_derived(measures: Collection[str]) -> set[str]:
    """Include derived variables whose measure and condition are available.

    Args:
        measures: Available measure and condition names.
    """
    names = set(measures)
    return names | {
        name
        for name, (measure, condition) in DERIVED_DVS.items()
        if measure in names and condition in names
    }


def canonical_measure(name: str) -> str:
    """Resolve a measure or release-column alias to its canonical name.

    Canonical names are unchanged. Match the supplied spelling first, then
    its lowercase form. Study-specific gaze column names require an explicit
    response choice; tabular source readers retain their original columns.

    Args:
        name: A measure name from a loader, a registry cell, a results table
            or a release column.

    Raises:
        KeyError: The name is unknown or requires source context.
    """
    if name.lower() in {"gaze_rt", "gazeduration", "firstrun.dur"}:
        raise KeyError(
            f"{name!r} requires source context; select 'first_run', "
            "'first_run_in_first_pass', and the response source or calculation. "
            "Tabular source columns remain available by name."
        )
    for candidate in (name, name.lower()):
        if candidate in MEASURE_SPELLINGS:
            return MEASURE_SPELLINGS[candidate]
        if candidate in CANONICAL_MEASURES or candidate in DERIVED_DVS:
            return candidate
    raise KeyError(f"{name!r} is not a known measure or measure spelling")


def canonical_measures(names: Iterable[str]) -> set[str]:
    """The canonical variables among a loader's declared names.

    A loader's ``measures`` also carry passthrough columns, condition flags and
    corpus-specific extras, which are not responses; those are left out rather
    than refused.

    Args:
        names: Measure names as a loader or catalog entry declares them.
    """
    out: set[str] = set()
    for name in names:
        try:
            out.add(canonical_measure(name))
        except KeyError:
            continue
    return out


#: Eye-tracking duration names, used consistently across loaders.
CANONICAL_RT_MEASURES: frozenset[str] = frozenset(
    name
    for name, spec in CANONICAL_MEASURES.items()
    if spec.family is ResponseFamily.EYE_TRACKING
) | frozenset(
    name
    for name, (measure, _) in DERIVED_DVS.items()
    if CANONICAL_MEASURES[measure].family is ResponseFamily.EYE_TRACKING
)

#: Shared mapping from EyeLink interest-area columns to canonical measures.
#: ``IA_FIRST_RUN_DWELL_TIME`` is the first run whenever it occurs; the
#: ``first_pass`` flag comes from :data:`EYELINK_SKIP_COLUMN`.
EYELINK_IA_MEASURES: dict[str, str] = {
    "IA_FIRST_FIXATION_DURATION": "first_fixation",
    "IA_FIRST_RUN_DWELL_TIME": "first_run",
    "IA_DWELL_TIME": "total_time",
    "IA_REGRESSION_PATH_DURATION": "go_past",
}

#: EyeLink's first-pass skip flag, 1 when the unit was not fixated before the
#: eye had moved past it.
EYELINK_SKIP_COLUMN = "IA_SKIP"

#: Measures and the flag an EyeLink IA report yields, in the order
#: :func:`eyelink_measures` returns them.
EYELINK_MEASURE_NAMES: tuple[str, ...] = (
    *EYELINK_IA_MEASURES.values(),
    "first_pass",
)


def eyelink_measures(
    lookup: Callable[[str], object], *, unfixated: bool
) -> tuple[float, ...]:
    """Parse one EyeLink interest-area row into :data:`EYELINK_MEASURE_NAMES` order.

    Args:
        lookup: Returns the row's raw cell for a column name, or a missing marker.
        unfixated: The row establishes that the area received no fixation, so
            missing durations read as zero and a missing flag as not reached.
    """
    values = [
        parse_reading_measure(lookup(column), unfixated=unfixated)
        for column in EYELINK_IA_MEASURES
    ]
    values.append(first_pass_flag(lookup(EYELINK_SKIP_COLUMN), unfixated=unfixated))
    return tuple(values)


#: ZuCo MATLAB per-word field -> canonical measure name, shared by the
#: ZuCo 1.0 and ZuCo 2.0 loaders. ``GD`` is populated for every fixated word,
#: so it is the first run whenever it occurs; ZuCo ships no first-pass flag.
ZUCO_MAT_MEASURES: dict[str, str] = {
    "FFD": "first_fixation",
    "GD": "first_run",
    "GPT": "go_past",
    "TRT": "total_time",
}

#: pymovements reading-measures column -> canonical measure name, shared by
#: the EMTeC and PoTeC loaders. FD is arrival-based; FFD/FPRT are
#: first-pass conditional. FRT is not used: the releases contain zeros
#: contradicted by positive FD, so first_run remains partial from FPRT.
PYMOVEMENTS_RM_MEASURES: dict[str, str] = {
    "FD": "first_fixation",
    "FFD": "first_fixation_in_first_pass",
    "FPRT": FIRST_PASS_RUN,
    "TFT": "total_time",
    "RPD_inc": FIRST_PASS_GO_PAST,
}

#: pymovements' first-pass reading time, the first run of a unit reached in
#: first pass and zero otherwise.
PYMOVEMENTS_FIRST_PASS_COLUMN = "FPRT"

#: Measures and the flag a pymovements table yields, in the order
#: :func:`pymovements_measures` returns them.
PYMOVEMENTS_MEASURE_NAMES: tuple[str, ...] = (
    *PYMOVEMENTS_RM_MEASURES.values(),
    "first_run",
    "first_pass",
)


def pymovements_measures(
    lookup: Callable[[str], object], *, unfixated: bool
) -> tuple[float, ...]:
    """Parse one pymovements row into :data:`PYMOVEMENTS_MEASURE_NAMES` order.

    Args:
        lookup: Returns the row's raw cell for a column name, or a missing marker.
        unfixated: The row establishes that the unit received no fixation, so
            missing durations read as zero.
    """
    values = [
        parse_reading_measure(lookup(column), unfixated=unfixated)
        for column in PYMOVEMENTS_RM_MEASURES
    ]
    values.extend(
        conditional_first_run(
            lookup(PYMOVEMENTS_FIRST_PASS_COLUMN), unfixated=unfixated
        )
    )
    return tuple(values)
