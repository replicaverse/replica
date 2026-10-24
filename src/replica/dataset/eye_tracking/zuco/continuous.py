"""Continuous ZuCo EEG and gaze with fixation-locked word onsets.

Each sentence has its EEG (``rawData``, 500 Hz, 105 channels) and, under
every fixated word, one EEG and one gaze fragment per fixation (``rawEEG``,
``rawET``) without their position in the trial. A fragment is an exact slice
of the sentence EEG, so matching samples recovers its onset; the last
fixation of a trial can run past the end of the EEG. A gaze fragment holds
the tracker clock, x, y and pupil size at the EEG sample rate.

The release assigns a fixation to a word when it lies inside the word's
rectangle and lasts at least 50 samples; ``fixPositions`` numbers these pairs
in trial order, and :func:`word_fixation_pairs` reproduces them.
"""

from dataclasses import dataclass

import numpy as np

from ....core import Modality, Recording, Stimulus

#: Both releases sample EEG and gaze at 500 Hz.
SAMPLE_RATE_HZ = 500
#: Shortest fixation the release assigns to a word, in samples.
MIN_FIXATION_SAMPLES = 50
#: Channels of the gaze recording: the gaze fragment's columns after its clock.
GAZE_CHANNELS = ("x", "y", "pupil")


class FragmentMismatch(ValueError):
    """A released fragment contradicts the sentence EEG or the fixation table."""


@dataclass(frozen=True)
class WordFragments:
    """Released fixation fragments of one word.

    Args:
        eeg: ``(samples, channels)`` EEG fragment per fixation.
        gaze: ``(samples, 4)`` gaze fragment per fixation: tracker clock in
            milliseconds, x, y, pupil size.
        positions: One-based ``fixPositions`` of each fixation.
    """

    eeg: tuple[np.ndarray, ...] = ()
    gaze: tuple[np.ndarray, ...] = ()
    positions: tuple[int, ...] = ()


def word_fixation_pairs(
    x: np.ndarray, y: np.ndarray, durations: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair fixations with the words the release assigns them to.

    Args:
        x: Fixation x positions, in trial order.
        y: Fixation y positions.
        durations: Fixation durations in samples.
        bounds: ``(n_words, 4)`` word rectangles ``x0, y0, x1, y1``.

    Returns:
        The ``(n_fixations, n_words)`` inclusion mask, then the fixation and
        word index of each pair in ``fixPositions`` order.
    """
    inside = (
        (x[:, None] >= bounds[:, 0])
        & (x[:, None] <= bounds[:, 2])
        & (y[:, None] >= bounds[:, 1])
        & (y[:, None] <= bounds[:, 3])
    )
    fixations, words = np.nonzero(inside & (durations[:, None] >= MIN_FIXATION_SAMPLES))
    return inside, fixations, words


def locate_fragment(recording: np.ndarray, fragment: np.ndarray) -> tuple[int, bool]:
    """Find the sample at which an exact fragment of a recording starts.

    A complete occurrence wins over a partial one at the end of the recording.

    Args:
        recording: ``(samples, channels)`` recording.
        fragment: ``(samples, channels)`` slice to locate.

    Returns:
        The first sample, and whether the fragment runs past the recording.

    Raises:
        FragmentMismatch: The fragment occurs nowhere or more than once.
    """
    candidates = np.flatnonzero(recording[:, 0] == fragment[0, 0])
    whole = [
        int(start)
        for start in candidates
        if start + len(fragment) <= len(recording)
        and np.array_equal(recording[start : start + len(fragment)], fragment)
    ]
    if len(whole) == 1:
        return whole[0], False
    if whole:
        raise FragmentMismatch(f"fragment occurs {len(whole)} times")
    tails = [
        int(start)
        for start in candidates
        if start + len(fragment) > len(recording)
        and np.array_equal(recording[start:], fragment[: len(recording) - start])
    ]
    if len(tails) == 1:
        return tails[0], True
    if not tails:
        raise FragmentMismatch("fragment occurs nowhere")
    raise FragmentMismatch(f"fragment ends the recording {len(tails)} ways")


def sentence_recordings(
    stimulus: Stimulus,
    participant_id: str,
    eeg: np.ndarray,
    words: list[WordFragments],
    fixations: dict[str, np.ndarray],
    bounds: np.ndarray,
    metadata: dict,
) -> tuple[Recording, Recording]:
    """Build the EEG and gaze recordings of one trial.

    Args:
        stimulus: Sentence whose ``words`` layer indexes ``words`` and ``bounds``.
        participant_id: Reader.
        eeg: ``(samples, channels)`` sentence EEG.
        words: Fragments of each word, empty for an unfixated word.
        fixations: ``x``, ``y`` and ``duration`` (samples) of every fixation.
        bounds: ``(n_words, 4)`` word rectangles.
        metadata: Metadata both recordings share.

    Returns:
        The EEG and gaze recordings on the trial clock. ``event_times`` holds
        each word's first-fixation onset; ``metadata["fixations"]`` lists the
        located fixations in onset order (one-based fixation index, word, onset
        sample, fragment length, truncated flag). Gaze is NaN outside fixations.

    Raises:
        FragmentMismatch: A fragment contradicts the fixation table or the EEG.
        ValueError: ``words`` or ``bounds`` do not match the stimulus.
    """
    n_words = len(stimulus.layers["words"])
    if len(words) != n_words or bounds.shape != (n_words, 4):
        raise ValueError("one fragment set and one rectangle per word are required")
    _, pair_fixation, pair_word = word_fixation_pairs(
        fixations["x"], fixations["y"], fixations["duration"], bounds
    )
    located = []
    for unit, fragments in enumerate(words):
        if len(fragments.eeg) != len(fragments.gaze) or len(fragments.eeg) != len(
            fragments.positions
        ):
            raise FragmentMismatch(
                f"word {unit}: fragments and positions differ in number"
            )
        for fragment, gaze, position in zip(
            fragments.eeg, fragments.gaze, fragments.positions, strict=True
        ):
            if (
                not 1 <= position <= len(pair_fixation)
                or pair_word[position - 1] != unit
            ):
                raise FragmentMismatch(
                    f"word {unit}: fixation position {position} is not assigned to it"
                )
            fixation = int(pair_fixation[position - 1])
            # A fragment spans its fixation's samples plus the end sample.
            samples = int(fixations["duration"][fixation]) + 1
            if fragment.shape != (samples, eeg.shape[1]):
                raise FragmentMismatch(
                    f"word {unit}: the EEG fragment of fixation {fixation + 1} has "
                    f"shape {fragment.shape}, not ({samples}, {eeg.shape[1]})"
                )
            if gaze.shape != (samples, 1 + len(GAZE_CHANNELS)):
                raise FragmentMismatch(
                    f"word {unit}: the gaze fragment of fixation {fixation + 1} has "
                    f"shape {gaze.shape}, not ({samples}, {1 + len(GAZE_CHANNELS)})"
                )
            try:
                onset, truncated = locate_fragment(eeg, fragment)
            except FragmentMismatch as exc:
                raise FragmentMismatch(
                    f"word {unit}, fixation {fixation + 1}: {exc}"
                ) from exc
            located.append((onset, unit, fixation, samples, truncated, gaze))
    located.sort(key=lambda entry: entry[:2])
    event_times = np.full(n_words, np.nan)
    gaze_values = np.full((len(eeg), len(GAZE_CHANNELS)), np.nan)
    for onset, unit, _fixation, samples, _truncated, gaze in located:
        if np.isnan(event_times[unit]):
            event_times[unit] = onset / SAMPLE_RATE_HZ
        stop = min(onset + samples, len(eeg))
        gaze_values[onset:stop] = gaze[: stop - onset, 1:]

    def table() -> dict[str, list]:
        return {
            "fixation_index": [fixation + 1 for _, _, fixation, _, _, _ in located],
            "unit_index": [unit for _, unit, _, _, _, _ in located],
            "onset_sample": [onset for onset, _, _, _, _, _ in located],
            "samples": [samples for _, _, _, samples, _, _ in located],
            "truncated": [truncated for _, _, _, _, truncated, _ in located],
        }

    shared = {
        "stimulus": stimulus,
        "participant_id": participant_id,
        "sample_interval": 1 / SAMPLE_RATE_HZ,
        "dimension_names": ("time", "channel"),
    }
    return (
        Recording(
            values=eeg,
            measure_name="eeg",
            modality=Modality.EEG,
            event_times=event_times,
            metadata={**metadata, "fixations": table()},
            **shared,
        ),
        Recording(
            values=gaze_values,
            measure_name="gaze",
            modality=Modality.EYE_TRACKING,
            event_times=event_times.copy(),
            dimension_labels={"channel": GAZE_CHANNELS},
            metadata={**metadata, "fixations": table()},
            **shared,
        ),
    )
