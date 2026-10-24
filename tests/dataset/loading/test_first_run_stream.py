"""Unit tests for the first run and the first-pass flag from a fixation stream.

The first run on word w is the sum of all fixation durations starting at the
eye's first arrival at w and ending at the first departure from w to any other
word, whenever that arrival occurs. Refixations after the eye has left w
(regression-and-return) do not contribute. The first-pass flag marks the words
the eye reached before it had ever been further right.

These exercise the shared stream helpers on constructed fixation lists, so
they pin the algorithms without a corpus on disk.
"""

import numpy as np
import pytest

from replica.dataset.loading.fixation_utils import (
    RunIdInferrer,
    first_pass_flags,
    first_run_durations,
)


@pytest.mark.parametrize(
    "ia_ids,durations,ia_to_idx,n_words,expected",
    [
        # Reader fixates word 0, then word 1, then word 2, one fixation each.
        ([10, 20, 30], [150, 200, 180], {10: 0, 20: 1, 30: 2}, 3, [150, 200, 180]),
        # Two consecutive fixations on word 0 before moving to word 1.
        ([10, 10, 20], [150, 120, 180], {10: 0, 20: 1}, 2, [270, 180]),
        # Word 0 (250 ms), word 1 (200 ms), regression to word 0 (150 ms): the
        # first runs remain 250 ms and 200 ms.
        ([10, 20, 10], [250, 200, 150], {10: 0, 20: 1}, 2, [250, 200]),
        # The same regression, then forward to word 2 (180 ms). Word 2 is new,
        # so it opens a first run although the eye came from a closed word.
        (
            [10, 20, 10, 30],
            [250, 200, 150, 180],
            {10: 0, 20: 1, 30: 2},
            3,
            [250, 200, 180],
        ),
        # Two fixations on word 0 (sum 300), one on word 1 (180), a regression
        # to word 0 (100, ignored), then forward to word 2 (220).
        (
            [10, 10, 20, 10, 30],
            [150, 150, 180, 100, 220],
            {10: 0, 20: 1, 30: 2},
            3,
            [300, 180, 220],
        ),
    ],
    ids=[
        "single_fixation_per_word",
        "consecutive_fixations_sum",
        "regression_does_not_contribute",
        "regression_then_new_word",
        "refixations_then_regression",
    ],
)
def test_first_run_durations(ia_ids, durations, ia_to_idx, n_words, expected):
    g = first_run_durations(
        ia_ids=ia_ids, durations=durations, ia_to_idx=ia_to_idx, n_words=n_words
    )
    np.testing.assert_allclose(g, expected)


def test_skipped_word_has_nan_first_run():
    # Reader fixates word 0, jumps over word 1, lands on word 2.
    # Word 1 gets NaN, never fixated at all.
    g = first_run_durations(
        ia_ids=[10, 30],
        durations=[200, 220],
        ia_to_idx={10: 0, 20: 1, 30: 2},
        n_words=3,
    )
    assert g[0] == 200
    assert np.isnan(g[1])
    assert g[2] == 220


@pytest.mark.parametrize("off_text_ends_run,expected", [(True, 150), (False, 270)])
def test_off_text_run_boundary_is_explicit(off_text_ends_run, expected):
    g = first_run_durations(
        ia_ids=[10, 999, 10, 20],
        durations=[150, 80, 120, 200],
        ia_to_idx={10: 0, 20: 1},  # 999 missing on purpose
        n_words=2,
        off_text_ends_run=off_text_ends_run,
    )
    np.testing.assert_allclose(g, [expected, 200])


def test_first_pass_flags_exclude_regression_entries_and_skips():
    # Word 0, jump to word 2, regress to word 1, forward to word 3. Word 1 is
    # first entered after the eye had been on word 2, so it has no first pass;
    # word 4 was never fixated.
    flags = first_pass_flags([0, 2, 1, 3], 5)
    np.testing.assert_array_equal(flags, [1.0, 0.0, 1.0, 1.0, 0.0])


def test_first_pass_flags_ignore_off_text_fixations():
    flags = first_pass_flags([None, 0, None, 1], 2)
    np.testing.assert_array_equal(flags, [1.0, 1.0])


def test_inferred_runs_end_at_off_text_landings_without_crossing_trials():
    inferrer = RunIdInferrer()
    assert [inferrer.next("trial1", unit) for unit in [0, 0, None, 0, 1, 0]] == [
        1,
        1,
        None,
        2,
        1,
        3,
    ]
    assert inferrer.next("trial2", 0) == 1
    assert inferrer.next("trial1", 0) == 3
    np.testing.assert_equal(
        first_run_durations([0, None, 0], [100, 50, 80], {0: 0}, 1), [100]
    )


@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_first_run_duration_is_not_replaced_by_a_later_fixation(missing):
    np.testing.assert_equal(
        first_run_durations([1, 1, 0], [missing, 80, 100], {0: 0, 1: 1}, 2),
        [100, np.nan],
    )
