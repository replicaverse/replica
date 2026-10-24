"""Enforce the measure vocabulary across loaders, on the data itself.

See :data:`replica.dataset.measures.CANONICAL_MEASURES`. Two checks:

1. Every loaded measure name is in the vocabulary.
2. The first run is a part of total dwell: per word ``first_run`` restricted to
   first pass (plain ``first_run`` on corpora without the flag) does not exceed
   ``total_time``, and on aggregate it is strictly smaller (re-reading exists).
"""

import numpy as np
import pytest

from replica.core import CONDITIONS, DERIVED_DVS
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.measures import (
    CANONICAL_MEASURES,
    FIRST_PASS_RUN,
    PASSTHROUGH_MEASURES,
)

pytestmark = pytest.mark.needs_data

# Eye-tracking datasets that expose both first-pass and total dwell, so the
# first-pass-vs-total invariant is checkable (on-disk in this env).
_GAZE_AND_TOTAL = [
    "bsc",
    "copco",
    "emtec",
    "sbsat",
    "onestop",
    "provo",
    "dundee",
    "geco",
    "meco_en",
    "rsc",
    "rastros",
]
# Datasets to check name-conformance on (broader; both fixation + word-level).
_VOCAB_DATASETS = [
    *_GAZE_AND_TOTAL,
    "ucl_et",
    "potec",
    "celer",
    "devarda2023",
    "meco_char_en",
]
_MAX_PAIRS = 300


@pytest.mark.parametrize("name", _VOCAB_DATASETS)
def test_measures_are_canonical(name):
    """The vocabulary is the responses, the per-observation controls a loader
    has no other channel for, and the condition flags."""
    ds = REGISTRY[name].factory()
    vocabulary = (
        set(CANONICAL_MEASURES)
        | set(PASSTHROUGH_MEASURES)
        | set(CONDITIONS)
        | set(DERIVED_DVS)
    )
    extra = set(ds.measures) - vocabulary
    assert not extra, f"{name} exposes measure(s) outside the vocabulary: {extra}"


@pytest.mark.parametrize("name", _GAZE_AND_TOTAL)
def test_first_pass_run_is_not_total(name):
    """The first run must be <= total_time per word, and strictly smaller on
    aggregate (otherwise it is not distinct from total dwell). Corpora with a
    ``first_pass`` flag are checked on the first run restricted to first pass,
    corpora without one on the unconditional first run."""
    ds = REGISTRY[name].factory()
    measure = FIRST_PASS_RUN if "first_pass" in ds.measures else "first_run"
    total = {
        (d.participant_id, d.stimulus.stimulus_id): np.asarray(d.values, float)
        for d in ds.data(measure="total_time")
    }
    g_sum = t_sum = 0.0
    n_words = n_violations = n_pairs = 0
    n_gaze_cov = n_total_cov = 0  # words fixated under each measure
    for d in ds.data(measure=measure):
        if n_pairs >= _MAX_PAIRS:
            break
        tt = total.get((d.participant_id, d.stimulus.stimulus_id))
        if tt is None:
            continue
        g = np.asarray(d.values, float)
        if g.shape != tt.shape:
            continue
        n_pairs += 1
        n_gaze_cov += int(np.sum(np.isfinite(g) & (g > 0)))
        n_total_cov += int(np.sum(np.isfinite(tt) & (tt > 0)))
        m = np.isfinite(g) & np.isfinite(tt) & (g > 0) & (tt > 0)
        if not m.any():
            continue
        # the first run never exceeds total dwell (1ms float tolerance)
        n_violations += int(np.sum(g[m] > tt[m] + 1.0))
        n_words += int(m.sum())
        g_sum += float(g[m].sum())
        t_sum += float(tt[m].sum())

    assert n_words > 0, f"{name}: no comparable first_run/total words found"
    assert n_violations == 0, (
        f"{name}: first_run > total_time in {n_violations}/{n_words} words "
        ", first_run is not a part of total dwell"
    )
    # first run and total must be distinct columns: total dwell either covers
    # more words (re-read-only) or is larger on the overlap. (When they are the
    # identical column, coverage matches and the overlap ratio is 1.0.)
    ratio = g_sum / t_sum
    assert n_total_cov > n_gaze_cov or ratio < 0.999, (
        f"{name}: first_run and total_time are indistinguishable "
        f"(coverage {n_gaze_cov} vs {n_total_cov}, overlap ratio {ratio:.4f})"
    )
