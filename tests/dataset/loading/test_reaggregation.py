"""Native-unit round trips through the dataset reaggregation API."""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from replica.dataset import FixationCalculation, load_dataset
from replica.dataset.catalog.registry import REGISTRY
from replica.dataset.eye_tracking.meco.character import LANGUAGES
from replica.dataset.eye_tracking.meco.releases import L1W2_SITE_LANGUAGES
from replica.dataset.eye_tracking.meco.releases import LANGUAGES as WORD_LANGUAGES
from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer
from tests.dataset._fixation_test_base import (
    assert_fixation_round_trip,
    group_fixations,
)


@pytest.mark.needs_data
@pytest.mark.slow
@pytest.mark.parametrize(
    "key, measures",
    [
        *(
            (key, None)
            for key in (
                "potec",
                "emtec",
                "sbsat",
                "bsc",
                "provo",
                "onestop",
                "dundee_official",
            )
        ),
        *((f"meco_char_{language}", None) for language in LANGUAGES),
        ("ucl_et", None),
        (
            "dundee",
            [
                "first_fixation",
                "first_run",
                "first_run_in_first_pass",
                "first_pass",
                "go_past",
                "go_past_in_first_pass",
                "total_time",
            ],
        ),
        ("copco", None),
    ],
)
def test_native_reaggregation_matches_exported_measures(data_root, key, measures):
    path = Path(data_root) / REGISTRY[key].default_subdir
    if not path.is_dir():
        pytest.skip(f"Source unavailable: {path}")
    dataset = load_dataset(key, path=os.fspath(path))
    [layer] = {datum.layer for datum in dataset._data}
    assert dataset.default_measure in dataset.measures
    events = group_fixations(dataset.fixations())
    cache = {}
    for drop_skips in (True, False):
        dataset.drop_skips = drop_skips
        for measure in dataset.measures if measures is None else measures:
            assert_fixation_round_trip(
                dataset,
                source_measure=measure,
                unit_layer=layer,
                fixations_by_key=events,
                via_dataset=True,
                aggregation_cache=cache,
                tolerance=1e-6,
                min_compared=1000,
            )


@pytest.mark.needs_data
@pytest.mark.slow
@pytest.mark.parametrize(
    "key",
    [
        *(f"meco_{language}" for language in WORD_LANGUAGES),
        *(f"meco_l1_w2_{site}" for site in L1W2_SITE_LANGUAGES),
        "meco_l2_w1",
        "meco_l2_w2",
    ],
)
def test_meco_word_reaggregation_matches_supported_measures(data_root, key):
    path = Path(data_root) / REGISTRY[key].default_subdir
    if not path.is_dir() or not list(path.glob("fixations_*.parquet")):
        pytest.skip(f"Prepared MECO fixations unavailable: {path}")
    dataset = load_dataset(key, path=os.fspath(path))
    measures = [
        "first_fixation",
        "first_fixation_in_first_pass",
        "first_pass",
        "total_time",
        "go_past",
        "go_past_in_first_pass",
    ]
    measures.extend(
        [
            "first_run",
            "first_run_in_first_pass",
        ]
    )
    references = defaultdict(list)
    for measure in measures:
        for datum in dataset._native_responses(measure).values():
            if key == "meco_l2_w2":
                simplified_chinese = datum.metadata["site"] == "ch_s"
                if (
                    measure in {"go_past", "go_past_in_first_pass"}
                    and simplified_chinese
                ):
                    assert np.isnan(datum.values).all()
                    continue
            references[datum.participant_id, datum.stimulus.stimulus_id].append(datum)
    for measure in ("first_run", "first_run_in_first_pass"):
        for datum in dataset._native_responses(measure, source="released").values():
            references[datum.participant_id, datum.stimulus.stimulus_id].append(datum)
    compared = 0
    for pair, events in dataset.fixation_trials():
        cache = {}
        for reference in references.pop(pair, []):
            if (
                reference.measure_name == "first_run"
                and reference.metadata.get("response_source") == "released"
            ):
                selected = dataset._reaggregation_fixations(reference, events)
                algorithm = (
                    "standard@1"
                    if key == "meco_l2_w2" and selected[0].metadata["site"] == "ch_s"
                    else "popeye_episode@1"
                )
                arrays = aggregate_fixations_to_layer(
                    selected,
                    reference.stimulus,
                    reference.layer,
                    calculation=FixationCalculation(first_run=algorithm),
                )
                arrays["_released_first_run"] = arrays["first_run"]
                np.testing.assert_allclose(
                    dataset._reaggregation_values(reference, arrays),
                    reference.values,
                    rtol=0,
                    atol=1e-6,
                    equal_nan=True,
                )
            for drop_skips in (False, True):
                dataset.drop_skips = drop_skips
                datum = (
                    dataset._skips_as_missing(reference) if drop_skips else reference
                )
                actual = dataset.reaggregate(
                    datum,
                    datum.layer,
                    fixations=events,
                    _reference=reference,
                    _cache=cache,
                )
                np.testing.assert_allclose(
                    actual.values, datum.values, rtol=0, atol=1e-6, equal_nan=True
                )
                compared += len(datum.values)
    assert not references, "Some summary trials have no source events"
    assert compared > 100_000
