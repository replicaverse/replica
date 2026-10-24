"""Real-data tests for the Dundee loader.

Tests the publicly reachable Shain et al. (2024) preprocessed Dundee release
(``dundee_X.csv``).

Fetch via ``replica fetch dundee``.
"""

from pathlib import Path

import numpy as np
import pytest

from replica.dataset import Dundee
from replica.dataset.measures import FIRST_PASS_RUN, with_derived


@pytest.fixture(scope="module")
def dundee_reference(data_root):
    path = Path(data_root) / "eye-tracking/processed/dundee_shain2024logarithmic"
    original = Path(data_root) / "eye-tracking/original/dundee/english"
    if not (path / "dundee_X.csv").exists() or not original.is_dir():
        pytest.skip(
            "Dundee processed release and original reference files are required"
        )
    dataset = Dundee()
    dataset.load(str(path))
    return dataset, original, path


@pytest.mark.slow
@pytest.mark.needs_data
class TestDundeeRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, dundee_reference):
        self.ds, self.original, self.processed = dundee_reference

    def test_participant_count(self):
        """Dundee ships data for 10 participants."""
        assert len(self.ds.participants) == 10

    def test_article_level_stimuli(self):
        """Kennedy et al. (2003) released 20 newspaper editorials.
        Each editorial is one Stimulus, the LM context for any word
        must be the cumulative document so far, not just the current
        sentence."""
        stims = self.ds.stimuli()
        assert len(stims) == 20, (
            f"Expected 20 editorial-level stimuli, got {len(stims)}"
        )
        # Articles are multi-thousand-word; ensure none collapsed to a
        # sentence by mistake.
        for s in stims:
            assert len(s.layers["words"]) > 100, (
                f"{s.stimulus_id}: only {len(s.layers['words'])} words, "
                "expected article-level grouping."
            )
            assert "sentence_boundaries" in s.metadata
            assert len(s.metadata["sentence_boundaries"]) > 1

    def test_measures(self):
        """Four canonical eye-tracking durations, the ``first_pass`` flag read
        from each region's first event, and the release's analysis controls, exposed as
        measures so consumers can discover them (the GECO pattern)."""
        assert sorted(self.ds.measures) == sorted(
            with_derived(
                [
                    "first_run",
                    "first_pass",
                    "first_fixation",
                    "go_past",
                    "go_past_in_first_pass",
                    "total_time",
                    "wdelta",
                    "prevwasfix",
                    "startofsentence",
                    "endofsentence",
                    "startoffile",
                    "endoffile",
                ]
            )
        )

    def test_default_measure(self):
        """The first run restricted to first pass is the canonical Dundee DV."""
        assert self.ds.default_measure == FIRST_PASS_RUN

    def test_first_run_is_nonneg(self):
        """Reading-time durations are non-negative."""
        for d in list(self.ds.data(measure="first_run"))[:200]:
            non_nan = d.values[~np.isnan(d.values)]
            assert (non_nan >= 0).all(), (
                f"negative first_run in {d.participant_id}/"
                f"{d.stimulus.stimulus_id}: {non_nan[non_nan < 0]}"
            )

    def test_first_pass_flag_is_binary(self):
        """The flag holds 1, 0 or NaN and is never masked as a skip."""
        flags = [d for d in self.ds.data() if d.measure_name == "first_pass"]
        assert flags
        for d in flags[:200]:
            assert set(np.unique(d.values[~np.isnan(d.values)])) <= {0.0, 1.0}

    def test_complete_display_text_and_measured_regions(self):
        total_words = total_regions = 0
        for stimulus in self.ds.stimuli():
            docid = stimulus.metadata["docid"]
            lines = (
                (self.original / f"items/text{docid:02d}.itm")
                .read_text("cp437")
                .splitlines()
            )
            displayed = " ".join(
                line[1:].rstrip().removesuffix(">")
                for line in lines
                if line.startswith("+")
            ).split()
            assert stimulus.text == " ".join(displayed)
            assert stimulus.unit_strings("source_words") == displayed
            groups = stimulus.metadata["source_word_groups"]
            assert [number for group in groups for number in group] == list(
                range(1, len(displayed) + 1)
            )
            for region, group in zip(stimulus.unit_strings(), groups, strict=True):
                assert region == " ".join(displayed[number - 1] for number in group)
            total_words += len(displayed)
            total_regions += len(groups)
        assert total_words == 51_502
        assert total_regions == 51_501

    def test_all_events_against_original_records_and_screen_geometry(self):
        from collections import defaultdict

        published = defaultdict(list)
        for fixation in self.ds.fixations():
            published[(fixation.stimulus_id, fixation.participant_id)].append(fixation)
        matched = unaligned = 0
        for stimulus in self.ds.stimuli():
            docid = stimulus.metadata["docid"]
            geometry = {}
            for line in (
                (self.original / f"items/tx{docid:02d}wrdp.dat")
                .read_text("cp437")
                .splitlines()
            ):
                fields = line.split()
                geometry[int(fields[12])] = (int(fields[2]), int(fields[3]))
            region_indices = {
                n: i
                for i, group in enumerate(stimulus.metadata["source_word_groups"])
                for n in group
            }
            for participant in self.ds.participants:
                reference = []
                elapsed = 0
                for line in (
                    (self.original / f"{participant}{docid:02d}ma1p.dat")
                    .read_text("cp437")
                    .splitlines()[1:]
                ):
                    fields = line.split()
                    if len(fields) < 8:
                        continue
                    number, duration = int(fields[6]), int(fields[7])
                    if number > 0 and not fields[0].startswith("*"):
                        reference.append(
                            (
                                number,
                                duration,
                                elapsed,
                                (int(fields[1]), int(fields[2])),
                            )
                        )
                    elapsed += duration
                events = published[(stimulus.stimulus_id, participant)]
                assert len(events) == len(reference)
                for event, (number, duration, onset, screen_line) in zip(
                    events, reference, strict=True
                ):
                    assert event.start_time == pytest.approx(onset, abs=1e-5)
                    assert event.duration == duration
                    assert event.metadata["source_word_number"] == number
                    expected_index = (
                        region_indices[number]
                        if screen_line == geometry[number]
                        else None
                    )
                    assert event.unit_index == expected_index
                    if expected_index is None:
                        assert event.char_offset is None
                        unaligned += 1
                    else:
                        assert (
                            event.char_offset
                            == stimulus.layers["source_words"].spans[number - 1].start
                        )
                    matched += 1
        assert matched == 408_439
        assert unaligned == 1_083

    def test_article18_source_discrepancies_do_not_misassign_measurements(self):
        def datum(measure, participant):
            return next(
                d
                for d in self.ds.data(participant=participant)
                if d.measure_name == measure and d.stimulus.metadata["docid"] == 18
            )

        # Original quotation objects 1169 and 1170 are one released region.
        first_run = datum("first_run", "sd")
        assert first_run.stimulus.metadata["source_word_groups"][1168] == [1169, 1170]
        assert first_run.values[1168] == 855
        assert datum("first_pass", "sd").values[1168] == 1
        # The published total for 'this' includes stale-screen events.
        total = datum("total_time", "sb")
        assert np.isnan(total.values[607])
        preserved = {
            v["unit_index"]: v["released_value"]
            for v in total.metadata["unavailable_source_values"]
        }
        assert preserved[607] == 7950
        assert datum("first_fixation", "sb").values[607] == 204
        assert np.isnan(datum("first_fixation", "sc").values[607])
        # A reliable 'case' reading remains on its own region after the merge.
        assert datum("first_run", "sc").values[1221] == 407
        assert datum("first_pass", "sc").values[1221] == 1

    def test_known_skips_include_words_absent_from_every_reader(self):
        # Original word 73 in article 1 has no event in any reader's MA1 file.
        for measure in ("first_fixation", "first_run", "go_past", "total_time"):
            stored = [
                d
                for d in self.ds._data
                if d.measure_name == measure and d.stimulus.metadata["docid"] == 1
            ]
            assert len(stored) == 10
            assert all(d.values[72] == 0 for d in stored)
            assert all(
                np.isnan(d.values[72])
                for d in self.ds.data(measure)
                if d.stimulus.metadata["docid"] == 1
            )

        # The flag reads 0 on the skip in both modes.
        def first_pass_flags():
            return [
                d
                for d in self.ds.data()
                if d.measure_name == "first_pass" and d.stimulus.metadata["docid"] == 1
            ]

        assert len(first_pass_flags()) == 10
        assert all(d.values[72] == 0 for d in first_pass_flags())
        try:
            self.ds.drop_skips = False
            assert all(
                d.values[72] == 0
                for d in self.ds.data("first_run")
                if d.stimulus.metadata["docid"] == 1
            )
            assert all(d.values[72] == 0 for d in first_pass_flags())
            # The invalid 'this' fixation is unavailable in include mode too.
            d = next(
                d
                for d in self.ds.data("first_fixation", "sc")
                if d.stimulus.metadata["docid"] == 18
            )
            assert np.isnan(d.values[607])
        finally:
            self.ds.drop_skips = True

    def test_released_values_are_preserved_or_explicitly_unavailable(self):
        import pandas as pd

        from replica.dataset.eye_tracking.dundee.shain2024 import (
            FIRST_PASS_COLUMN,
            MEASURE_COLUMNS,
        )

        measure_columns = {**MEASURE_COLUMNS, FIRST_PASS_COLUMN: "first_run"}
        columns = ["subject", "docid", "sentid", "sentpos", "time", *measure_columns]
        frame = pd.read_csv(
            self.processed / "dundee_X.csv", sep=r"\s+", usecols=columns
        )
        events = {}
        for fixation in self.ds.fixations():
            events.setdefault(
                (fixation.stimulus_id, fixation.participant_id), []
            ).append(fixation)
        datums = {
            (d.stimulus.metadata["docid"], d.participant_id, d.measure_name): d
            for d in self.ds._data
        }
        checked = preserved = 0
        for (docid, subject), trial in frame.groupby(["docid", "subject"]):
            trial = trial.sort_values("time").copy()
            trial["region"] = [
                e.metadata["released_unit_index"]
                for e in events[(f"dundee_doc{docid}", subject)]
            ]
            for column, measure in measure_columns.items():
                datum = datums[(docid, subject, measure)]
                unavailable = {
                    v["unit_index"]: v["released_value"]
                    for v in datum.metadata.get("unavailable_source_values", [])
                }
                source = trial.dropna(subset=[column]).drop_duplicates("region")
                for region, value in source[["region", column]].itertuples(index=False):
                    if region in unavailable:
                        assert np.isnan(datum.values[region])
                        assert unavailable[region] == value
                        preserved += 1
                    else:
                        assert datum.values[region] == value
                    checked += 1
        assert checked > 2_000_000
        assert preserved > 0


def test_refixated_word_keeps_its_first_pass_row(tmp_path):
    """The release has one row per fixation.

    A re-read word has one row per fixation in ``time`` order; only the first
    carries the word-level aggregates (fdurFP/GP/TT), while the regression
    rows carry that fixation's own fdurSP and NA elsewhere. Taking the last
    row instead would erase the first-pass durations of re-read words and
    report a regression fixation as the first fixation.
    """
    columns = (
        "word docid sentid sentpos subject time fdurSP fdurFP fdurGP fdurTT wdelta"
    )
    rows = [
        # first fixation, carries the aggregates
        "alpha 1 0 1 sa 0.10 200 200 200 350 1",
        # regression back to the same word: only its own duration
        "alpha 1 0 1 sa 2.50 170 NA NA NA -1",
        # a word read once
        "beta 1 0 2 sa 0.30 180 180 180 180 1",
    ]
    directory = tmp_path / "eye-tracking" / "processed" / "dundee_shain2024logarithmic"
    directory.mkdir(parents=True)
    (directory / "dundee_X.csv").write_text("\n".join([columns, *rows]) + "\n")
    items = directory / "items"
    items.mkdir()
    (items / "tx01wrdp.dat").write_text(
        "alpha 1 1 1 0 0 0 0 0 0 0 0 1 0\nbeta 1 1 1 0 0 0 0 0 0 0 0 2 0\n"
    )
    (items / "text01.itm").write_text("+alpha beta >\n")
    (directory / "sa01ma1p.dat").write_text(
        "*Off-screen -99 -99 0 0 0 -99 100 0 0 0 0\n"
        "alpha 1 1 0 0 0 1 200 0 0 0 0\n"
        "beta 1 1 0 0 0 2 180 0 0 0 0\n"
        "*Off-screen -99 -99 0 0 0 -99 2020 0 0 0 0\n"
        "alpha 1 1 0 0 0 1 170 0 0 0 0\n"
    )

    ds = Dundee()
    ds.load(str(directory))

    def values(measure):
        datum = next(d for d in ds.data(participant="sa") if d.measure_name == measure)
        return np.asarray(datum.values, dtype=float)

    # The regression row must not erase the first-pass aggregates.
    assert values("first_run")[0] == 200.0
    assert values("first_pass")[0] == 1.0
    assert values("go_past")[0] == 200.0
    assert values("total_time")[0] == 350.0
    # ... nor be mistaken for the first fixation.
    assert values("first_fixation")[0] == 200.0
    assert values("first_run")[1] == 180.0
    assert values("first_pass")[1] == 1.0
