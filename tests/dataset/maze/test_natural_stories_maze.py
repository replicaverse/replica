"""Real-data tests for the Natural Stories Maze loader.

Run with: pytest tests/dataset/maze/test_natural_stories_maze.py -v --data-root <path>
"""

import os

import numpy as np
import pytest

from replica.dataset import NaturalStoriesMaze

pytestmark = pytest.mark.needs_data


class TestNaturalStoriesMazeRealData:
    @pytest.fixture(autouse=True)
    def load_dataset(self, data_root):
        path = os.path.join(data_root, "maze", "original", "natural_stories_maze")
        if not os.path.isdir(path):
            pytest.skip(f"Natural Stories Maze data not found at {path}")
        self.ds = NaturalStoriesMaze()
        try:
            self.ds.load(path)
        except FileNotFoundError as e:
            pytest.skip(f"Natural Stories Maze data not loadable: {e}")

    def test_stimulus_count(self):
        """One stimulus per Natural Stories sentence."""
        assert len(self.ds.stimuli()) == 474

    def test_participant_count(self):
        """The study's analysed sample, its exclusions reapplied here."""
        assert len(self.ds.participants) == 63

    def test_full_sample_is_everyone_recruited(self, data_root):
        import os

        from replica.dataset import NaturalStoriesMaze

        ds = NaturalStoriesMaze(analysed_sample=False)
        ds.load(os.path.join(data_root, "maze", "original", "natural_stories_maze"))
        assert len(ds.participants) == 100

    def test_measures(self):
        assert self.ds.measures == ["maze_rt"]

    @pytest.mark.parametrize(
        ("post_error", "expected"),
        [("drop_two", 55384), ("drop_rest", 40809)],
    )
    def test_reproduces_the_studys_analysis_datasets(
        self, data_root, post_error, expected
    ):
        """Row-for-row against the authors' manuscript.Rmd.

        ``drop_two`` is their ``data_ready``, ``drop_rest`` their
        ``data_pre_error``. Their analysis then drops the first word of each
        sentence and RTs outside (100, 5000) ms, which REPLICA leaves to the
        analysis layer, so both are applied here to line the counts up.
        """
        import os

        from replica.dataset import NaturalStoriesMaze

        ds = NaturalStoriesMaze(post_error=post_error)
        ds.load(os.path.join(data_root, "maze", "original", "natural_stories_maze"))
        kept = 0
        for datum in ds._data:
            values = datum.values.copy()
            values[0] = np.nan  # the study's word_num != 0
            kept += int(((values > 100) & (values < 5000)).sum())
        assert kept == expected

    def test_rejects_an_unknown_error_policy(self):
        from replica.dataset import NaturalStoriesMaze

        with pytest.raises(ValueError, match="post_error"):
            NaturalStoriesMaze(post_error="whatever")

    def test_language_is_declared(self):
        assert self.ds.info.language == "en"

    def test_first_stimulus_words(self):
        stim = next(
            s for s in self.ds.stimuli() if s.text.startswith("If you were to journey")
        )
        assert stim.layers["words"].values[:5] == [
            "If",
            "you",
            "were",
            "to",
            "journey",
        ]

    def test_first_word_is_kept_but_excluded_from_analysis(self):
        """The first word is paired with a nonce distractor, so the study
        excludes it. The release still records the latency, and the mask
        applies the exclusion."""
        assert any(not np.isnan(d.values[0]) for d in self.ds._data)
        assert all(not self.ds.exclusion_mask(d)[0] for d in self.ds._data)

    def test_latencies_are_positive(self):
        for datum in self.ds._data[:200]:
            observed = datum.values[~np.isnan(datum.values)]
            assert (observed > 0).all()

    def test_stimuli_match_natural_stories_text(self, data_root):
        """Maze reads the Natural Stories passages, so every sentence but one
        appears verbatim in that corpus.

        The maze materials read "peeked" where the Natural Stories release has
        "peaked", so the expected count is 473 of 474.
        """
        import re

        ns_path = os.path.join(
            data_root, "self-paced-reading", "original", "natural_stories"
        )
        if not os.path.isdir(ns_path):
            pytest.skip("Natural Stories data not available for the cross-check")
        from replica.dataset import NaturalStories

        ns = NaturalStories()
        ns.load(ns_path)

        def norm(text):
            return re.sub(r"\s+", " ", text).strip()

        ns_text = norm(" ".join(s.text for s in ns.stimuli()))
        found = [s for s in self.ds.stimuli() if norm(s.text) in ns_text]
        missing = [s for s in self.ds.stimuli() if norm(s.text) not in ns_text]
        assert len(found) == 473
        assert len(missing) == 1
        assert "peeked" in missing[0].text

    def test_stimuli_carry_their_story(self):
        stories = {s.metadata["story"] for s in self.ds.stimuli()}
        assert len(stories) == 10


def test_stimulus_id_is_independent_of_participant_and_error_selection(data_root):
    path = os.path.join(data_root, "maze/original/natural_stories_maze")
    identities = []
    for analysed, policy in [(True, "drop_two"), (False, "drop_two"), (False, "keep")]:
        dataset = NaturalStoriesMaze(analysed_sample=analysed, post_error=policy)
        dataset.load(path)
        identities.append({stim.text: stim.stimulus_id for stim in dataset.stimuli()})
    assert len(identities[0]) == 474
    assert identities[0] == identities[1] == identities[2]


def test_full_sample_retains_source_errors_and_zeros_but_excludes_zero_latency(
    data_root,
):
    dataset = NaturalStoriesMaze(analysed_sample=False, post_error="keep")
    dataset.load(os.path.join(data_root, "maze/original/natural_stories_maze"))
    import pyreadr

    source = pyreadr.read_r(
        os.path.join(data_root, "maze/original/natural_stories_maze/cleaned.rds")
    )[None]
    source = source[source["type"] != "practice"].copy()
    source["subject"] = source["subject"].astype(str)
    trials = dict(tuple(source.groupby(["sentence", "subject"], sort=False)))
    zeros = 0
    for datum in dataset.data("maze_rt"):
        zero = datum.values == 0
        zeros += zero.sum()
        assert not dataset.exclusion_mask(datum)[zero].any()
        rows = trials[
            (datum.stimulus.metadata["sentence"], datum.participant_id)
        ].sort_values("word_num")
        for field in ("word_num", "correct", "rt"):
            np.testing.assert_equal(
                datum.metadata[f"source_{field}"], rows[field].tolist()
            )
    expected_zeros = ((source["correct"] == "yes") & (source["rt"] == 0)).sum()
    assert zeros == expected_zeros > 0
