"""Tests for Narratives fMRI dataset loader."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from replica.dataset.fmri.narratives import Narratives


def test_other_stories_require_an_explicit_trim():
    from replica.core import Stimulus

    dataset = Narratives(story="another")
    with pytest.raises(ValueError, match="No verified trimming"):
        dataset._trim_and_store_bold(
            np.zeros((30, 2)), Stimulus("", stimulus_id="s"), "p"
        )


@pytest.mark.parametrize("trim", [(0, 0), (2, 3)])
@pytest.mark.parametrize(
    "interval, dtype",
    [(1.5, np.float64), (2.0, np.float64), (None, np.float64), (0.1, np.float32)],
)
def test_explicit_trim_preserves_sample_alignment(trim, interval, dtype):
    from replica.core import Stimulus

    dataset = Narratives(story="another", trim_trs=trim)
    values = np.arange(60).reshape(30, 2)
    times = np.arange(30, dtype=dtype) * (interval or 1.5)
    if interval is None:
        times[::2] += 0.1
    dataset._trim_and_store_bold(
        values, Stimulus("", stimulus_id="s"), "p", sample_times=times
    )
    recording = dataset.recordings()[0]
    end = -trim[1] if trim[1] else None
    np.testing.assert_equal(recording.values, values[trim[0] : end])
    np.testing.assert_equal(recording.times, times[trim[0] : end])
    assert recording.sample_times.dtype == np.dtype(dtype)
    if dtype is np.float32:
        assert recording.sample_interval == pytest.approx(
            interval, rel=4 * np.finfo(dtype).eps
        )
    else:
        assert recording.sample_interval == interval


@pytest.mark.parametrize("trim", [(-1, 0), (0, -1), (1.5, 0), (0,)])
def test_invalid_trimming_is_rejected(trim):
    with pytest.raises(ValueError, match="two non-negative integers"):
        Narratives(trim_trs=trim)


class TestNarrativesAlignment:
    """Test stimulus loading from a synthetic align.csv."""

    def test_load_alignment(self):
        """Parse alignment CSV into Stimulus with words + word_times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic align.csv
            align_file = os.path.join(tmpdir, "test_align.csv")
            with open(align_file, "w") as f:
                f.write("This,this,21.22,21.29\n")
                f.write("is,is,21.46,21.65\n")
                f.write("Los,los,21.66,21.94\n")
                f.write("Angeles,angeles,21.94,22.77\n")

            ds = Narratives(story="test")
            stimulus = ds._load_alignment(align_file)

            # Check words layer
            assert "words" in stimulus.layers
            words = stimulus.unit_strings("words")
            assert words == ["This", "is", "Los", "Angeles"]

            # Check word_times layer
            assert "word_times" in stimulus.layers
            times_layer = stimulus.layers["word_times"]
            assert len(times_layer) == 4

            _, first_time = times_layer[0]
            assert first_time == (21.22, 21.29)

            _, last_time = times_layer[3]
            assert last_time == (21.94, 22.77)

    def test_stimulus_id(self):
        """Stimulus ID matches story name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            align_file = os.path.join(tmpdir, "story_align.csv")
            with open(align_file, "w") as f:
                f.write("Hello,hello,0.0,0.5\n")
                f.write("world,world,0.6,1.0\n")

            ds = Narratives(story="story")
            stimulus = ds._load_alignment(align_file)
            assert stimulus.stimulus_id == "story"

    def test_alignment_column_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            align_file = os.path.join(tmpdir, "story_align.csv")
            with open(align_file, "w") as f:
                f.write("surface,<unk>,0.0,0.5\n")

            surface = Narratives(story="story", alignment_word_column=0)
            author = Narratives(story="story", alignment_word_column=1)

            assert surface._load_alignment(align_file).unit_strings() == ["surface"]
            author_stimulus = author._load_alignment(align_file)
            assert author_stimulus.unit_strings() == ["<unk>"]
            assert author_stimulus.metadata["alignment_word_column"] == 1

    def test_report_trigger_repair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            align_file = os.path.join(tmpdir, "story_align.csv")
            with open(align_file, "w") as f:
                f.write("word,word,0.0,0.5\n")
            with open(os.path.join(tmpdir, "story.report"), "w") as f:
                f.write("0.0 init-trigger\n")
                f.write("1.5 trigger\n")
                f.write("3.0 trigger\n")
                f.write("6.0 trigger\n")
                f.write("7.5 trigger\n")

            dataset = Narratives(story="story")
            times = dataset._load_report_times(Path(align_file))
            np.testing.assert_array_equal(times, [0.0, 1.5, 3.0, 4.5, 6.0, 7.5])


class TestNarrativesLoadNpy:
    """Test BOLD loading from .npy files."""

    def test_load_npy_subjects(self):
        """Load BOLD data from .npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create align.csv
            align_dir = os.path.join(tmpdir, "stimulus_resources")
            os.makedirs(align_dir)
            align_file = os.path.join(align_dir, "test_align.csv")
            with open(align_file, "w") as f:
                f.write("Hello,hello,0.0,0.5\n")
                f.write("world,world,0.6,1.0\n")

            # Create BOLD .npy files
            story_dir = os.path.join(tmpdir, "test")
            os.makedirs(story_dir)
            rng = np.random.default_rng(42)
            for i in range(3):
                bold = rng.standard_normal((100, 50))
                np.save(os.path.join(story_dir, f"sub_{i}.npy"), bold)

            ds = Narratives(story="test", trim_trs=(14, 9))
            ds.load(tmpdir, align_path=align_dir)

            # Check native fMRI recordings loaded
            recordings = ds.recordings(measure="bold", modality="fmri")
            assert len(recordings) == 3
            assert len(ds.participants) == 3

            # Check trimming applied (14 start + 9 end = 23 removed from 100)
            recording = recordings[0]
            assert recording.values.shape == (77, 50)
            assert recording.metadata["trim_start_samples"] == 14
            assert recording.metadata["trim_end_samples"] == 9
            assert recording.metadata["timing_source"] == "nominal-tr"
            assert recording.times[0] == 14 * recording.sample_interval

    def test_max_subjects(self):
        """max_subjects limits loaded participants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            align_dir = os.path.join(tmpdir, "stimulus_resources")
            os.makedirs(align_dir)
            with open(os.path.join(align_dir, "test_align.csv"), "w") as f:
                f.write("Hello,hello,0.0,0.5\n")

            story_dir = os.path.join(tmpdir, "test")
            os.makedirs(story_dir)
            for i in range(5):
                np.save(
                    os.path.join(story_dir, f"sub_{i}.npy"),
                    np.zeros((100, 10)),
                )

            ds = Narratives(story="test", trim_trs=(14, 9))
            ds.load(tmpdir, align_path=align_dir, max_subjects=2)

            assert len(ds.recordings(measure="bold", modality="fmri")) == 2


class TestNativeFMRIQueries:
    """Test fMRI through the shared recording query."""

    def test_recording_filtering(self):
        """recordings() filters fMRI by participant and stimulus ID."""
        from replica.core import (
            AnnotationLayer,
            Recording,
            Span,
            Stimulus,
        )

        # Build minimal stimulus
        stim = Stimulus(text="hello", stimulus_id="s1")
        stim.add_layer(AnnotationLayer("words", [Span(0, 5)], ["hello"]))
        stim.add_layer(AnnotationLayer("word_times", [Span(0, 5)], [(0.0, 0.5)]))

        ds = Narratives(story="test")
        ds._stimuli = [stim]
        for participant in ("p1", "p2"):
            ds.add_recording(
                Recording(
                    stimulus=stim,
                    values=np.zeros((10, 5)),
                    measure_name="bold",
                    modality="fmri",
                    participant_id=participant,
                    sample_interval=1.5,
                    dimension_names=("time", "vertex"),
                )
            )

        assert len(ds.recordings(participant="p1", modality="fmri")) == 1
        assert len(ds.recordings(participant="p3", modality="fmri")) == 0
