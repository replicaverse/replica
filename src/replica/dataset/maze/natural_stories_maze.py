"""Natural Stories Maze (Boyce & Levy, 2023).

The Natural Stories passages read in the A-maze task
of Boyce et al. (2020): at every position the reader chooses between
the true next word and an automatically generated distractor, and the latency
of that choice is the measure. Sentences are presented one at a time, so each
sentence is one Stimulus.

The release presents the passages in *error-correction* Maze, in which a
reader who picks the distractor is corrected and continues rather than
losing the rest of the item. The first word of a sentence is paired with a
nonce ``x-x-x`` distractor because there is no context to choose on, and the
authors exclude it; :meth:`exclusion_mask` drops it as the first unit anyway.

The practice story every reader sees first is not part of the Natural Stories
corpus and is dropped.

Data format expected, in ``Data/cleaned.rds`` of the authors' release
(https://github.com/vboyce/natural-stories-maze), with columns
[type, word_num, word, distractor, rt, correct, sentence, native,
num_correct, subject].
"""

import hashlib
from pathlib import Path

import numpy as np

from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

from ...core import Datum, Stimulus
from .base import MazeDataset

#: Free-text native-language answers the authors accept as English.
_NATIVE_ENGLISH = ("ENG", "English", "ENGLISH", "english")

#: Accuracy a participant must exceed to enter the authors' analysis.
_MIN_ACCURACY = 0.8

#: Latency above which a trial is treated as the reader being distracted;
#: the authors drop these before computing per-participant accuracy.
_MAX_RT_FOR_ACCURACY = 5000.0

#: Words after a wrong choice that the study's main analysis discards:
#: the released analysis drops exactly these two lags, which
#: ``post_error="drop_two"`` reproduces.
_POST_ERROR_LAG = (1, 2)

#: What to do with words following a wrong choice. ``drop_two`` is the study's
#: main dataset, ``drop_rest`` its pre-mistake subset (used for the GAMs and
#: shipped as ``maze_pre_error.rds``), ``keep`` every correct response.
_POST_ERROR_POLICIES = ("drop_two", "drop_rest", "keep")


class NaturalStoriesMaze(MazeDataset):
    """Natural Stories Maze (Boyce & Levy, 2023).

    Args:
        analysed_sample: Keep only the participants the study analyses,
            i.e. those reporting English as their native language and
            exceeding 80% accuracy. ``False`` loads every recruited
            reader.
        post_error: Which words after a wrong choice to discard, one of
            ``_POST_ERROR_POLICIES``. The wrong choice itself is always
            discarded. Source choices and latencies remain in datum metadata.

    Latency windows are analysis exclusions, separate from participant and
    post-error selection. The default mask excludes sentence-initial and
    non-positive responses; the paper's (100, 5000) ms window is not imposed.
    """

    info = DatasetInfo(name="natural_stories_maze", language="en")

    def __init__(self, analysed_sample: bool = True, post_error: str = "drop_two"):
        """Record which participants and post-error words :meth:`load` keeps."""
        super().__init__()
        if post_error not in _POST_ERROR_POLICIES:
            raise ValueError(
                f"post_error must be one of {_POST_ERROR_POLICIES}, got {post_error!r}"
            )
        self.analysed_sample = analysed_sample
        self.post_error = post_error

    def exclusion_mask(self, datum: Datum) -> np.ndarray:
        """Exclude missing, sentence-initial and non-positive response latencies.

        Args:
            datum: Observation whose source latencies remain available in data().
        """
        return super().exclusion_mask(datum) & (datum.values > 0)

    def load(self, path: str, rt_file: str = "cleaned.rds") -> None:
        """Load Natural Stories Maze from the authors' release.

        Args:
            path: Directory containing the data file.
            rt_file: Serialized trial table from the release.
        """
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        frame = self._read(Path(path) / rt_file)
        if self.analysed_sample:
            frame = self._analysed_sample(frame)
        self._build(frame)

    @staticmethod
    def _read(rt_path: Path):
        import pyreadr

        frame = pyreadr.read_r(str(rt_path))[None]
        frame = frame[frame["type"] != "practice"].copy()
        frame["subject"] = frame["subject"].astype(str)
        frame["word_num"] = frame["word_num"].astype(int)
        frame["rt"] = frame["rt"].astype(float)
        return frame

    @staticmethod
    def _analysed_sample(frame):
        """Apply the study's participant exclusions."""
        frame = frame[frame["native"].isin(_NATIVE_ENGLISH)]
        scored = frame[frame["rt"] < _MAX_RT_FOR_ACCURACY]
        accuracy = (
            scored.assign(hit=(scored["correct"] == "yes").astype(float))
            .groupby("subject")["hit"]
            .mean()
        )
        keep = accuracy[accuracy > _MIN_ACCURACY].index
        return frame[frame["subject"].isin(keep)]

    @staticmethod
    def _distance_since_mistake(frame):
        """Words elapsed since the last wrong choice in the same sentence.

        0 marks a word with no preceding mistake, and the mistake itself.
        """
        frame = frame.sort_values(["sentence", "subject", "word_num"])
        at_mistake = np.where(frame["correct"].ne("yes"), frame["word_num"], np.nan)
        filled = (
            frame.assign(_at=at_mistake).groupby(["sentence", "subject"])["_at"].ffill()
        )
        return frame, (frame["word_num"] - filled).fillna(0)

    def _keep_mask(self, frame):
        """Which trials yield a usable latency under the error policy."""
        frame, distance = self._distance_since_mistake(frame)
        chosen_correctly = frame["correct"] == "yes"
        if self.post_error == "keep":
            return frame, chosen_correctly
        if self.post_error == "drop_rest":
            return frame, chosen_correctly & (distance == 0)
        return frame, chosen_correctly & ~distance.isin(_POST_ERROR_LAG)

    def _build(self, frame) -> None:
        """Build one Stimulus per sentence and one Datum per trial."""
        self._stimuli: list[Stimulus] = []
        self._data: list[Datum] = []
        sentence_map: dict[str, Stimulus] = {}

        for sentence, rows in frame.groupby("sentence", sort=False):
            words = (
                rows.drop_duplicates("word_num")
                .sort_values("word_num")["word"]
                .tolist()
            )
            sentence_map[sentence] = self._make_stimulus(
                words,
                f"nsmaze_{hashlib.sha256(str(sentence).encode()).hexdigest()[:16]}",
                {"story": str(rows["type"].iloc[0]), "sentence": sentence},
            )

        frame, keep = self._keep_mask(frame)
        source_trials = dict(tuple(frame.groupby(["sentence", "subject"], sort=False)))
        for (sentence, subject), rows in frame[keep].groupby(
            ["sentence", "subject"], sort=False
        ):
            stimulus = sentence_map[sentence]
            values = np.full(len(stimulus.layers["words"]), np.nan)
            for idx, rt in zip(rows["word_num"], rows["rt"], strict=True):
                if 0 <= idx < len(values):
                    values[int(idx)] = float(rt)
            source = source_trials[(sentence, subject)].sort_values("word_num")
            self._data.append(
                Datum(
                    stimulus=stimulus,
                    measure_name="maze_rt",
                    values=values,
                    participant_id=str(subject),
                    metadata={
                        "source_word_num": source["word_num"].tolist(),
                        "source_correct": source["correct"].tolist(),
                        "source_rt": source["rt"].tolist(),
                    },
                )
            )
