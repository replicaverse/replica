"""Base class for maze-task corpora."""

from ...core import Presentation
from ..base import ReadingTimeDataset


class MazeDataset(ReadingTimeDataset):
    """Maze-task dataset with a single response-latency measure.

    In the A-maze task the reader picks the true continuation over a
    distractor at each position, so the measure is the latency of that
    choice.
    """

    default_measure = "maze_rt"
    default_presentation = Presentation.MAZE
    additive_measures = frozenset({"maze_rt"})
