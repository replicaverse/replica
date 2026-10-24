"""Maze-task reading corpora."""

from .base import MazeDataset
from .natural_stories_maze import NaturalStoriesMaze

__all__ = ["MazeDataset", "NaturalStoriesMaze"]
