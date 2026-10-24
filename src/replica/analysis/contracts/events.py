"""Impulse and response timeline contracts for continuous-time models."""

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EventIndex:
    """Index of one event stream, sorted by series then time.

    One row per event: a presented word per participant for self-paced
    reading, a fixation for eye tracking. ``series`` is a dense code of the
    grouping whose timeline is continuous (usually participant and stimulus),
    matching the reference implementations' ``series_ids``.
    """

    participant_id: np.ndarray
    stimulus_id: np.ndarray
    unit_index: np.ndarray
    time: np.ndarray
    series: np.ndarray
    #: Optional identity of the item random-effect level.  This is distinct
    #: from ``unit_index`` when a corpus presents the same source item more
    #: than once.
    item_id: np.ndarray | None = None
    layer: str = "words"

    def __post_init__(self) -> None:
        n = len(self.time)
        for name in ("participant_id", "stimulus_id", "unit_index", "series"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"EventIndex.{name} must have one entry per event")
        if self.item_id is not None and len(self.item_id) != n:
            raise ValueError("EventIndex.item_id must have one entry per event")

    def __len__(self) -> int:
        return len(self.time)


@dataclass(frozen=True)
class EventStream:
    """An :class:`EventIndex` with one predictor column per impulse."""

    index: EventIndex
    values: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        for name, column in self.values.items():
            if len(column) != len(self.index):
                raise ValueError(
                    f"impulse column {name!r} must have one entry per event"
                )

    def matrix(self, names: list[str]) -> np.ndarray:
        """Impulse columns in the given order, shape ``(n_events, len(names))``.

        Args:
            names: Impulse columns to stack left to right. An empty list gives
                a zero-column matrix that still has one row per event.
        """
        if not names:
            return np.zeros((len(self.index), 0), dtype=np.float32)
        return np.column_stack([self.values[name] for name in names]).astype(np.float32)


@dataclass(frozen=True)
class ResponseEvents:
    """The response side, row-aligned to the flat arrays, plus its impulses.

    ``time`` and ``series`` place each analyzed observation on the same
    timeline as the impulses. The impulse table is shared by every fold: only
    the response side is split, mirroring releases whose impulse file is the
    same for train, development and test.
    """

    time: np.ndarray
    series: np.ndarray
    impulses: EventStream
    control_names: tuple[str, ...] = ()
    focal_names: tuple[str, ...] = ()
    #: Dense code of the word each response is to, for models that give the
    #: word a random effect of its own.
    item: np.ndarray | None = None
    #: Response values and grouping codes, so a model can build a design for
    #: rows a fold assigns to neither side, such as a development partition.
    y: np.ndarray | None = None
    group: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.time) != len(self.series):
            raise ValueError("ResponseEvents.time and .series must align")

    def __len__(self) -> int:
        return len(self.time)

    def impulse_names(self, *, is_full: bool) -> list[str]:
        """Baseline impulses, plus the focal ones for the full model.

        Args:
            is_full: True for the full model, which appends the focal impulses
                the controls do not already cover; False leaves the baseline
                term set alone.
        """
        names = list(self.control_names)
        if is_full:
            names += [n for n in self.focal_names if n not in names]
        return names

    def subset(self, mask: np.ndarray) -> "ResponseEvents":
        """Select response rows; the impulse table is left whole.

        Args:
            mask: Boolean mask over the response rows, true where the row is
                kept, as a fold's train or test membership.
        """
        return replace(
            self,
            time=self.time[mask],
            series=self.series[mask],
            item=None if self.item is None else self.item[mask],
            y=None if self.y is None else self.y[mask],
            group=None if self.group is None else self.group[mask],
        )


@runtime_checkable
class TimedDataset(Protocol):
    """A dataset that can place its observations on a participant timeline."""

    def event_index(self, measure: str, *, layer: str = "words") -> EventIndex:
        """Impulse-stream index for one measure.

        Args:
            measure: Reading measure the timeline is built for, the dependent
                variable an analysis is about to score.
            layer: Stimulus annotation layer the ``unit_index`` counts in.
        """
        ...


@runtime_checkable
class EventCovariateDataset(Protocol):
    """A timed dataset that owns impulses of its own.

    Word-level predictors reach the event stream through the feature
    pipeline, which knows a value per word rather than per presentation. An
    impulse that is a property of the presentation instead, such as how far
    into a session a word appeared, has no such value and is supplied here.
    These impulses join the baseline term set, since a covariate the dataset
    defines is not what an ablation removes.
    """

    def event_impulses(self) -> tuple[str, ...]:
        """Names of the impulses this dataset supplies."""
        ...

    def event_values(
        self, names: list[str], index: EventIndex
    ) -> dict[str, np.ndarray]:
        """Per-event columns for whichever of ``names`` the dataset owns.

        Args:
            names: Every impulse the analysis asks for; the dataset returns
                entries only for the ones it defines, leaving the rest to the
                feature pipeline.
            index: Event rows the returned columns must align with.
        """
        ...
