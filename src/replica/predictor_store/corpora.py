"""Corpus text and source-identity checks shared by store export and readback."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Protocol

from replica.compute.units import TOKEN_UNIT, WORD_JOINED, verify_token_rows
from replica.core import Stimulus
from replica.dataset.catalog.provenance import corpus_fingerprint
from replica.feature.core.annotation.characters import CHARACTER_LAYER

from . import layout
from .alignment import RowCheck, describe
from .cells import Cell


class Corpora(Protocol):
    """Where the export reads the text units of a corpus."""

    def names(self) -> list[str]:
        """Every dataset name a tag may end in."""

    def stimuli(self, dataset: str) -> list[Stimulus]:
        """The stimuli of one corpus, as the loader builds them.

        Args:
            dataset: Registry name of the corpus.
        """

    def word_spacing(self, dataset: str) -> bool:
        """Whether the corpus text separates its words with whitespace.

        Args:
            dataset: Registry name of the corpus.
        """


class RegistryCorpora:
    """The dataset registry, loading each corpus from ``PSYCHO_DATA_DIR``."""

    def __init__(self) -> None:
        """Keep each loaded dataset's info, so its flags cost no second load."""
        self._info: dict[str, object] = {}

    def names(self) -> list[str]:
        """Every registered dataset name."""
        from replica.dataset.catalog.registry import REGISTRY

        return sorted(REGISTRY)

    def stimuli(self, dataset: str) -> list[Stimulus]:
        """Load one corpus through its registered loader.

        Args:
            dataset: Registry name of the corpus.
        """
        from replica.dataset.catalog.registry import REGISTRY

        loaded = REGISTRY[dataset].factory()
        self._info[dataset] = loaded.info
        return list(loaded.stimuli())

    def word_spacing(self, dataset: str) -> bool:
        """The loader's word spacing flag.

        Args:
            dataset: Registry name of the corpus.
        """
        from replica.dataset.catalog.registry import REGISTRY

        if dataset not in self._info:
            self._info[dataset] = REGISTRY[dataset].factory().info
        return bool(getattr(self._info[dataset], "word_spacing", True))


class TextIndex:
    """Unit strings of one corpus by stimulus id, for the word and character grains.

    The word grain reads the loader's ``words`` layer, or its single layer
    when it has no ``words`` layer; the character grain is every character
    of the text, whitespace included, which is the ``characters`` layer the
    compute driver attaches.
    """

    def __init__(self, stimuli: Sequence[Stimulus], word_spacing: bool = True):
        """Index the unit strings of *stimuli*.

        Args:
            stimuli: The corpus, as the loader builds it.
            word_spacing: The loader's word spacing flag.

        Raises:
            ValueError: When no ``words`` layer exists and the stimuli carry
                more than one layer between them.
        """
        layers = {name for s in stimuli for name in s.layers}
        if "words" in layers:
            self.layer = "words"
        elif len(layers) == 1:
            self.layer = next(iter(layers))
        else:
            raise ValueError(f"ambiguous unit layer: {sorted(layers)}")
        self.stimuli = list(stimuli)
        self.word_spacing = word_spacing
        self.words = {s.stimulus_id: s.unit_strings(self.layer) for s in stimuli}
        self.chars = {s.stimulus_id: list(s.text) for s in stimuli}
        self.layers: dict[str, dict[str, list[str]]] = {}
        self.fingerprints: dict[tuple[str, str], str] = {}

    def grid(self, unit: str) -> dict[str, list[str]]:
        """Unit strings of every stimulus at *unit* grain.

        ``words`` and ``characters`` are built here; any other grain is the
        stimulus layer of that name, so a cell computed on another
        segmentation is checked against that segmentation.

        Args:
            unit: The grain's layer name.

        Raises:
            ValueError: When a stimulus has no layer of that name.
        """
        if unit == "words":
            return self.words
        if unit == CHARACTER_LAYER:
            return self.chars
        if unit not in self.layers:
            from replica.segmentation import SEGMENTERS, attach_segmentation

            stimuli = self.stimuli
            if unit in SEGMENTERS:
                stimuli = deepcopy(stimuli)
                attach_segmentation(unit, stimuli)
            missing = [s.stimulus_id for s in stimuli if unit not in s.layers]
            if missing:
                raise ValueError(
                    f"the loader has no {unit!r} layer on {len(missing)} stimuli "
                    f"({missing[0]}), so cells at that grain cannot be checked; a "
                    "segmentation registers with replica.segmentation.register_segmenter "
                    "in a module named with --import"
                )
            self.layers[unit] = {s.stimulus_id: s.unit_strings(unit) for s in stimuli}
        return self.layers[unit]

    def units(self, unit: str, stimulus_id: str, n_records: int) -> list[str] | None:
        """Strings of one stimulus at *unit* grain, or ``None`` when they cannot be named.

        A cell writes one record per unit, so a stimulus whose record count
        differs from the corpus was scored against another version of the
        text and no position in it can be named.

        Args:
            unit: ``words`` or ``characters``.
            stimulus_id: Stimulus to look up.
            n_records: Records the cell holds for it.
        """
        strings = self.grid(unit).get(stimulus_id)
        if strings is None or len(strings) != n_records:
            return None
        return strings


class MisalignedCell(ValueError):
    """A cell's records index other units than its corpus loader produces."""


def check_payload(cell: Cell, payload: dict, text: TextIndex) -> None:
    """Check the recorded corpus digest and each payload's unit positions.

    Args:
        cell: Source cell and its declared computational identity.
        payload: Source records grouped by stimulus ID.
        text: Current corpus text and unit grids.
    """
    layout.check_ids(payload, cell.path.name)
    stimuli = {stimulus.stimulus_id: stimulus for stimulus in text.stimuli}
    unknown = set(payload) - stimuli.keys()
    if unknown:
        raise MisalignedCell(f"stimulus IDs absent from the loader: {sorted(unknown)}")
    key = (cell.unit, cell.context_scope)
    if key not in text.fingerprints:
        text.fingerprints[key] = corpus_fingerprint(cell.dataset, text.stimuli, *key)
    if cell.identity.get("corpus_sha256") != text.fingerprints[key]:
        raise MisalignedCell("corpus_sha256 differs from the loader's corpus")
    if cell.unit == TOKEN_UNIT:
        for stimulus_id, records in payload.items():
            for index, record in enumerate(records):
                layout.token_identity(record.metadata, index)
            verify_token_rows(
                stimuli[stimulus_id],
                records,
                "words" if cell.method in WORD_JOINED else None,
            )
        return
    check = RowCheck(text.grid(cell.unit))
    for stimulus_id, records in payload.items():
        for index in range(len(records)):
            check.add(stimulus_id, index)
    problems = check.problems()
    if problems:
        raise MisalignedCell(describe(problems))
