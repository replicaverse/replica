"""Russian Sentence Corpus (Laurinavichyute et al., 2019).

Russian sentence-reading eye-tracking corpus published at
https://osf.io/x5q2r/. Native speakers read 144 isolated sentences built
around a target word; the release is a tab-separated per-interest-area
table (tab-delimited despite the ``.csv`` suffix).

Two releases ship side by side and are selected with ``release=``:

  * ``"published"`` (``data.csv``), the version the paper reports: 96
    participants, sentence-final words removed.
  * ``"extended"`` (``data_103.csv``), a later upload: 102 participants
    (the filename says 103) with sentence-final words retained.

Columns read: ``item.id`` (sentence), ``word.serial.no`` (1-indexed word
position), ``DATA_FILE`` (participant, e.g. ``90.edf``), ``item.text``
(the full sentence), four EyeLink duration columns, and ``IA_SKIP``.
Other source columns, including cloze predictability and saccade geometry,
are not loaded. Frequency and length controls are computed by feature providers.

Reference: Laurinavichyute et al. (2019). *Behavior Research Methods.*
https://doi.org/10.3758/s13428-018-1051-6
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.measures import (
    EYELINK_IA_MEASURES,
    EYELINK_MEASURE_NAMES,
    eyelink_measures,
)
from replica.dataset.metadata import io as metadata_io

from ...core import Datum, Stimulus
from ..loading.utils import (
    append_measure_datums,
    parse_reading_measure,
    scatter_first_occurrence,
)

# The release contains go-past durations shorter than first-run durations.
# Preserve released values and exclude contradictory go-past observations from analysis.

# Release name → filename under the dataset directory.
_RELEASES = {
    "published": "data.csv",
    "extended": "data_103.csv",
}

_MEASURE_NAMES = EYELINK_MEASURE_NAMES

# item id → sentence text
_Texts = dict[int, str]
# item id → participant → [(word position, per-measure tuple)]
_ItemData = dict[int, dict[str, list[tuple[int, tuple[float, ...]]]]]


def inconsistent_go_past(first_run, go_past) -> np.ndarray:
    """Flag finite regression paths shorter than the corresponding first run.

    Args:
        first_run: Released first-run durations.
        go_past: Released regression-path durations at the same observations.
    """
    first_run, go_past = np.asarray(first_run, float), np.asarray(go_past, float)
    if first_run.shape != go_past.shape:
        raise ValueError("First-run and go-past observations must have matching shapes")
    return np.isfinite(first_run) & np.isfinite(go_past) & (go_past < first_run)


class RSC(EyeTrackingDataset):
    """Russian Sentence Corpus (Laurinavichyute et al., 2019).

    Source file (download with ``replica fetch rsc``, which brings both):
    ``data.csv`` for the default ``release="published"``, ``data_103.csv``
    for ``release="extended"``. Either loads without the other present.

    One ``Stimulus`` per sentence; participant ids are the ``DATA_FILE``
    stem (``"90.edf"`` → ``"90"``). Default measure: ``first_run_in_first_pass``.

    Usage::

        ds = RSC()
        ds.load("datasets/eye-tracking/original/rsc")
        data = ds.data(measure="first_run_in_first_pass")
    """

    info = DatasetInfo(name="rsc", language="ru")
    _declared_measures = _MEASURE_NAMES
    RELEASES = tuple(_RELEASES)

    def __init__(
        self, release: str = "published", exclude_inconsistent_go_past: bool = True
    ):
        """Pick which of the two released measure files this loader reads.

        Args:
            release: ``"published"`` for the version the paper reports, or
                ``"extended"`` for the later upload with more participants and
                sentence-final words retained. One of ``RELEASES``.
            exclude_inconsistent_go_past: Exclude regression paths shorter than
                the same observation's first run from analysis. Released values
                remain available through :meth:`data`.

        Raises:
            ValueError: When ``release`` is not one of ``RELEASES``.
        """
        super().__init__()
        if release not in self.RELEASES:
            raise ValueError(
                f"Unknown RSC release {release!r}. Valid: {list(self.RELEASES)}"
            )
        self.release = release
        self.exclude_inconsistent_go_past = exclude_inconsistent_go_past

    def load(self, path: str) -> None:
        """Load one RSC release from its dataset directory.

        Args:
            path: Directory holding the release file for ``self.release``.
        """
        # The YAML declares both releases; verify only the one being read, so
        # either can be loaded on its own.
        self._load_metadata(
            path,
            files=[_RELEASES[self.release]],
            metadata_path=metadata_io.find_dataset_yaml(path),
        )
        source = self._source_path(path)
        texts, item_data = self._read_measures(source)
        self._build(texts, item_data)

    def _source_path(self, path: str) -> Path:
        source = Path(path) / _RELEASES[self.release]
        if not source.exists():
            raise FileNotFoundError(
                f"RSC: {source.name} not found in {path}. Run `replica fetch rsc`."
            )
        return source

    @staticmethod
    def _read_measures(source: Path) -> tuple[_Texts, _ItemData]:
        """Parse the per-interest-area table into per-item, per-participant rows.

        ``item.text`` is constant within an item, so the first row of each
        item fixes the sentence.
        """
        texts: _Texts = {}
        item_data: _ItemData = defaultdict(lambda: defaultdict(list))
        with open(source, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            required = {
                "item.id",
                "word.serial.no",
                "DATA_FILE",
                "item.text",
                *EYELINK_IA_MEASURES,
                "IA_SKIP",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"RSC {source.name}: missing columns {sorted(missing)}"
                )
            for row in reader:
                try:
                    item_id = int(row["item.id"])
                    position = int(row["word.serial.no"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"RSC {source.name}:{reader.line_num}: invalid item or word index"
                    ) from error
                text = row["item.text"]
                if (
                    not text
                    or not row["DATA_FILE"]
                    or not 1 <= position <= len(text.split())
                ):
                    raise ValueError(
                        f"RSC {source.name}:{reader.line_num}: invalid trial identity or word position"
                    )
                if texts.setdefault(item_id, text) != text:
                    raise ValueError(
                        f"RSC {source.name}: conflicting text for item {item_id}"
                    )
                participant = row["DATA_FILE"].removesuffix(".edf")
                values = eyelink_measures(
                    row.get,
                    unfixated=parse_reading_measure(row["IA_DWELL_TIME"]) == 0.0,
                )
                item_data[item_id][participant].append((position, values))
        return texts, item_data

    def _build(self, texts: _Texts, item_data: _ItemData) -> None:
        """Build one Stimulus per sentence and one Datum per participant-measure.

        Word positions are 1-indexed against ``item.text`` split on spaces,
        so a measure lands at ``position - 1``. Positions the release omits
        stay NaN: in the ``published`` release the sentence-initial word (no
        valid preceding region) and the sentence-final word of every item,
        and in both releases a handful of interior words the authors
        excluded.

        Duplicates: a few dozen (participant, sentence) trials per release
        appear twice, sometimes with contradicting values for the same word.
        The first occurrence in file order wins, which keeps a whole trial
        from one recording rather than mixing two.
        """
        self._stimuli: list[Stimulus] = []
        self._data: list[Datum] = []
        for item_id in sorted(texts):
            words = texts[item_id].split()
            stimulus = self._make_stimulus(
                words,
                stimulus_id=f"rsc_s{item_id}",
                metadata={"item_id": item_id},
            )
            index_of = {position: position - 1 for position in range(1, len(words) + 1)}
            for participant, rows in item_data[item_id].items():
                arrays = scatter_first_occurrence(
                    rows, index_of, len(words), len(_MEASURE_NAMES)
                )
                first_run = arrays[_MEASURE_NAMES.index("first_run")]
                go_past = arrays[_MEASURE_NAMES.index("go_past")]
                inconsistent = inconsistent_go_past(first_run, go_past)
                before = len(self._data)
                append_measure_datums(
                    self._data,
                    stimulus=stimulus,
                    participant_id=participant,
                    measure_names=list(_MEASURE_NAMES),
                    measure_values=arrays,
                )
                for datum in self._data[before:]:
                    if datum.measure_name == "go_past":
                        datum.metadata["inconsistent_go_past_indices"] = np.flatnonzero(
                            inconsistent
                        ).tolist()

    def exclusion_mask(self, datum: Datum) -> np.ndarray:
        """Exclude source regression paths incompatible with their first runs.

        Args:
            datum: Released observation or a derived first-pass view.
        """
        return super().exclusion_mask(datum) & self.go_past_consistency_mask(datum)

    def go_past_consistency_mask(self, datum: Datum) -> np.ndarray:
        """Apply only the optional duration screen, without position exclusions.

        Args:
            datum: Released observation or a derived first-pass view.
        """
        mask = np.ones(len(datum.values), dtype=bool)
        if self.exclude_inconsistent_go_past and datum.measure_name in {
            "go_past",
            "go_past_in_first_pass",
        }:
            mask[datum.metadata.get("inconsistent_go_past_indices", [])] = False
        return mask

    def fixations(
        self,
        participant: str | None = None,
        stimulus_id: str | None = None,
    ) -> list[Fixation]:
        """Reject event access: RSC's fixation reports cover a different recording set.

        Args:
            participant: Accepted for signature compatibility, never used.
            stimulus_id: Accepted for signature compatibility, never used.

        Raises:
            NotImplementedError: Always; use :meth:`data` for per-word measures.
        """
        raise NotImplementedError(
            "RSC's fixation reports are a separate OSF folder covering a "
            "different recording set and preprocessing windows. They have not "
            "been reconciled to this interest-area release."
        )
