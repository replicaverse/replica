"""PoTeC: Potsdam Textbook Corpus (Jakobi et al., 2025).

German eye-tracking corpus of textbook reading. 75 native German speakers
reading 12 scientific texts (6 biology, 6 physics), recorded at 1000 Hz
with an EyeLink 1000.

Reference: https://doi.org/10.3758/s13428-024-02536-8
Data: https://osf.io/dn5hp/
"""

import bisect
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.measures import PYMOVEMENTS_MEASURE_NAMES, pymovements_measures
from replica.dataset.metadata import io as metadata_io

from ...core import Stimulus
from ..loading.fixation_utils import (
    SyntheticTimer,
    append_fixation,
    replace_with_fixation_measures,
)
from ..loading.utils import append_measure_datums, parse_reading_measure

TEXT_IDS = ["b0", "b1", "b2", "b3", "b4", "b5", "p0", "p1", "p2", "p3", "p4", "p5"]


class PoTeC(EyeTrackingDataset):
    """PoTeC eye-tracking corpus (Jakobi et al., 2025).

    Complete measures are computed from the corrected character fixations.
    The optional released summaries omit the final fixation in each trial;
    their values remain in datum metadata when complete measures are selected.
    Measures retain the author calculator's integer-millisecond conversion and
    exclusion of off-word events before determining runs.

    Example::

        ds = PoTeC()
        ds.load("datasets/eye-tracking/original/potec")
        data = ds.data(measure="first_run_in_first_pass")
    """

    info = DatasetInfo(name="potec", language="de")
    _declared_measures = (*PYMOVEMENTS_MEASURE_NAMES, "go_past")

    def __init__(self, measurement_source: str = "fixations"):
        """Select complete fixation measures or the original released summaries.

        Args:
            measurement_source: ``"fixations"`` to include all corrected events;
                ``"released"`` to retain the original summary-table values.
        """
        super().__init__()
        if measurement_source not in {"fixations", "released"}:
            raise ValueError("measurement_source must be 'fixations' or 'released'")
        self.measurement_source = measurement_source

    def load(self, path: str) -> None:
        """Load PoTeC data from the downloaded directory.

        Reads ``{path}/reader{N}_{text_id}_merged.tsv``, or a prepared
        ``reading_measures/reader*_rm.tsv`` layout when no merged files are
        present. ``measurement_source="fixations"`` also needs the corrected
        ``reader{N}_{text_id}_fixations.tsv`` files and ``word_limits.json``;
        released mode loads no fixation events.

        Args:
            path: Root directory of the downloaded PoTeC data.
        """

        files = ["reader*_b*_merged.tsv", "reader*_p*_merged.tsv"]
        if self.measurement_source == "fixations":
            files += [
                "reader*_b[0-5]_fixations.tsv",
                "reader*_p[0-5]_fixations.tsv",
                "word_limits.json",
            ]
        self._load_metadata(
            path, files=files, metadata_path=metadata_io.find_dataset_yaml(path)
        )
        self._fixations: list[Fixation] = []
        limits = None
        if self.measurement_source == "fixations":
            limits_file = Path(path) / "word_limits.json"
            if not limits_file.exists():
                raise FileNotFoundError(
                    "PoTeC complete measures require corrected fixations and word_limits.json; run `replica fetch potec`"
                )
            limits = json.loads(limits_file.read_text())

        rm_dir = path
        rm_pattern = re.compile(r"^reader(\d+)_([a-z]\d+)_merged\.tsv$")
        if not any(rm_pattern.match(fname) for fname in os.listdir(path)):
            rm_dir = os.path.join(path, "reading_measures")
            rm_pattern = re.compile(r"^reader(\d+)_([a-z]\d+)_rm\.tsv$")
        if not os.path.isdir(rm_dir):
            raise FileNotFoundError(f"No PoTeC merged reading measures found in {path}")

        # Corrected reading-measure files carry the displayed text. The older
        # AOI geometry files have 15 corrupted labels (e.g. Agarose-Sieb*v).
        stimuli_by_text: dict[str, Stimulus] = {}
        for fname in sorted(os.listdir(rm_dir)):
            match = rm_pattern.match(fname)
            if match is None or match[2] not in TEXT_IDS:
                continue
            reader_id, text_id = match.groups()
            with open(os.path.join(rm_dir, fname), newline="", encoding="utf-8") as f:
                rows = sorted(
                    csv.DictReader(f, delimiter="\t"),
                    key=lambda row: int(row["word_index_in_text"]),
                )
            numbers = [int(row["word_index_in_text"]) for row in rows]
            if not rows or len(set(numbers)) != len(numbers):
                raise ValueError(f"Empty or duplicate word positions in {fname}")
            words = [row["word_with_punct"] for row in rows]
            # The published stimulus says "null". Newer source exports
            # accidentally parse that literal as missing, then fill it with 0.
            # Pin the repair to the verified source text and word position.
            if text_id == "p3" and 140 in numbers:
                index = numbers.index(140)
                if words[index] not in {"null", "0", "nan"}:
                    raise ValueError(f"Unexpected PoTeC p3 word 140: {words[index]!r}")
                words[index] = "null"
            sentences = [int(row["sent_index_in_text"]) for row in rows]
            if text_id not in stimuli_by_text:
                boundaries = [
                    i
                    for i, sid in enumerate(sentences)
                    if i == 0 or sid != sentences[i - 1]
                ]
                stimuli_by_text[text_id] = self._make_stimulus(
                    words,
                    f"potec_{text_id}",
                    {
                        "text_id": text_id,
                        "source_word_numbers": numbers,
                        "sentence_boundaries": boundaries,
                        "word_source": "word_with_punct",
                        "source_sentence_numbers": sentences,
                    },
                )
            stimulus = stimuli_by_text[text_id]
            if (
                words != stimulus.layers["words"].values
                or numbers != stimulus.metadata["source_word_numbers"]
                or sentences != stimulus.metadata["source_sentence_numbers"]
            ):
                raise ValueError(
                    f"Word alignment mismatch for reader {reader_id}, text {text_id}"
                )
            arrays = np.array(
                [
                    pymovements_measures(
                        row.__getitem__,
                        unfixated=parse_reading_measure(row["TFT"]) == 0.0,
                    )
                    for row in rows
                ]
            ).T
            append_measure_datums(
                self._data,
                stimulus=stimulus,
                participant_id=reader_id,
                measure_names=PYMOVEMENTS_MEASURE_NAMES,
                measure_values=list(arrays),
                metadata={"measurement_source": "released"},
            )
            if limits is not None:
                self._load_fixations(
                    Path(path), stimulus, reader_id, text_id, limits[text_id]
                )
        if not stimuli_by_text:
            raise FileNotFoundError(f"No PoTeC reading-measure files found in {rm_dir}")

        self._add_arrival_go_past()
        if self.measurement_source == "fixations":
            replace_with_fixation_measures(
                self._data,
                self._fixations,
                floor_durations=True,
                off_text_ends_run=False,
            )

    def _load_fixations(
        self, path: Path, stimulus: Stimulus, reader: str, text: str, limits: list
    ) -> None:
        """Align the corrected character events to the displayed word spans."""
        source = path / f"reader{reader}_{text}_fixations.tsv"
        with source.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if not rows:
            raise ValueError(f"Empty PoTeC fixation file: {source.name}")
        starts, ends = limits
        if len(starts) != len(stimulus.layers["words"]) or len(ends) != len(starts):
            raise ValueError(f"PoTeC character/word mapping mismatch for {text}")
        if starts != sorted(set(starts)) or any(
            a > b for a, b in zip(starts, ends, strict=True)
        ):
            raise ValueError(f"Invalid PoTeC character intervals for {text}")
        if stimulus.metadata["source_word_numbers"] != list(range(1, len(starts) + 1)):
            raise ValueError(f"Non-contiguous PoTeC word positions for {text}")
        timer = SyntheticTimer()
        for row in sorted(rows, key=lambda row: int(row["fixation_index"])):
            if str(row["reader_id"]) != reader or row["text_id"] != text:
                raise ValueError(f"PoTeC fixation identity mismatch in {source.name}")
            duration = float(row["fixation_duration"])
            if not np.isfinite(duration) or duration < 0:
                raise ValueError(f"Invalid PoTeC fixation duration in {source.name}")
            try:
                aoi = int(row["aoi"])
            except ValueError as error:
                if row["aoi"] not in {"", "NA", "nan"}:
                    raise ValueError(
                        f"Invalid PoTeC AOI in {source.name}: {row['aoi']!r}"
                    ) from error
                aoi = -1
            index = bisect.bisect_right(starts, aoi) - 1
            unit = index if 0 <= index < len(starts) and aoi <= ends[index] else None
            offset = None
            if unit is not None:
                span = stimulus.layers["words"].spans[unit]
                offset = span.start + aoi - starts[unit]
                if offset >= span.end:
                    raise ValueError(
                        f"PoTeC fixation exceeds displayed word in {source.name}"
                    )
            start, end = timer.next(row["trial"], duration)
            append_fixation(
                self._fixations,
                participant_id=reader,
                stimulus_id=stimulus.stimulus_id,
                start_time=start,
                end_time=end,
                duration=duration,
                x=np.nan,
                y=np.nan,
                unit_index=unit,
                char_offset=offset,
                char_offset_is_exact=unit is not None,
                fixation_index=int(row["fixation_index"]),
                metadata={
                    "trial_index": row["trial"],
                    "synthetic_start_time": True,
                    "source_file": source.name,
                    "source_aoi": aoi,
                },
            )
