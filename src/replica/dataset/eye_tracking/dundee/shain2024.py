"""Shain et al. (2024) Dundee measures aligned to the original stimuli.

The public dundee_X.csv supplies fixation events and released aggregates.
The author-distributed Dundee V.1 english directory supplies display text,
word objects, and MA1 events for independent alignment. Both are required.
"""

import csv
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation, FixationCalculation
from replica.dataset.metadata import io as metadata_io

from ....core import AnnotationLayer, Datum, Stimulus
from ...loading.fixation_utils import append_fixation
from ...loading.utils import build_word_spans
from .source import DundeeWord as _Word
from .source import normalize_quotes as _normalized
from .source import read_display_words

# SP is one physical fixation; FP/GP/TT are source interval aggregates.
# Native loading preserves them; reaggregation validates the source calculation.
MEASURE_COLUMNS: dict[str, str] = {
    "fdurSP": "first_fixation",
    "fdurGP": "go_past_in_first_pass",
    "fdurTT": "total_time",
    "wdelta": "wdelta",
    "prevwasfix": "prevwasfix",
    "startofsentence": "startofsentence",
    "endofsentence": "endofsentence",
    "startoffile": "startoffile",
    "endoffile": "endoffile",
}
#: ModelBlocks' first-pass duration: the first run of a word reached in first
#: pass, empty where the word had none. It yields ``first_run`` on those words
#: and the ``first_pass`` flag.
FIRST_PASS_COLUMN = "fdurFP"
DUNDEE_MEASURES = [*MEASURE_COLUMNS.values(), "first_run", "first_pass"]
_DURATIONS = frozenset({"fdurSP", "fdurGP", "fdurTT"})
_KEY = ["sentid", "sentpos"]


def _append_interruption(events, label, number, duration):
    """Attach a non-fixation interval to its preceding physical event."""
    if events and (label.startswith("*") or number == 0):
        events[-1][-1].append(
            {
                "duration": float(duration),
                "kind": label if label.startswith("*") else "unassigned",
                "closes_windows": label == "*Off-screen" and number > 1,
            }
        )


class Dundee(EyeTrackingDataset):
    """Released Dundee measures on complete, independently aligned articles.

    Source screen/line conflicts have no reliable word assignment. Events are
    retained with unit_index=None; aggregates involving those events are NaN,
    with their released values retained in the datum's metadata.
    """

    info = DatasetInfo(name="dundee", language="en")
    _declared_measures = (*DUNDEE_MEASURES, "go_past")
    additive_measures = frozenset({"total_time"})

    def load(self, path: str, word_path: str | Path | None = None) -> None:
        """Read the processed CSV and original Dundee reference files.

        Args:
            path: Directory holding Shain's dundee_X.csv.
            word_path: Original english/items directory containing
                textNN.itm and txNNwrdp.dat. Its parent must contain the
                corresponding sXNNma1p.dat event files. Defaults to items
                beside the CSV or the registered original directory.
        """
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        csv_path = Path(path) / "dundee_X.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dundee data not found at {csv_path}. Run: replica fetch dundee"
            )
        if word_path is None:
            candidates = [
                Path(path) / "items",
                Path(path).parent.parent / "original/dundee/english/items",
            ]
            word_path = next((p for p in candidates if p.is_dir()), candidates[0])
        self._load_from_shain_csv(csv_path, Path(word_path))
        self._add_arrival_go_past()

    def _load_from_shain_csv(self, csv_path: Path, word_path: Path) -> None:
        wanted = {
            "word",
            "subject",
            "docid",
            "sentid",
            "sentpos",
            "time",
            "inregression",
            *MEASURE_COLUMNS,
            FIRST_PASS_COLUMN,
        }
        # Numeric NA parsing is separate so the literal word "None" survives.
        frame = pd.read_csv(
            csv_path,
            sep=r"\s+",
            usecols=lambda c: c in wanted,
            keep_default_na=False,
            quoting=csv.QUOTE_NONE,
        )
        required = {"word", "subject", "docid", "sentid", "sentpos", "time", "fdurSP"}
        if missing := required - set(frame):
            raise ValueError(
                f"Dundee release missing required columns: {sorted(missing)}"
            )
        for col in set(frame) - {"word", "subject"}:
            frame[col] = pd.to_numeric(
                frame[col].replace(
                    {"None": np.nan, "NaN": np.nan, "NA": np.nan, "": np.nan}
                ),
                errors="raise",
            )
        identifiers = frame[["docid", "sentid", "sentpos"]]
        if identifiers.isna().any().any() or (identifiers % 1 != 0).any().any():
            raise ValueError("Dundee release has missing or noninteger position IDs")
        self._stimuli = []
        self._data = []
        self._fixations: list[Fixation] = []
        for docid, doc_frame in frame.groupby("docid"):
            docid = int(docid)
            original = self._read_text(word_path, docid)
            aligned = self._align_events(doc_frame, original, word_path.parent, docid)
            groups = self._region_groups(original, docid)
            original_to_region = {
                number: i for i, group in enumerate(groups) for number in group
            }
            key_to_idx = self._source_positions(
                aligned, original, groups, original_to_region, docid
            )
            aligned["_region"] = [
                key_to_idx[(int(sid), int(pos))]
                for sid, pos in aligned[_KEY].itertuples(index=False)
            ]
            boundaries: dict[int, int] = {}
            for (sid, pos), index in key_to_idx.items():
                start = index - pos + 1
                if boundaries.setdefault(sid, start) != start:
                    raise ValueError(
                        f"Dundee text {docid}: inconsistent sentence positions"
                    )
            starts = [boundaries[sid] for sid in sorted(boundaries)]
            if not starts or starts[0] != 0 or starts != sorted(set(starts)):
                raise ValueError(f"Dundee text {docid}: incomplete sentence boundaries")
            regions = [
                " ".join(original[n - 1].text for n in group) for group in groups
            ]
            stimulus = self._make_stimulus(
                regions,
                f"dundee_doc{docid}",
                {
                    "docid": docid,
                    "sentence_boundaries": starts,
                    "source_word_numbers": [g[0] for g in groups],
                    "source_word_groups": groups,
                    "text_source": "Dundee V.1 display files",
                    "measure_source": "Shain et al. (2024) released region aggregates",
                },
            )
            words = [word.text for word in original]
            stimulus.add_layer(
                AnnotationLayer("source_words", build_word_spans(words), words)
            )
            for subject, part in aligned.groupby("subject"):
                ordered = part.sort_values("time").reset_index(drop=True)
                uncertain = self._uncertain_first_pass(
                    ordered, original, original_to_region
                )
                self._append_measures(
                    stimulus, str(subject), ordered, len(groups), uncertain
                )
                self._append_fixations(stimulus, str(subject), ordered)

    _read_text = staticmethod(read_display_words)

    @staticmethod
    def _region_groups(words: list[_Word], docid: int) -> list[list[int]]:
        groups = [[i] for i in range(1, len(words) + 1)]
        # The released aggregates merge this punctuation object into the
        # preceding word. Keep both original positions in one measured span.
        if docid == 18:
            if [_normalized(w.text) for w in words[1168:1170]] != ["years!'", "'"]:
                raise ValueError(
                    "Dundee text 18: expected released quotation region is absent"
                )
            groups[1168:1170] = [[1169, 1170]]
        return groups

    @staticmethod
    def _align_events(
        frame: pd.DataFrame, words: list[_Word], english: Path, docid: int
    ) -> pd.DataFrame:
        trials = []
        for subject, part in frame.groupby("subject"):
            if not re.fullmatch(r"s[a-z]", str(subject)):
                raise ValueError(f"Dundee: invalid source participant {subject!r}")
            path = english / f"{subject}{docid:02d}ma1p.dat"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Dundee original event reference not found: {path}"
                )
            events = []
            elapsed = 0.0
            for line in path.read_text("cp437").splitlines():
                fields = line.rsplit(None, 11)
                if len(fields) != 12:
                    raise ValueError(f"Malformed Dundee event in {path}: {line!r}")
                try:
                    numbers = [int(v) for v in fields[1:]]
                except ValueError:
                    if fields[0].strip() == "WORD":
                        continue
                    raise ValueError(f"Dundee {path.name}: malformed event") from None
                number, duration = numbers[5], numbers[6]
                label = fields[0].strip()
                if label.startswith("*"):
                    if label not in {"*Blink", "*Off-screen"}:
                        raise ValueError(f"Dundee {path.name}: unknown event {label!r}")
                elif number > 0:
                    if number > len(words):
                        raise ValueError(f"Dundee {path.name}: event outside stimulus")
                    word = words[number - 1]
                    valid = (numbers[0], numbers[1]) == (word.screen, word.line)
                    events.append(
                        (
                            number,
                            elapsed / 1000,
                            float(duration),
                            numbers[0],
                            numbers[1],
                            valid,
                            [],
                        )
                    )
                elif number == 0 and not label.strip("\x00"):
                    # Some NUL-padded records have a positive duration but
                    # no word. Their elapsed time is present in the release.
                    pass
                else:
                    raise ValueError(f"Dundee {path.name}: unidentified event")
                if duration < 0:
                    raise ValueError(f"Dundee {path.name}: negative event duration")
                _append_interruption(events, label, number, duration)
                elapsed += duration
            ordered = part.sort_values("time").copy()
            reference = pd.DataFrame(
                events,
                columns=[
                    "_source_number",
                    "_time",
                    "_duration",
                    "_screen",
                    "_line",
                    "_aligned",
                    "_interruptions",
                ],
            )
            if (
                len(reference) != len(ordered)
                or ordered.time.duplicated().any()
                or not np.allclose(reference._time, ordered.time, rtol=0, atol=1e-8)
                or not np.array_equal(reference._duration, ordered.fdurSP)
            ):
                raise ValueError(
                    f"Dundee {path.name}: released events do not match original timing/duration"
                )
            for column in (
                "_source_number",
                "_screen",
                "_line",
                "_aligned",
                "_interruptions",
            ):
                ordered[column] = reference[column].to_numpy()
            trials.append(ordered)
        return pd.concat(trials, ignore_index=True)

    @staticmethod
    def _source_positions(
        frame: pd.DataFrame,
        words: list[_Word],
        groups: list[list[int]],
        original_to_region: dict[int, int],
        docid: int,
    ) -> dict:
        mapping = {}
        for key, events in frame.groupby(_KEY):
            reliable = events[events._aligned]
            candidates = reliable if len(reliable) else events
            indices = {original_to_region[int(n)] for n in candidates._source_number}
            if len(indices) != 1:
                raise ValueError(
                    f"Dundee {docid} {key}: conflicting source word positions"
                )
            index = indices.pop()
            expected = "".join(_normalized(words[n - 1].text) for n in groups[index])
            # Original display text and measured span stay unchanged.
            if docid == 18 and groups[index] == [1169, 1170]:
                expected = "years!'''"
            elif docid == 4 and groups[index] == [2338] and expected == "'":
                expected = "-"
            if any(events.word != expected):
                raise ValueError(
                    f"Dundee {docid} {key}: released word does not match source {expected!r}"
                )
            mapping[tuple(map(int, key))] = index
        return mapping

    @staticmethod
    def _uncertain_first_pass(
        ordered: pd.DataFrame, words: list[_Word], original_to_region: dict[int, int]
    ) -> set[int]:
        """Keep first-pass outcomes unresolved where source geometry conflicts.

        The word number and screen/line give alternative landing positions.
        A possible forward landing also affects the intervening words. Already
        settled first visits retain their earlier first-pass status.
        """
        uncertain: set[int] = set()
        settled: set[int] = set()
        frontier = -1
        for row in ordered.to_dict("records"):
            region = int(row["_region"])
            if row["_aligned"]:
                settled.update(range(frontier + 1, region + 1))
                settled.add(region)
                frontier = max(frontier, region)
            else:
                candidates = {
                    original_to_region[i]
                    for i, word in enumerate(words, 1)
                    if (word.screen, word.line) == (row["_screen"], row["_line"])
                }
                candidates.add(original_to_region[int(row["_source_number"])])
                affected = candidates | set(range(frontier + 1, max(candidates) + 1))
                uncertain.update(affected - settled)
        return uncertain

    def _append_measures(
        self,
        stimulus: Stimulus,
        subject: str,
        ordered: pd.DataFrame,
        n_regions: int,
        uncertain_first_pass: set[int] | None = None,
    ) -> None:
        uncertain_first_pass = uncertain_first_pass or set()
        bad = ~ordered._aligned.to_numpy(dtype=bool)
        bad_times = ordered.time.to_numpy()[bad]
        bad_regions = set(ordered.loc[bad, "_region"])
        previous_bad = np.r_[False, bad[:-1]]
        # Full original streams were matched: positions absent for every
        # reader are known skips too.
        unvisited = list(set(range(n_regions)) - set(ordered._region))
        columns = {**MEASURE_COLUMNS, FIRST_PASS_COLUMN: "first_run"}
        for column, canonical in columns.items():
            if column not in ordered:
                continue
            values = np.full(n_regions, np.nan)
            if column in _DURATIONS or column == FIRST_PASS_COLUMN:
                values[unvisited] = 0.0
            unavailable = []
            rows = ordered.dropna(subset=[column]).drop_duplicates("_region")
            for row_index, region, released, aligned, time in rows[
                ["_region", column, "_aligned", "time"]
            ].itertuples(name=None):
                index, value = int(region), float(released)
                reason = ""
                if not aligned:
                    reason = "source_screen_or_line_mismatch"
                elif column == "fdurTT" and index in bad_regions:
                    reason = "total_includes_unaligned_event"
                elif column in {"fdurFP", "fdurGP"} and np.any(
                    (bad_times >= float(time) - 1e-8)
                    & (bad_times < float(time) + value / 1000 - 1e-8)
                ):
                    reason = "duration_interval_includes_unaligned_event"
                elif column in {"wdelta", "prevwasfix"} and previous_bad[row_index]:
                    reason = "previous_event_unaligned"
                if reason:
                    unavailable.append(
                        {"unit_index": index, "released_value": value, "reason": reason}
                    )
                else:
                    values[index] = value
            metadata = {
                "source_algorithm": "modelblocks@1",
                "upstream_revisions": {
                    "modelblocks-release": "3ae5120327005cde1c02576eda24876997e20550"
                },
            }
            metadata.update(
                {"unavailable_source_values": unavailable} if unavailable else {}
            )
            if column == FIRST_PASS_COLUMN:
                skipped = values == 0
                first_pass = np.where(values > 0, 1.0, np.where(skipped, 0.0, np.nan))
                # A regression on the first event establishes a first-pass skip
                # even when the release has no first-pass duration for the word.
                if "inregression" in ordered:
                    first_events = ordered.drop_duplicates("_region")
                    for row in first_events.to_dict("records"):
                        if row["_aligned"] and row["inregression"] == 1:
                            first_pass[int(row["_region"])] = 0.0
                metadata["uncertain_first_pass_indices"] = sorted(uncertain_first_pass)
                flag_metadata = dict(metadata)
                flag_metadata["uncertain_first_pass_values"] = [
                    {"unit_index": index, "released_value": float(first_pass[index])}
                    for index in sorted(uncertain_first_pass)
                    if np.isfinite(first_pass[index])
                ]
                first_pass[list(uncertain_first_pass)] = np.nan
                metadata["unavailable_first_pass_indices"] = np.flatnonzero(
                    np.isnan(first_pass)
                ).tolist()
                flag_metadata["unavailable_first_pass_indices"] = metadata[
                    "unavailable_first_pass_indices"
                ]
                values = np.where(skipped, np.nan, values)
                values[unvisited] = 0.0
                self._data.append(
                    Datum(
                        stimulus=stimulus,
                        measure_name="first_pass",
                        values=first_pass,
                        participant_id=subject,
                        metadata=flag_metadata,
                    )
                )
            if column in {"fdurFP", "fdurGP"}:
                self._mask_uncertain_first_pass(
                    values, uncertain_first_pass, unavailable
                )
                if unavailable:
                    metadata["unavailable_source_values"] = unavailable
            self._data.append(
                Datum(
                    stimulus=stimulus,
                    measure_name=canonical,
                    values=values,
                    participant_id=subject,
                    metadata=metadata,
                )
            )

    @staticmethod
    def _mask_uncertain_first_pass(values, uncertain, unavailable) -> None:
        """Retain source values before masking uncertain conditional durations."""
        for index in sorted(uncertain):
            if np.isfinite(values[index]):
                unavailable.append(
                    {
                        "unit_index": index,
                        "released_value": float(values[index]),
                        "reason": "first_pass_depends_on_unaligned_event",
                    }
                )
        values[list(uncertain)] = np.nan

    def _append_fixations(
        self, stimulus: Stimulus, subject: str, ordered: pd.DataFrame
    ) -> None:
        offsets = [span.start for span in stimulus.layers["source_words"].spans]
        for index, row in enumerate(ordered.to_dict("records"), 1):
            aligned = bool(row["_aligned"])
            source_number = int(row["_source_number"])
            start, duration = float(row["time"]) * 1000, float(row["fdurSP"])
            metadata = {
                "source_word_number": source_number,
                "following_interruptions": row["_interruptions"],
                "source_screen": int(row["_screen"]),
                "source_line": int(row["_line"]),
                "released_unit_index": int(row["_region"]),
                "alignment_status": "matched"
                if aligned
                else "source_screen_or_line_mismatch",
            }
            if "inregression" in row:
                metadata["in_regression"] = bool(row["inregression"])
            append_fixation(
                self._fixations,
                participant_id=subject,
                stimulus_id=stimulus.stimulus_id,
                start_time=start,
                end_time=start + duration,
                duration=duration,
                x=float("nan"),
                y=float("nan"),
                unit_index=int(row["_region"]) if aligned else None,
                char_offset=offsets[source_number - 1] if aligned else None,
                fixation_index=index,
                metadata=metadata,
            )

    def _reaggregation_arrays(self, events, stimulus, layer, **options):
        """Reproduce ModelBlocks intervals after mapping the requested units."""
        from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer

        native = stimulus.layers[layer].spans == stimulus.layers["words"].spans
        if native:
            events = [
                replace(
                    e, unit_index=e.metadata["released_unit_index"], char_offset=None
                )
                for e in events
            ]
        arrays = aggregate_fixations_to_layer(
            events,
            stimulus,
            layer,
            calculation=FixationCalculation(
                first_run="modelblocks@1",
                go_past="modelblocks@1",
                total_time="modelblocks@1",
            ),
        )
        arrays["_native_units"] = np.asarray([native])
        return arrays

    def _reaggregation_values(self, datum, arrays):
        """Keep the release's conditional availability and alignment exclusions."""
        values = super()._reaggregation_values(datum, arrays)
        if datum.measure_name == "go_past_in_first_pass":
            values[(arrays["first_pass"] == 0) & (arrays["total_time"] > 0)] = np.nan
        if arrays["_native_units"].item():
            for entry in datum.metadata.get("unavailable_source_values", []):
                values[entry["unit_index"]] = np.nan
            values[datum.metadata.get("uncertain_first_pass_indices", [])] = np.nan
            if datum.measure_name in {
                "first_pass",
                "first_run_in_first_pass",
                "first_fixation_in_first_pass",
            }:
                values[datum.metadata.get("unavailable_first_pass_indices", [])] = (
                    np.nan
                )
            for entry in datum.metadata.get("uncertain_first_pass_values", []):
                values[entry["unit_index"]] = np.nan
        return values

    def _calculate_response(self, datum, layer, events, measure, calculation, cache):
        """Reject unresolved Dundee locations before an alternative calculation."""
        if any(e.unit_index is None for e in events):
            raise ValueError(
                "Dundee source screen/line conflicts do not establish target landings"
            )
        return super()._calculate_response(
            datum, layer, events, measure, calculation, cache
        )

    def _reaggregation_provenance(self, datum, layer):
        """Name the interval calculation used for native and custom responses."""
        return {
            "source_algorithm": "modelblocks@1",
            "upstream_revisions": {
                "modelblocks-release": "3ae5120327005cde1c02576eda24876997e20550"
            },
        }
