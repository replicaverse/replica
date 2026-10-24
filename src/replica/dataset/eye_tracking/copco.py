"""CopCo: Copenhagen Corpus of eye-tracking recordings (Hollenstein et al., 2022).

Danish eye-tracking corpus (natural reading; L1, L1-dyslexia, and L2
readers). Per-word measures extracted from interest area reports.

Reference: https://aclanthology.org/2022.lrec-1.182
"""

import ast
import csv
import glob
import os
from dataclasses import replace

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation, FixationCalculation
from replica.dataset.metadata import io as metadata_io

from ...core import Datum, Stimulus
from ..loading.fixation_utils import (
    RunIdInferrer,
    SyntheticTimer,
    append_fixation,
    copco_extractor_go_past,
    first_pass_units,
    first_run_durations,
)
from ..loading.utils import (
    append_measure_datums,
    parse_float,
    parse_reading_measure,
    scatter_measure_tuples,
    set_optional_meta,
)

# Column name in CSV → measure name used in Datum
MEASURE_COLUMNS = {
    "word_first_fix_dur": "first_fixation",
    "word_first_pass_dur": "first_run",
    "word_go_past_time": "go_past",
    "word_total_fix_dur": "total_time",
}


def _fixation_rows(file):
    """Read EyeLink fields whose outer quotes do not escape quote labels."""
    rows = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)

    def unquote(value):
        return (
            value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value
        )

    header = [unquote(value) for value in next(rows, [])]
    required = {
        "RECORDING_SESSION_LABEL",
        "paragraphid",
        "speechid",
        "TRIAL_INDEX",
        "CURRENT_FIX_DURATION",
        "CURRENT_FIX_INDEX",
    }
    if not required.issubset(header):
        raise ValueError(
            f"CopCo fixation report missing columns: {sorted(required - set(header))}"
        )
    for line, values in enumerate(rows, 2):
        if len(values) != len(header):
            raise ValueError(
                f"CopCo fixation report has {len(values)} fields at line {line}; expected {len(header)}"
            )
        yield dict(zip(header, (unquote(value) for value in values), strict=True))


def _missing_characters(word: str, char_ids: list[int]) -> list[tuple[int, int, str]]:
    """Infer omitted non-ASCII character IDs only from consistent ASCII anchors.

    The event label must additionally match the inferred character at lookup.
    Releases that already include non-ASCII IDs need no reconstruction.
    """
    anchors = [i for i, c in enumerate(word) if ord(c) < 128 and not c.isspace()]
    if not anchors or len(anchors) != len(char_ids):
        return []
    origins = {cid - i for cid, i in zip(char_ids, anchors, strict=True)}
    if len(origins) != 1:
        return []
    start = origins.pop()
    result = []
    part, in_word = -1, False
    for i, char in enumerate(word):
        if char.isspace():
            in_word = False
            continue
        if not in_word:
            part, in_word = part + 1, True
        if ord(char) >= 128:
            result.append((start + i, part, char))
    return result


class CopCo(EyeTrackingDataset):
    """CopCo eye-tracking corpus (Hollenstein et al., 2022).

    Danish eye-tracking data. Each paragraph is one Stimulus; per-word
    measures include first fixation duration, first run, go-past time, and
    total fixation time. ``go_past`` uses the full fixation-derived regression
    path; ``source="released"`` retains the released extractor's shorter window.
    The standard path requires raw reports and stays unavailable without them.
    Known skips are zero when ``drop_skips=False``; source blanks remain blank
    in the released view. The authors' feature extractor sums the first run of
    fixations on a word whenever it occurs and records no first-pass flag.
    The loader derives the ``first_pass`` flag from the fixation reports
    under the extractor's own assignment rules, for every trial whose rebuilt
    stream reproduces the released first run. The release joins some
    multiword interest areas with a nonbreaking space. Each stays one measured
    region, with natural whitespace and one span covering all its words.

    Usage::

        ds = CopCo()
        ds.load("datasets/eye-tracking/original/copco")
        data = ds.data(measure="first_run_in_first_pass")
    """

    info = DatasetInfo(name="copco", language="da")
    _declared_measures = (
        *MEASURE_COLUMNS.values(),
        "first_pass",
        "go_past",
        "go_past_in_first_pass",
    )

    @staticmethod
    def _input_files(path: str) -> list[str]:
        """Declared files a load reads; the fixation reports only when present."""
        files = ["ExtractedFeatures/P*.csv"]
        if os.path.isdir(os.path.join(path, "FixationReports")):
            files.append("FixationReports/FIX_report_P*.txt")
        return files

    def load(self, path: str) -> None:
        """Load CopCo data from the dataset directory.

        Reads the 57 per-participant ``ExtractedFeatures/P*.csv`` files under
        *path* and, when present, verifies and reads ``FixationReports/``. One
        Stimulus is one paragraph, which Hollenstein et al. (2022) show on one
        screen, with its sentences joined in ``(sentenceId, wordId)`` order.

        Args:
            path: Root directory of the downloaded CopCo data.
        """

        self._load_metadata(
            path,
            files=self._input_files(path),
            metadata_path=metadata_io.find_dataset_yaml(
                "eye-tracking/original/copco", required=True
            ),
        )

        features_dir = os.path.join(path, "ExtractedFeatures")
        csv_files = sorted(glob.glob(os.path.join(features_dir, "P*.csv")))

        if not csv_files:
            raise FileNotFoundError(
                f"No P*.csv files found in {features_dir}.\n"
                "Download the data from https://copco.io/"
            )

        # (speechId, paragraphId) → {(sentenceId, wordId, part) → word_str}
        # ``paragraphId`` is local to each speech (every speech starts its
        # paragraphs at 0/1/...), so it must be combined with ``speechId`` to
        # identify a paragraph. Keying on ``paragraphId`` alone would merge
        # paragraphs from different speeches into one stimulus.
        paragraph_words: dict[tuple[int, int], dict[tuple[int, int, int], str]] = {}
        # (speechId, paragraphId) → participant →
        #   list of ((sentenceId, wordId, part), m_tuple)
        paragraph_part_data: dict[
            tuple[int, int],
            dict[str, list[tuple[tuple[int, int, int], tuple[float, ...]]]],
        ] = {}
        # (participant, trialId, char_IA_index) →
        #   (speechId, paragraphId, sentenceId, wordId, part).
        # CopCo's fixation reports use character-level interest areas (one
        # IA per character of a paragraph). The mapping is keyed on trialId
        # which is unique per paragraph-presentation, TRIAL_INDEX in the
        # fixation file matches trialId in ExtractedFeatures.
        char_to_word: dict[tuple[str, int, int], tuple[int, int, int, int, int]] = {}
        self._inferred_characters: dict = {}

        for csv_file in csv_files:
            # Extract participant ID from filename (e.g. "P01" from "P01.csv")
            participant = os.path.splitext(os.path.basename(csv_file))[0]

            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        speech_id = int(row["speechId"])
                        paragraph_id = int(row["paragraphId"])
                    except (KeyError, ValueError):
                        raise ValueError(
                            f"Invalid CopCo speech or paragraph ID in {csv_file}: {row!r}"
                        ) from None
                    sentence_id = int(row["sentenceId"])
                    word_id = int(row["wordId"])
                    # Preserve the source measured region. Character IDs from
                    # every constituent word map to this same response unit.
                    raw_word = str(row["word"]).strip()
                    word_str = " ".join(raw_word.split())
                    if not word_str:
                        raise ValueError(f"Empty CopCo area in {csv_file}")
                    stim_key = (speech_id, paragraph_id)
                    words_dict = paragraph_words.setdefault(stim_key, {})
                    area_key = (sentence_id, word_id, 0)
                    if words_dict.setdefault(area_key, word_str) != word_str:
                        raise ValueError(
                            f"Conflicting CopCo labels at {stim_key}, {area_key}"
                        )

                    # Char-level IA → word mapping for the fixation parser.
                    trial_id = -1
                    try:
                        trial_id = int(row["trialId"])
                        char_ids_raw = row.get("char_IA_ids", "")
                        char_ids = (
                            ast.literal_eval(char_ids_raw) if char_ids_raw else []
                        )
                    except (KeyError, ValueError, SyntaxError):
                        char_ids = []
                    for cid in char_ids:
                        ck = (participant, trial_id, int(cid))
                        char_to_word.setdefault(
                            ck, (speech_id, paragraph_id, sentence_id, word_id, 0)
                        )
                    for cid, _, char in _missing_characters(raw_word, char_ids):
                        self._inferred_characters[(participant, trial_id, cid)] = (
                            (speech_id, paragraph_id, sentence_id, word_id, 0),
                            char,
                        )

                    m_tuple = tuple(
                        parse_reading_measure(
                            row[csv_col],
                            unfixated=parse_float(row["word_total_fix_dur"]) == 0.0,
                        )
                        for csv_col in MEASURE_COLUMNS
                    )
                    part_dict = paragraph_part_data.setdefault(stim_key, {})
                    records = part_dict.setdefault(participant, [])
                    records.append((area_key, m_tuple))

        # Build Stimulus and Datum objects, sorted by (speech_id, paragraph_id)
        stim_by_paragraph: dict[tuple[int, int], Stimulus] = {}
        # (speech_id, paragraph_id) → {(sentenceId, wordId, part) → flat word index}
        wid_to_idx_by_paragraph: dict[
            tuple[int, int], dict[tuple[int, int, int], int]
        ] = {}
        for stim_key in sorted(paragraph_words.keys()):
            speech_id, paragraph_id = stim_key
            words_dict = paragraph_words[stim_key]
            sorted_keys = sorted(words_dict.keys())  # (sentence_id, word_id, part)
            words = [words_dict[k] for k in sorted_keys]
            # sentence boundaries: word indices where a new sentenceId starts
            sentence_boundaries: list[int] = []
            prev_sid: int | None = None
            for i, (sid, _, _) in enumerate(sorted_keys):
                if sid != prev_sid:
                    sentence_boundaries.append(i)
                    prev_sid = sid
            stimulus = self._make_stimulus(
                words,
                f"copco_s{speech_id}_p{paragraph_id}",
                {
                    "speech_id": speech_id,
                    "paragraph_id": paragraph_id,
                    "language": "da",
                    "sentence_boundaries": sentence_boundaries,
                    "source_word_ids": [(sid, wid) for sid, wid, _ in sorted_keys],
                },
            )
            stim_by_paragraph[stim_key] = stimulus
            n_words = len(words)
            wid_to_idx = {k: i for i, k in enumerate(sorted_keys)}
            wid_to_idx_by_paragraph[stim_key] = wid_to_idx

            part_dict = paragraph_part_data.get(stim_key, {})
            measure_names = tuple(MEASURE_COLUMNS.values())
            for participant in sorted(part_dict.keys()):
                wmeas_list = part_dict[participant]
                measure_vals = scatter_measure_tuples(
                    wmeas_list, wid_to_idx, n_words, len(measure_names)
                )

                append_measure_datums(
                    self._data,
                    stimulus=stimulus,
                    participant_id=participant,
                    measure_names=measure_names,
                    measure_values=measure_vals,
                    metadata={"unfixated_is_missing": True},
                )

        # ---- Step 2: load raw fixations ----
        self._fixations: list[Fixation] = []
        fix_dir = os.path.join(path, "FixationReports")
        if os.path.isdir(fix_dir):
            self._load_fixations(
                fix_dir,
                char_to_word,
                stim_by_paragraph,
                wid_to_idx_by_paragraph,
            )

        self._add_standard_go_past()

    def _reaggregation_arrays(self, events, stimulus, layer, **options):
        """Calculate standard windows and CopCo's separate released quantity."""
        from replica.dataset.loading.reaggregation import map_fixations_to_layer

        arrays = super()._reaggregation_arrays(events, stimulus, layer, **options)
        mapped = map_fixations_to_layer(events, stimulus, layer)
        arrays["_released_go_past"] = copco_extractor_go_past(
            mapped, len(stimulus.layers[layer])
        )
        return arrays

    def _reaggregation_values(self, datum, arrays):
        """Select the released extractor only for its retained source response."""
        if (
            datum.metadata.get("response_source") == "released"
            and datum.measure_name == "go_past"
        ):
            arrays = {**arrays, "go_past": arrays["_released_go_past"]}
        return super()._reaggregation_values(datum, arrays)

    def _add_standard_go_past(self):
        """Add standard regression paths after validating the released source stream."""
        trials = dict(self.fixation_trials())
        references = {
            measure: self._native_responses(measure)
            for measure in ("first_fixation", "first_run", "total_time")
        }
        self._released_data = [d for d in self._data if d.measure_name == "go_past"]
        self._data = [d for d in self._data if d.measure_name != "go_past"]
        for released in self._released_data:
            released.metadata = {
                **released.metadata,
                "response_source": "released",
                "source_field": "word_go_past_time",
                "source_algorithm": "copco_extractor@1",
                "upstream_revisions": FixationCalculation(
                    go_past="copco_extractor@1"
                ).to_dict()["upstream_revisions"],
            }
            if released.measure_name != "go_past":
                continue
            pair = (released.participant_id, released.stimulus.stimulus_id)
            events = self._reaggregation_fixations(released, trials.get(pair, []))
            metadata = {
                "measurement_source": "fixations",
                "calculation": FixationCalculation(
                    include_inter_unit_fixations=False
                ).to_dict(),
            }
            if not events:
                arrays = {
                    name: np.full(len(released.values), np.nan)
                    for name in ("go_past", "go_past_in_first_pass")
                }
                metadata["unavailable_reason"] = "source_fixations_absent"
            else:
                arrays = self._reaggregation_arrays(
                    events, released.stimulus, released.layer, use_source_runs=False
                )
                expected = {"go_past": released}
                for name, lookup in references.items():
                    key = (
                        released.stimulus.stimulus_id,
                        released.participant_id,
                        released.layer,
                        name,
                    )
                    expected[name] = lookup[key]
                for name, reference in expected.items():
                    if not np.allclose(
                        self._reaggregation_values(reference, arrays),
                        reference.values,
                        rtol=0,
                        atol=1e-6,
                        equal_nan=True,
                    ):
                        raise ValueError(
                            f"CopCo source events do not reproduce {name} for {pair}"
                        )
            for name in ("go_past", "go_past_in_first_pass"):
                self._data.append(
                    replace(
                        released,
                        measure_name=name,
                        values=arrays[name].copy(),
                        metadata=dict(metadata),
                    )
                )

    # Fixation-level loading. CopCo's EyeLink fixation reports
    # (FixationReports/FIX_report_P{NN}.txt) use character-level IAs:
    # ``CURRENT_FIX_INTEREST_AREA_INDEX`` indexes characters.
    # The ``char_to_word`` map built above (from ExtractedFeatures'
    # ``char_IA_ids``) translates those to word ids.
    #
    # Source columns used:
    #   RECORDING_SESSION_LABEL                participant ID
    #   paragraphid                            CopCo paragraph identifier
    #   TRIAL_INDEX                            trial within session
    #   CURRENT_FIX_INDEX                      1-based fixation ordinal
    #   CURRENT_FIX_DURATION                   ms
    #   CURRENT_FIX_X / _Y                     pixel coords
    #   CURRENT_FIX_INTEREST_AREA_INDEX        char-level IA, "." if off-text
    #   CURRENT_FIX_INTEREST_AREA_DWELL_TIME   EyeLink's per-char dwell
    def _resolve_character(self, key, label, char_to_word):
        mapping = char_to_word.get(key)
        if mapping is not None:
            return mapping, False
        inferred = getattr(self, "_inferred_characters", {}).get(key)
        if inferred is None:
            return None, False
        # Some UTF-8 reports contain a second encoding of the original label.
        try:
            label = label.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        mapping, expected = inferred
        return (mapping, True) if label == expected else (None, False)

    def _load_fixations(
        self,
        fix_dir: str,
        char_to_word: dict[tuple[str, int, int], tuple[int, int, int, int, int]],
        stim_by_paragraph: dict[tuple[int, int], Stimulus],
        wid_to_idx_by_paragraph: dict[tuple[int, int], dict[tuple[int, int, int], int]],
    ) -> None:

        run_ids = RunIdInferrer()
        timer = SyntheticTimer()
        # (participant, trial) -> [(paragraph key, unit under the authors' rules, ms)]
        author_streams: dict[
            tuple[str, int], list[tuple[tuple[int, int] | None, int | None, float]]
        ] = {}

        for fix_path in sorted(glob.glob(os.path.join(fix_dir, "FIX_report_P*.txt"))):
            # The fixation reports start with a UTF-8 BOM that pollutes
            # the first column header (``RECORDING_SESSION_LABEL``).
            # ``utf-8-sig`` strips the BOM during reading.
            with open(fix_path, encoding="utf-8-sig") as f:
                reader = _fixation_rows(f)
                for row in reader:
                    participant = row.get("RECORDING_SESSION_LABEL", "").strip()
                    if row["speechid"] == row["paragraphid"] == "UNDEFINEDnull":
                        continue
                    try:
                        trial_index = int(row["TRIAL_INDEX"])
                        fix_paragraph_id = int(row["paragraphid"])
                        speech_id = int(row["speechid"])
                        dur = float(row["CURRENT_FIX_DURATION"])
                        fix_idx = int(row["CURRENT_FIX_INDEX"])
                    except (KeyError, ValueError, TypeError):
                        raise ValueError(
                            f"Invalid CopCo fixation identity or duration in {fix_path}: {row!r}"
                        ) from None
                    if not participant or not np.isfinite(dur) or dur < 0:
                        raise ValueError(
                            f"Invalid CopCo participant or duration in {fix_path}: {row!r}"
                        )
                    # The release uses -1 for calibration and 1327 for practice.
                    if fix_paragraph_id < 0 or speech_id == 1327:
                        continue

                    # Resolve char-IA → (speech, paragraph, sentence, word).
                    ia_raw = row.get("CURRENT_FIX_INTEREST_AREA_INDEX", ".")
                    stim_key: tuple[int, int] | None = None
                    sentence_id: int | None = None
                    unit_index: int | None = None
                    char_offset: int | None = None
                    inferred = False
                    if ia_raw not in (".", "", "NA"):
                        try:
                            char_ia = int(ia_raw)
                        except ValueError:
                            char_ia = None
                        if char_ia is not None:
                            mapping, inferred = self._resolve_character(
                                (participant, trial_index, char_ia),
                                row.get("CURRENT_FIX_INTEREST_AREA_LABEL", ""),
                                char_to_word,
                            )
                            if mapping is not None:
                                speech, pid, sid, wid, part = mapping
                                key = (speech, pid)
                                stim = stim_by_paragraph.get(key)
                                if stim is not None:
                                    idx = wid_to_idx_by_paragraph.get(key, {}).get(
                                        (sid, wid, part)
                                    )
                                    if idx is not None:
                                        stim_key = key
                                        sentence_id = sid
                                        unit_index = idx
                                        char_offset = (
                                            stim.layers["words"].spans[idx].start
                                        )

                    # The feature extractor reads a period label as off-text
                    # and has no ids for the characters recovered by inference.
                    author_unit = (
                        unit_index
                        if stim_key is not None
                        and not inferred
                        and row.get("CURRENT_FIX_INTEREST_AREA_LABEL", "") != "."
                        else None
                    )
                    author_streams.setdefault((participant, trial_index), []).append(
                        (stim_key, author_unit, dur)
                    )

                    if stim_key is None:
                        stim_key = (speech_id, fix_paragraph_id)
                        if stim_key not in stim_by_paragraph:
                            raise ValueError(
                                f"CopCo fixation has no stimulus for paragraph {stim_key}"
                            )
                    elif stim_key != (speech_id, fix_paragraph_id):
                        raise ValueError(
                            "CopCo character mapping disagrees with source paragraph"
                        )
                    stim = stim_by_paragraph[stim_key]
                    paragraph_id = stim_key[1]

                    x = parse_float(row.get("CURRENT_FIX_X", "").replace(",", "."))
                    y = parse_float(row.get("CURRENT_FIX_Y", "").replace(",", "."))

                    trial_key = (participant, stim_key, trial_index)
                    start, end = timer.next(trial_key, dur)
                    run_id = run_ids.next(trial_key, unit_index)

                    meta: dict = {
                        "trial_index": trial_index,
                        "speech_id": stim_key[0],
                        "paragraph_id": paragraph_id,
                        "sentence_id": sentence_id,
                        **({"run_id": run_id} if run_id is not None else {}),
                        "synthetic_start_time": True,
                        "inferred_character_mapping": inferred,
                        "source_ia_index": ia_raw,
                        "source_ia_label": row.get(
                            "CURRENT_FIX_INTEREST_AREA_LABEL", ""
                        ),
                    }
                    # EyeLink IA dwell, per-character in CopCo, but useful
                    # for comparison against the source.
                    set_optional_meta(
                        meta,
                        "char_ia_dwell",
                        row.get("CURRENT_FIX_INTEREST_AREA_DWELL_TIME", ""),
                        float,
                    )

                    append_fixation(
                        self._fixations,
                        participant_id=participant,
                        stimulus_id=stim.stimulus_id,
                        start_time=start,
                        end_time=end,
                        duration=dur,
                        x=x,
                        y=y,
                        unit_index=unit_index,
                        char_offset=char_offset,
                        fixation_index=fix_idx,
                        metadata=meta,
                    )

        self._derive_first_pass(author_streams, stim_by_paragraph)

    def _derive_first_pass(
        self,
        author_streams: dict[
            tuple[str, int], list[tuple[tuple[int, int] | None, int | None, float]]
        ],
        stim_by_paragraph: dict[tuple[int, int], Stimulus],
    ) -> None:
        """The first-pass flag from the fixation order, checked against the release.

        The stream of one trial is rebuilt as the feature extractor assigns
        fixations: a fixation counts for a word when its character id belongs
        to the word and its label is not a period, and any other fixation ends
        a run. Where that stream reproduces the released first run on every
        fixated word, a fixated word is in first pass unless the eye had
        already passed it, and a never-fixated word is not. Trials the stream
        does not reproduce leave the flag unavailable.

        Args:
            author_streams: Per participant and trial, each fixation's paragraph,
                its word under the extractor's rules, and its duration.
            stim_by_paragraph: Paragraph key to its stimulus.
        """
        first_runs = {
            (datum.participant_id, datum.stimulus.stimulus_id): datum
            for datum in self._data
            if datum.measure_name == "first_run"
        }
        for (participant, _trial), rows in author_streams.items():
            keys = {key for key, _, _ in rows if key is not None}
            if len(keys) != 1:
                continue
            stimulus = stim_by_paragraph[keys.pop()]
            datum = first_runs.get((participant, stimulus.stimulus_id))
            if datum is None:
                continue
            units = [unit for _, unit, _ in rows]
            n_words = len(datum.values)
            derived = first_run_durations(
                units,
                [duration for _, _, duration in rows],
                {index: index for index in range(n_words)},
                n_words,
                off_text_ends_run=True,
            )
            derived[np.isnan(derived)] = 0.0
            released = datum.values
            fixated = np.isfinite(released) & (released > 0)
            reproduced = bool(
                np.allclose(derived[fixated], released[fixated], rtol=0, atol=0.5)
            )
            first_pass = np.full(n_words, np.nan)
            if reproduced:
                eligible = first_pass_units(units)
                first_pass[np.isfinite(released)] = 0.0
                for index in np.flatnonzero(fixated):
                    if int(index) in eligible:
                        first_pass[index] = 1.0
            append_measure_datums(
                self._data,
                stimulus=stimulus,
                participant_id=participant,
                measure_names=("first_pass",),
                measure_values=(first_pass,),
                metadata={
                    "first_run_reproduced": reproduced,
                    "unfixated_is_missing": True,
                },
            )

    def _reaggregation_fixations(
        self, datum: Datum, events: list[Fixation]
    ) -> list[Fixation]:
        """Apply the author extractor's character exclusions to the event stream.

        Excluded landings still interrupt runs. Raw access retains their
        recovered word assignments; summary reaggregation uses only released
        character IDs and treats period labels as off-text.
        """
        result = []
        for event in super()._reaggregation_fixations(datum, events):
            metadata = {k: v for k, v in event.metadata.items() if k != "run_id"}
            if metadata.get("inferred_character_mapping") or (
                metadata.get("source_ia_label") == "."
            ):
                event = replace(event, unit_index=None, char_offset=None)
            result.append(replace(event, metadata=metadata))
        return result

    # ``fixations()`` inherits from EyeTrackingDataset.
    # ``unit_index`` and ``char_offset`` align to the ``"words"`` layer.
    # CopCo's source IAs are character-level; each fixation is resolved to
    # the containing word via ExtractedFeatures' ``char_IA_ids`` mapping.
    # Fixations outside those words retain their duration with no assigned unit.
