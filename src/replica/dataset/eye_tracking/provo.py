"""PROVO Corpus dataset.

The Provo Corpus: A large eye-tracking corpus with predictability norms.
Luke & Christianson (2018). Behavior Research Methods.

The corpus includes:
- Eye-tracking data (first fixation, gaze duration, total reading time, etc.)
- Human cloze probabilities for each word position

Cloze probabilities are stored as a Stimulus layer named "human_cloze"
so that ClozeLM can read them directly.

Reference: https://osf.io/sjefs/
"""

import csv
import os

import numpy as np

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.measures import EYELINK_MEASURE_NAMES, eyelink_measures
from replica.dataset.metadata import io as metadata_io

from ...core import AnnotationLayer, Datum
from ..loading.fixation_utils import append_fixation
from ..loading.utils import parse_float, set_optional_meta

# The cloze norms and the fixation report are Mac OS Roman. Their only
# non-ASCII bytes are 0xd5 and 0xd1, which decode there as U+2019 (right single
# quote) and U+2014 (em dash) but as "Õ" and "Ñ" under latin-1, corrupting
# words like "doesn’t" into "doesnÕt". Neither read raises under latin-1, so a
# wrong encoding shows up only as corrupted text.
CLOZE_ENCODING = "mac_roman"


class PROVO(EyeTrackingDataset):
    """Eye-tracking corpus with human cloze norms (English).

    When norms are available, the ``human_cloze`` stimulus layer stores per-word
    observed response proportions.

    Raw fixations are exposed via :meth:`fixations` when
    ``Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv``
    is present.
    """

    info = DatasetInfo(name="PROVO", language="en")
    _declared_measures = EYELINK_MEASURE_NAMES

    def __init__(self, first_pass_filter: bool = False):
        """Create the loader.

        Args:
            first_pass_filter: Keep observations whose first fixation advances
                beyond every word position fixated before it, excluding forward
                jumps exceeding four aligned word positions. Add ``wdelta``
                (position difference between retained first-pass fixations,
                before applying the jump limit) and ``prevwasfix`` (whether
                that difference is one). The first retained fixation has NaN
                controls and is exempt from the jump limit. Excluded duration
                observations become NaN; retained durations and stimulus text
                are unchanged. Defaults to ``False``. This uses the same
                ModelBlocks-based observation filter as GECO.
        """
        super().__init__()
        self.first_pass_filter = bool(first_pass_filter)

    def load(self, path: str) -> None:
        """Load PROVO data from the OSF directory layout.

        Expected files (as distributed on OSF):
            {path}/Provo_Corpus-Eyetracking_Data.csv             (per-word IA)
            {path}/Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv
                (optional; per-fixation EyeLink Data Viewer export, populates
                ``ds.fixations()`` when present)
            {path}/Provo_Corpus-Predictability_Norms.csv         (optional, cloze)

        If cloze data is absent, the ``human_cloze`` layer is omitted.
        If the fixation report is absent, ``ds.fixations()`` returns ``[]``.
        Checksums are verified for the optional files that are present.

        Args:
            path: Root directory of the downloaded PROVO corpus.
        """
        et_file = os.path.join(path, "Provo_Corpus-Eyetracking_Data.csv")
        cloze_file = os.path.join(path, "Provo_Corpus-Predictability_Norms.csv")
        fix_file = os.path.join(
            path,
            "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv",
        )
        used_files = [et_file] + [
            name for name in (cloze_file, fix_file) if os.path.exists(name)
        ]
        self._load_metadata(
            path,
            files=[os.path.basename(name) for name in used_files],
            metadata_path=metadata_io.find_dataset_yaml(path),
        )

        if not os.path.exists(et_file):
            raise FileNotFoundError(
                f"Eye-tracking data not found: {et_file}\n"
                "Download the PROVO corpus from https://osf.io/sjefs/"
            )

        # Load the canonical passage text first (cloze norms file). The IA
        # file's per-word ``Word`` column has known release-level corruptions
        # (e.g. text 25 word 45 "90%" stored as "0.9"; text 18 word 3 has a
        # misnumbered final 'evolution' row). The cloze file's ``Text``
        # column preserves the verbatim passage as displayed to participants,
        # so it is the source of truth for the word list.
        passages: dict[int, list[str]] = {}
        if os.path.exists(cloze_file):
            passages = self._load_passages(cloze_file)

        self._load_et(et_file, passages=passages)

        if os.path.exists(cloze_file):
            self._load_cloze(cloze_file)

        # Eager-load the fixation report into a list so ``fixations()`` is a
        # filter without re-parsing.
        self._fixations: list[Fixation] = []
        if os.path.exists(fix_file):
            self._load_fixations(fix_file)

    def _load_passages(self, cloze_file: str) -> dict[int, list[str]]:
        """Read the canonical passage text per text_id from the cloze file.

        The cloze file's ``Text`` column preserves the verbatim passage as
        displayed to participants, including the leading word. Its cloze
        metadata is absent, but IA_ID=1 has a label and measured durations.
        This is the source of truth for the word list. The IA
        file's per-word ``Word``/``Word_Cleaned`` columns carry mangled and
        duplicated labels at a few positions; ``Text`` does not.
        """

        passages: dict[int, str] = {}
        with open(cloze_file, newline="", encoding=CLOZE_ENCODING) as f:
            for row in csv.DictReader(f):
                text_id_str = row.get("Text_ID", "")
                if text_id_str in ("", "NA", "."):
                    continue
                text_id = int(text_id_str)
                t = row.get("Text", "")
                if text_id in passages:
                    if passages[text_id] != t and not (
                        text_id == 27
                        and passages[text_id].replace("’", "'") == t.replace("’", "'")
                    ):
                        raise ValueError(f"Conflicting Provo passage: {text_id}")
                else:
                    passages[text_id] = t

        result: dict[int, list[str]] = {}
        for text_id, text in passages.items():
            if text_id == 55:
                # Both EyeLink reports contain these whole measured areas.
                # The cloze materials instead use comma-separated words.
                for cloze, display in [
                    ("livres, a", "livres--a"),
                    ("profession, writing.", "profession--writing."),
                ]:
                    text = text.replace(cloze, display)
            tokens = text.split()
            if len(tokens) < 2:
                continue
            result[text_id] = tokens
        return result

    def _load_et(
        self,
        et_file: str,
        passages: dict[int, list[str]] | None = None,
    ) -> None:
        """Parse the eye-tracking CSV and populate _stimuli and _data.

        When ``passages`` is provided (from :meth:`_load_passages`), the
        word list is taken from the canonical passage text and the
        IA file's per-word RT rows are mapped to word positions via
        label alignment against the passage. Without passages, falls back
        to the IA file's ``Word`` column (only used when the cloze norms
        file is absent).
        """

        if passages is None:
            passages = {}

        # text_id → participant_id → list of (source area ID, label, measures)
        text_participant_data: dict[
            int, dict[str, list[tuple[int, str, dict[str, float]]]]
        ] = {}
        # Source area IDs distinguish measured rows even when Word_Number
        # metadata is duplicated. Exports without IA_ID use Word_Number.
        text_ia_words: dict[int, list[tuple[int, str]]] = {}
        seen_wn: dict[int, set[int]] = {}

        with open(et_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_id_str = row.get("Text_ID", row.get("ARTICLE_ID", ""))
                wn_str = row.get("Word_Number", row.get("IA_ID", ""))
                area_id = row.get("IA_ID", "")
                initial_area = area_id == "1" and wn_str in ("", "NA", ".")
                if text_id_str in ("", "NA", ".") or (
                    wn_str in ("", "NA", ".") and area_id in ("", "NA", ".")
                ):
                    continue
                text_id = int(text_id_str)
                # As in ModelBlocks' fixation-index alignment, use the measured
                # area identity rather than treating Word_Number as a row key.
                # The official text-18 "evolution" row has Word_Number=3 but
                # IA_ID=51. Area IDs are not passage offsets: gaps/compounds
                # elsewhere still require the label alignment below. Areas
                # without cloze metadata still retain their measured durations.
                wn = int(area_id if area_id not in ("", "NA", ".") else wn_str)
                participant = row.get(
                    "RECORDING_SESSION_LABEL", row.get("Participant_ID", "")
                )
                word_str = row.get("IA_LABEL", "").strip() or row.get(
                    "Word_Cleaned", row.get("Word", "")
                )
                if (text_id, wn, word_str) == (36, 24, "?"):
                    # The Mac-Roman fixation report and cloze Text agree:
                    # this is an em dash, lost in the summary CSV encoding.
                    word_str = "—"
                if initial_area:
                    word_str = self._initial_label(row, passages.get(text_id))
                # The same source identity keys text and measurements.
                seen_for_text = seen_wn.setdefault(text_id, set())
                ia_words_list = text_ia_words.setdefault(text_id, [])
                if wn not in seen_for_text:
                    seen_for_text.add(wn)
                    ia_words_list.append((wn, word_str))
                elif dict(ia_words_list)[wn] != word_str:
                    raise ValueError(f"Conflicting Provo area label: {text_id}/{wn}")

                part_data = text_participant_data.setdefault(text_id, {})
                word_list = part_data.setdefault(participant, [])
                unfixated = parse_float(row.get("IA_DWELL_TIME")) == 0.0
                measure_vals: dict[str, float] = dict(
                    zip(
                        EYELINK_MEASURE_NAMES,
                        eyelink_measures(row.get, unfixated=unfixated),
                        strict=True,
                    )
                )
                measure_vals["__first_fixation_time"] = parse_float(
                    row.get("IA_FIRST_FIXATION_TIME", "")
                )
                measure_vals["__trial_index"] = parse_float(row.get("TRIAL_INDEX"))
                word_list.append((wn, word_str, measure_vals))

        # Build stimuli + per-word RT arrays

        for text_id in sorted(text_participant_data.keys()):
            # Pick the word list: passage if available, else fall back
            # to the IA file's deduped words.
            if passages.get(text_id):
                words = list(passages[text_id])
                source = "passage"
            else:
                ia_sorted = sorted(text_ia_words.get(text_id, []))
                words = [w for _, w in ia_sorted]
                source = "ia_file"

            stimulus = self._make_stimulus(
                words,
                f"provo_{text_id}",
                {"text_id": text_id, "word_source": source},
            )
            n_words = len(words)

            # Map source area ID → word position via label-walk.
            wn_to_pos = self._align_wn_to_passage(text_ia_words.get(text_id, []), words)
            stimulus.metadata["source_ia_to_unit"] = wn_to_pos

            # Per-participant per-measure RT array.
            part_data = text_participant_data.get(text_id, {})
            for participant, word_meas in part_data.items():
                metadata = self._presentation_metadata(word_meas)
                measure_values: dict[str, np.ndarray] = {}
                for out_name in EYELINK_MEASURE_NAMES:
                    vals = np.full(n_words, np.nan)
                    for wn, _wstr, mdict in word_meas:
                        pos = wn_to_pos.get(wn)
                        if pos is not None and 0 <= pos < n_words:
                            vals[pos] = mdict.get(out_name, np.nan)
                    measure_values[out_name] = vals
                if self.first_pass_filter:
                    # Reuse the corpus-independent first-pass frontier logic
                    # validated by the GECO loader.
                    from .geco import _first_pass_filter_trial

                    fix_times = {
                        pos: mdict.get("__first_fixation_time")
                        for wn, _wstr, mdict in word_meas
                        if (pos := wn_to_pos.get(wn)) is not None
                    }
                    positions = list(range(n_words))
                    keep, wdelta, prevwasfix = _first_pass_filter_trial(
                        positions, fix_times, {pos: pos for pos in positions}
                    )
                    measure_values = {
                        name: np.where(keep, values, np.nan)
                        for name, values in measure_values.items()
                    }
                    measure_values["wdelta"] = wdelta
                    measure_values["prevwasfix"] = prevwasfix
                for out_name, vals in measure_values.items():
                    self._data.append(
                        Datum(
                            stimulus=stimulus,
                            measure_name=out_name,
                            values=vals,
                            participant_id=participant,
                            metadata=dict(metadata),
                        )
                    )

    @staticmethod
    def _presentation_metadata(word_measures) -> dict:
        """Identify the unique presentation represented by a summary trial."""
        trials = {
            int(measures["__trial_index"])
            for _, _, measures in word_measures
            if np.isfinite(measures["__trial_index"])
        }
        if len(trials) > 1:
            raise ValueError(f"Multiple Provo summary presentations: {sorted(trials)}")
        return {"source_trial_index": trials.pop()} if trials else {}

    @staticmethod
    def _initial_label(row: dict, passage: list[str] | None) -> str:
        label = row.get("IA_LABEL", "").strip()
        if not label or (passage and label != passage[0]):
            raise ValueError(f"Unverified Provo initial area: {row.get('Text_ID')}")
        return label

    def _align_wn_to_passage(
        self,
        ia_rows: list[tuple[int, str]],
        passage: list[str],
    ) -> dict[int, int | None]:
        """Match ordered measured areas to verified passage labels.

        IA_ID identifies the measured region; Word_Number is cloze metadata
        and contains gaps and a duplicated value in the official release.
        An unrecognized label fails instead of assigning its duration to an
        arbitrary neighbouring word.
        """
        rows = sorted(ia_rows)
        passage_norms = [self._normalize_word(w) for w in passage]
        mapping: dict[int, int | None] = {}
        j = 0
        for wn, label in rows:
            if j >= len(passage_norms):
                mapping[wn] = None
                continue
            target = self._normalize_word(label)
            if passage_norms[j] == target:
                mapping[wn] = j
                j += 1
            else:
                # Try a small lookahead.
                matched = None
                for jj in range(j + 1, min(j + 3, len(passage_norms))):
                    if passage_norms[jj] == target:
                        matched = jj
                        break
                if matched is not None:
                    mapping[wn] = matched
                    j = matched + 1
                else:
                    raise ValueError(
                        f"Unaligned Provo area {wn}: {label!r}; expected {passage[j]!r}"
                    )
        return mapping

    def _load_cloze(self, cloze_file: str) -> None:
        """Parse predictability norms CSV and add 'human_cloze' layer.

        The file has multiple rows per word position (one per cloze response).
        The cloze probability is the proportion on the row where Response
        matches Word (the orthographic match). A tested position whose target
        word no participant produced gets cloze 0.
        """

        # text_id → word_idx → cloze_prob
        cloze_data: dict[int, dict[int, float]] = {}
        # text_id → positions present in the norms (target may be unproduced)
        seen_positions: dict[int, set[int]] = {}
        stimuli = {s.metadata["text_id"]: s for s in self._stimuli}
        source_positions: dict[int, dict[int, int | None]] = {}
        targets: dict[tuple[int, int], str] = {}
        unaligned: dict[int, list[dict]] = {}

        with open(cloze_file, newline="", encoding=CLOZE_ENCODING) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_id_str = row.get("Text_ID", "")
                word_idx_str = row.get("Word_Number", "")
                if text_id_str in ("", "NA", ".") or word_idx_str in ("", "NA", "."):
                    continue
                text_id = int(text_id_str)
                if text_id not in stimuli:
                    continue
                number = int(word_idx_str)
                if row.get("Word_Unique_ID") == "QID2687":
                    # This released question says text 18 / word 3, but its
                    # target is "probably"; text 18 has "the" there. Do not
                    # contaminate the valid QID929 responses at that position
                    # or infer another question's context from its row order.
                    if (text_id, number, row.get("Word")) != (18, 3, "probably"):
                        raise ValueError("Changed Provo QID2687 source anomaly")
                    unaligned.setdefault(text_id, []).append(dict(row))
                    continue
                word_idx = number - 1
                # Release numbering includes a missing punctuation position
                # in texts 3/13 and omits the displayed em dash in text 36.
                if (text_id == 3 and number >= 46) or (text_id == 13 and number >= 20):
                    word_idx -= 1
                elif text_id == 36 and number >= 24:
                    word_idx += 1
                elif text_id == 55:
                    if number in (9, 10, 61, 62):
                        source_positions.setdefault(text_id, {})[number] = None
                        continue  # Separate cloze words differ from the measured area.
                    if number >= 11:
                        word_idx -= 1
                source_positions.setdefault(text_id, {})[number] = word_idx
                word = row.get("Word", "").strip().lower()
                key = (text_id, number)
                if key in targets and targets[key] != word:
                    raise ValueError(f"Conflicting Provo cloze target: {key}")
                targets[key] = word
                words = stimuli[text_id].unit_strings()
                if not 0 <= word_idx < len(words) or not self._cloze_target_matches(
                    word, words[word_idx], key
                ):
                    raise ValueError(f"Unaligned Provo cloze target: {key}: {word!r}")
                seen_positions.setdefault(text_id, set()).add(word_idx)
                response = row.get("Response", "").strip().lower()
                if word != response:
                    continue
                prob = parse_float(row.get("Response_Proportion", ""))
                probs = cloze_data.setdefault(text_id, {})
                probs[word_idx] = probs.get(word_idx, 0.0) + prob

        # Attach layers to stimuli
        for stimulus in self._stimuli:
            text_id = stimulus.metadata.get("text_id")
            stimulus.metadata["source_cloze_to_unit"] = source_positions.get(
                text_id, {}
            )
            stimulus.metadata["source_unaligned_cloze"] = unaligned.get(text_id, [])
            if text_id not in seen_positions:
                continue
            n_words = len(stimulus.layers["words"])
            probs = cloze_data.get(text_id, {})
            seen = seen_positions[text_id]
            spans = list(stimulus.layers["words"].spans)
            # Position tested in the norms but whose target word no participant
            # produced has cloze 0; positions absent from the file stay NaN.
            values = [
                probs.get(i, 0.0 if i in seen else np.nan) for i in range(n_words)
            ]
            cloze_layer = AnnotationLayer("human_cloze", spans, values)
            stimulus.add_layer(cloze_layer)

    @classmethod
    def _cloze_target_matches(
        cls, target: str, display: str, key: tuple[int, int]
    ) -> bool:
        """Validate the norms' released contraction/possessive stems."""
        target = cls._normalize_word(target)
        display = cls._normalize_word(display)
        forms = {display, display.split("'")[0]}
        if display.endswith("n't"):
            forms.add(display[:-3])
        if display == "cannot":
            forms.add("can")
        exceptions = {
            (25, 45): ("0.9", "90%"),
            (34, 27): ("inc", "inc.-owned"),
            (45, 28): ("womens", "women's"),
        }
        return target in forms or exceptions.get(key) == (target, display)

    @staticmethod
    def _normalize_word(s: str) -> str:
        """Strip whitespace + outer punctuation, lowercase, and fold the
        apostrophe variants (Õ, backtick) PROVO uses inconsistently across
        files to ASCII '. Used to match fixation IA labels against stimulus
        words.

        PROVO's two files differ on:
          * outer punctuation: the fixation file keeps periods attached
            ('California.') while the IA file sometimes splits them
            ('california' + '.').
          * apostrophe style: the fixation file's IA labels render the
            apostrophe as Õ (e.g. 'doesnÕt', U+00D5) while the IA file uses
            straight ASCII apostrophes. Both must collapse to the same
            token here.
        """
        # Fold the apostrophe variants PROVO uses (Õ, backtick) to ASCII '.
        for ch in ("’", "Õ", "`"):
            s = s.replace(ch, "'")
        # PROVO's IA file's Word_Cleaned column has '?' where the original
        # had a curly apostrophe (encoding round-trip damage during CSV
        # export, e.g. 'women?s' for "women's"). Fold those too.
        if "?" in s and not (s.startswith("?") or s.endswith("?")):
            s = s.replace("?", "'")
        s = s.strip().lower()
        while s and s[-1] in ".,;:!?\"')]":
            s = s[:-1]
        while s and s[0] in "\"'([":
            s = s[1:]
        return s

    def _build_ia_to_position(self, fix_file: str) -> dict[int, dict[int, int | None]]:
        """Align each (text_id, IA_INDEX) to a word position in the stimulus.

        Returns ``{text_id: {ia_index: position or None}}``. ``None`` means the
        IA exists in the fixation file but has no matching stimulus word (e.g.
        an unrecorded source area). Verified compounds remain whole regions.
        The walk is order-preserving: IA indices are stepped through in
        ascending order, word positions resuming where the previous match
        left off, with a small lookahead window for transient mismatches.

        Args:
            fix_file: Path of the fixation report.
        """

        # Collect IA labels per text_id. The same IA can appear on many
        # rows (one per fixation); the label is identical, so first-seen
        # wins.
        ia_labels: dict[int, dict[int, str]] = {}
        with open(fix_file, newline="", encoding=CLOZE_ENCODING) as f:
            for row in csv.DictReader(f):
                try:
                    text_id = int(row["trial"])
                    ia = int(row["CURRENT_FIX_INTEREST_AREA_INDEX"])
                except (KeyError, ValueError):
                    continue
                label = row.get("CURRENT_FIX_INTEREST_AREA_LABEL", "")
                ia_labels.setdefault(text_id, {}).setdefault(ia, label)

        result: dict[int, dict[int, int | None]] = {}
        for stim in self._stimuli:
            text_id = stim.metadata["text_id"]
            replica_words = [
                self._normalize_word(w) for w in stim.layers["words"].values
            ]
            ia_for_text = ia_labels.get(text_id, {})
            mapping: dict[int, int | None] = {}
            replica_cursor = 0
            for ia_n in sorted(ia_for_text):
                raw_label = ia_for_text[ia_n]
                target = self._normalize_word(raw_label)
                matched: int | None = None
                # Look ahead a few words to absorb tiny segmentation
                # differences without runaway drift.
                for j in range(
                    replica_cursor, min(replica_cursor + 4, len(replica_words))
                ):
                    if replica_words[j] == target:
                        matched = j
                        break
                mapping[ia_n] = matched
                if matched is not None:
                    replica_cursor = matched + 1
            result[text_id] = mapping
        return result

    # Fixation-level loading. CAVEAT: PROVO sometimes records the same
    # paragraph twice per participant (different TRIAL_INDEX). The IA
    # report keeps only the smallest TRIAL_INDEX; the fixation report
    # logs all of them. ``fixations()`` returns all events with
    # TRIAL_INDEX in metadata; consumers matching the IA file must filter
    # to the smallest trial_index per (participant_id, stimulus_id).
    #
    # EyeLink Data Viewer Fixation_Report columns (subset used):
    #
    #   RECORDING_SESSION_LABEL     participant ID (matches the per-word file)
    #   trial                       paragraph ID (matches per-word Text_ID)
    #   TRIAL_INDEX                 1-based trial within session (kept as metadata)
    #   CURRENT_FIX_INDEX           1-based fixation ordinal within trial
    #   CURRENT_FIX_START / _END    ms relative to trial onset (already
    #                               trial-relative, no IP_START_TIME subtraction)
    #   CURRENT_FIX_DURATION        end - start
    #   CURRENT_FIX_X / _Y          pixel coordinates
    #   CURRENT_FIX_INTEREST_AREA_INDEX  1-based word index, "." if off-text
    #   CURRENT_FIX_INTEREST_AREA_RUN_ID 1=first pass, 2+=re-reading (metadata)
    #   CURRENT_FIX_PUPIL           pupil diameter (metadata)
    def _load_fixations(self, fix_file: str) -> None:

        # Build the IA → word-position map by label alignment: PROVO's
        # fixation IA_INDEX numbering doesn't align 1:1 with the IA file's
        # Word_Number (different segmentation of punctuation), so a constant
        # offset is insufficient.
        ia_lookup = self._build_ia_to_position(fix_file)
        char_offsets: dict[int, list[int]] = {}
        for stim in self._stimuli:
            text_id = stim.metadata["text_id"]
            char_offsets[text_id] = [span.start for span in stim.layers["words"].spans]

        with open(fix_file, newline="", encoding=CLOZE_ENCODING) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Identify trial; some rows may be malformed/blank.
                trial_str = row.get("trial", "")
                participant = row.get("RECORDING_SESSION_LABEL", "")
                if not trial_str or trial_str in (".", "NA") or not participant:
                    continue
                try:
                    text_id = int(trial_str)
                except ValueError:
                    continue
                stim_id = f"provo_{text_id}"

                # Time + space, the report stores them as numeric strings.
                try:
                    start = float(row["CURRENT_FIX_START"])
                    end = float(row["CURRENT_FIX_END"])
                    dur = float(row["CURRENT_FIX_DURATION"])
                    x = float(row["CURRENT_FIX_X"])
                    y = float(row["CURRENT_FIX_Y"])
                    fix_idx = int(row["CURRENT_FIX_INDEX"])
                except (KeyError, ValueError):
                    continue

                # Linguistic association: look up the fixation's IA_INDEX
                # in the per-text alignment map built by
                # ``_build_ia_to_position`` (label-matched against the
                # stimulus words). Fixations on IAs with no stimulus word
                # (segmentation-mismatched
                # punctuation) return None, an unaligned fixation rather
                # than a parse error.
                ia_str = row.get("CURRENT_FIX_INTEREST_AREA_INDEX", ".")
                unit_index: int | None = None
                char_offset: int | None = None
                if ia_str not in (".", "", "NA"):
                    try:
                        ia_n = int(ia_str)
                    except ValueError:
                        ia_n = None
                    if ia_n is not None:
                        unit_index = ia_lookup.get(text_id, {}).get(ia_n)
                        if unit_index is not None:
                            char_offset = char_offsets[text_id][unit_index]

                # TRIAL_INDEX is essential metadata for matching the IA
                # report (which drops re-presentations), store as int.
                # ia_index is the raw EyeLink IA number from the fixation file.
                meta: dict = {}
                set_optional_meta(meta, "trial_index", row.get("TRIAL_INDEX", ""), int)
                set_optional_meta(meta, "ia_index", ia_str, int)
                set_optional_meta(
                    meta,
                    "run_id",
                    row.get("CURRENT_FIX_INTEREST_AREA_RUN_ID", ""),
                    int,
                )
                set_optional_meta(
                    meta, "pupil", row.get("CURRENT_FIX_PUPIL", ""), float
                )

                append_fixation(
                    self._fixations,
                    participant_id=participant,
                    stimulus_id=stim_id,
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
