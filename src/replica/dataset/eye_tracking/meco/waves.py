"""MECO wave/cohort releases beyond the L1 Wave 1 loader in :mod:`.l1`.

The upstream releases and PyMovements call these datasets MECOL1W2,
MECOL2W1, and MECOL2W2.  L1W2 is exposed once per collection site because
its stimuli are in the participants' native language.  L2 stimuli are English;
their releases are exposed as one dataset per wave.
"""

import os
from typing import Any

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

from ....core import AnnotationLayer, Span
from ...loading.utils import append_measure_datums
from ...loading.word_regions import preserve_interest_areas
from .fixations import MECOLazyFixationMixin
from .releases import L1W2_SITE_LANGUAGES, WORD_RELEASES, MECOWordRelease
from .word_utils import (
    MEASURE_COLUMNS,
    SKIP_COLUMN,
    clean_word,
    measures_from_row,
    present_measures,
    read_r_dataframe,
    scatter_frame_measures,
    twelve_item_inventory,
)

_WordSequence = tuple[tuple[int, str], ...]

# Sites whose AOIs are linguist-defined words in an unspaced display, so the
# stimulus text joins the words without a separator.
_UNSPACED_SITES = frozenset({"ch_s", "ch_t"})
_ParticipantFrame = tuple[str, Any]

# Literal blank AOIs in the native-reader wave-2 word report. They have
# source measurements, but supply no lexical target or model context.
_L1W2_NONWORD_AREAS = {
    ("ba", 6, 57),
    ("ba", 4, 19),
    ("bp", 7, 68),
    ("bp", 7, 195),
    ("ge_po", 5, 22),
    ("ge_po", 7, 130),
    ("ge_zu", 5, 22),
    ("ge_zu", 7, 130),
    ("ic", 11, 22),
    ("tr", 8, 29),
}

# Orthographic compounds in the twelve author L2 passages (also the texts
# used by Eyebench). Some sites assign their halves separate measured areas.
_L2_COMPOUNDS = {
    1: {"Grief-stricken,"},
    5: {"anti-virus", "anti-spyware"},
    9: {"present-day"},
    10: {"file-sharing", "widely-used", "music-sharing"},
    11: {"modern-day", "centuries-old", "wood-encased", "lead-based."},
    12: {"multi-person", "multi-function"},
}


def _l2_display_regions(stimulus, item_id: int) -> None:
    """Restore source compound spelling without combining measured areas.

    The Chinese-site pencil passage repeats the whole ``wood-encased`` label
    on two areas across a line break. The official passage and fixation
    coordinates establish the order of its ``wood-`` and ``encased`` halves.
    Source area IDs and their individual response values remain unchanged.
    """
    words = list(stimulus.layers["words"].values)
    numbers = stimulus.metadata["source_word_numbers"]
    if item_id == 11 and 20 in numbers and 21 in numbers:
        i = numbers.index(20)
        if numbers[i : i + 2] == [20, 21] and words[i : i + 2] == ["wood-encased"] * 2:
            words[i : i + 2] = ["wood-", "encased"]
    compounds = _L2_COMPOUNDS.get(item_id, set())
    text = ""
    spans = []
    for i, word in enumerate(words):
        joined = (
            i > 0 and words[i - 1].endswith("-") and words[i - 1] + word in compounds
        )
        if i and not joined:
            text += " "
        start = len(text)
        text += word
        spans.append(Span(start, len(text)))
    stimulus.text = text
    stimulus.layers["words"] = AnnotationLayer("words", spans, words)


class _MECOWave(MECOLazyFixationMixin, EyeTrackingDataset):
    """Shared reader for MECO's word-level R reading-measure releases."""

    _declared_measures = (
        *MEASURE_COLUMNS.values(),
        "first_run",
        "first_pass",
        "go_past",
    )

    release: MECOWordRelease
    site: str | None

    def __init__(self, site: str | None = None):
        super().__init__()
        self.site = site
        language = "en" if self.release.reader_group == "L2" else "multilingual"
        self.info = DatasetInfo(name=self.release.name, language=language)

    def load(self, path: str) -> None:
        self._load_metadata(
            path,
            metadata_path=metadata_io.find_dataset_yaml(
                self.release.data_subdir, required=True
            ),
        )
        if self.site is not None:
            self.info.name = f"{self.release.name}-{self.site}"
            self.info.language = L1W2_SITE_LANGUAGES[self.site]
            self.info.word_spacing = self.site not in _UNSPACED_SITES
        else:
            self.info.name = self.release.name
            self.info.language = "en"
        rda_path = os.path.join(path, self.release.filename)
        if not os.path.exists(rda_path):
            raise FileNotFoundError(
                f"{self.release.name} data not found at {rda_path!r}. "
                f"Run: replica fetch {self.fixation_release_key}"
            )
        frame = read_r_dataframe(rda_path)
        self._load_frame(frame)
        self._add_arrival_go_past()
        self._configure_fixation_store(path)

    def _load_frame(self, frame) -> None:
        required = {"uniform_id", "lang", "itemid", "wordnum", "word"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{self.release.name} is missing required columns: {sorted(missing)}"
            )

        # The R frames are wide and long. Keep only the identity and
        # dependent-variable columns before any grouping so the all-site L2
        # loaders do not hold large transient frames in RAM.
        supported_columns = [
            column
            for column in (*MEASURE_COLUMNS, SKIP_COLUMN)
            if column in frame.columns
        ]
        frame = frame[[*sorted(required), *supported_columns]].drop_duplicates()
        if self.site is not None:
            available = sorted(frame["lang"].unique())
            frame = frame[frame["lang"] == self.site]
            if frame.empty:
                raise ValueError(
                    f"Site {self.site!r} not present in {self.release.name}; "
                    f"available: {available}"
                )

        present = present_measures(frame, release_name=self.release.name)

        blank = frame.word.isna() | frame.word.astype(str).str.strip().eq("")
        self._nonword_measures = {}
        if blank.any():
            keys = {
                (str(row.lang), int(row.itemid), int(row.wordnum))
                for row in frame[blank].itertuples()
            }
            if self.release.name != "MECOL1W2" or not keys <= _L1W2_NONWORD_AREAS:
                raise ValueError(f"Unverified blank MECO areas: {sorted(keys)}")
            for row in frame[~blank].itertuples():
                if (str(row.lang), int(row.itemid), int(row.wordnum)) in keys:
                    raise ValueError("MECO blank area differs between participants")
            for (participant, item), rows in frame[blank].groupby(
                ["uniform_id", "itemid"]
            ):
                self._nonword_measures[(str(participant), int(item))] = [
                    {
                        "source_word_number": int(row["wordnum"]),
                        **measures_from_row(row.__getitem__, present),
                    }
                    for _, row in rows.iterrows()
                ]
            frame = frame[~blank]

        # itemid is stable text identity. L1 sites use different numeric
        # offsets (e.g. Basque 0..11, Russian 2..13), so normalize their
        # sorted 12-item inventory to public text IDs 1..12.
        item_ids = twelve_item_inventory(frame, release_name=self.release.name)

        for text_id, item_id in enumerate(item_ids, start=1):
            item_frame = frame[frame["itemid"] == item_id]
            variants = self._tokenization_variants(item_frame)
            for variant_index, sequence in enumerate(
                sorted(variants, key=repr), start=1
            ):
                self._append_variant(
                    text_id=text_id,
                    item_id=int(item_id),
                    variant_index=variant_index,
                    variant_count=len(variants),
                    sequence=sequence,
                    participant_frames=variants[sequence],
                    present=present,
                )

        self._retain_released_first_runs()

    @staticmethod
    def _tokenization_variants(
        item_frame: Any,
    ) -> dict[_WordSequence, list[_ParticipantFrame]]:
        """Group readers that saw exactly the same source AOI sequence."""
        variants: dict[_WordSequence, list[_ParticipantFrame]] = {}
        for participant_id, participant_frame in item_frame.groupby("uniform_id"):
            participant_frame = participant_frame.sort_values("wordnum")
            sequence = tuple(
                (int(row.wordnum), clean_word(row.word))
                for row in participant_frame.itertuples()
            )
            variants.setdefault(sequence, []).append(
                (str(participant_id), participant_frame)
            )
        return variants

    def _append_variant(
        self,
        *,
        text_id: int,
        item_id: int,
        variant_index: int,
        variant_count: int,
        sequence: _WordSequence,
        participant_frames: list[_ParticipantFrame],
        present: list[tuple[str, str]],
    ) -> None:
        """Create one tokenization-compatible stimulus and its measurements."""
        words, source_numbers, word_index = preserve_interest_areas(sequence)
        suffix = f"_v{variant_index}" if variant_count > 1 else ""
        site_label = self.site or "all"
        stimulus = self._make_stimulus(
            words,
            f"{self.release.name.lower()}_{site_label}_{text_id}{suffix}",
            {
                "text_id": text_id,
                "source_item_id": item_id,
                "source_word_numbers": source_numbers,
                "source_word_labels": [label for _, label in sequence],
                "source_nonword_numbers": sorted(
                    {
                        row["source_word_number"]
                        for participant, _ in participant_frames
                        for row in self._nonword_measures.get(
                            (participant, item_id), []
                        )
                    }
                ),
                "reader_group": self.release.reader_group,
                "wave": self.release.wave,
                "site": self.site or "multiple",
                "tokenization_variant": variant_index,
                "language": (
                    L1W2_SITE_LANGUAGES[self.site] if self.site is not None else "en"
                ),
            },
            separator="" if self.site in _UNSPACED_SITES else " ",
        )
        if self.release.reader_group == "L2":
            _l2_display_regions(stimulus, item_id)
        measure_names = [measure for _, measure in present]
        for participant_id, participant_frame in participant_frames:
            arrays = scatter_frame_measures(
                participant_frame, present, word_index, len(words)
            )
            append_measure_datums(
                self._data,
                stimulus=stimulus,
                participant_id=participant_id,
                measure_names=measure_names,
                measure_values=arrays,
                metadata={
                    "site": str(participant_frame["lang"].iloc[0]),
                    "source_nonword_measures": self._nonword_measures.get(
                        (participant_id, item_id), []
                    ),
                    "source_item_id": getattr(self, "_original_item_ids", {}).get(
                        (participant_id, item_id), item_id
                    ),
                },
            )


class MECOL1W2(_MECOWave):
    """MECO native-reader wave 2 for one upstream collection site."""

    release = WORD_RELEASES["meco_l1_w2"]
    SITES = tuple(L1W2_SITE_LANGUAGES)
    fixation_release_key = "meco_l1_w2"

    def __init__(self, site: str = "en_uk"):
        """Restrict the release to one site and adopt that site's language.

        Args:
            site: Collection-site key of :data:`L1W2_SITE_LANGUAGES`, which also
                selects the stimulus language. An unknown key raises.
        """
        if site not in L1W2_SITE_LANGUAGES:
            raise ValueError(f"Unknown MECOL1W2 site {site!r}; valid: {self.SITES}")
        super().__init__(site=site)
        self.info.language = L1W2_SITE_LANGUAGES[site]
        self.info.name = f"MECOL1W2-{site}"
        self.info.word_spacing = site not in _UNSPACED_SITES


class MECOL2W1(_MECOWave):
    """MECO second-language readers, wave 1 (English stimuli)."""

    release = WORD_RELEASES["meco_l2_w1"]
    fixation_release_key = "meco_l2_w1"

    def __init__(self):
        """Pool every collection site, since the L2 stimuli are English throughout."""
        super().__init__()


class MECOL2W2(_MECOWave):
    """MECO second-language readers, wave 2 (English stimuli)."""

    release = WORD_RELEASES["meco_l2_w2"]
    fixation_release_key = "meco_l2_w2"

    def __init__(self):
        """Pool every collection site, since the L2 stimuli are English throughout."""
        super().__init__()

    def _load_frame(self, frame) -> None:
        # This reader's three mislabeled trials exactly match complete texts
        # 5, 10 and 11 read by other participants. Keep the reported IDs for
        # joining the raw event release, but group the word data by true text.
        frame = frame.copy()
        self._original_item_ids = {}
        item_numbers = frame["itemid"].astype(int)
        for reported, actual in ((3, 5), (6, 10), (7, 11)):
            mask = (frame["uniform_id"] == "ch_s_1") & (item_numbers == reported)
            if not mask.any():
                continue
            source = self._tokenization_variants(frame[mask].drop_duplicates())
            reference = self._tokenization_variants(
                frame[
                    (frame["uniform_id"] != "ch_s_1") & (item_numbers == actual)
                ].drop_duplicates()
            )
            if len(source) != 1 or not set(source) <= set(reference):
                raise ValueError(
                    f"MECO L2W2 item correction {reported}->{actual} does not match the release"
                )
            if ((frame["uniform_id"] == "ch_s_1") & (item_numbers == actual)).any():
                raise ValueError(f"MECO L2W2 ch_s_1 already has text {actual}")
            # R item IDs are strings in this release; retain their dtype.
            frame.loc[mask, "itemid"] = frame.loc[
                item_numbers == actual, "itemid"
            ].iloc[0]
            self._original_item_ids[("ch_s_1", actual)] = reported
        super()._load_frame(frame)
