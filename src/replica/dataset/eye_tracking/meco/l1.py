"""MECO L1 (Multilingual Eye-tracking Corpus, native readers) dataset.

Siegelman et al. (2022). Behavior Research Methods.
https://doi.org/10.3758/s13428-021-01772-6

MECO L1 contains eye-tracking data from native (L1) readers reading the
same 12 texts translated into 13 languages, enabling cross-linguistic
comparisons. Each MECO_L1 instance corresponds to one language; run
evaluate() 13 times and aggregate.

The full MECO project also collected L2 (second-language) data and a
second wave (W2). Following pymovements' naming convention (MECOL1W1,
MECOL1W2, MECOL2W1, MECOL2W2), this class corresponds to L1 wave 1.

Languages (MECO release codes, not all ISO 639-1, e.g. Dutch ``du``,
Estonian ``ee``, Greek ``gr``, Spanish ``sp``):
    de, du, ee, en, fi, gr, he, it, ko, no, ru, sp, tr
"""

import os

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

from ...loading.utils import append_measure_datums
from ...loading.word_regions import preserve_interest_areas
from .fixations import MECOLazyFixationMixin
from .releases import (
    FILE_TO_ISO as _FILE_TO_ISO,
)
from .releases import (
    ISO_TO_FILE as _ISO_TO_RDA,
)
from .releases import (
    LANGUAGES as _SHARED_LANGUAGES,
)
from .word_utils import (
    MEASURE_COLUMNS,
    clean_word,
    measures_from_row,
    present_measures,
    read_r_dataframe,
    scatter_frame_measures,
    twelve_item_inventory,
)

# Keep the implementation and public factory on the same immutable vocabulary.
LANGUAGES = _SHARED_LANGUAGES

# Blank interest areas are spaces in the official supplementary texts, not
# words. Keep their original measurements separately, without inventing "_".
_NONWORD_AREAS = {
    ("ee", 8, 9),
    ("ee", 8, 29),
    ("fi", 11, 95),
    ("fi", 7, 41),
    ("gr", 11, 134),
    ("gr", 7, 7),
    ("it", 9, 147),
    ("tr", 8, 29),
}


class MECO_L1(MECOLazyFixationMixin, EyeTrackingDataset):
    """MECO L1 (native reader) eye-tracking corpus for a single language.

    Loads data from the first wave of the Multilingual Eye-movement Corpus
    for native (L1) readers. Instantiate once per language and call
    load(path) for each.

    The primary data format is ``joint_l1_data_trimmed_version2.1.rda``
    (an R data file read via pyreadr).

    Args:
        language: Language code (one of MECO_L1.LANGUAGES).

    Example::

        for lang in MECO_L1.LANGUAGES:
            ds = MECO_L1(language=lang)
            ds.load(path)
            results[lang] = evaluate(ds, evaluation_spec, analysis_config)

    Standard first-run requires the matching prepared fixation report. Released
    measures remain available without it, using ``source="released"``. Use
    ``fixation_table()`` / ``iter_fixations()`` for lazy event access or filtered
    ``fixations(participant=...)`` for canonical objects.
    """

    _declared_measures = (
        *MEASURE_COLUMNS.values(),
        "first_run",
        "first_pass",
        "go_past",
    )

    LANGUAGES = LANGUAGES
    fixation_release_key = "meco_l1_w1"

    def __init__(self, language: str = "en"):
        """Bind the instance to one MECO language across measures and events.

        Args:
            language: MECO release code in :data:`LANGUAGES`, which selects the
                rows kept from the joint release, the ISO code reported as the
                dataset language, and the site the fixation store is filtered
                to. An unknown code raises.
        """
        super().__init__()
        if language not in LANGUAGES:
            raise ValueError(
                f"Unknown MECO language {language!r}. Valid options: {LANGUAGES}"
            )
        self.language = language
        self.fixation_site = _ISO_TO_RDA.get(language, language)
        self.info = DatasetInfo(
            name=f"MECO_L1-{language}",
            language=_FILE_TO_ISO.get(language, language),
        )

    def load(self, path: str) -> None:
        """Load MECO L1 data for this language from the release archive.

        Args:
            path: Root directory of the downloaded MECO data.

        Raises:
            FileNotFoundError: When the release archive is absent.
        """
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        # _load_metadata replaces info with yaml contents, where language is
        # the corpus-level "multilingual". Re-pin to the per-instance language.
        self.info.language = _FILE_TO_ISO.get(self.language, self.language)
        # Override name with language-specific variant
        self.info.name = f"MECO_L1-{self.language}"

        rda_path = os.path.join(path, "joint_l1_data_trimmed_version2.1.rda")
        if not os.path.exists(rda_path):
            raise FileNotFoundError(
                f"MECO L1 release not found at {rda_path}. "
                "Run `replica fetch meco`, or download it from "
                "https://osf.io/3527a/"
            )
        self._load_rda(rda_path)
        self._add_arrival_go_past()
        self._configure_fixation_store(path)

    # ------------------------------------------------------------------
    # .rda loading (primary)
    # ------------------------------------------------------------------

    def _load_rda(self, rda_path: str) -> None:
        """Load MECO L1 data from the joint .rda file via pyreadr."""
        df = read_r_dataframe(rda_path)

        # Map the language code to the .rda code
        rda_lang = _ISO_TO_RDA.get(self.language, self.language)

        # ``trialid`` is presentation order in this release and is shifted for
        # some participants.  ``itemid`` is the stable text identity (also the
        # identity used by the MECO source texts), so grouping uses ``itemid``.
        lang_df = df[df["lang"] == rda_lang].drop_duplicates()
        if lang_df.empty:
            available_langs = sorted(df["lang"].unique())
            raise ValueError(
                f"Language '{rda_lang}' (from '{self.language}') not found in .rda. "
                f"Available: {available_langs}"
            )

        item_ids = twelve_item_inventory(
            lang_df, release_name=f"MECO_L1-{self.language}"
        )

        # Build stimuli and data per trial
        for text_id, item_id in enumerate(item_ids, start=1):
            trial_df = lang_df[lang_df["itemid"] == item_id]

            # Get the word list from the first subject (words are identical
            # across subjects within a language+trial).
            first_subj = trial_df["subid"].iloc[0]
            template = trial_df[trial_df["subid"] == first_subj].sort_values("wordnum")
            nonwords = {
                int(row.wordnum)
                for row in template.itertuples()
                if not str(row.word).strip() or str(row.word).lower() == "nan"
            }
            if any(
                (rda_lang, int(item_id), number) not in _NONWORD_AREAS
                for number in nonwords
            ):
                raise ValueError(f"Unverified blank MECO area in {rda_lang}/{item_id}")
            template = template[~template.wordnum.isin(nonwords)]
            words, source_numbers, word_index = preserve_interest_areas(
                (int(row.wordnum), clean_word(row.word))
                for row in template.itertuples()
            )

            stimulus = self._make_stimulus(
                words,
                f"meco_{self.language}_{text_id}",
                {
                    "text_id": text_id,
                    "source_item_id": int(item_id),
                    "source_word_numbers": source_numbers,
                    "source_nonword_numbers": sorted(nonwords),
                    "language": self.language,
                },
            )

            n_words = len(words)

            # Measures present in this trial's columns (identical across
            # subjects, since each subj_df is a slice of trial_df).
            present = present_measures(
                trial_df, release_name=f"MECO_L1-{self.language}"
            )
            measure_names = [name for _, name in present]

            # Group by subject and scatter each subject's per-word measures
            # into dense arrays via the shared helper. ``wordnum`` is 1-based
            # in the .rda; preserve 0.0 for skipped/unfixated words.
            participant_col = "uniform_id" if "uniform_id" in trial_df else "subid"
            for subj_id, subj_df in trial_df.groupby(participant_col):
                subj_df = subj_df.sort_values("wordnum")
                nonword_rows = subj_df[subj_df.wordnum.isin(nonwords)]
                if any(
                    str(word).strip() not in {"", "nan"} for word in nonword_rows.word
                ):
                    raise ValueError(
                        f"MECO nonword area differs for participant {subj_id}"
                    )
                nonword_values = [
                    {
                        "source_word_number": int(row["wordnum"]),
                        **measures_from_row(row.__getitem__, present),
                    }
                    for _, row in nonword_rows.iterrows()
                ]
                subj_df = subj_df[~subj_df.wordnum.isin(nonwords)]
                observed = [
                    (int(row.wordnum), clean_word(row.word))
                    for row in subj_df.itertuples()
                ]
                if observed != list(zip(source_numbers, words, strict=True)):
                    raise ValueError(
                        f"MECO {self.language} text {text_id}: source areas differ "
                        f"for participant {subj_id}"
                    )
                arrays = scatter_frame_measures(subj_df, present, word_index, n_words)
                append_measure_datums(
                    self._data,
                    stimulus=stimulus,
                    participant_id=str(subj_id),
                    measure_names=measure_names,
                    measure_values=arrays,
                    metadata={"source_nonword_measures": nonword_values},
                )

        self._retain_released_first_runs()
