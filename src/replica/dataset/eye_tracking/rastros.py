"""RastrOS: Brazilian Portuguese paragraph reading with human cloze norms.

Leal et al. (2022), https://doi.org/10.1007/s10579-022-09609-0.
The authors' OSF release (https://osf.io/9jxg3/) contains 50 paragraphs,
37 eye-tracking participants and exact response counts from the cloze task.
Interest-area words supply the displayed text, including punctuation and
nonbreaking spaces. Stable Word_Unique_ID keys join the files because some
Text_ID values in the norms are shifted. No study-specific smoothing,
trial exclusions, duration cutoffs or predictors are applied here.
"""

import csv
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo, Fixation
from replica.dataset.measures import (
    EYELINK_IA_MEASURES,
    EYELINK_MEASURE_NAMES,
    EYELINK_SKIP_COLUMN,
)
from replica.dataset.metadata import io as metadata_io

from ...core import AnnotationLayer, Span
from ..loading.utils import append_measure_datums
from ..loading.word_regions import normalize_c1_controls

RESPONSE_FILE = "RastrOS_Corpus_Eyetracking_Data.tsv"
NORMS_FILE = "RastrOS_Corpus_Predictability_Norms.tsv"
MEASURE_COLUMNS = dict(EYELINK_IA_MEASURES)


def read_table(path: Path) -> pd.DataFrame:
    """Read the release's UTF-8 TSV with literal quotation marks.

    Args:
        path: Eye-tracking or cloze-norm file from the RastrOS release.
    """
    return pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=["", "."],
        low_memory=False,
    )


def read_responses(path: Path) -> pd.DataFrame:
    """Read the word-level response table without filtering or averaging trials.

    Published column names and file order are retained, including IA_SKIP
    and display coordinates needed by analyses with explicit boundary rules.
    Zero durations remain zero in this table and in the Dataset's duration
    arrays; the default view masks them as skips.

    Args:
        path: Published eye-tracking response TSV.
    """
    frame = read_table(path)
    if (
        frame[["RECORDING_SESSION_LABEL", "Word_Unique_ID", "IA_LABEL"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Missing RastrOS participant, interest-area ID or word")
    if frame.duplicated(["RECORDING_SESSION_LABEL", "Word_Unique_ID"]).any():
        raise ValueError("Duplicate RastrOS observations")
    if frame.groupby("Word_Unique_ID").IA_LABEL.nunique().ne(1).any():
        raise ValueError("Conflicting RastrOS display words")
    return frame


def _display_word(label: str, reference: str) -> str:
    """Restore commas only where the other source column proves the export loss.

    The authors' cloze-eye-merge.go replaces commas with periods in every
    eye-report field, including IA_LABEL. Its separate Word column is not
    transformed. Preserve the reading-area spelling and capitalization when
    the two source strings differ in other ways.
    """
    label = normalize_c1_controls(label).strip()
    reference = normalize_c1_controls(reference).strip()
    quotes = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

    def normalized(value):
        return value.translate(quotes).casefold()

    if len(label) == len(reference) and normalized(
        reference.replace(",", ".")
    ) == normalized(label):
        return "".join(
            "," if b == "," else a for a, b in zip(label, reference, strict=True)
        )
    return label


def _restore_display_words(items: pd.DataFrame) -> None:
    """Restore verified punctuation and the two areas of a wrapped compound."""
    items["word"] = [
        _display_word(label, reference)
        for label, reference in zip(
            items.source_ia_label, items.source_word, strict=True
        )
    ]
    items["separator"] = np.where(items.word_index.eq(0), "", " ")
    compound = items[items.Word_Unique_ID.isin(["UID_44_34", "UID_44_35"])]
    if len(compound):
        if compound.word.tolist() != ["embebendo-", "se"]:
            raise ValueError("RastrOS split compound does not match the source")
        if compound.source_word.tolist() != ["embebendo-se", "embebendo-se"]:
            raise ValueError("RastrOS compound reference does not match the source")
        items.loc[items.Word_Unique_ID.eq("UID_44_35"), "separator"] = ""


def _word_table(norms: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    keys = responses.Word_Unique_ID.str.extract(r"^UID_(\d+)_(\d+)$")
    if keys.isna().any().any():
        raise ValueError("Invalid RastrOS interest-area ID")
    responses = responses.assign(
        context_id=keys[0].astype(int), word_index=keys[1].astype(int) - 1
    )
    items = (
        responses.drop_duplicates("Word_Unique_ID")[
            ["Word_Unique_ID", "context_id", "word_index", "IA_LABEL", "Word"]
        ]
        .sort_values(["context_id", "word_index"])
        .rename(columns={"IA_LABEL": "source_ia_label", "Word": "source_word"})
    )
    _restore_display_words(items)
    for _, group in items.groupby("context_id"):
        if group.word_index.tolist() != list(range(len(group))):
            raise ValueError("Incomplete RastrOS paragraph positions")
    if not set(norms.Word_Unique_ID) <= set(items.Word_Unique_ID):
        raise ValueError("RastrOS norms reference unknown interest areas")
    grouped = norms.groupby("Word_Unique_ID")
    for column in ["Total_Response_Count", "Sentence_Number", "Word"]:
        if grouped[column].nunique(dropna=False).ne(1).any():
            raise ValueError(f"Conflicting RastrOS {column}")
    counts = norms[["Response_Count", "Total_Response_Count"]]
    if (
        counts.isna().any().any()
        or not np.isfinite(counts.to_numpy()).all()
        or counts.lt(0).any().any()
        or counts.mod(1).ne(0).any().any()
        or norms.Total_Response_Count.le(0).any()
    ):
        raise ValueError("Invalid RastrOS cloze counts")
    # For 122 targets the release lists fewer responses than its stated total.
    # Keep that published denominator, as Response_Proportion does, instead of
    # renormalizing the listed alternatives into a different cloze estimate.
    if grouped.Response_Count.sum().gt(grouped.Total_Response_Count.first()).any():
        raise ValueError("RastrOS response counts exceed the stated total")
    norms = norms.assign(hit=norms.Response.eq(norms.Word) * norms.Response_Count)
    counts = norms.groupby("Word_Unique_ID").agg(
        n_responses=("Total_Response_Count", "first"),
        hits=("hit", "sum"),
        sentence=("Sentence_Number", "first"),
        target=("Word", "first"),
    )
    items = items.join(counts, on="Word_Unique_ID").reset_index(drop=True)

    # Some released norm IDs repeat a neighbouring target, including at
    # punctuation-only areas. Never attach that target's counts to this word.
    def target_key(word):
        if not isinstance(word, str):
            return ""
        word = " ".join(
            unicodedata.normalize("NFC", word).casefold().replace("_", " ").split()
        )
        while word and unicodedata.category(word[0]).startswith("P"):
            word = word[1:]
        while word and unicodedata.category(word[-1]).startswith("P"):
            word = word[:-1]
        return word

    display = items.word.map(target_key)
    items["cloze_aligned"] = display.ne("") & display.eq(items.target.map(target_key))
    items.loc[~items.cloze_aligned, ["hits", "n_responses"]] = np.nan
    if items.loc[items.word_index.gt(0), "sentence"].isna().any():
        raise ValueError("Missing RastrOS sentence membership")
    items["sentence"] = items.sentence.fillna(1).astype(int)
    items["sentid"] = items.groupby(["context_id", "sentence"], sort=True).ngroup()
    items["sentpos"] = items.groupby("sentid").cumcount() + 1
    return items


def read_words(norms_path: Path, responses_path: Path) -> pd.DataFrame:
    """Read complete ordered contexts and exact, unsmoothed human response counts.

    The denominator is the published Total_Response_Count, even when the
    released rows do not enumerate every response included in that total.

    Args:
        norms_path: Published cloze-norm TSV with response frequencies.
        responses_path: Published response TSV supplying complete displayed words.

    Returns:
        One row per word with Word_Unique_ID, context_id, zero-based
        word_index, displayed word, n_responses, hits, sentence, normalized
        target, and the sentence/position keys sentid and sentpos. Unprompted
        words have missing counts; a prompted word nobody guessed has zero hits
        and a positive total.
    """
    return _word_table(read_table(norms_path), read_responses(responses_path))


class RastrOS(EyeTrackingDataset):
    """Brazilian Portuguese eye-tracking and cloze data, one Stimulus per paragraph.

    Fetch with ``replica fetch rastros`` and load with
    ``load_dataset("rastros")``. Each paragraph has aligned human_cloze,
    human_cloze_hits and human_cloze_responses layers. The two released
    multiword interest areas stay intact, with their nonbreaking spaces,
    so response durations never get duplicated across invented word regions.
    """

    info = DatasetInfo(name="RastrOS", language="pt", stimulus_type="paragraph")
    _declared_measures = EYELINK_MEASURE_NAMES

    def load(self, path: str) -> None:
        """Load the two checksum-verified author files from a dataset directory.

        Args:
            path: Directory populated by ``replica fetch rastros``.
        """
        root = Path(path)
        for filename in [RESPONSE_FILE, NORMS_FILE]:
            if not (root / filename).is_file():
                raise FileNotFoundError(
                    f"RastrOS: {filename} not found in {root}. Run `replica fetch rastros`."
                )
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        self._responses = read_responses(root / RESPONSE_FILE)
        self._words = _word_table(read_table(root / NORMS_FILE), self._responses)
        self._build()

    def word_table(self) -> pd.DataFrame:
        """Return a copy of the ordered word table with exact human cloze counts."""
        return self._words.copy()

    def response_table(self) -> pd.DataFrame:
        """Return a copy of unfiltered responses with published columns and row order."""
        return self._responses.copy()

    def _build(self) -> None:
        self._stimuli, self._data = [], []
        stimuli = {}
        for context, group in self._words.groupby("context_id", sort=True):
            stimulus = self._make_stimulus(
                group.word.tolist(),
                stimulus_id=f"rastros_p{context}",
                metadata={
                    "text_id": int(context),
                    "word_uids": group.Word_Unique_ID.tolist(),
                },
            )
            text = ""
            spans = []
            for word, separator in zip(group.word, group.separator, strict=True):
                text += separator
                spans.append(Span(len(text), len(text) + len(word)))
                text += word
            stimulus.text = text
            stimulus.layers["words"] = AnnotationLayer(
                "words", spans, group.word.tolist()
            )
            spans = stimulus.layers["words"].spans
            for name, values in [
                ("human_cloze", group.hits / group.n_responses),
                ("human_cloze_hits", group.hits),
                ("human_cloze_responses", group.n_responses),
            ]:
                stimulus.add_layer(AnnotationLayer(name, spans, values.tolist()))
            stimuli[context] = stimulus
        rows = self._responses.merge(
            self._words[["Word_Unique_ID", "context_id", "word_index"]],
            on="Word_Unique_ID",
            how="left",
            validate="many_to_one",
        )
        for (context, participant), group in rows.groupby(
            ["context_id", "RECORDING_SESSION_LABEL"], sort=False
        ):
            stimulus = stimuli[context]
            n = len(stimulus.layers["words"])
            positions = group.word_index.to_numpy(int)
            arrays = []
            unfixated = pd.to_numeric(group["IA_DWELL_TIME"]).to_numpy() == 0.0
            for column in MEASURE_COLUMNS:
                raw = pd.to_numeric(group[column], errors="raise").to_numpy(float)
                array = np.full(n, np.nan)
                array[positions] = np.where(
                    np.isfinite(raw), raw, np.where(unfixated, 0.0, np.nan)
                )
                arrays.append(array)
            skipped = np.full(n, np.nan)
            skipped[positions] = pd.to_numeric(
                group[EYELINK_SKIP_COLUMN], errors="raise"
            ).to_numpy(float)
            arrays.append(
                np.where(np.isnan(skipped), np.nan, np.where(skipped != 0, 0.0, 1.0))
            )
            append_measure_datums(
                self._data,
                stimulus=stimulus,
                participant_id=str(participant),
                measure_names=list(EYELINK_MEASURE_NAMES),
                measure_values=arrays,
            )

    def fixations(
        self, participant: str | None = None, stimulus_id: str | None = None
    ) -> list[Fixation]:
        """Reject event-level access because this release contains word-level summaries.

        Args:
            participant: Participant filter accepted by the shared interface.
            stimulus_id: Stimulus filter accepted by the shared interface.

        Raises:
            NotImplementedError: The release supplies no sequence of fixation events.
        """
        raise NotImplementedError(
            "RastrOS releases only word-level interest-area summaries, with no "
            "fixation events."
        )
