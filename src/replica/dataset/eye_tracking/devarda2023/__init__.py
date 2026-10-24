"""Aligned multi-measure psycholinguistic corpus of de Varda et al. (2023).

Combines, in a single per-test-word table, behavioural and electrophysiological
measures originally collected by Frank et al. (2013, 2015):

  * Eye-tracking, RTfirstfix, RTfirstpass, RTrightbound, RTgopast.
  * Self-paced reading, self_paced_reading_time.
  * ERP, ELAN, LAN, N400, EPNP, P600, PNP.

The reading measures are means across participants at each test-word position.
The list field identifies the cloze item assignment.
Values are stored under one ``averaged`` participant, with NaN at unmeasured
positions. Duration means include the source's zeros; individual skips cannot
be removed from an already averaged value. Use UCL for participant-level data.

Reference (this aligned release):
    de Varda, A. G., Marelli, M., & Amenta, S. (2023). Cloze probability,
    predictability ratings, and computational estimates for 205 English
    sentences, aligned with existing EEG and reading time data.
    *Behavior Research Methods*. https://doi.org/10.3758/s13428-023-02261-8

Underlying experiments: Frank et al. (2013) eye-tracking + self-paced
reading and Frank et al. (2015) ERP releases.

Source repo (data lives here): https://github.com/Andrea-de-Varda/prediction-resource
"""

import csv
from pathlib import Path

import numpy as np

from replica.core import AnnotationLayer, Datum, Modality, Presentation, Span, Stimulus
from replica.dataset.base import EyeTrackingDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.eye_tracking.devarda2023.covariates import (
    DeVardaCovariates,
    find_devarda_table,
    has_norm_columns,
)
from replica.dataset.loading.utils import scatter_values
from replica.dataset.metadata import io as metadata_io

# Released participant means, stored at their source test-word positions.
MEASURE_COLUMNS: dict[str, str] = {
    # Eye-tracking (Frank et al., 2013)
    "RTfirstpass": "first_run_in_first_pass",
    "RTfirstfix": "first_fixation_in_first_pass",
    "RTrightbound": "right_bounded",
    "RTgopast": "go_past_in_first_pass",
    # Self-paced reading (Frank et al., 2013)
    "self_paced_reading_time": "spr_rt",
    # ERP (Frank et al., 2015)
    "ELAN": "elan",
    "LAN": "lan",
    "N400": "n400",
    "EPNP": "epnp",
    "P600": "p600",
    "PNP": "pnp",
}
FIRST_PASS_COLUMN = "RTfirstpass"
MEASURE_NAMES = tuple(MEASURE_COLUMNS.values())
# How each non-eye-tracking mean was recorded: button presses for self-paced
# reading, EEG for the ERP components.
_MODALITIES: dict[str, str] = {
    "spr_rt": str(Modality.BEHAVIORAL),
    **dict.fromkeys(("elan", "lan", "n400", "epnp", "p600", "pnp"), str(Modality.EEG)),
}


class DeVarda2023(DeVardaCovariates, EyeTrackingDataset):
    """de Varda et al. (2023) aligned ERP + eye-tracking + SPR corpus.

    Exposes four eye-tracking duration means, SPR mean reading time and six
    ERP components. The default is ``first_run_in_first_pass`` (RTfirstpass).
    The release does not establish individual first-pass flags or unconditional
    first-run durations. Cloze list assignments are retained per test position,
    and the release's own norms (cloze probability, predictability ratings and
    their summaries) are attached as covariates on load. The participant-level
    sources of the means are ``UCL_ET``, ``UCL_SPR`` and ``Frank2015ERP``.
    """

    def fixations(self, participant=None, stimulus_id=None):
        """Reject event-level access: the release is aggregated per cell.

        Args:
            participant: Accepted for signature compatibility, never used.
            stimulus_id: Accepted for signature compatibility, never used.

        Raises:
            NotImplementedError: Always; the release has no fixation events.
        """
        raise NotImplementedError(
            "DeVarda2023 ships one row per (sentence, list, test-word) "
            "cell with measures pre-averaged across participants. "
            "Individual fixations are not in the release."
        )

    # Frank's eye-tracking and self-paced-reading experiments; how his ERP
    # experiment presented the same sentences is not recorded here.
    info = DatasetInfo(
        name="devarda2023",
        language="en",
        presentation=[
            Presentation.NATURAL_READING,
            Presentation.SELF_PACED_READING,
        ],
    )
    _declared_measures = MEASURE_NAMES
    additive_measures = frozenset({"spr_rt"})

    def load(self, path: str) -> None:
        """Load the aligned per-cell measure table.

        When the table carries the norm columns, they are attached as covariates
        and every row's ``word`` must be the word at its ``context_length``.

        Args:
            path: Directory holding the official ``all_measures.csv``, or the
                local ``aligned.csv`` alias for the same file; the official
                name wins when both are present.

        Raises:
            ValueError: A required column is absent, a position is invalid or
                repeated, or a normed row names a word that is not at its
                position.
        """
        csv_path = find_devarda_table(path)
        self._load_metadata(
            path,
            files={"aligned.csv": csv_path.name},
            metadata_path=metadata_io.find_dataset_yaml(path),
        )
        self._load_csv(csv_path)
        # A synthetic table may carry the responses alone; the release carries
        # the norms as well.
        if has_norm_columns(csv_path):
            self.attach_predictability_norms(csv_path)

    def _load_csv(self, csv_path: Path) -> None:
        # First pass: group rows by sent_id, retain insertion order.
        sentences: dict[int, dict] = {}
        sentence_order: list[int] = []
        rows_by_sent: dict[int, list[dict]] = {}

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            required = {
                "sent_id",
                "sentence",
                "list",
                "context_length",
                *MEASURE_COLUMNS,
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"DeVarda2023: missing columns {sorted(missing)}")
            for row in reader:
                sid = int(row["sent_id"])
                if sid not in sentences:
                    sentences[sid] = {"sentence": row["sentence"]}
                    sentence_order.append(sid)
                    rows_by_sent[sid] = []
                rows_by_sent[sid].append(row)

        self._stimuli = []
        self._data = []

        for sid in sentence_order:
            sentence_text = sentences[sid]["sentence"]
            words = sentence_text.split()

            # Build word spans from the sentence text. The data uses
            # whitespace tokenisation; positions in the data align to
            # ``sentence.split()[context_length]``.
            spans: list[Span] = []
            offset = 0
            for w in words:
                start = sentence_text.index(w, offset)
                end = start + len(w)
                spans.append(Span(start, end))
                offset = end

            word_layer = AnnotationLayer(name="words", spans=spans, values=words)
            stimulus = Stimulus(
                text=sentence_text,
                stimulus_id=f"devarda_s{sid}",
                metadata={"sent_id": sid},
            )
            stimulus.add_layer(word_layer)
            self._stimuli.append(stimulus)

            columns: dict[str, dict[int, float]] = {name: {} for name in MEASURE_NAMES}
            lists: dict[int, int] = {}
            for row in rows_by_sent[sid]:
                position = int(row["context_length"])
                if not 0 <= position < len(words) or position in lists:
                    raise ValueError(
                        f"DeVarda2023: invalid or duplicate position {sid}/{position}"
                    )
                if row["sentence"] != sentence_text:
                    raise ValueError(
                        f"DeVarda2023: conflicting text for sentence {sid}"
                    )
                lists[position] = int(row["list"])
                for raw_col, canonical in MEASURE_COLUMNS.items():
                    value = row[raw_col]
                    columns[canonical][position] = float(value) if value else np.nan
            stimulus.metadata["cloze_lists"] = lists
            for canonical in MEASURE_NAMES:
                values = scatter_values(columns[canonical], len(words))
                if np.any(~np.isnan(values)):
                    self._data.append(
                        Datum(
                            stimulus=stimulus,
                            measure_name=canonical,
                            values=values,
                            participant_id="averaged",
                            metadata={"aggregation": "participant_mean"},
                            modality=_MODALITIES.get(
                                canonical, str(Modality.EYE_TRACKING)
                            ),
                        )
                    )
