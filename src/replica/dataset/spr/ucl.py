"""UCL SPR Corpus (Frank et al., 2013).

Self-paced reading data: 361 sentences, 117 participants.

Word-text source:
    Stimulus word strings are built from the ``word`` column in
    ``selfpacedreading.RT.txt``, the text displayed to participants.

Reference: https://doi.org/10.3758/s13428-012-0313-y
"""

import csv
from pathlib import Path

from replica.dataset.base import SPRDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.eye_tracking.devarda2023.covariates import DeVardaCovariates
from replica.dataset.metadata import io as metadata_io

from ...core import Datum, Stimulus
from ..loading.utils import scatter_values


def _parse_ucl_stimuli_metadata(path: Path) -> dict[int, dict[str, str]]:
    """Parse ``stimuli.txt`` for metadata only (question, answer)."""
    metadata: dict[int, dict[str, str]] = {}
    with open(path, encoding="cp1252") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sent_nr = int(row["sent_nr"])
            metadata[sent_nr] = {
                "question": row.get("question", "-"),
                "answer": row.get("answer", "-"),
            }
    return metadata


def _words_from_positions(word_positions: dict[int, str]) -> list[str]:
    """Convert a word_pos -> word mapping to a list of words."""
    max_pos = max(word_positions.keys())
    return [word_positions.get(i, "") for i in range(max_pos + 1)]


# ---------------------------------------------------------------------------
# UCL_SPR dataset
# ---------------------------------------------------------------------------


class UCL_SPR(DeVardaCovariates, SPRDataset):
    """UCL self-paced reading corpus (Frank et al., 2013).

    361 short English sentences read by 117 participants in a
    word-by-word self-paced reading paradigm.

    Stimulus word strings are built from the ``word`` column in the RT file,
    the text displayed to participants. The stimulus ids (``ucl_<sent_nr>``)
    are shared with ``UCL_ET`` and ``Frank2015ERP``, and :meth:`attach_predictability_norms`
    adds de Varda et al.'s (2023) predictability norms as covariates on the
    sentences whose displayed text equals the UCL sentence.
    """

    info = DatasetInfo(name="ucl_spr", language="en")

    def load(
        self,
        path: str,
        stimuli_file: str = "stimuli.txt",
        rt_file: str = "selfpacedreading.RT.txt",
    ) -> None:
        """Load UCL SPR data from disk.

        Args:
            path: Directory holding the release's tab-separated files.
            stimuli_file: File under *path* carrying the comprehension question
                and answer per sentence.
            rt_file: File under *path* carrying one row per (subject, sentence,
                word position) with the displayed word and its reading time.
        """
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        self._reset_devarda()
        base = Path(path)

        stim_metadata = _parse_ucl_stimuli_metadata(base / stimuli_file)
        self._load_rts(base / rt_file, stim_metadata)

    def _load_rts(
        self,
        rt_path: Path,
        stim_metadata: dict[int, dict[str, str]],
    ) -> None:
        """Parse SPR reading times into Stimulus and Datum objects."""
        sent_words: dict[int, dict[int, str]] = {}
        rt_data: dict[tuple[str, int], dict[int, float]] = {}
        # Comprehension-question outcome per trial: "c" correct, "e" error,
        # "-" no question. Studies that follow the release's accuracy filter
        # read it off the datum metadata.
        trial_correct: dict[tuple[str, int], str] = {}

        with open(rt_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                subj = row["subj_nr"].strip()
                sent_nr = int(row["sent_nr"])
                word_pos = int(row["word_pos"]) - 1
                word = row["word"].strip()
                rt_str = row["RT"].strip()
                trial_correct[(subj, sent_nr)] = (row.get("correct") or "-").strip()

                if sent_nr not in sent_words:
                    sent_words[sent_nr] = {}
                if word_pos not in sent_words[sent_nr]:
                    sent_words[sent_nr][word_pos] = word

                try:
                    rt_val = float(rt_str)
                except ValueError:
                    continue

                key = (subj, sent_nr)
                if key not in rt_data:
                    rt_data[key] = {}
                rt_data[key][word_pos] = rt_val

        self._sent_map: dict[int, Stimulus] = {}
        self._stimuli = []
        for sent_nr in sorted(sent_words.keys()):
            meta = stim_metadata.get(sent_nr, {})
            words = _words_from_positions(sent_words[sent_nr])
            stim_metadata_dict = {"sent_nr": sent_nr, **meta}
            stimulus = self._make_stimulus(
                words,
                f"ucl_{sent_nr}",
                stim_metadata_dict,
            )
            self._sent_map[sent_nr] = stimulus

        self._data = []
        for (subj, sent_nr), positions in rt_data.items():
            stimulus = self._sent_map.get(sent_nr)
            if stimulus is None:
                continue

            n_words = len(stimulus.layers["words"])
            values = scatter_values(positions, n_words)

            self._data.append(
                Datum(
                    stimulus=stimulus,
                    measure_name="spr_rt",
                    values=values,
                    participant_id=subj,
                    metadata={"correct": trial_correct.get((subj, sent_nr), "-")},
                )
            )
