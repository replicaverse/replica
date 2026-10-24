"""Natural Stories Corpus (Futrell et al., 2021).

Self-paced reading data for 10 naturalistic stories. Standard benchmark for
evaluating surprisal-based predictors.

Data format expected:
- ``all_stories.tok``: tokenized stories, one word per line with story/zone IDs
- ``processed_RTs.tsv``: reading times with columns
  [WorkerId, item, zone, word, RT, correct, ...] (``item`` is the story ID)
"""

import csv
from pathlib import Path

from replica.dataset.base import SPRDataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.metadata import io as metadata_io

from ...core import Datum, Stimulus
from ..loading.utils import scatter_values


class NaturalStories(SPRDataset):
    """Natural Stories Corpus (Futrell et al., 2021)."""

    info = DatasetInfo(name="natural_stories", language="en")

    def load(
        self,
        path: str,
        stories_file: str = "all_stories.tok",
        rt_file: str = "processed_RTs.tsv",
    ) -> None:
        """Load Natural Stories from disk.

        Args:
            path: Directory containing the data files.
            stories_file: Tokenized stories file.
            rt_file: Processed reading times file.
        """
        self._load_metadata(path, metadata_path=metadata_io.find_dataset_yaml(path))
        base = Path(path)
        self._load_stimuli(base / stories_file)
        self._load_rts(base / rt_file)

    def _load_stimuli(self, stories_path: Path) -> None:
        """Parse the tokenized stories file into Stimulus objects.

        Expected TSV columns: word, zone, item
        where 'item' is the story ID and 'zone' is the 1-indexed word position.
        """
        # Group words by story, preserving zone order
        stories: dict[str, list[tuple[int, str]]] = {}
        story_order: list[str] = []

        with open(stories_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                story_id = row.get("item", "").strip()
                word = row.get("word", "").strip()
                zone = row.get("zone", "").strip()
                if not (story_id and word and zone):
                    raise ValueError(
                        f"Natural Stories: missing token identity at line {reader.line_num}"
                    )
                if story_id not in stories:
                    stories[story_id] = []
                    story_order.append(story_id)
                stories[story_id].append((int(zone), word))

        self._stimuli: list[Stimulus] = []
        self._story_id_map: dict[str, Stimulus] = {}

        for story_id in story_order:
            # Sort by zone index and extract words
            ordered = sorted(stories[story_id], key=lambda x: x[0])
            if [zone for zone, _ in ordered] != list(range(1, len(ordered) + 1)):
                raise ValueError(
                    f"Natural Stories: duplicate or missing zones in story {story_id}"
                )
            words = [word for _, word in ordered]
            stimulus = self._make_stimulus(
                words,
                f"ns_{story_id}",
                {"story_id": story_id},
            )
            self._story_id_map[story_id] = stimulus

    def _load_rts(self, rt_path: Path) -> None:
        """Parse reading times into Datum objects."""
        # Read all RT data, grouped by (worker, story)
        rt_data: dict[tuple[str, str], dict[int, float]] = {}

        with open(rt_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                worker = row.get("WorkerId", "").strip()
                story_id = row.get("item", row.get("story", "")).strip()
                zone = row.get("zone", "").strip()
                rt = row.get("RT", "").strip()

                if not (worker and story_id and zone and rt):
                    continue

                try:
                    zone_idx = int(zone) - 1  # 1-indexed → 0-indexed
                    rt_val = float(rt)
                except ValueError:
                    continue

                key = (worker, story_id)
                if key not in rt_data:
                    rt_data[key] = {}
                rt_data[key][zone_idx] = rt_val

        self._data = []
        for (worker, story_id), zones in rt_data.items():
            stimulus = self._story_id_map.get(story_id)
            if stimulus is None:
                continue

            n_words = len(stimulus.layers["words"])
            values = scatter_values(zones, n_words)

            datum = Datum(
                stimulus=stimulus,
                measure_name="spr_rt",
                values=values,
                participant_id=worker,
            )
            self._data.append(datum)
