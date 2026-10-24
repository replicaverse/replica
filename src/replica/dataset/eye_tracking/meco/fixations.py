"""Lazy, columnar access to the large MECO fixation-event releases.

The upstream RDA files are monolithic, so preparation is a one-time operation:
select the canonical columns in a child process and write compressed Parquet.
Normal dataset loading then uses Arrow predicate pushdown and never creates the
complete 1--2 million-event corpus as Python objects.
"""

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from replica.dataset.contracts import Fixation, FixationCalculation

from ....core import AnnotationLayer, Datum, Span, Stimulus
from .prepare import validate_parquet
from .releases import (
    FIXATION_RELEASES as RELEASES,
)


@dataclass(frozen=True)
class _StimulusMapping:
    stimulus: Stimulus
    word_indices: dict[int, int]
    char_offsets: list[int]
    unit_layer: str = "words"


class MECOLazyFixationMixin:
    """Mixin adding lazy Arrow-backed fixation access to MECO loaders."""

    fixation_release_key: str
    _data: list[Datum]
    # These released columns store NaN on never-fixated words.
    _unfixated_nan_measures = frozenset(
        {
            "first_fixation",
            "go_past",
            "go_past_in_first_pass",
            "total_time",
        }
    )

    def data(
        self,
        measure: str | None = None,
        participant: str | None = None,
        *,
        source: str = "default",
        calculation=None,
    ) -> list[Datum]:
        """Calculate standard first-run on demand; retain source-only access.

        Args:
            measure: Response name; ``None`` requests every measure.
            participant: Keep one participant's responses; ``None`` keeps all.
            source: Default standard runs or the released calculation.
            calculation: Optional versioned calculation on native units.
        """
        if source == "default" and measure in {
            None,
            "first_run",
            "first_run_in_first_pass",
        }:
            self._ensure_standard_first_run()
        return super().data(
            measure, participant, source=source, calculation=calculation
        )

    def _native_responses(self, measure, *, source="default"):
        """Materialize standard runs before indexing their native references."""
        if source == "default" and measure in {"first_run", "first_run_in_first_pass"}:
            self._ensure_standard_first_run()
        return super()._native_responses(measure, source=source)

    def _ensure_standard_first_run(self) -> None:
        """Validate each source stream before exposing its contiguous first-run."""
        stored = {datum.measure_name for datum in self._data}
        if "first_run" in stored or not self._released_data:
            return
        if not hasattr(self, "_fixation_parquet_path"):
            raise FileNotFoundError(
                "Standard MECO first-run requires the matching prepared fixation "
                f"report. Load the dataset, then run `replica fetch "
                f"{self.fixation_release_key}_fixations` and "
                f"`replica prepare-fixations {self.fixation_release_key}`."
            )
        references = defaultdict(dict)
        for name in ("first_fixation", "first_pass", "total_time", "first_run"):
            for datum in super()._native_responses(name, source="released").values():
                references[datum.participant_id, datum.stimulus.stimulus_id][name] = (
                    datum
                )
        standard = []
        for pair, raw in self.fixation_trials():
            source = references.pop(pair, None)
            if source is None:
                continue
            released = source["first_run"]
            events = self._reaggregation_fixations(released, raw)
            arrays = self._reaggregation_arrays(
                events, released.stimulus, released.layer, use_source_runs=False
            )
            for name, reference in source.items():
                if not np.allclose(
                    self._reaggregation_values(reference, arrays),
                    reference.values,
                    rtol=0,
                    atol=1e-6,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"MECO source events do not reproduce {name} for {pair}"
                    )
            standard.append(
                replace(
                    released,
                    measure_name="first_run",
                    values=arrays["first_run"].copy(),
                    metadata={
                        **{
                            k: v
                            for k, v in released.metadata.items()
                            if k
                            not in {
                                "source_algorithm",
                                "upstream_revisions",
                                "source_field",
                            }
                        },
                        "source_metadata": dict(released.metadata),
                        "measurement_source": "fixations",
                        "source_measure": "first_run",
                        "response_source": "default",
                        "calculation": FixationCalculation().to_dict(),
                        "fixation_source_sha256": RELEASES[
                            self.fixation_release_key
                        ].source_sha256,
                    },
                )
            )
        if references:
            raise ValueError("Some MECO summary trials have no source fixations")
        self._data.extend(standard)

    def _reaggregation_fixations(
        self, datum: Datum, events: list[Fixation]
    ) -> list[Fixation]:
        """Count exact repeated export rows once; retain them in raw access.

        The L2 wave 1 release repeats entire fixation sequences. Equal event
        IDs must also agree on every exported field before a copy is removed.
        """
        unique: dict[tuple, Fixation] = {}
        for event in events:
            key = (
                event.participant_id,
                event.stimulus_id,
                event.metadata.get("source_trial_id"),
                event.fixation_index,
            )
            previous = unique.get(key)
            if previous is not None and previous != event:
                raise ValueError(f"Conflicting MECO fixation rows for {key}")
            unique[key] = event
        return list(unique.values())

    def _reaggregation_arrays(self, events, stimulus, layer, **options):
        """Retain known blank-area landings in regression paths and first-pass order."""
        from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer

        arrays = aggregate_fixations_to_layer(
            events, stimulus, layer, include_inter_unit_fixations=True, **options
        )
        if stimulus.layers[layer].spans == stimulus.layers["words"].spans:
            arrays["_released_first_run"] = self._released_first_run(
                events, stimulus, arrays
            )
            arrays["_released_first_run_in_first_pass"] = np.where(
                arrays["first_pass"] == 1, arrays["_released_first_run"], 0.0
            )
        return arrays

    def _reaggregation_values(self, datum, arrays):
        """Validate released episode values separately from standard runs."""
        if datum.metadata.get(
            "response_source"
        ) == "released" and datum.measure_name in {
            "first_run",
            "first_run_in_first_pass",
        }:
            key = "_released_" + datum.measure_name
            if key not in arrays:
                raise ValueError(
                    "Released MECO episodes require native units; select a calculation"
                )
            arrays = {**arrays, datum.measure_name: arrays[key]}
            datum = replace(
                datum,
                metadata={
                    **datum.metadata,
                    "unfixated_is_missing": datum.measure_name == "first_run",
                },
            )
        return super()._reaggregation_values(datum, arrays)

    def _released_first_run(self, events, stimulus, arrays):
        """Reproduce released native runs using their own area's episode labels."""
        from replica.dataset.loading.reaggregation import map_fixations_to_layer

        from ...loading.fixation_utils import first_run_durations

        if (
            self.fixation_release_key == "meco_l2_w2"
            and events[0].metadata["site"] == "ch_s"
        ):
            return arrays["first_run"].copy()
        mapped = map_fixations_to_layer(
            events, stimulus, "words", use_source_runs=False
        )
        first_episode = {}
        ranks = []
        for event in mapped:
            episode = event.metadata.get("source_word_run_id")
            if event.unit_index is not None and episode is None:
                raise ValueError("MECO fixation lacks its released word episode")
            key = (event.metadata.get("trial_index"), episode)
            first_episode.setdefault(event.unit_index, key)
            ranks.append(1 if key == first_episode[event.unit_index] else 2)
        return first_run_durations(
            [event.unit_index for event in mapped],
            [event.duration for event in mapped],
            {i: i for i in range(len(stimulus.layers["words"]))},
            len(stimulus.layers["words"]),
            run_ids=ranks,
        )

    def fixation_trials(self) -> Iterator[tuple[tuple[str, str], list[Fixation]]]:
        """Read participant/stimulus groups without materializing the release."""
        for key, events in groupby(
            self.iter_fixations(),
            key=lambda event: (event.participant_id, event.stimulus_id),
        ):
            yield key, list(events)

    def _retain_released_first_runs(self):
        """Keep source summaries separate from the standard default response."""
        released = [
            d
            for d in self._data
            if d.measure_name == "first_run"
            and d.metadata.get("measurement_source") != "fixations"
        ]
        if not released:
            return
        self._released_data = released
        self._data = [d for d in self._data if d.measure_name != "first_run"]
        for datum in self._released_data:
            site = datum.metadata.get("site", datum.stimulus.metadata.get("site"))
            algorithm = (
                "standard@1"
                if self.fixation_release_key == "meco_l2_w2" and site == "ch_s"
                else "popeye_episode@1"
            )
            datum.metadata = {
                **datum.metadata,
                "response_source": "released",
                "source_field": "firstrun.dur",
                "source_release": self.fixation_release_key,
                "source_algorithm": algorithm,
                "upstream_revisions": FixationCalculation(
                    first_run=algorithm
                ).to_dict()["upstream_revisions"],
            }

    def _configure_fixation_store(self, path: str | Path) -> None:
        self._retain_released_first_runs()
        release = RELEASES[self.fixation_release_key]
        directory = Path(path)
        self._fixation_rda_path = directory / release.rda_filename
        self._fixation_parquet_path = directory / release.parquet_filename
        self._fixation_lookup: dict[tuple[str, int], _StimulusMapping] = {}
        self._fixation_alignment_method: dict[tuple[str, int], str] = {}
        self._fixation_candidates: dict[int, dict[str, _StimulusMapping]] = defaultdict(
            dict
        )
        self._fixation_site_candidates: dict[tuple[str, int], Counter[str]] = (
            defaultdict(Counter)
        )
        seen: set[tuple[str, int]] = set()
        mappings: dict[str, _StimulusMapping] = {}
        for datum in self._data:
            if datum.measure_name in {"go_past", "go_past_in_first_pass"}:
                datum.metadata["go_past_requires_right_exit"] = True
            item_id = int(
                datum.metadata.get(
                    "source_item_id", datum.stimulus.metadata["source_item_id"]
                )
            )
            key = (datum.participant_id, item_id)
            stimulus = datum.stimulus
            if stimulus.stimulus_id not in mappings:
                mappings[stimulus.stimulus_id] = self._source_mapping(stimulus)
            mapping = mappings[stimulus.stimulus_id]
            self._fixation_candidates[item_id][datum.stimulus.stimulus_id] = mapping
            site = str(
                datum.metadata.get("site", datum.stimulus.metadata.get("site", ""))
            )
            self._fixation_site_candidates[(site, item_id)][
                datum.stimulus.stimulus_id
            ] += 1
            if key in seen:
                continue
            seen.add(key)
            self._fixation_lookup[key] = mapping
            self._fixation_alignment_method[key] = "reading_measures"

    def _source_mapping(self, stimulus: Stimulus) -> _StimulusMapping:
        """Map released word and blank-area IDs to their source spans."""
        numbers = stimulus.metadata["source_word_numbers"]
        layer = "words"
        nonwords = stimulus.metadata.get("source_nonword_numbers", [])
        if nonwords:
            regions = dict(zip(numbers, stimulus.layers[layer].spans, strict=True))
            for number in nonwords:
                before = [span.end for n, span in regions.items() if n < number]
                after = [span.start for n, span in regions.items() if n > number]
                start = max(before, default=0)
                end = min(after, default=len(stimulus.text))
                if start >= end or not stimulus.text[start:end].isspace():
                    raise ValueError(f"MECO blank area {number} has no whitespace span")
                regions[number] = Span(start, end)
            numbers = sorted(regions)
            spans = [regions[n] for n in numbers]
            layer = "fixation_areas"
            stimulus.add_layer(
                AnnotationLayer(
                    layer, spans, [span.extract(stimulus.text) for span in spans]
                )
            )
        counts = Counter(numbers)
        return _StimulusMapping(
            stimulus=stimulus,
            word_indices={
                int(number): i
                for i, number in enumerate(numbers)
                if counts[number] == 1
            },
            char_offsets=[span.start for span in stimulus.layers[layer].spans],
            unit_layer=layer,
        )

    def _fixation_dataset(self) -> pads.Dataset:
        release = RELEASES[self.fixation_release_key]
        site = getattr(self, "site", None) or getattr(self, "fixation_site", None)
        if (
            release.available_sites is not None
            and site is not None
            and site not in release.available_sites
        ):
            raise NotImplementedError(
                f"The upstream {release.name} fixation RDA excludes site {site!r}; "
                "its word-level reading measures remain available."
            )
        if not self._fixation_parquet_path.is_file():
            raw_hint = (
                f" Then run `replica prepare-fixations {self.fixation_release_key}`."
                if self._fixation_rda_path.is_file()
                else f" Run `replica fetch {self.fixation_release_key}_fixations` first,"
                f" then `replica prepare-fixations {self.fixation_release_key}`."
            )
            raise FileNotFoundError(
                f"Prepared {release.name} fixations not found at "
                f"{self._fixation_parquet_path}.{raw_hint}"
            )
        validate_parquet(self._fixation_parquet_path, release)
        return pads.dataset(self._fixation_parquet_path, format="parquet")

    def _fixation_filter(
        self,
        participant: str | None,
        stimulus_id: str | None,
        dataset: pads.Dataset | None = None,
    ):
        expression = None
        if participant is not None:
            expression = pads.field("participant_id") == participant
        site = getattr(self, "site", None) or getattr(self, "fixation_site", None)
        if site is not None:
            site_filter = pads.field("site") == site
            expression = site_filter if expression is None else expression & site_filter
        if stimulus_id is not None:
            if dataset is None:
                raise ValueError("dataset is required for a stimulus fixation filter")
            item_ids = [
                item_id
                for item_id, candidates in self._fixation_candidates.items()
                if stimulus_id in candidates
            ]
            if not item_ids:
                return pads.field("participant_id") == "__no_such_participant__"
            item_filter = pads.field("source_item_id").isin(item_ids)
            scan_filter = (
                item_filter if expression is None else expression & item_filter
            )
            trials = dataset.to_table(
                columns=["participant_id", "site", "source_item_id"],
                filter=scan_filter,
            )
            raw_keys = set(
                zip(
                    trials["participant_id"].to_pylist(),
                    trials["site"].to_pylist(),
                    trials["source_item_id"].to_pylist(),
                    strict=True,
                )
            )
            keys = []
            for participant_id, trial_site, item_id in raw_keys:
                mapping = self._resolve_fixation_mapping(
                    participant_id, item_id, trial_site, dataset
                )
                if mapping is not None and mapping.stimulus.stimulus_id == stimulus_id:
                    keys.append((participant_id, item_id))
            if not keys:
                return pads.field("participant_id") == "__no_such_participant__"
            pair_filter = None
            for participant_id, item_id in keys:
                condition = (pads.field("participant_id") == participant_id) & (
                    pads.field("source_item_id") == item_id
                )
                pair_filter = (
                    condition if pair_filter is None else pair_filter | condition
                )
            expression = pair_filter if expression is None else expression & pair_filter
        return expression

    @staticmethod
    def _clean_source_word(value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(str(value).strip().split())

    def _resolve_fixation_mapping(
        self,
        participant_id: str,
        item_id: int,
        site: str,
        dataset: pads.Dataset,
    ) -> _StimulusMapping | None:
        """Resolve trials absent from the trimmed reading-measure frame.

        Some participants have raw events for all 12 texts but retained word
        measures for only a subset. A single-template item is unambiguous. For
        L2W2's split/joined hyphen variants, compare the raw word labels and use
        the collection site's majority only when the fixated labels tie.
        """
        key = (participant_id, item_id)
        if key in self._fixation_lookup:
            return self._fixation_lookup[key]
        candidates = list(self._fixation_candidates.get(item_id, {}).values())
        if not candidates:
            return None
        if len(candidates) == 1:
            chosen = candidates[0]
            method = "single_item_template"
        else:
            trial = dataset.to_table(
                columns=["source_word_number", "source_word"],
                filter=(pads.field("participant_id") == participant_id)
                & (pads.field("source_item_id") == item_id),
            )
            labels: dict[int, str] = {}
            for number, word in zip(
                trial["source_word_number"].to_pylist(),
                trial["source_word"].to_pylist(),
                strict=True,
            ):
                if number is not None and word is not None:
                    labels.setdefault(number, self._clean_source_word(word) or "")

            scored: list[tuple[int, int, _StimulusMapping]] = []
            for candidate in candidates:
                words = candidate.stimulus.layers[candidate.unit_layer].values
                if candidate.unit_layer == "words":
                    words = candidate.stimulus.metadata.get("source_word_labels", words)
                matches = contradictions = 0
                for number, word in labels.items():
                    index = candidate.word_indices.get(number)
                    if index is None or words[index] != word:
                        contradictions += 1
                    else:
                        matches += 1
                scored.append((contradictions, -matches, candidate))
            best_score = min(
                (contradictions, negative_matches)
                for contradictions, negative_matches, _ in scored
            )
            tied = [
                candidate
                for contradictions, negative_matches, candidate in scored
                if (contradictions, negative_matches) == best_score
            ]
            if len(tied) == 1:
                chosen = tied[0]
                method = "raw_word_labels"
            else:
                site_counts = self._fixation_site_candidates.get(
                    (site, item_id), Counter()
                )
                tied.sort(
                    key=lambda candidate: (
                        -site_counts[candidate.stimulus.stimulus_id],
                        candidate.stimulus.stimulus_id,
                    )
                )
                chosen = tied[0]
                method = "site_majority_tokenization"
        self._fixation_lookup[key] = chosen
        self._fixation_alignment_method[key] = method
        return chosen

    def fixation_table(
        self,
        participant: str | None = None,
        stimulus_id: str | None = None,
        *,
        columns: list[str] | None = None,
        aligned: bool = False,
    ) -> pa.Table:
        """Return a lazily-filtered Arrow table of fixation rows.

        Alignment is columnar but requires visiting the selected rows; predicate
        pushdown still happens first.

        Args:
            participant: Keep only this participant's rows; ``None`` keeps every
                participant the site filter admits.
            stimulus_id: Keep only the trials that align to this stimulus;
                ``None`` keeps every text.
            columns: Source columns to read, naming the canonical Parquet
                fields. ``None`` reads all of them, and a custom selection
                cannot be combined with *aligned*.
            aligned: Append ``unit_layer``, ``stimulus_id``, ``unit_index``, ``char_offset``,
                and ``alignment_method``, resolving each row against the
                reading-measure stimuli.

        Raises:
            ValueError: When *aligned* is combined with *columns*, or when a
                selected trial has no stimulus to align against.
        """
        if aligned and columns is not None:
            raise ValueError("aligned=True cannot be combined with custom columns")
        dataset = self._fixation_dataset()
        table = dataset.to_table(
            columns=columns,
            filter=self._fixation_filter(participant, stimulus_id, dataset),
        )
        if not aligned:
            return table

        stimulus_ids: list[str] = []
        unit_indices: list[int | None] = []
        char_offsets: list[int | None] = []
        methods: list[str] = []
        unit_layers: list[str] = []
        for participant_id, site, item_id, word_number in zip(
            table["participant_id"].to_pylist(),
            table["site"].to_pylist(),
            table["source_item_id"].to_pylist(),
            table["source_word_number"].to_pylist(),
            strict=True,
        ):
            mapping = self._resolve_fixation_mapping(
                participant_id, item_id, site, dataset
            )
            if mapping is None:
                raise ValueError(
                    f"No reading-measure stimulus can align fixation trial "
                    f"participant={participant_id!r}, item={item_id}"
                )
            unit_index = (
                mapping.word_indices.get(word_number)
                if word_number is not None
                else None
            )
            stimulus_ids.append(mapping.stimulus.stimulus_id)
            unit_layers.append(mapping.unit_layer)
            unit_indices.append(unit_index)
            char_offsets.append(
                mapping.char_offsets[unit_index] if unit_index is not None else None
            )
            methods.append(self._fixation_alignment_method[(participant_id, item_id)])
        return (
            table.append_column("unit_layer", pa.array(unit_layers))
            .append_column("stimulus_id", pa.array(stimulus_ids))
            .append_column("unit_index", pa.array(unit_indices, type=pa.int32()))
            .append_column("char_offset", pa.array(char_offsets, type=pa.int32()))
            .append_column("alignment_method", pa.array(methods))
        )

    def iter_fixations(
        self,
        participant: str | None = None,
        stimulus_id: str | None = None,
        *,
        batch_size: int = 65_536,
    ) -> Iterator[Fixation]:
        """Stream canonical fixation objects without materializing the corpus.

        Args:
            participant: Keep only this participant's events; ``None`` keeps
                every participant the site filter admits.
            stimulus_id: Keep only the trials that align to this stimulus;
                ``None`` keeps every text.
            batch_size: Rows per Arrow scan batch, which bounds how many events
                are held as Python objects at once.

        Returns:
            One :class:`Fixation` per event, in the store's temporal sort order,
            with ``unit_index`` left as ``None`` outside the source areas.
            Known blank areas retain their whitespace spans in ``unit_layer``.
        """
        dataset = self._fixation_dataset()
        scanner = dataset.scanner(
            filter=self._fixation_filter(participant, stimulus_id, dataset),
            batch_size=batch_size,
        )
        for batch in scanner.to_batches():
            data = batch.to_pydict()
            for i, participant_id in enumerate(data["participant_id"]):
                item_id = data["source_item_id"][i]
                mapping = self._resolve_fixation_mapping(
                    participant_id,
                    item_id,
                    data["site"][i],
                    dataset,
                )
                if mapping is None:
                    raise ValueError(
                        f"No reading-measure stimulus can align fixation trial "
                        f"participant={participant_id!r}, item={item_id}"
                    )
                source_word_number = data["source_word_number"][i]
                unit_index = (
                    mapping.word_indices.get(source_word_number)
                    if source_word_number is not None
                    else None
                )
                char_offset = (
                    mapping.char_offsets[unit_index] if unit_index is not None else None
                )
                yield Fixation(
                    participant_id=participant_id,
                    stimulus_id=mapping.stimulus.stimulus_id,
                    start_time=data["start_time"][i],
                    end_time=data["end_time"][i],
                    duration=data["duration"][i],
                    x=data["x"][i],
                    y=data["y"][i],
                    unit_index=unit_index,
                    char_offset=char_offset,
                    fixation_index=data["fixation_index"][i],
                    unit_layer=mapping.unit_layer,
                    metadata={
                        "site": data["site"][i],
                        "source_item_id": item_id,
                        "source_trial_id": data["source_trial_id"][i],
                        "trial_index": data["source_trial_id"][i],
                        "source_word_number": source_word_number,
                        "source_word": data["source_word"][i],
                        "source_ia_number": data["source_ia_number"][i],
                        "source_ia": data["source_ia"][i],
                        "pupil_size": data["pupil_size"][i],
                        "blink": data["blink"][i],
                        # MECO's word.runid is not an EyeLink per-area visit ID.
                        "source_word_run_id": data["run_id"][i],
                        "alignment_method": self._fixation_alignment_method[
                            (participant_id, item_id)
                        ],
                    },
                )

    def fixations(
        self,
        participant: str | None = None,
        stimulus_id: str | None = None,
    ) -> list[Fixation]:
        """Materialize a filtered fixation list; reject unsafe all-corpus use.

        Args:
            participant: Participant whose events to collect; ``None`` admits
                every participant who read *stimulus_id*.
            stimulus_id: Stimulus whose trials to collect; ``None`` admits every
                text *participant* read.

        Raises:
            ValueError: When both filters are ``None``. The unfiltered release
                is too large to hold as Python objects, so stream it through
                :meth:`iter_fixations` or read it with :meth:`fixation_table`.
        """
        if participant is None and stimulus_id is None:
            raise ValueError(
                "MECO has more than one million fixation events. Use "
                "fixation_table() or iter_fixations() for all events, or pass "
                "participant= / stimulus_id= to fixations()."
            )
        return list(self.iter_fixations(participant, stimulus_id))
