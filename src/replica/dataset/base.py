"""Dataset hierarchy for REPLICA.

The base classes here cover self-paced reading, eye tracking and EEG; the
maze base class lives in ``maze.base`` and fMRI loaders subclass ``Dataset``
directly under ``fmri``.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import pandas as pd

from replica.contracts.features import Feature
from replica.dataset.contracts import FixationCalculation
from replica.dataset.features import _dataset_feature
from replica.dataset.measures.sources import measure_sources

from ..core import (
    DERIVED_DVS,
    AnnotationLayer,
    Datum,
    Modality,
    Presentation,
    Recording,
    Stimulus,
)
from .contracts import DatasetInfo, Fixation
from .loading.utils import build_word_spans
from .measures import (
    CANONICAL_RT_MEASURES,
    FIRST_PASS_GO_PAST,
    FIRST_PASS_RUN,
    with_derived,
)


class Dataset(ABC):
    """Abstract base for all psycholinguistic datasets.

    Contract: a subclass provides ``info`` (:class:`DatasetInfo`)
    and after ``load()`` exposes its data via ``stimuli()`` and ``data()``.

    Unit granularity is controlled by the ``segmentation`` field of the
    ``EvaluationSpec`` passed to ``evaluate()``.

    Subclasses should override ``additive_measures`` to declare which
    measures can be meaningfully summed across finer-grained units when
    aggregating to a coarser segmentation.  Measures left out of that set
    require raw data at the target granularity and cannot be aggregated.
    """

    info: DatasetInfo
    additive_measures: frozenset[str] = frozenset()
    default_modality: str | Modality = Modality.BEHAVIORAL
    #: Presentation method shared by this loader's corpora; empty if unspecified.
    default_presentation: str | Presentation = ""

    _stimuli: list[Stimulus]
    _data: list[Datum]
    _recordings: list[Recording]

    def __init__(self):
        """Start empty; ``load()`` fills the stimuli, datums, and recordings."""
        self._stimuli = []
        self._data = []
        self._released_data: list[Datum] = []
        self._recordings = []

    def _load_metadata(
        self,
        path: str,
        verify: bool = True,
        files: Collection[str] | Mapping[str, str] | None = None,
        data_path: str | None = None,
        *,
        metadata_path: str | Path | None = None,
    ) -> None:
        """Load explicitly selected YAML metadata and optionally verify its files.

        Without ``metadata_path``, keep the loader's existing ``info``.
        Bundled loaders explicitly select their packaged metadata; custom
        loaders may supply any YAML file.

        Args:
            path: Dataset directory containing files to verify.
            verify: Verify file checksums listed in the YAML.
            files: Restrict verification to these YAML paths. Pass the files
                the current options actually read when the YAML declares
                alternatives (one entry per release, say), so selecting one
                does not require the others on disk.
                A mapping resolves declared paths to actual input filenames.
            data_path: Directory containing the selected inputs when it differs
                from ``path``.
            metadata_path: YAML file to read; ``None`` keeps existing metadata.
        """
        from replica.dataset.metadata.io import (
            load_yaml,
            verify_checksums,
        )

        if metadata_path is None:
            return

        info = load_yaml(metadata_path)

        # ``PSYCHO_SKIP_CHECKSUMS=1`` skips the SHA-256 recompute of every raw
        # file. Off by default; the test suite sets it.
        skip = os.environ.get("PSYCHO_SKIP_CHECKSUMS") == "1"
        if verify and not skip and info.files:
            entries = info.files
            if files is not None:
                wanted = set(files)
                entries = [entry for entry in entries if entry.path in wanted]
                if isinstance(files, Mapping):
                    entries = [
                        _replace(entry, path=files[entry.path]) for entry in entries
                    ]
            verify_checksums(data_path or path, entries)
        self.info = info

    def _make_stimulus(
        self,
        units: list[str],
        stimulus_id: str,
        metadata: dict | None = None,
        *,
        separator: str = " ",
    ) -> Stimulus:
        """Build a Stimulus from a list of units, register it, and return it.

        Joins units with the given separator, builds character-offset spans,
        attaches a ``"words"`` annotation layer, and appends the stimulus
        to ``self._stimuli``. Material fields declared in ``info`` derive a
        ``cv_group`` unless the stimulus already supplies one.

        Raises:
            ValueError: A declared material field is missing from the metadata.
        """
        metadata = dict(metadata or {})
        if "cv_group" not in metadata:
            material = self._material_identity(metadata)
            if material is not None:
                metadata["cv_group"] = material
        text = separator.join(units)
        spans = build_word_spans(units, separator=separator)
        layer = AnnotationLayer("words", spans, units)
        stimulus = Stimulus(text=text, stimulus_id=stimulus_id, metadata=metadata)
        stimulus.add_layer(layer)
        self._stimuli.append(stimulus)
        return stimulus

    def _material_identity(self, metadata: dict) -> str | None:
        r"""Derive an identity from ``info``, or None if no material is declared.

        Segments join with ``:``; a ``:`` or ``\`` inside a value is escaped
        with a backslash, so distinct field tuples never share an identity.

        Args:
            metadata: Stimulus metadata holding every declared material field.

        Raises:
            ValueError: A malformed declaration, absent field, or non-scalar,
                missing or blank field value.
        """
        name = type(self).__name__
        fields = self.info.material_fields
        namespace = self.info.material_namespace
        if not isinstance(fields, list) or any(
            not isinstance(f, str) or not f.strip() for f in fields
        ):
            raise ValueError(f"{name}: material_fields must be a list of field names")
        if not fields and namespace == "":
            return None
        if not fields:
            raise ValueError(f"{name}: material_fields is required with a namespace")
        if (
            not isinstance(namespace, str)
            or not namespace.strip()
            or ":" in namespace
            or "\\" in namespace
        ):
            raise ValueError(f"{name}: material_namespace {namespace!r} is not allowed")
        missing = [f for f in fields if f not in metadata]
        if missing:
            raise ValueError(
                f"{name} stimulus metadata lacks material fields {missing}"
            )
        segments = [namespace]
        for field_name in fields:
            value = metadata[field_name]
            if not pd.api.types.is_scalar(value):
                raise ValueError(
                    f"{name}: material field {field_name!r} is not a scalar"
                )
            if pd.isna(value):
                raise ValueError(f"{name}: material field {field_name!r} is missing")
            text = str(value)
            if not text.strip():
                raise ValueError(f"{name}: material field {field_name!r} is blank")
            segments.append(text.replace("\\", "\\\\").replace(":", "\\:"))
        return ":".join(segments)

    @abstractmethod
    def load(self, path: str) -> None:
        """Load dataset from disk. Populates _stimuli and _data.

        Args:
            path: Directory holding this corpus's data files.
        """
        ...

    def stimuli(self) -> list[Stimulus]:
        """All stimuli in the dataset."""
        return list(self._stimuli)

    def feature(
        self,
        quantity: str,
        *,
        source: str = "default",
        modality: str | None = None,
        name: str | None = None,
    ) -> Feature:
        """Select a stored quantity for any role in an evaluation.

        Args:
            quantity: Measurement name, or an exact declared field when
                ``source="released"``.
            source: Loader values or retained ``released`` values.
            modality: Acquisition modality when the name is ambiguous.
            name: Feature display name; defaults to the quantity name.
        """
        return _dataset_feature(
            self, quantity, source=source, modality=modality, name=name
        )

    def add_datum(self, datum: Datum) -> None:
        """Attach an extra Datum (e.g. an observation-level feature measure).

        Args:
            datum: Unit-aligned observation; an empty modality is filled in from
                ``default_modality``.
        """
        if not datum.modality:
            datum.modality = str(self.default_modality)
        self._data.append(datum)

    def add_recording(self, recording: Recording) -> None:
        """Attach a sampled response such as EEG, MEG, ECoG, or fMRI.

        Args:
            recording: Continuously sampled response, carrying its own modality.
        """
        self._recordings.append(recording)

    def recordings(
        self,
        measure: str | None = None,
        participant: str | None = None,
        modality: str | Modality | None = None,
        stimulus_id: str | None = None,
    ) -> list[Recording]:
        """Filter continuous recordings through the shared modality API.

        Args:
            measure: Keep recordings of this measure name; ``None`` keeps all.
            participant: Keep one participant's recordings; ``None`` keeps all.
            modality: Keep recordings acquired with this modality; ``None`` keeps
                all.
            stimulus_id: Keep recordings against this stimulus; ``None`` keeps all.

        Returns:
            The surviving recordings, in the order they were added.
        """
        result = self._recordings
        if measure is not None:
            result = [r for r in result if r.measure_name == measure]
        if participant is not None:
            result = [r for r in result if r.participant_id == participant]
        if modality is not None:
            modality = str(modality)
            result = [r for r in result if r.modality == modality]
        if stimulus_id is not None:
            result = [r for r in result if r.stimulus.stimulus_id == stimulus_id]
        return list(result)

    def responses(
        self,
        measure: str | None = None,
        participant: str | None = None,
        modality: str | Modality | None = None,
        stimulus_id: str | None = None,
    ) -> list[Datum | Recording]:
        """Return unit-aligned data and sampled recordings through one API.

        A datum without an explicit modality inherits the dataset's
        ``default_modality`` at query time.

        Args:
            measure: Keep responses of this measure name; ``None`` keeps all.
            participant: Keep one participant's responses; ``None`` keeps all.
            modality: Keep responses acquired with this modality; ``None`` keeps
                all.
            stimulus_id: Keep responses against this stimulus; ``None`` keeps all.

        Returns:
            The matching datums, then the matching recordings.
        """
        wanted_modality = str(modality) if modality is not None else None
        datums = self.data(measure=measure, participant=participant)
        if stimulus_id is not None:
            datums = [d for d in datums if d.stimulus.stimulus_id == stimulus_id]
        if wanted_modality is not None:
            datums = [
                d
                for d in datums
                if str(d.modality or self.default_modality) == wanted_modality
            ]
        recordings = self.recordings(
            measure=measure,
            participant=participant,
            modality=wanted_modality,
            stimulus_id=stimulus_id,
        )
        return [*datums, *recordings]

    def data(
        self,
        measure: str | None = None,
        participant: str | None = None,
        *,
        source: str = "default",
    ) -> list[Datum]:
        """Filter data by dependent variable and/or participant ID.

        Args:
            measure: Measure name to keep, or a derived dependent variable
                from :data:`~replica.core.DERIVED_DVS` such as
                ``first_run_in_first_pass``; ``None`` keeps all.
            participant: Keep one participant's datums; ``None`` keeps all.
            source: Default loader responses or retained released values.

        Returns:
            The matching datums, each with an empty modality filled in from
            ``default_modality``. A derived variable reads its measure where
            the flag is 1, zero where it is 0 and NaN where the flag is NaN or
            absent; those values are copies and the stored datums are
            untouched. A datum stored under the derived name itself, such as
            a participant aggregate, is already restricted and is returned as
            stored.

        Raises:
            ValueError: A derived variable was requested and no datum of this
                dataset carries its flag.
        """
        if source not in {"default", "released"}:
            raise ValueError("Response source must be 'default' or 'released'")
        result = self._data
        if source == "released":
            stream_measures = {
                name
                for name, declaration in measure_sources(self).items()
                if declaration.rule == "fixation_stream"
            }
            overrides = {
                (d.stimulus.stimulus_id, d.participant_id, d.layer, d.measure_name): d
                for d in self._released_data
            }
            for datum in self._data:
                if (
                    datum.metadata.get("measurement_source") == "fixations"
                    and "released_metadata" in datum.metadata
                ):
                    key = (
                        datum.stimulus.stimulus_id,
                        datum.participant_id,
                        datum.layer,
                        datum.measure_name,
                    )
                    overrides.setdefault(
                        key,
                        _replace(
                            datum,
                            values=np.asarray(
                                datum.metadata["released_values"], dtype=float
                            ),
                            metadata=dict(datum.metadata["released_metadata"]),
                        ),
                    )
            result = [
                d
                for d in self._data
                if d.metadata.get("measurement_source")
                not in {"fixations", "reaggregated_fixations"}
                and (
                    d.measure_name not in stream_measures
                    or d.metadata.get("measurement_source") == "released"
                )
                and (d.stimulus.stimulus_id, d.participant_id, d.layer, d.measure_name)
                not in overrides
            ] + list(overrides.values())
        requested_measure = measure
        condition = None
        if measure is not None:
            stored = [d for d in result if d.measure_name == measure]
            if not stored and measure in DERIVED_DVS:
                measure, condition = DERIVED_DVS[measure]
                stored = [d for d in result if d.measure_name == measure]
            result = stored
        if participant is not None:
            result = [d for d in result if d.participant_id == participant]
        if condition is not None:
            assert requested_measure is not None
            result = self._restricted_to(result, condition, requested_measure)
        for datum in result:
            if not datum.modality:
                datum.modality = str(self.default_modality)
        if source == "released":
            result = [
                _replace(d, metadata={**d.metadata, "response_source": "released"})
                for d in result
            ]
        return list(result)

    def _restricted_to(
        self, datums: list[Datum], condition: str, measure_name: str
    ) -> list[Datum]:
        """Copy datums, retaining values where the condition flag is 1.

        Use zero where the flag is 0 and NaN where it is missing.
        """
        flags = {
            (d.stimulus.stimulus_id, d.participant_id): d.values
            for d in self._data
            if d.measure_name == condition
        }
        if not flags:
            raise ValueError(
                f"{self.info.name} records no {condition} flag, so its measures "
                f"cannot be restricted to {condition}"
            )
        restricted = []
        for datum in datums:
            flag = flags.get((datum.stimulus.stimulus_id, datum.participant_id))
            values = np.asarray(datum.values, dtype=np.float64)
            if flag is None:
                values = np.full(values.shape, np.nan)
            else:
                values = np.where(
                    flag == 1.0, values, np.where(flag == 0.0, 0.0, np.nan)
                )
            restricted.append(_replace(datum, values=values, measure_name=measure_name))
        return restricted

    # Optional hint declared on subclasses: the canonical measures the loader
    # is expected to expose. It stands in for the stored measures only while
    # nothing is loaded, which is how ``describe_dataset`` lists them.
    _declared_measures: tuple[str, ...] = ()

    @property
    def measures(self) -> list[str]:
        """Stored measures, condition flags and the variables they derive.

        Before anything is loaded, the loader's declared measures stand in.
        """
        stored: set[str] = {d.measure_name for d in [*self._data, *self._released_data]}
        stored |= {r.measure_name for r in self._recordings}
        if not stored:
            stored = set(self._declared_measures)
        return sorted(with_derived(stored))

    @property
    def participants(self) -> list[str]:
        """Available participant IDs."""
        datum_ids = {d.participant_id for d in self._data if d.participant_id}
        recording_ids = {r.participant_id for r in self._recordings if r.participant_id}
        return sorted(datum_ids | recording_ids)

    @property
    def modalities(self) -> list[str]:
        """Acquisition modalities exposed by this dataset.

        Metadata declarations, loaded observations, and the loader's default
        are merged so modality discovery works both before and after ``load``.
        """
        declared = set(self.info.modalities)
        declared.update(str(d.modality) for d in self._data if d.modality)
        declared.update(r.modality for r in self._recordings)
        if self._data or not declared:
            declared.add(str(self.default_modality))
        return sorted(declared)

    @property
    def presentation(self) -> list[str]:
        """Return presentation methods from metadata and loader defaults.

        Return an empty list when neither source specifies a method.
        """
        declared = {str(p) for p in self.info.presentation if p}
        if self.default_presentation:
            declared.add(str(self.default_presentation))
        return sorted(declared)

    def is_additive(self, measure: str) -> bool:
        """True if *measure* is in ``additive_measures`` (can be summed).

        Args:
            measure: Measure name to test for summability across finer units.
        """
        if measure in DERIVED_DVS or measure in {
            "first_fixation",
            "first_run",
            "go_past",
            "right_bounded",
            "first_pass",
        }:
            return False
        return measure in self.additive_measures


class ReadingTimeDataset(Dataset):
    """Per-unit reading time dataset. Shared covariate computation and exclusion logic."""

    STANDARD_COVARIATES = [
        "unit_length",
        "log_frequency",
        "unit_position",
        "prev_unit_length",
    ]

    # Subclasses may override
    default_measure: str = ""

    # Word frequency table (log frequencies), assigned by the caller; while it
    # is None the ``log_frequency`` covariate is NaN.
    _word_frequencies: dict[str, float] | None = None

    def covariates(
        self, stimulus: Stimulus, target_layer: str = "words"
    ) -> dict[str, np.ndarray]:
        """Compute standard covariates for a stimulus, aligned to target layer.

        Args:
            stimulus: Stimulus whose units the covariates are computed over.
            target_layer: Annotation layer naming the units to align to.

        Returns:
            One array per name in ``STANDARD_COVARIATES``, each as long as the
            layer. ``prev_unit_length`` is NaN at the first unit, and
            ``log_frequency`` is all-NaN until a frequency table is attached.
        """
        layer = stimulus.layers[target_layer]
        strings = layer.extract_strings(stimulus.text)
        n = len(strings)
        covs: dict[str, np.ndarray] = {}

        # Unit length (characters)
        lengths = np.array([len(s) for s in strings], dtype=float)
        covs["unit_length"] = lengths

        # Previous unit length (NaN for first unit)
        prev_lengths = np.empty(n)
        prev_lengths[0] = np.nan
        prev_lengths[1:] = lengths[:-1]
        covs["prev_unit_length"] = prev_lengths

        # Unit position (0-indexed)
        covs["unit_position"] = np.arange(n, dtype=float)

        # Log frequency
        if self._word_frequencies is not None:
            covs["log_frequency"] = np.array(
                [self._word_frequencies.get(s.lower(), np.nan) for s in strings],
                dtype=float,
            )
        else:
            covs["log_frequency"] = np.full(n, np.nan)

        return covs

    #: First-unit exclusion. Set False when no independent boundary
    #: exclusion is intended; requested lag features still mask missing values.
    exclude_first_unit: bool = True

    def exclusion_mask(self, datum: Datum) -> np.ndarray:
        """Boolean keep-mask over units (True = include this observation).

        The legacy ``exclude_first_unit=True`` policy drops the first unit
        even without lags. Set it to False to let required lag features determine
        boundary eligibility. NaNs are always excluded. RT windows and outlier
        trimming are researcher choices the
        analysis layer applies.

        Args:
            datum: Observation whose per-unit values are screened.
        """
        values = datum.values
        mask: np.ndarray = np.ones(len(values), dtype=bool)
        if self.exclude_first_unit and len(mask):
            mask[0] = False
        mask &= ~np.isnan(values)
        return mask


class EyeTrackingDataset(ReadingTimeDataset):
    """Eye-tracking dataset with multiple reading time measures.

    Most loaders read released word-level interest-area measures.
    Loaders with fixation data expose :class:`Fixation` objects through
    :meth:`fixations`.

    :data:`CANONICAL_RT_MEASURES` defines the measurement names.
    ``first_run`` measures the first visit regardless of reading pass;
    ``first_run_in_first_pass`` restricts it to first-pass entry. The same
    distinction applies to ``go_past`` and ``go_past_in_first_pass``.
    Released conditional measurements are stored directly. Unconditional
    values remain unavailable where the source does not establish them.

    The default measure is ``first_run_in_first_pass``, which needs a
    ``first_pass`` flag datum per response; a loader without that flag
    overrides ``default_measure`` (``"first_run"`` for ZuCo).
    """

    default_measure = FIRST_PASS_RUN
    default_modality = Modality.EYE_TRACKING
    default_presentation = Presentation.NATURAL_READING
    additive_measures = frozenset({"total_time", "total_fixation_duration"})

    #: How a word a participant never fixated enters a duration measure. Loaders
    #: store observed skips as zero and unavailable measurements as NaN.
    #: With ``drop_skips`` the zero is masked in the returned view; otherwise
    #: it remains 0 ms. Set the attribute on a loaded dataset, or pass
    #: ``drop_skips=`` to
    #: :func:`replica.dataset.load_dataset`.
    drop_skips: bool = True
    _unfixated_nan_measures: frozenset[str] = frozenset()

    def data(
        self,
        measure: str | None = None,
        participant: str | None = None,
        *,
        source: str = "default",
        calculation: FixationCalculation | None = None,
    ) -> list[Datum]:
        """Select native responses or recompute them with explicit algorithms.

        Args:
            measure: Generic measure name; ``None`` keeps all native measures.
            participant: Keep one participant's responses; ``None`` keeps all.
            source: Default loader responses or retained released values.
            calculation: Versioned calculation on native units, after validation
                against the selected source. Requires a specific measure.

        Returns:
            Responses with duration zeros masked when ``drop_skips`` is true.
            Stored values and unavailable observations remain unchanged.
        """
        result = super().data(measure, participant, source=source)
        if self.drop_skips:
            result = [self._skips_as_missing(datum) for datum in result]
        if calculation is not None:
            if measure is None:
                raise ValueError("An explicit calculation requires a measure")
            references = self._native_responses(measure, source=source)
            if not result:
                raise ValueError(
                    "Calculation requires source responses and fixation events"
                )
            requests = []
            for datum in result:
                key = (
                    datum.stimulus.stimulus_id,
                    datum.participant_id,
                    datum.layer,
                    measure,
                )
                requests.append((key, datum, references[key]))
            calculated = self._reaggregate_responses(
                requests, None, {measure: calculation}
            )
            result = [calculated[key] for key, _, _ in requests]
        return result

    def _add_arrival_go_past(self) -> None:
        """Add arrival-based go-past where the released measurements establish it.

        Retain positive conditional durations and values on first-pass units.
        Retain a conditional zero when total dwell is also zero; use NaN for
        other observations. Preserve the stored first-pass measurement.
        """
        observations = {
            (d.stimulus.stimulus_id, d.participant_id, d.measure_name): d.values
            for d in self._data
        }
        for datum in list(self._data):
            if datum.measure_name != FIRST_PASS_GO_PAST:
                continue
            key = (datum.stimulus.stimulus_id, datum.participant_id)
            if (*key, "go_past") in observations:
                raise ValueError(
                    "Arrival-based go-past already exists for this observation"
                )
            source = np.asarray(datum.values, dtype=float)
            flag = observations.get((*key, "first_pass"), np.full(source.shape, np.nan))
            total = observations.get(
                (*key, "total_time"), np.full(source.shape, np.nan)
            )
            known = (source > 0) | (flag == 1) | ((source == 0) & (total == 0))
            self._data.append(
                _replace(
                    datum,
                    measure_name="go_past",
                    values=np.where(known, source, np.nan),
                    metadata={
                        **datum.metadata,
                        "source_measure": FIRST_PASS_GO_PAST,
                        "unavailable_outside_first_pass": True,
                    },
                )
            )

    @staticmethod
    def _skips_as_missing(datum: Datum) -> Datum:
        """Read released duration zeros as missing without modifying storage."""
        if datum.measure_name not in CANONICAL_RT_MEASURES:
            return datum
        values = np.asarray(datum.values, dtype=np.float64)
        if not (values == 0.0).any():
            return datum
        return _replace(datum, values=np.where(values == 0.0, np.nan, values))

    def exclusion_mask(self, datum: Datum) -> np.ndarray:
        """Apply structural exclusions and the selected duration skip policy.

        Args:
            datum: Observation to screen, including an unconverted stored datum.
        """
        mask = super().exclusion_mask(datum)
        if self.drop_skips and datum.measure_name in CANONICAL_RT_MEASURES:
            mask &= datum.values != 0.0
        return mask

    STANDARD_COVARIATES = [
        *ReadingTimeDataset.STANDARD_COVARIATES,
        "prev_word_fixated",
        "saccade_launch_distance",
    ]

    def covariates(
        self, stimulus: Stimulus, target_layer: str = "words"
    ) -> dict[str, np.ndarray]:
        """Extend base covariates with eye-tracking specific ones.

        ``prev_word_fixated`` and ``saccade_launch_distance`` depend on
        participant fixation streams. This stimulus-level method returns NaN
        placeholders for them.

        Args:
            stimulus: Stimulus whose units the covariates are computed over.
            target_layer: Annotation layer naming the units to align to.
        """
        covs = super().covariates(stimulus, target_layer)
        n = len(stimulus.layers[target_layer])
        covs["prev_word_fixated"] = np.full(n, np.nan)
        covs["saccade_launch_distance"] = np.full(n, np.nan)
        return covs

    def _native_responses(
        self, measure: str, *, source: str = "default"
    ) -> dict[tuple[str, str, str, str], Datum]:
        """Index unmasked native responses once for a batch of reaggregations."""
        result = {}
        for datum in Dataset.data(self, measure, source=source):
            key = (
                datum.stimulus.stimulus_id,
                datum.participant_id,
                datum.layer,
                measure,
            )
            if key in result:
                raise ValueError(
                    f"Duplicate response identity for reaggregation: {key}"
                )
            result[key] = datum
        return result

    def _reaggregation_fixations(
        self, datum: Datum, events: list[Fixation]
    ) -> list[Fixation]:
        """Select the source presentation used by the released response.

        Loaders may additionally apply their documented event assignment rules.
        Selection must be the same for every measure of a presentation.
        """
        trial = datum.metadata.get("source_trial_index")
        if trial is None:
            return events
        return [event for event in events if event.metadata.get("trial_index") == trial]

    def fixation_trials(self) -> Iterator[tuple[tuple[str, str], list[Fixation]]]:
        """Yield each participant/stimulus key with its complete fixation stream.

        Loaders backed by sorted event stores can override this to read one
        trial at a time. Repeated presentations remain in the same group.
        """
        groups: dict[tuple[str, str], list[Fixation]] = {}
        for event in self.fixations():
            groups.setdefault((event.participant_id, event.stimulus_id), []).append(
                event
            )
        yield from groups.items()

    def _reaggregate_responses(self, requests, target_layer, calculations=None):
        """Recompute all requested measures while retaining one trial's event cache."""
        calculations = calculations or {}
        explicit = any(value is not None for value in calculations.values())
        pending: dict[tuple[str, str], list] = {}
        for key, datum, reference in requests:
            pending.setdefault((datum.participant_id, key[0]), []).append(
                (key, datum, reference)
            )
        result: dict[tuple[str, str, str, str], Datum] = {}
        seen = set()
        try:
            trials = iter(self.fixation_trials())
            first = next(trials, None)
        except NotImplementedError:
            if explicit:
                raise
            return result
        if first is None:
            if explicit:
                raise ValueError("Explicit calculations require raw fixation events")
            return result
        from itertools import chain

        for pair, events in chain([first], trials):
            if pair in seen:
                raise ValueError(
                    f"Duplicate fixation group for participant/stimulus: {pair}"
                )
            seen.add(pair)
            cache: dict = {}
            for key, datum, reference in pending.pop(pair, []):
                if key in result:
                    raise ValueError(
                        f"Duplicate response identity for reaggregation: {key}"
                    )
                result[key] = self.reaggregate(
                    datum,
                    datum.layer if target_layer is None else target_layer,
                    fixations=events,
                    measure=key[3],
                    _reference=reference,
                    _cache=cache,
                    calculation=calculations.get(key[3]),
                )
        if pending:
            raise ValueError(
                "Reaggregation requires source events; an absent trial is not a skip"
            )
        return result

    def _reaggregation_values(
        self, datum: Datum, arrays: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply the release's measure availability rules to recomputed arrays."""
        name = datum.measure_name
        key = "total_time" if name == "total_fixation_duration" else name
        if key not in arrays:
            raise NotImplementedError(
                f"Fixation reaggregation does not implement {name!r}"
            )
        values = arrays[key].copy()
        unfixated = (
            np.isnan(arrays["first_fixation"])
            & (arrays["total_time"] == 0)
            & np.isfinite(arrays["first_pass"])
        )
        if key == "first_fixation":
            values[unfixated] = 0.0
        if name in self._unfixated_nan_measures or datum.metadata.get(
            "unfixated_is_missing", False
        ):
            values[unfixated] = np.nan
        if key == "go_past" and datum.metadata.get(
            "unavailable_outside_first_pass", False
        ):
            values[(arrays["first_pass"] == 0) & (arrays["total_time"] > 0)] = np.nan
        return values

    def _reaggregation_arrays(self, events, stimulus, layer, **options):
        """Calculate the shared reading measures and any loader-specific quantities."""
        from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer

        return aggregate_fixations_to_layer(events, stimulus, layer, **options)

    def _has_empty_fixation_record(self, datum: Datum, *, selected: bool) -> bool:
        """Whether the source establishes a complete trial with no retained events."""
        return False

    def _reaggregation_provenance(self, datum: Datum, layer: str) -> dict:
        """Record loader-specific calculation and event-assignment policies."""
        return {}

    def _reaggregation_measure_response(self, datum: Datum, measure: str) -> Datum:
        """Find another native measure of the same source presentation."""
        key = (datum.stimulus.stimulus_id, datum.participant_id, datum.layer, measure)
        requested = self._native_responses(measure).get(key)
        if requested is None:
            raise ValueError(
                "Requested measure has no native response for this presentation"
            )
        if requested.metadata.get("source_trial_index") != datum.metadata.get(
            "source_trial_index"
        ):
            raise ValueError("Requested measure refers to another presentation")
        return requested

    def reaggregate(
        self,
        datum: Datum,
        target_layer: str,
        *,
        fixations: list[Fixation] | None = None,
        measure: str | None = None,
        calculation: FixationCalculation | None = None,
        _reference: Datum | None = None,
        _cache: dict | None = None,
    ) -> Datum:
        """Recompute a response on custom spans from its source fixation stream.

        The source calculation must first reproduce the native response, including
        missingness, with the current source and skip policies. An explicit
        calculation then uses the validated events with its own windows and
        policies. Released summaries are never substituted or summed here.

        Args:
            datum: Native response identifying the participant, presentation
                and source layer; its measure supplies the validation reference.
            target_layer: Existing annotation layer on the datum's stimulus.
            fixations: Optional preselected events for this participant/stimulus.
            measure: Requested fixation measure; defaults to the datum's measure.
                If exported, its own native response must also round-trip. An
                unexported measure uses the validated datum's event selection
                and records that it has no native reference.
            calculation: Explicit versioned algorithms and event policies. These
                apply after native validation, including on native spans. The
                requested measure must have a generic name; released source
                names do not identify an alternative calculation.

        Returns:
            A new target-layer datum; native measurements remain unchanged.

        Raises:
            ValueError: Missing events or a native round-trip disagreement.
            NotImplementedError: The source or requested measure lacks event support.
        """
        events = (
            self.fixations(datum.participant_id, datum.stimulus.stimulus_id)
            if fixations is None
            else fixations
        )
        if not events and not self._has_empty_fixation_record(datum, selected=False):
            raise ValueError(
                "Reaggregation requires source events; an absent trial is not a skip"
            )
        if any(
            e.participant_id != datum.participant_id
            or e.stimulus_id != datum.stimulus.stimulus_id
            for e in events
        ):
            raise ValueError(
                "Fixations do not match the datum's participant and stimulus"
            )
        source_events = events
        selection_key = ("events", id(events), datum.metadata.get("source_trial_index"))
        selection = None if _cache is None else _cache.get(selection_key)
        if selection is None:
            events = self._reaggregation_fixations(datum, events)
            if _cache is not None:
                # Retain both lists so their identities cannot be reused in a batch.
                _cache[selection_key] = (source_events, events)
        else:
            _, events = selection
        if not events and not self._has_empty_fixation_record(datum, selected=True):
            raise ValueError("Source events do not contain the released presentation")
        options = dict(
            floor_durations=datum.metadata.get("duration_rounding") == "floor_ms",
            off_text_ends_run=datum.metadata.get("off_text_ends_run", True),
            go_past_includes_off_text=datum.metadata.get(
                "go_past_includes_off_text", False
            ),
            go_past_requires_right_exit=datum.metadata.get(
                "go_past_requires_right_exit", False
            ),
            use_source_runs=datum.metadata.get("measurement_source") != "fixations",
        )
        name = datum.measure_name

        def compute(layer: str, response: Datum = datum) -> Datum:
            calculation = dict(options)
            if response.measure_name != name:
                calculation["use_source_runs"] = False
            cache_key = (id(events), id(datum.stimulus), layer, *calculation.values())
            arrays = None if _cache is None else _cache.get(cache_key)
            if arrays is None:
                arrays = self._reaggregation_arrays(
                    events, datum.stimulus, layer, **calculation
                )
                if _cache is not None:
                    _cache[cache_key] = arrays
            result = _replace(
                response,
                layer=layer,
                values=self._reaggregation_values(response, arrays),
                metadata={
                    "measurement_source": "reaggregated_fixations",
                    "source_layer": datum.layer,
                    "source_metadata": dict(datum.metadata),
                    "native_reference_measure": name,
                    "has_native_reference": response.measure_name == name,
                    **calculation,
                    **self._reaggregation_provenance(datum, layer),
                },
            )
            return result

        native = compute(datum.layer)
        self._validate_native_response(datum, native.values, _reference)
        if calculation is not None:
            return self._calculate_response(
                datum,
                target_layer,
                events,
                name if measure is None else measure,
                calculation,
                _cache,
            )
        if measure is not None and measure != name:
            if measure in self.measures:
                requested = self._reaggregation_measure_response(datum, measure)
                return self.reaggregate(
                    self._skips_as_missing(requested) if self.drop_skips else requested,
                    target_layer,
                    fixations=source_events,
                    _reference=requested,
                    _cache=_cache,
                )
            # Availability annotations belong to the released measure. A new
            # fixation-derived quantity retains unknown events and known skips.
            response = _replace(datum, measure_name=measure, metadata={})
            result = compute(target_layer, response)
        else:
            result = native if target_layer == datum.layer else compute(target_layer)
        return self._skips_as_missing(result) if self.drop_skips else result

    def _validate_native_response(
        self, datum: Datum, values: np.ndarray, reference: Datum | None
    ) -> None:
        """Require agreement with the stored source response and its visible view."""
        name = datum.measure_name
        if reference is None:
            references = [
                d
                for d in Dataset.data(
                    self,
                    name,
                    datum.participant_id,
                    source=datum.metadata.get("response_source", "default"),
                )
                if d.stimulus.stimulus_id == datum.stimulus.stimulus_id
                and d.layer == datum.layer
            ]
            if len(references) != 1:
                raise ValueError("Reaggregation requires a unique native response")
            reference = references[0]
        if (
            reference.stimulus is not datum.stimulus
            or reference.layer != datum.layer
            or reference.participant_id != datum.participant_id
            or reference.measure_name != name
        ):
            raise ValueError("Native reference does not match the requested response")
        visible = self._skips_as_missing(reference) if self.drop_skips else reference
        if not (
            np.allclose(values, reference.values, rtol=0, atol=1e-6, equal_nan=True)
            and np.allclose(
                datum.values, visible.values, rtol=0, atol=1e-6, equal_nan=True
            )
        ):
            raise ValueError(
                f"Cannot reaggregate {self.info.name} {name!r} for "
                f"{datum.participant_id}/{datum.stimulus.stimulus_id}: "
                "source fixations do not reproduce the native measure and missingness"
            )

    def _calculate_response(
        self,
        datum: Datum,
        layer: str,
        events: list[Fixation],
        measure: str,
        calculation: FixationCalculation,
        cache: dict | None,
    ) -> Datum:
        """Apply an explicit calculation to events whose native anchor passed."""
        from replica.dataset.loading.reaggregation import aggregate_fixations_to_layer

        spans = tuple((s.start, s.end) for s in datum.stimulus.layers[layer].spans)
        key = ("calculation", id(events), id(datum.stimulus), layer, spans, calculation)
        arrays = None if cache is None else cache.get(key)
        if arrays is None:
            arrays = aggregate_fixations_to_layer(
                events, datum.stimulus, layer, calculation=calculation
            )
            if cache is not None:
                cache[key] = arrays
        if measure not in arrays:
            raise ValueError(
                f"Explicit calculations require a generic fixation measure, got {measure!r}"
            )
        response = _replace(datum, measure_name=measure, metadata={})
        values = arrays[measure].copy()
        if measure == "first_fixation":
            values[(arrays["total_time"] == 0) & np.isfinite(arrays["first_pass"])] = 0
        result = _replace(
            response,
            layer=layer,
            values=values,
            metadata={
                "measurement_source": "reaggregated_fixations",
                "calculation": calculation.to_dict(),
                "source_dataset": self.info.name,
                "source_layer": datum.layer,
                "source_metadata": dict(datum.metadata),
                "native_reference_measure": datum.measure_name,
                "has_native_reference": False,
                "target_spans": spans,
                "drop_skips": self.drop_skips,
            },
        )
        return self._skips_as_missing(result) if self.drop_skips else result

    def fixations(
        self,
        participant: str | None = None,
        stimulus_id: str | None = None,
    ) -> list[Fixation]:
        """Return raw fixation events, optionally filtered.

        A filtered view of the ``self._fixations`` that :meth:`load` fills, or
        ``[]`` for a loader without a fixation-level source; a loader that needs
        more overrides this method. The aggregated :meth:`data` is the
        canonical API, and this one gives event-level access.

        Args:
            participant: Keep one participant's fixations; ``None`` keeps all.
            stimulus_id: Keep fixations on this stimulus; ``None`` keeps all.
        """
        result = getattr(self, "_fixations", None)
        if not result:
            return []
        if participant is not None:
            result = [f for f in result if f.participant_id == participant]
        if stimulus_id is not None:
            result = [f for f in result if f.stimulus_id == stimulus_id]
        return result


class SPRDataset(ReadingTimeDataset):
    """Self-paced reading dataset with a single RT measure."""

    default_measure = "spr_rt"
    default_presentation = Presentation.SELF_PACED_READING
    additive_measures = frozenset({"spr_rt"})


class EEGDataset(ReadingTimeDataset):
    """EEG dataset (e.g. frequency-band power from sentence reading).

    Shares the sentence-reading paradigm with eye-tracking and SPR
    (covariates, evaluate pipeline, exclusion mask).
    """

    default_measure: str = ""
    default_modality = Modality.EEG


@dataclass
class TabularDatasetConfig:
    """Configuration for building a TabularDataset from tabular data."""

    name: str
    language: str = "en"
    text_col: str = "ia"
    trial_col: str = "trialid"
    unit_index_col: str = "ianum"
    measure_col: str = "gaze_rt"
    #: Name the dataset exposes the measure under, when the frame's column is
    #: named after the release rather than the quantity.
    measure_name: str = ""
    participant_col: str | None = None


class TabularDataset(ReadingTimeDataset):
    """Dataset built from tabular data (DataFrame or dict of arrays).

    Used when data is already in memory (e.g., loaded from CSV) instead of loaded
    from a canonical dataset format. Each unique value in ``trial_col`` becomes
    a Stimulus; rows within a trial become units.
    """

    def __init__(
        self,
        data: pd.DataFrame | dict,
        config: TabularDatasetConfig,
    ):
        """Build stimuli and datums from an in-memory table.

        Args:
            data: One row per unit, and per participant as well when
                ``config.participant_col`` is set. A dict of columns is wrapped
                in a DataFrame.
            config: Which columns carry the trial, unit index, text, measure, and
                participant, plus the dataset's name and language.
        """
        super().__init__()

        self.info = DatasetInfo(name=config.name, language=config.language)
        self.default_measure = config.measure_name or config.measure_col
        self._config = config
        self._build(data if isinstance(data, pd.DataFrame) else pd.DataFrame(data))

    def load(self, path: str) -> None:
        """Unsupported: the table arrives through the constructor instead.

        Args:
            path: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use constructor with data")

    def exclusion_mask(self, datum: Datum) -> np.ndarray:
        """Exclude NaN only, no additional exclusions.

        Args:
            datum: Observation whose per-unit values are screened.
        """
        return ~np.isnan(datum.values)

    def _build(self, df: pd.DataFrame) -> None:

        cfg = self._config
        for trial_id, trial_df in df.groupby(cfg.trial_col):
            # Deduplicate units: one row per unit position
            unit_df = trial_df.drop_duplicates(subset=[cfg.unit_index_col]).sort_values(
                cfg.unit_index_col
            )
            words = [
                str(w) if pd.notna(w) else "?" for w in unit_df[cfg.text_col].values
            ]
            stimulus = self._make_stimulus(
                words,
                f"trial_{trial_id}",
                {"trial_id": trial_id},
            )

            if cfg.participant_col is not None:
                # Canonical unit order from unit_df; reindex per participant so
                # values align with words even with missing/duplicate unit rows.
                unit_order = unit_df[cfg.unit_index_col].values
                for pid, p_df in trial_df.groupby(cfg.participant_col):
                    p_units = p_df.drop_duplicates(
                        subset=[cfg.unit_index_col]
                    ).set_index(cfg.unit_index_col)[cfg.measure_col]
                    values = p_units.reindex(unit_order).values.astype(float)
                    datum = Datum(
                        stimulus=stimulus,
                        measure_name=cfg.measure_name or cfg.measure_col,
                        values=values,
                        participant_id=str(pid),
                    )
                    self._data.append(datum)
            else:
                # Read from unit_df so values align with words (deduped + unit-index sorted).
                values = unit_df[cfg.measure_col].values.astype(float)
                datum = Datum(
                    stimulus=stimulus,
                    measure_name=cfg.measure_name or cfg.measure_col,
                    values=values,
                    participant_id="",
                )
                self._data.append(datum)
