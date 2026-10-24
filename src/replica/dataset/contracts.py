"""Dataset metadata and fixation records independent of loader execution."""

from dataclasses import asdict, dataclass, field, fields


@dataclass
class FileEntry:
    r"""One file (or glob group) in a dataset directory.

    When ``n_files == 0`` (default), ``path`` is a single relative file path and
    ``checksum`` is its SHA-256. When ``n_files > 0``, ``path`` is a glob pattern
    (e.g. ``FixationReports/FIX_report_P*.txt``) and ``checksum`` is a
    deterministic aggregate over the matching files: SHA-256 of the sorted
    ``"<relpath>\\t<sha256>\\n"`` lines. ``original_path`` records the stable
    upstream URL or archive-member path. If that information genuinely cannot
    be published, ``provenance_unavailable`` must explain why instead.
    """

    path: str
    role: str = ""
    checksum: str = ""
    original_path: str = ""
    provenance_unavailable: str = ""
    n_files: int = 0


@dataclass
class IndexMapping:
    """Column mapping from data files to stimulus structure."""

    stimulus_col: str = ""
    unit_col: str = ""
    participant_col: str = ""
    stimulus_text_file: str = ""


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""

    name: str
    citation: str = ""
    bibtex: str = ""
    url: str = ""
    download_url: str = ""
    paradigm: str = ""
    modalities: list[str] = field(default_factory=list)
    #: How the stimuli were presented (see :class:`Presentation`). A corpus can
    #: carry more than one, and stays empty where the release does not say.
    presentation: list[str] = field(default_factory=list)
    language: str = "en"
    #: Whether the stimulus text separates its words with whitespace. False for
    #: a script written without word spacing (Chinese), where the loader joins
    #: the words directly and a subword token can cover parts of two words.
    word_spacing: bool = True
    # The kind of linguistic chunk that constitutes one Stimulus. One of:
    #   "single-unit"  one word or region per Stimulus
    #   "phrase"       short phrase, ≤10 units
    #   "sentence"     ~10-30 units
    #   "paragraph"    ~30-80 units
    #   "passage"      ≥80 units (multi-paragraph or full chapter)
    # Empty string = unspecified. The unit type itself (words vs. characters)
    # is recorded on the stimulus's annotation layer.
    stimulus_type: str = ""
    description: str = ""
    #: Prefix of material identities shared by related stimuli.
    material_namespace: str = field(default="", kw_only=True)
    #: Ordered stimulus-metadata fields identifying one source material.
    material_fields: list[str] = field(default_factory=list, kw_only=True)

    # Provenance
    source: str = ""
    original: bool = True
    version: str = ""
    release_date: str = ""
    access_date: str = ""
    license: str = ""
    derived_from: str = ""
    processing: str = ""
    #: Who released the files this loader reads: ``authors`` (the corpus
    #: authors' own release), ``third-party`` (someone else's redistribution
    #: or repackaging) or ``replica``. Orthogonal to ``original``, which says
    #: whether the data was reshaped.
    distribution: str = ""
    #: Who that third party is; required unless the distribution is authors.
    distributed_by: str = ""
    #: How anyone obtains the files this loader reads: ``free`` (a public
    #: download, without registration or request), ``on request`` (the authors
    #: send them on request) or ``licence required`` (a licence comes first,
    #: such as an LDC licence for the texts). A loader that also reads files
    #: of another release takes the more restricted of the two.
    availability: str = ""

    # Structured
    measures: list[str] = field(default_factory=list)
    files: list[FileEntry] = field(default_factory=list)
    index_mapping: IndexMapping | None = None


@dataclass(slots=True)
class Fixation:
    """One fixation event from an eye-tracking record.

    Time origin is trial onset in milliseconds. Coordinates are in the
    upstream release's pixel space; origin and screen resolution vary by
    corpus. Loaders convert ad-hoc upstream formats into this canonical
    schema.

    Linguistic association is provided whenever the upstream's interest-area
    information identifies which unit was being fixated. ``unit_index`` is
    a 0-based position into ``stimulus.layers[unit_layer]``; ``char_offset``
    is an absolute offset into ``stimulus.text``, usually the source span's
    start. ``unit_layer`` names that source layer. Set
    ``char_offset_is_exact`` only when the source establishes the character
    landing itself; an area-start anchor cannot locate subword fixations.
    A source unit alone is sufficient for native or coarser aggregation;
    its character offset may be omitted.
    Both indices are ``None`` for fixations that fell off-text.

    ``candidate_unit_indices`` records conflicting source assignments in
    ``unit_layer``. The first candidate is the reported position; ``None``
    represents a possible off-text landing. Such events have no assigned
    ``unit_index`` or ``char_offset`` until the target spans resolve them.

    ``fixation_index`` is 1-based within trial (the canonical EyeLink
    convention).
    """

    participant_id: str
    stimulus_id: str

    # Timing, trial-relative, milliseconds
    start_time: float
    end_time: float
    duration: float

    # Screen-space, pixels
    x: float
    y: float

    # Linguistic association (None when off-text)
    unit_index: int | None
    char_offset: int | None

    # 1-based ordinal within the trial
    fixation_index: int

    # Loose bag for corpus-specific extras (pupil size, run id, etc.)
    metadata: dict = field(default_factory=dict)
    unit_layer: str = "words"
    char_offset_is_exact: bool = False
    candidate_unit_indices: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class FixationCalculation:
    """Versioned reading windows and event policies for recomputation.

    Measure names and first-pass restrictions remain separate from algorithms.
    The specification applies after the loader selects source events and the
    shared mapper assigns them to the requested spans. Source run labels are
    never used by an explicit calculation.

    Args:
        first_run: ``standard@1``, ``popeye_episode@1`` or ``modelblocks@1``.
        go_past: ``standard@1``, ``copco_extractor@1`` or ``modelblocks@1``.
        total_time: ``standard@1`` or interruption-inclusive ``modelblocks@1``.
        floor_durations: Floor each event duration to integer milliseconds.
        off_text_ends_run: End a standard first run on an off-text event.
        go_past_includes_off_text: Count off-text dwell in standard regression paths.
        go_past_requires_right_exit: Require a rightward exit for standard go-past.
        include_inter_unit_fixations: Retain known whitespace landings in windows.
    """

    first_run: str = "standard@1"
    go_past: str = "standard@1"
    total_time: str = "standard@1"
    floor_durations: bool = False
    off_text_ends_run: bool = True
    go_past_includes_off_text: bool = False
    go_past_requires_right_exit: bool = False
    include_inter_unit_fixations: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                name not in {"first_run", "go_past", "total_time"}
                and type(value) is not bool
            ):
                raise TypeError(f"{name} must be a boolean")
        if self.first_run not in {"standard@1", "popeye_episode@1", "modelblocks@1"}:
            raise ValueError(f"Unknown first-run algorithm: {self.first_run!r}")
        if self.go_past not in {"standard@1", "copco_extractor@1", "modelblocks@1"}:
            raise ValueError(f"Unknown go-past algorithm: {self.go_past!r}")
        if self.total_time not in {"standard@1", "modelblocks@1"}:
            raise ValueError(f"Unknown total-time algorithm: {self.total_time!r}")
        if self.first_run != "standard@1" and not self.off_text_ends_run:
            raise ValueError("off_text_ends_run only configures standard first-run")
        if self.go_past != "standard@1" and (
            self.go_past_includes_off_text or self.go_past_requires_right_exit
        ):
            raise ValueError("Off-text and right-exit options require standard go-past")

    def to_dict(self) -> dict:
        """Serialize the resolved algorithms, policies and upstream code pins."""
        revisions = {}
        if self.first_run == "popeye_episode@1":
            revisions["popEye"] = "93ce662b571eda30162d24e2da34433c1a146e77"
        if self.go_past == "copco_extractor@1":
            revisions["copco-processing"] = "0b3bd294a3c09e186c551c085b709423181947f9"
        if "modelblocks@1" in {self.first_run, self.go_past, self.total_time}:
            revisions["modelblocks-release"] = (
                "3ae5120327005cde1c02576eda24876997e20550"
            )
        return {
            **asdict(self),
            "schema_version": 2,
            "other_measures": "standard@1",
            "duration_unit": "ms",
            "use_source_runs": False,
            "upstream_revisions": revisions,
        }

    @classmethod
    def from_dict(cls, record: dict) -> "FixationCalculation":
        """Restore a serialized specification, checking versions and code pins.

        Args:
            record: Complete dictionary emitted by :meth:`to_dict`.
        """
        names = {f.name for f in fields(cls)}
        if not names <= record.keys():
            raise ValueError("Calculation record is missing resolved options")
        calculation = cls(**{name: record[name] for name in names})
        if calculation.to_dict() != record:
            raise ValueError(
                "Calculation record has unknown or inconsistent provenance"
            )
        return calculation
