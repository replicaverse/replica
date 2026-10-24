"""Single source of truth for MECO language, site, and release metadata."""

from dataclasses import dataclass

# Public codes follow the MECO release. ``du``/``ee``/``gr``/``sp`` are
# MECO-specific codes (ISO would be nl/et/el/es).
LANGUAGES: tuple[str, ...] = (
    "de",
    "du",
    "ee",
    "en",
    "fi",
    "gr",
    "he",
    "it",
    "ko",
    "no",
    "ru",
    "sp",
    "tr",
)

# The RDA and character CSV releases use ``ge`` for German.
ISO_TO_FILE: dict[str, str] = {"de": "ge"}

# ``DatasetInfo.language`` must be ISO 639-1; the release codes above are
# not all ISO, so loaders translate through this map before declaring it.
FILE_TO_ISO: dict[str, str] = {"du": "nl", "ee": "et", "gr": "el", "sp": "es"}

L1W2_SITE_LANGUAGES: dict[str, str] = {
    "ba": "eu",
    "bp": "pt",
    "ch_s": "zh",
    "sp_ch": "es",
    "da": "da",
    "ge_po": "de",
    "ge_zu": "de",
    "hi_iiith": "hi",
    "hi_iitk": "hi",
    "ic": "is",
    "no": "no",
    "ru_mo": "ru",
    # The se site is Serbia; the released texts are Serbian in Latin script.
    "se": "sr",
    "tr": "tr",
    "ch_t": "zh",
    "en_uk": "en",
}


@dataclass(frozen=True)
class MECOWordRelease:
    """One upstream release of MECO's word-level reading measures.

    Attributes:
        name: Loader name the release is exposed under, also the
            ``DatasetInfo.name`` prefix.
        data_subdir: Directory below the data root the release is fetched into.
        filename: The release's R dataframe file inside that directory.
        osf_id: OSF file identifier the fetcher downloads it by.
        reader_group: ``"L1"`` for native readers, ``"L2"`` for
            second-language readers on English stimuli.
        wave: Collection wave, 1 or 2.
    """

    name: str
    data_subdir: str
    filename: str
    osf_id: str
    reader_group: str
    wave: int


WORD_RELEASES: dict[str, MECOWordRelease] = {
    "meco_l1_w2": MECOWordRelease(
        "MECOL1W2",
        "eye-tracking/original/meco_l1_w2",
        "joint_data_trimmed_wave2_version2.1.rda",
        "6996d9113ccc685cd17b0813",
        "L1",
        2,
    ),
    "meco_l2_w1": MECOWordRelease(
        "MECOL2W1",
        "eye-tracking/original/meco_l2_w1",
        "joint_data_l2_trimmed_version2.0.rda",
        "re7uy",
        "L2",
        1,
    ),
    "meco_l2_w2": MECOWordRelease(
        "MECOL2W2",
        "eye-tracking/original/meco_l2_w2",
        "joint_data_trimmed_L2_wave2_2025_01_03.rda",
        "tma46",
        "L2",
        2,
    ),
}


@dataclass(frozen=True)
class MECOFixationRelease:
    """One upstream release of MECO's raw fixation events.

    Attributes:
        name: Loader name the matching word-level release is exposed under.
        data_subdir: Directory below the data root the release is fetched into.
        rda_filename: The release's R dataframe file inside that directory.
        osf_id: OSF identifier the fetcher downloads it by.
        r_key: Name of the dataframe object inside the R file.
        parquet_filename: Converted store written beside the source file.
        expected_rows: Fixation count the converted store has to carry, the
            row-level check that the conversion lost nothing.
        source_sha256: Digest of the upstream file, pinning which revision a
            store was built from.
        available_sites: Collection sites the release covers, or ``None`` when
            it covers every site of its wave.
    """

    name: str
    data_subdir: str
    rda_filename: str
    osf_id: str
    r_key: str
    parquet_filename: str
    expected_rows: int
    source_sha256: str
    available_sites: frozenset[str] | None = None


FIXATION_RELEASES: dict[str, MECOFixationRelease] = {
    "meco_l1_w1": MECOFixationRelease(
        "MECOL1W1",
        "eye-tracking/original/meco",
        "joint_l1_fixation_report_2.1.rda",
        "c3tkj",
        "joint.fix",
        "fixations_l1_w1_v2.1.parquet",
        1_197_535,
        "be7002625ff99aaa20b691559cd4ddc0525464c4c210de7c678e9c8bf8600fd0",
    ),
    "meco_l1_w2": MECOFixationRelease(
        "MECOL1W2",
        "eye-tracking/original/meco_l1_w2",
        "joint_fix_trimmed_l1_wave2_version2.1.rda",
        "6996d9162f3ce6907e505d0d",
        "joint.fix",
        "fixations_l1_w2_v2.1.parquet",
        1_888_773,
        "d1de22e59b1d95dc430beb02b90dbf14f9505646e56988947e259daa954f691c",
    ),
    "meco_l2_w1": MECOFixationRelease(
        "MECOL2W1",
        "eye-tracking/original/meco_l2_w1",
        "joint_l2_fixation_trimmed_report_2.0.rda",
        "8rygu",
        "joint.fix.l2",
        "fixations_l2_w1.parquet",
        1_221_387,
        "49c7507cb4570485e2b2f33beace5063fb5ee9da100001c6adeec90531358b60",
    ),
    "meco_l2_w2": MECOFixationRelease(
        "MECOL2W2",
        "eye-tracking/original/meco_l2_w2",
        "joint_fix_trimmed_L2_wave2.rda",
        "677e22d30788b62a2dd10a52",
        "joint.fix.l2_w2",
        "fixations_l2_w2.parquet",
        1_408_291,
        "096de7423425c7340b26d940d4dbc4637a5a8cbe42b512ecb484713b733c489a",
    ),
}
