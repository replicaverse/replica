"""Conversion and validation for optional MECO fixation Parquet stores."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .releases import FIXATION_RELEASES as RELEASES
from .releases import MECOFixationRelease

SOURCE_COLUMNS = {
    "uniform_id": "participant_id",
    "lang": "site",
    "itemid": "source_item_id",
    "trialid": "source_trial_id",
    "fixid": "fixation_index",
    "start": "start_time",
    "stop": "end_time",
    "dur": "duration",
    "xn": "x",
    "yn": "y",
    "ps": "pupil_size",
    "blink": "blink",
    "wordnum": "source_word_number",
    "word": "source_word",
    "ianum": "source_ia_number",
    "ia": "source_ia",
    "word.runid": "run_id",
}

PARQUET_SCHEMA = pa.schema(
    [
        ("participant_id", pa.string()),
        ("site", pa.string()),
        ("source_item_id", pa.int32()),
        ("source_trial_id", pa.int32()),
        ("fixation_index", pa.int32()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("pupil_size", pa.float64()),
        ("blink", pa.float64()),
        ("source_word_number", pa.int32()),
        ("source_word", pa.string()),
        ("source_ia_number", pa.int32()),
        ("source_ia", pa.string()),
        ("run_id", pa.int32()),
    ]
)
PARQUET_FORMAT_VERSION = "1"


def _parquet_schema(release: MECOFixationRelease) -> pa.Schema:
    return PARQUET_SCHEMA.with_metadata(
        {
            b"meco_release": release.name.encode(),
            b"source_sha256": release.source_sha256.encode(),
            b"expected_rows": str(release.expected_rows).encode(),
            b"format_version": PARQUET_FORMAT_VERSION.encode(),
            b"sort_order": b"participant_id,source_item_id,fixation_index",
        }
    )


def _r_available() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    check = subprocess.run(
        [rscript, "-e", 'quit(status=!requireNamespace("data.table", quietly=TRUE))'],
        check=False,
        capture_output=True,
    )
    return check.returncode == 0


def _rda_to_csv_with_r(rda_path: Path, r_key: str, csv_path: Path) -> None:
    """Use R/data.table to select columns before crossing into Python."""
    program = r"""
args <- commandArgs(TRUE)
env <- new.env()
load(args[[1]], envir=env)
key <- args[[2]]
if (!exists(key, envir=env, inherits=FALSE)) {
  stop(sprintf("R object %s not found; available: %s", key, paste(ls(env), collapse=", ")))
}
frame <- get(key, envir=env, inherits=FALSE)
columns <- strsplit(args[[4]], ",", fixed=TRUE)[[1]]
missing <- setdiff(columns, names(frame))
if (length(missing)) stop(sprintf("missing columns: %s", paste(missing, collapse=", ")))
frame <- data.table::as.data.table(frame[, columns, drop=FALSE])
data.table::setorderv(frame, c("uniform_id", "itemid", "fixid"))
data.table::fwrite(frame, args[[3]], na="")
"""
    subprocess.run(
        [
            shutil.which("Rscript") or "Rscript",
            "-e",
            program,
            str(rda_path),
            r_key,
            str(csv_path),
            ",".join(SOURCE_COLUMNS),
        ],
        check=True,
    )


def _csv_to_parquet(
    csv_path: Path, parquet_path: Path, release: MECOFixationRelease
) -> None:
    source_schema = pa.schema(
        [
            (source, PARQUET_SCHEMA.field(target).type)
            for source, target in SOURCE_COLUMNS.items()
        ]
    )
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=16 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(column_types=source_schema),
    )
    writer = pq.ParquetWriter(
        parquet_path, _parquet_schema(release), compression="zstd"
    )
    try:
        for batch in reader:
            renamed = pa.RecordBatch.from_arrays(
                [
                    batch.column(batch.schema.get_field_index(source))
                    for source in SOURCE_COLUMNS
                ],
                schema=PARQUET_SCHEMA,
            )
            writer.write_batch(renamed)
    finally:
        writer.close()


def _rda_to_parquet_with_pyreadr(
    rda_path: Path, parquet_path: Path, release: MECOFixationRelease
) -> None:
    import pyreadr

    result = pyreadr.read_r(str(rda_path))
    if release.r_key not in result:
        raise ValueError(
            f"Expected {release.r_key!r} in {rda_path}; found {list(result)}"
        )
    frame = result[release.r_key][list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    frame = frame.sort_values(["participant_id", "source_item_id", "fixation_index"])
    table = pa.Table.from_pandas(
        frame,
        schema=PARQUET_SCHEMA,
        preserve_index=False,
        safe=False,
    ).replace_schema_metadata(_parquet_schema(release).metadata)
    pq.write_table(table, parquet_path, compression="zstd", row_group_size=100_000)


def convert_release(
    release_key: str, data_dir: str | Path, *, force: bool = False
) -> Path:
    """Convert one downloaded MECO fixation RDA to canonical Parquet.

    The source file is checked against its pinned digest first, the Parquet is
    written to a temporary name and validated before it replaces the target, so
    an interrupted conversion never leaves a usable-looking store behind.

    Args:
        release_key: Key into the fixation-release table, e.g. ``meco_l1_w1``.
        data_dir: Directory holding the downloaded RDA, where the Parquet is
            written beside it.
        force: Rebuild even when a valid Parquet is already present.

    Returns:
        Path of the canonical Parquet store.

    Raises:
        FileNotFoundError: When the RDA has not been fetched.
        ValueError: When the RDA's digest does not match the pinned one, or
            the converted store fails validation.
    """
    release = RELEASES[release_key]
    directory = Path(data_dir)
    rda_path = directory / release.rda_filename
    parquet_path = directory / release.parquet_filename
    if not rda_path.is_file():
        raise FileNotFoundError(
            f"Raw fixations not found at {rda_path}. Run: "
            f"replica fetch {release_key}_fixations"
        )
    digest = hashlib.sha256()
    with rda_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != release.source_sha256:
        raise ValueError(
            f"{rda_path} does not match the pinned upstream SHA-256; "
            f"expected {release.source_sha256}, got {digest.hexdigest()}"
        )
    if parquet_path.exists() and not force:
        validate_parquet(parquet_path, release)
        return parquet_path

    temporary = parquet_path.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        if _r_available():
            with tempfile.TemporaryDirectory(prefix="meco-fixations-") as temp_dir:
                csv_path = Path(temp_dir) / "selected.csv"
                _rda_to_csv_with_r(rda_path, release.r_key, csv_path)
                _csv_to_parquet(csv_path, temporary, release)
        else:
            _rda_to_parquet_with_pyreadr(rda_path, temporary, release)
        validate_parquet(temporary, release)
        temporary.replace(parquet_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return parquet_path


def validate_parquet(path: str | Path, release: MECOFixationRelease) -> None:
    """Validate schema, provenance, row count, version, and sort declaration.

    Args:
        path: Parquet store to check; only its footer metadata is read.
        release: Release the store claims to hold, supplying the expected row
            count and the pinned source digest.

    Raises:
        ValueError: When the row count, schema, source digest, format version,
            or declared sort order disagrees with ``release``.
    """
    metadata = pq.read_metadata(path)
    if metadata.num_rows != release.expected_rows:
        raise ValueError(
            f"{path} has {metadata.num_rows:,} fixation rows; "
            f"expected {release.expected_rows:,}"
        )
    if metadata.schema.to_arrow_schema().remove_metadata() != PARQUET_SCHEMA:
        raise ValueError(f"{path} does not have the canonical MECO fixation schema")
    file_metadata = metadata.metadata or {}
    if file_metadata.get(b"source_sha256") != release.source_sha256.encode():
        raise ValueError(f"{path} was not prepared from the pinned upstream source")
    if file_metadata.get(b"format_version") != PARQUET_FORMAT_VERSION.encode():
        raise ValueError(
            f"{path} uses an obsolete MECO fixation cache format; rebuild it"
        )
    if (
        file_metadata.get(b"sort_order")
        != b"participant_id,source_item_id,fixation_index"
    ):
        raise ValueError(f"{path} does not declare the required temporal sort order")


def prepare_in_subprocess(
    release_key: str, data_dir: str | Path, *, force: bool = False
) -> Path:
    """Prepare Parquet in a child process so RDA conversion memory is released.

    Args:
        release_key: Key into the fixation-release table, passed to
            :func:`convert_release` in the child.
        data_dir: Directory holding the downloaded RDA.
        force: Rebuild even when a valid Parquet is already present.

    Returns:
        Path of the Parquet store, re-validated in this process after the
        child exits.
    """
    release = RELEASES[release_key]
    worker = (
        "import sys; "
        "from replica.dataset.eye_tracking.meco.prepare "
        "import convert_release; "
        "path = convert_release(sys.argv[1], sys.argv[2], force=sys.argv[3] == '1'); "
        "print(f'Prepared {path}')"
    )
    command = [
        sys.executable,
        "-c",
        worker,
        release_key,
        str(data_dir),
        str(int(force)),
    ]
    subprocess.run(command, check=True)
    output = Path(data_dir) / release.parquet_filename
    validate_parquet(output, release)
    return output


def run_prepare(release_keys: list[str], *, force: bool = False) -> None:
    """CLI driver for one or more MECO fixation releases.

    Args:
        release_keys: Keys into the fixation-release table, each prepared under
            the configured data directory. An unknown key aborts before any
            conversion starts.
        force: Rebuild every named store even where a valid Parquet is present.

    Raises:
        SystemExit: When any key is not a known fixation release.
    """
    unknown = [key for key in release_keys if key not in RELEASES]
    if unknown:
        raise SystemExit(
            f"unknown MECO fixation release(s): {', '.join(unknown)}; "
            f"available: {', '.join(sorted(RELEASES))}"
        )
    from replica.dataset.locations import data_dir

    for key in release_keys:
        release = RELEASES[key]
        directory = data_dir() / release.data_subdir
        print(f"\n=== preparing {key} ===")
        output = prepare_in_subprocess(key, directory, force=force)
        metadata = pq.read_metadata(output)
        print(f"  {output}: {metadata.num_rows:,} rows")
