"""CELER release files written for loader tests."""

from replica.dataset.measures import EYELINK_IA_MEASURES


def write_sent_ia(path, rows) -> None:
    """Write ``sent_ia.tsv`` into ``path`` under the release's header.

    The header holds the interest-area keys, the Eyelink measure columns and
    the sentence.
    """
    header = [
        "sentenceid",
        "DATA_FILE",
        "IA_ID",
        "IA_LABEL",
        "shared_text",
        *EYELINK_IA_MEASURES,
        "sentence",
    ]
    path.joinpath("sent_ia.tsv").write_text(
        "\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in rows)
    )
