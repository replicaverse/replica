"""Original Dundee display text and word-object geometry shared by both loaders."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DundeeWord:
    """One displayed word object and its source screen and line numbers."""

    text: str
    screen: int
    line: int


def normalize_quotes(text: str) -> str:
    """The release normalizes double quotes to single quotes.

    Args:
        text: Display text or word label from the original items.
    """
    return text.replace('"', "'")


def read_display_words(items: Path, docid: int) -> list[DundeeWord]:
    """Match the complete ITM display text against ordered WRDP word objects.

    Args:
        items: Original Dundee ``english/items`` directory.
        docid: One-based text number used in the original filenames.
    """
    word_file = items / f"tx{docid:02d}wrdp.dat"
    display_file = items / f"text{docid:02d}.itm"
    for path in (word_file, display_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Dundee complete stimulus reference not found: {path}. "
                "Supply the original english/items directory via word_path."
            )
    indexed = {}
    for line in word_file.read_text("cp437").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 14:
            raise ValueError(f"Dundee {word_file.name}: malformed word object")
        number = int(fields[12])
        if number in indexed:
            raise ValueError(f"Dundee {word_file.name}: duplicate source word index")
        indexed[number] = DundeeWord(fields[0], int(fields[2]), int(fields[3]))
    if not indexed or sorted(indexed) != list(range(1, len(indexed) + 1)):
        raise ValueError(f"Dundee {word_file.name}: incomplete source word indices")
    displayed = " ".join(
        line[1:].rstrip().removesuffix(">")
        for line in display_file.read_text("cp437").splitlines()
        if line.startswith("+")
    ).split()
    if len(displayed) != len(indexed) or any(
        normalize_quotes(word) != normalize_quotes(indexed[i].text)
        for i, word in enumerate(displayed, 1)
    ):
        raise ValueError(f"Dundee {docid}: display text does not match word objects")
    return [
        DundeeWord(word, indexed[i].screen, indexed[i].line)
        for i, word in enumerate(displayed, 1)
    ]
