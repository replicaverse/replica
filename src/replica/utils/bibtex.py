"""Bibtex field extraction and LaTeX cleanup.

Kept out of ``studies`` so reading a citation does not import every recipe.
"""

import re


def bibtex_field(text: str, field: str) -> str | None:
    """Extract a bibtex field value, handling nested braces.

    Args:
        text: Bibtex source to search, one entry or a whole file.
        field: Field name to read, matched case-insensitively and anchored at a
            word boundary so ``title`` does not also match ``booktitle``.

    Returns:
        The value with its wrapping braces or quotes stripped, ``None`` when
        the field is absent or its opening brace never closes.
    """
    # \b anchors the field name so "title" does not match "booktitle".
    pattern = rf"\b{field}\s*=\s*"
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    rest = text[m.end() :]
    if rest.startswith("{"):
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return rest[1:i]
        return None
    elif rest.startswith('"'):
        end = rest.index('"', 1)
        return rest[1:end]
    else:
        m2 = re.match(r"(\S+)", rest)
        return m2.group(1).rstrip(",") if m2 else None


def clean_tex(s: str) -> str:
    r"""Whitespace, LaTeX accents in either order (``\\'{a}``/``{\\'a}``), braces.

    Args:
        s: A bibtex field value as extracted, LaTeX markup included.

    Returns:
        The value with whitespace runs collapsed to single spaces, acute and
        umlaut accents reduced to the bare letter, and all braces removed.
    """
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\\'\{([a-zA-Z])\}", r"\1", s)
    s = re.sub(r'\\"\{([a-zA-Z])\}', r"\1", s)
    s = re.sub(r"\\'([a-zA-Z])", r"\1", s)
    s = re.sub(r'\\"([a-zA-Z])', r"\1", s)
    return re.sub(r"[{}]", "", s)
