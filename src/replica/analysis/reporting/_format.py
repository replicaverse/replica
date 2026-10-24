"""Escaped text and numeric formatting shared by report renderers."""

import json
import math
from html import escape
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list)):
        return escape(json.dumps(value, ensure_ascii=False, indent=2))
    return escape(str(value))


def _number(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return f"{value:,.6g}" if _finite(value) else "Unavailable"


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _table(rows: list[tuple[Any, Any]]) -> str:
    return (
        "<table><tbody>"
        + "".join(
            f'<tr><th scope="row">{_text(key)}</th><td><pre>{_text(value)}</pre></td></tr>'
            for key, value in rows
        )
        + "</tbody></table>"
    )


def _details(label: str, value: Any) -> str:
    return f"<details><summary>{escape(label)}</summary><pre>{_text(value)}</pre></details>"
