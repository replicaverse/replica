"""Run a notebook's code cells in order, the way a kernel runs them."""

import json
from pathlib import Path


def run_code_cells(path: Path) -> dict:
    """Execute every code cell of the notebook at *path* in one namespace.

    Args:
        path: The ``.ipynb`` file to run.

    Returns:
        The namespace the cells leave behind.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{path.name}, cell {index}", "exec"), namespace)
    return namespace
