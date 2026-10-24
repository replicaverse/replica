"""Importing the analysis, dataset and feature layers stays model-free.

The model stack (torch, transformers) loads inside the backends that use it,
so a caller with precomputed predictors, or one who only loads a corpus,
never pays for it. A subprocess keeps the check independent of whatever this
test process has already imported.
"""

import subprocess
import sys

_PROBE = """
import sys
import replica.analysis, replica.dataset, replica.feature, replica.predictors
from replica import evaluate
print(sorted(m for m in ("torch", "transformers") if m in sys.modules))
"""


def test_core_layers_import_without_the_model_stack():
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout
