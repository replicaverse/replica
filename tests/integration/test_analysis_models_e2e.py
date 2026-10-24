"""End-to-end tests for analysis models (OLS, LMM, GAM).

Runs the full evaluate() pipeline on Natural Stories with each analysis
model to verify they all produce valid results through the complete pipeline.

Run with: pytest tests/integration/test_analysis_models_e2e.py -v \
    --data-root datasets
"""

import os

import numpy as np
import pytest

from replica import (
    OLS,
    AnalysisConfig,
    EvaluationSpec,
    ItemSplitter,
    ModelFeature,
    evaluate,
)
from replica.analysis import AnalysisResult
from replica.analysis.models import RGamModel, RLmerModel
from replica.dataset import NaturalStories
from replica.feature import Surprisal, UnitLength

pytestmark = [pytest.mark.needs_r, pytest.mark.needs_data]

from .conftest import _VaryingLM


@pytest.fixture(scope="module")
def natural_stories(data_root):
    """Load Natural Stories dataset (SPR, reasonably small)."""
    path = os.path.join(data_root, "self-paced-reading", "original", "natural_stories")
    if not os.path.isdir(path):
        pytest.skip(f"Natural Stories data not found at {path}")
    ds = NaturalStories()
    ds.load(path)
    return ds


@pytest.fixture(scope="module")
def varying_lm_module():
    return _VaryingLM()


def _run_pipeline(dataset, lm, analysis_model, n_folds=5):
    """Helper: run evaluate() and return results."""
    spec = EvaluationSpec(
        predictors=[ModelFeature(lm, Surprisal())],
        controls=[UnitLength()],
        targets=[dataset.feature(dataset.default_measure)],
    )
    cfg = AnalysisConfig(
        design=None,
        transform_dv=np.log2,
        model=analysis_model,
        cv=ItemSplitter(n_folds),
        n_folds=n_folds,
    )
    return evaluate(dataset, spec, cfg)


# Each model is run once at module scope and the result is shared by every
# downstream assertion, without this, `test_models_disagree_numerically` would
# re-run all three pipelines a second time.
@pytest.fixture(scope="module")
def pipeline_results(natural_stories, varying_lm_module):
    return {
        "ols": _run_pipeline(natural_stories, varying_lm_module, OLS()),
        "lmer": _run_pipeline(natural_stories, varying_lm_module, RLmerModel()),
        "gam": _run_pipeline(natural_stories, varying_lm_module, RGamModel()),
    }


@pytest.mark.parametrize("model_key", ["ols", "lmer", "gam"])
def test_pipeline_produces_finite_delta_ll(pipeline_results, model_key):
    """Each analysis model produces a finite, well-formed result through evaluate()."""
    results = pipeline_results[model_key]
    assert len(results) == 1
    assert isinstance(results[0], AnalysisResult)
    assert np.isfinite(results[0].delta_ll)
    assert results[0].n_observations > 0


def test_models_disagree_numerically(pipeline_results):
    """OLS, LMM, and GAM must not collapse to identical delta_ll, that
    would mean a bug in the dispatcher or all three pipelines fitting OLS."""
    deltas = {k: v[0].delta_ll for k, v in pipeline_results.items()}
    assert len(set(deltas.values())) > 1, (
        f"All three models produced identical delta_ll: {deltas}"
    )
