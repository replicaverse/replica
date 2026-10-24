"""Bound specifications and dataset factories in Cartesian benchmarks."""

import numpy as np

from replica.analysis import AnalysisResult, benchmark
from replica.analysis.models import OLS
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.feature import ModelFeature, Surprisal
from tests.analysis._fixtures import ConstantLM, make_toy_dataset


def _make_dataset(name, **kwargs):
    """Shorthand for make_toy_dataset with custom name and seed."""
    return make_toy_dataset(
        dataset_name=name,
        n_stimuli=kwargs.get("n_stimuli", 6),
        n_words=kwargs.get("n_words", 4),
        measures=kwargs.get("measures"),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_bound_spec_retains_its_source_and_result_identity(self):
        """A bound spec preserves dataset and model names in its result."""
        ds = _make_dataset("toy_a")
        lm = ConstantLM()
        results = benchmark(
            ds,
            EvaluationSpec(
                targets=[ds.feature("spr_rt")],
                predictors=[ModelFeature(lm, Surprisal())],
            ),
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), n_folds=3
            ),
        )
        assert len(results) == 1
        assert isinstance(results[0], AnalysisResult)
        assert results[0].dataset_name == "toy_a"
        assert results[0].model_name == "constant"

    def test_cartesian_product(self):
        """2 datasets x 2 models = 4 AnalysisResult objects."""
        ds_a = _make_dataset("toy_a")
        ds_b = _make_dataset("toy_b")
        results = benchmark(
            [ds_a, ds_b],
            [
                lambda ds: EvaluationSpec(
                    targets=[ds.feature("spr_rt")],
                    predictors=[ModelFeature(ConstantLM(), Surprisal())],
                ),
                lambda ds: EvaluationSpec(
                    targets=[ds.feature("spr_rt")],
                    predictors=[
                        ModelFeature(
                            ConstantLM(name="constant2", log_p=-0.5), Surprisal()
                        )
                    ],
                ),
            ],
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), n_folds=3
            ),
        )
        assert len(results) == 4
        pairs = {(r.dataset_name, r.model_name) for r in results}
        assert pairs == {
            ("toy_a", "constant"),
            ("toy_a", "constant2"),
            ("toy_b", "constant"),
            ("toy_b", "constant2"),
        }

    def test_dvs_per_spec(self):
        """Different EvaluationSpecs can carry different DVs."""
        ds_a = _make_dataset("toy_a", measures=["spr_rt", "total_rt"])
        lm = ConstantLM()
        spec_a = EvaluationSpec(
            targets=[ds_a.feature("spr_rt"), ds_a.feature("total_rt")],
            predictors=[ModelFeature(lm, Surprisal())],
        )
        spec_b = EvaluationSpec(
            targets=[ds_a.feature("spr_rt")],
            predictors=[ModelFeature(lm, Surprisal())],
        )
        results = benchmark(
            ds_a,
            [spec_a, spec_b],
            analysis=AnalysisConfig(
                design=None, transform_dv=np.log2, model=OLS(), n_folds=3
            ),
        )
        assert len(results) == 3
