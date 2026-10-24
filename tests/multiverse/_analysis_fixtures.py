"""Small analysis and corpus examples shared by row-formatting tests."""

from types import SimpleNamespace

from replica.analysis.contracts.results import AnalysisResult, FoldResult
from replica.dataset.measures import FIRST_PASS_RUN


def _analysis() -> AnalysisResult:
    # 10 folds, each ΔLL/N = 0.5 nats (delta_ll 5.0 over n_test 10).
    folds = [
        FoldResult(
            fold_index=i,
            n_test=10,
            baseline_ll=-100.0,
            full_ll=-95.0,
            delta_ll=5.0,
            baseline_r2=0.18,
            full_r2=0.20,
            delta_r2=0.02,
        )
        for i in range(10)
    ]
    return AnalysisResult(
        dataset_name="dummy",
        model_name="gpt2",
        feature_name="surprisal",
        dv_name=FIRST_PASS_RUN,
        metrics={
            "delta_ll": 50.0,
            "delta_ll_per_unit": 0.5,
            "full_r2": 0.20,
            "baseline_r2": 0.18,
        },
        n_observations=100,
        coefficient=1.0,
        coefficient_se=0.1,
        fold_results=folds,
    )


def _dataset():
    return SimpleNamespace(
        stimuli=lambda: [object()], participants=[object(), object()]
    )
