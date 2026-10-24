"""Analysis provenance distinguishes declared methods without storing fitted state."""

import gc
import json
import weakref

import numpy as np
import pytest

from replica.analysis.contracts.configuration import component_options
from replica.analysis.cv.group_holdout import CrossedGroupSplitter
from replica.analysis.cv.splitters import (
    FixedSplitter,
    ItemSplitter,
    MultiSeedSplitter,
    ShuffledKFoldSplitter,
)
from replica.analysis.models.fixed_effects.ridge import RidgeCVModel
from replica.analysis.models.mixed_effects.lmer import RLmerModel
from replica.analysis.models.mixed_effects.statsmodels import StatsmodelsLME
from replica.analysis.specification.config import AnalysisConfig


def test_ridge_penalties_and_fold_counts_are_distinct_and_exclude_fit_history():
    model = RidgeCVModel(alphas=[0.01, 0.1])
    first = AnalysisConfig(model=model, design=None, cv=ItemSplitter(3)).metadata()
    assert first["backend_options"] == {"alphas": [0.01, 0.1]}
    assert first["cv_options"] == {"n_folds": 3}
    second = AnalysisConfig(
        model=RidgeCVModel(alphas=[100, 1000]), design=None, cv=ItemSplitter(10)
    ).metadata()
    assert first["backend_options"] != second["backend_options"]
    assert first["cv_options"] != second["cv_options"]
    model.chosen_alphas.append(0.1)
    assert (
        AnalysisConfig(model=model, design=None, cv=ItemSplitter(3)).metadata() == first
    )
    json.dumps(first, allow_nan=False)


def test_random_structure_and_numerical_options_are_recorded_without_input_arrays():
    model = RLmerModel(
        random_slope_cols_subject=[1, 3], re_design=np.arange(6.0).reshape(3, 2)
    )
    options = component_options(model)
    assert options["random_slope_cols_subject"] == [1, 3]
    assert options["optimizer"] == "bobyqa"
    assert options["re_design"]["shape"] == [3, 2]
    before = options["re_design"]["sha256"]
    model.re_design[0, 0] = 1
    assert component_options(model)["re_design"]["sha256"] != before
    assert component_options(StatsmodelsLME(method=["powell"], reml=False)) == {
        "method": ["powell"],
        "nan_policy": "raise",
        "group_col": "item",
        "reml": False,
        "fit_kwargs": {},
        "integrated_loglik": False,
    }


def test_crossed_and_repeated_splitter_settings_survive_splitting():
    crossed = CrossedGroupSplitter(train_fraction={"reader": 0.8, "text": 0.9}, seed=7)
    assert component_options(crossed) == {
        "n_folds": None,
        "train_fraction": {"reader": 0.8, "text": 0.9},
        "seed": 7,
    }
    repeated = MultiSeedSplitter([1, 2], lambda seed: ShuffledKFoldSplitter(3, seed))
    before = component_options(repeated)
    for _ in range(2):
        list(repeated.split(*[np.arange(12)] * 4))
        assert component_options(repeated) == before
    assert [inner["options"] for inner in before["splitters"]] == [
        {"n_folds": 3, "seed": 1},
        {"n_folds": 3, "seed": 2},
    ]


def test_repeated_splitter_snapshots_do_not_retain_fitted_components():
    references = []

    class StatefulSplitter:
        def __init__(self, seed):
            self.settings = {
                "seed": seed,
                "generation": len(references) // 2,
                "used": False,
            }
            references.append(weakref.ref(self))

        def configuration(self):
            return self.settings

        def split(self, y, *args):
            self.y = y
            self.settings["used"] = True
            yield y < 2, y >= 2

    repeated = MultiSeedSplitter([1, 2], StatefulSplitter)
    for generation in range(2):
        list(repeated.split(*[np.arange(4)] * 4))
        gc.collect()
        assert all(reference() is None for reference in references)
        assert [
            inner["options"] for inner in component_options(repeated)["splitters"]
        ] == [
            {"seed": 1, "generation": generation, "used": False},
            {"seed": 2, "generation": generation, "used": False},
        ]


def test_foreign_fit_options_are_recorded_without_guessing_their_configuration():
    from statsmodels.base._penalties import L2

    metadata = AnalysisConfig(
        model=StatsmodelsLME(fit_kwargs={"fe_pen": L2(1.0)}), design=None
    ).metadata()
    assert metadata["backend_options"]["fit_kwargs"]["fe_pen"] == {
        "type": "statsmodels.base._penalties.L2",
        "options": None,
    }


def test_fixed_splitters_bind_masks_or_explicit_item_sets():
    left = component_options(FixedSplitter(np.array([True, False])))
    right = component_options(FixedSplitter(np.array([False, True])))
    assert left["train_mask"]["sha256"] != right["train_mask"]["sha256"]
    assert component_options(FixedSplitter.from_items([3, 1]))["train_items"] == [1, 3]
    assert (
        component_options(FixedSplitter(mask_fn=lambda *args: [True]))["mask_function"][
            "options"
        ]
        is None
    )


def test_custom_configuration_is_explicit_and_inherited_settings_are_not_guessed():
    class Custom:
        hidden_setting = 3

    class Derived(RidgeCVModel):
        extra_penalty = 10

    class Declared(Custom):
        def configuration(self):
            return {"hidden_setting": self.hidden_setting}

    assert component_options(Custom()) is None
    assert component_options(Derived()) is None
    assert component_options(Declared()) == {"hidden_setting": 3}
    with pytest.raises(TypeError, match="JSON"):
        component_options(type("Invalid", (), {"configuration": lambda self: []})())
