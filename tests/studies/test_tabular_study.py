"""A tabular study's features follow the units of each trial in its table."""

import pandas as pd

from replica.studies.base import TabularRecipe, TabularStudy


class _Toy(TabularStudy):
    name = "toy"
    bibtex = "@misc{toy, title={Toy}}"
    dependencies = []
    recipe = TabularRecipe(
        dataset_name="toy",
        measure_col="rt",
        predictors=("surprisal",),
        controls=("length",),
    )

    def load_df(self, root=None):
        return pd.DataFrame(
            {
                "trialid": [20, 10, 10, 20],
                "ianum": [1, 1, 0, 0],
                "ia": ["d", "b", "a", "c"],
                "rt": [240.0, 220.0, 210.0, 230.0],
                "surprisal": [4.0, 2.0, 1.0, 3.0],
                "length": [1.0, 1.0, 1.0, 1.0],
            }
        )

    def analysis_config(self):
        raise NotImplementedError


def test_evaluation_spec_aligns_values_with_the_units_of_each_trial():
    study = _Toy()
    dataset = study.dataset()
    spec = study.evaluation_spec(dataset=dataset)
    values = {
        stimulus.stimulus_id: (
            list(stimulus.unit_strings("words")),
            [
                output.values["surprisal"]
                for output in spec.predictors[0].model(stimulus)
            ],
        )
        for stimulus in dataset.stimuli()
    }
    assert values == {
        "trial_10": (["a", "b"], [1.0, 2.0]),
        "trial_20": (["c", "d"], [3.0, 4.0]),
    }
