"""Author-branch replay and independent manuscript table checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from replica.studies.hopton2026unigram import (
    ESTIMATORS,
    LANGUAGES,
    Hopton2026Unigram,
    PrecomputedFoldSplitter,
    _flat_frame,
    _resolve_flat_folds,
)
from replica.studies.hopton2026unigram.reporting import runtime_summary

FIXTURES = Path(__file__).parent / "fixtures"
PAPER = json.loads((FIXTURES / "paper_reference.json").read_text())["tables"]
LANGUAGE_NAMES = dict(
    zip(
        LANGUAGES,
        [
            "Dutch",
            "English",
            "Finnish",
            "German",
            "Greek",
            "Hebrew",
            "Italian",
            "Korean",
            "Russian",
            "Spanish",
            "Turkish",
        ],
        strict=True,
    )
)


@pytest.mark.artifact
def test_retained_timings_reproduce_manuscript_runtime_table():
    actual = runtime_summary().to_numpy(dtype=float)
    expected = np.asarray(PAPER["runtime"]["rows"], dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0005)


@pytest.mark.parametrize("estimator", ["nmc", "priority_mmc_pathwise"])
@pytest.mark.artifact
def test_retained_language_summaries_reproduce_manuscript_macro_table(estimator):
    source = pd.read_csv(FIXTURES / "author_gamm_means.tsv", sep="\t")
    source = source[source.estimator == estimator]
    summary = source.groupby(["condition", "comp"]).mean_dll.agg(["mean", "std"])
    names = {
        "unigram_surprisal": "freq",
        "pmi": "pmi",
        "contextual_surprisal": "surp",
        "contextual+unigram_surprisal": "surp+freq",
        "pmi+unigram_surprisal": "pmi+freq",
    }
    for condition, comparison, mean, sd in PAPER[f"gamm_macro_{estimator}"]["rows"]:
        key = (
            "Standard" if condition == "corpus" else "RB_Estimate",
            names[comparison],
        )
        np.testing.assert_allclose(
            summary.loc[key].to_numpy(), [float(mean), float(sd)], rtol=0, atol=0.00005
        )


@pytest.mark.needs_data
@pytest.mark.parametrize(
    "estimator", [name for name in ESTIMATORS if name != "reference"]
)
def test_paper_estimator_support_and_fold_partition(estimator):
    reference = {row[0]: row[1:] for row in PAPER[f"support_{estimator}"]["rows"]}
    for language in LANGUAGES:
        frame = Hopton2026Unigram()._load_language(language, estimator=estimator)
        participants, types, observations, train, test = reference[
            LANGUAGE_NAMES[language]
        ]
        assert (frame.subid.nunique(), frame.ia.nunique(dropna=False), len(frame)) == (
            participants,
            types,
            observations,
        )
        sizes = frame.groupby("fold").size()
        assert len(sizes) == 10 and sizes.max() - sizes.min() <= 1
        assert (len(frame) - round(len(frame) / 10), round(len(frame) / 10)) == (
            train,
            test,
        )


@pytest.fixture
def participant_frame():
    return pd.DataFrame(
        {
            "subid": ["a", "b", "a"],
            "trialid": [1, 1, 1],
            "ianum": [1, 1, 3],
            "ia": ["one", "one", "two"],
            "gaze_rt": [100.0, 200.0, 300.0],
            "fold": [1, 2, 2],
            "surp": [1.0, 1.0, 2.0],
            "prev_surp": [0.0, 0.0, 1.0],
            "freq": [4.0, 4.0, 5.0],
            "prev_freq": [0.0, 0.0, 4.0],
            "pmi": [3.0, 3.0, 3.0],
            "prev_pmi": [0.0, 0.0, 3.0],
            "freq_BW": [8.0, 8.0, 9.0],
            "prev_freq_BW": [6.0, 7.0, 8.0],
            "pmi_BW": [7.0, 7.0, 7.0],
            "prev_pmi_BW": [6.0, 7.0, 7.0],
        }
    )


@pytest.mark.parametrize("condition,suffix", [("Standard", ""), ("RB_Estimate", "_BW")])
def test_predictors_preserve_participant_spillover_and_missing_units(
    monkeypatch, participant_frame, condition, suffix
):
    from replica.analysis.data.preparation import prepare_observations

    study = Hopton2026Unigram()
    frame = participant_frame.iloc[[1, 2, 0]]
    monkeypatch.setattr(study, "_load_language", lambda *args, **kwargs: frame)
    dataset = study.dataset()
    spec = study.evaluation_spec(dataset=dataset, condition=condition, comp="pmi+freq")
    actual = prepare_observations(dataset, spec, study.analysis_config())
    expected = participant_frame.set_index("gaze_rt").loc[actual.y]
    assert len(actual.y) == 3
    assert set(actual.columns) == {"pmi", "prev_pmi", "freq", "prev_freq"}
    for name, values in actual.columns.items():
        np.testing.assert_array_equal(values, expected[name + suffix])


def test_word_scores_weight_words_and_reject_other_fold_assignments(
    monkeypatch, participant_frame
):
    study = Hopton2026Unigram()
    frame = participant_frame
    monkeypatch.setattr(study, "_load_language", lambda *args, **kwargs: frame)
    dataset = study.dataset()
    flat = _flat_frame(dataset, frame, "gaze_rt")
    fold_ids, participants, items = _resolve_flat_folds(dataset, frame, "gaze_rt")
    options = PrecomputedFoldSplitter(
        fold_ids, expected_participants=participants, expected_items=items
    ).configuration()
    scores = np.where(flat.ianum == 3, 9.0, np.where(flat.subid == "a", 1.0, 3.0))
    folds = [
        SimpleNamespace(
            fold_index=index,
            n_test=int((fold_ids == fold).sum()),
            per_obs_delta_ll=scores[fold_ids == fold],
        )
        for index, fold in enumerate(sorted(set(fold_ids)))
    ]
    result = SimpleNamespace(
        dataset_name=dataset.info.name,
        dv_name="gaze_rt",
        metadata={"configuration": {"cv_options": options}},
        fold_results=folds,
    )
    actual = study.word_scores(result)
    np.testing.assert_array_equal(actual.dll, [2.0, 9.0])
    wrong = PrecomputedFoldSplitter(
        np.roll(fold_ids, 1), expected_participants=participants, expected_items=items
    ).configuration()
    result.metadata["configuration"]["cv_options"] = wrong
    with pytest.raises(ValueError, match="fold assignment"):
        study.word_scores(result)


@pytest.mark.needs_r
@pytest.mark.needs_data
@pytest.mark.parametrize(
    "estimator,comparison",
    [
        ("reference", "surp"),
        ("reference", "freq"),
        ("nmc", "freq"),
        ("mmc_pathwise", "freq"),
        ("priority_mmc_pathwise", "freq"),
    ],
)
def test_native_gamm_replays_author_reference(estimator, comparison):
    from replica import evaluate

    study = Hopton2026Unigram()
    options = {"estimator": estimator}
    dataset = study.dataset(**options)
    spec = study.evaluation_spec(dataset=dataset, comp=comparison, **options)
    (result,) = evaluate(dataset, spec, study.analysis_config(**options))
    actual = study.word_scores(result, **options)
    if estimator == "reference":
        words = pd.read_parquet(FIXTURES / "author_r_words.parquet")
        expected_words = (
            words[words.comp == comparison].set_index(["trialid", "ianum"]).sort_index()
        )
        actual_words = actual.set_index(["trialid", "ianum"]).sort_index()
        pd.testing.assert_index_equal(actual_words.index, expected_words.index)
        np.testing.assert_allclose(
            actual_words.dll, expected_words.dll, rtol=1e-6, atol=1e-8
        )
        reference = pd.read_csv(
            Path(__file__).parents[1] / "fixtures/hopton2026_r_reference.tsv", sep="\t"
        )
        row = reference[
            (reference.lang == "en")
            & (reference.psychometric == "gaze_rt")
            & (reference.condition == "RB_Estimate")
            & (reference.comp == comparison)
        ].iloc[0]
        expected = row.mean_dll
        assert len(actual) == row.n
        np.testing.assert_allclose(
            [actual.dll.std(ddof=1), actual.dll.sem()],
            [row.sd_dll, row.se_dll],
            rtol=1e-6,
            atol=3e-5,
        )
    else:
        reference = pd.read_csv(FIXTURES / "author_gamm_means.tsv", sep="\t")
        expected = (
            reference[
                (reference.estimator == estimator)
                & (reference.language == "en")
                & (reference.condition == "RB_Estimate")
                & (reference.comp == comparison)
            ]
            .iloc[0]
            .mean_dll
        )
    assert actual.dll.mean() == pytest.approx(expected, rel=1e-6, abs=3e-5)
