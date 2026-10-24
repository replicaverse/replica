"""Registered EEG N400 datasets from the Kuribayashi et al. (2025) release.

Guards that zuco_n400 / szewczyk_n400 / michaelov_n400 are discoverable through the registry,
load their signed ERP amplitudes (per subject by default for the critical-word
corpora), expose the classical N400 predictors, and match the release values.
"""

import numpy as np
import pandas as pd
import pytest

from replica.dataset.catalog import registry as R
from replica.dataset.eeg import MichaelovN400, SzewczykN400

# (name, n_stimuli, measures). The mega-N400 is registered whole
# (``szewczyk_n400``, all five studies) and once per source study; each stimulus
# is one full sentence per item.
_CASES = [
    ("zuco_n400", 300, {"ant", "n400", "p600"}),
    ("michaelov_n400", 500, {"n400"}),
    ("szewczyk_n400", 1331, {"n400", "n400_baseline"}),
    ("federmeier2007_n400", 564, {"n400", "n400_baseline"}),
    ("hubbard2019_n400", 192, {"n400", "n400_baseline"}),
    ("wlotko_federmeier2012_n400", 300, {"n400", "n400_baseline"}),
    ("szewczyk_federmeier2022_n400", 600, {"n400", "n400_baseline"}),
    ("szewczyk2022_n400", 672, {"n400", "n400_baseline"}),
]

# The five source studies pooled in the mega-N400, and their subject counts.
_SZEWCZYK_STUDIES = [
    "federmeier2007_n400",
    "hubbard2019_n400",
    "wlotko_federmeier2012_n400",
    "szewczyk_federmeier2022_n400",
    "szewczyk2022_n400",
]


def _factory(name):
    spec = R.get(name)
    if not spec.default_path().exists():
        pytest.skip(f"{name} data not present ({spec.default_path()})")
    return spec.factory()


def _release_dir(name):
    return (
        (R.get(name).default_path() / "all.txt.averaged_rt.annotation").resolve().parent
    )


@pytest.mark.needs_data
@pytest.mark.parametrize("name,n_stim,measures", _CASES)
def test_registered_eeg_loads(name, n_stim, measures):
    ds = _factory(name)
    assert len(ds.stimuli()) == n_stim
    assert set(ds.measures) >= measures
    assert ds.info.paradigm == "eeg"
    vals = np.concatenate(
        [np.asarray(d.values, dtype=float) for d in ds.data(measure="n400")]
    )
    assert (vals < 0).any(), "N400 amplitudes are signed, unlike reading times"


@pytest.mark.needs_data
def test_zuco_n400_values_match_annotation():
    ds = _factory("zuco_n400")
    ann = pd.read_csv(
        R.get("zuco_n400").default_path() / "all.txt.averaged_rt.annotation"
    ).sort_values(["article", "sent_id", "tokenN_in_sent"])
    loaded = {
        (d.stimulus.metadata["article"], d.stimulus.metadata["sent_id"]): np.asarray(
            d.values, dtype=float
        )
        for d in ds.data(measure="n400")
    }
    for (article, sent_id), grp in ann.groupby(["article", "sent_id"]):
        expected = grp.sort_values("tokenN_in_sent")["N400"].to_numpy(dtype=float)
        np.testing.assert_allclose(loaded[(article, sent_id)], expected, equal_nan=True)


@pytest.mark.needs_data
def test_zuco_erp_windows_and_values_match_the_release():
    dataset = _factory("zuco_n400")
    frame = pd.read_csv(
        R.get("zuco_n400").default_path() / "all.txt.averaged_rt.annotation"
    )
    # From the release's calc_N400_mean, calc_P600_mean and calc_ANT_mean.
    definitions = {
        "n400": ("N400", (300, 500), "central"),
        "p600": ("P600", (600, 700), "posterior"),
        "ant": ("ANT", (200, 400), "anterior"),
    }
    for measure, (column, window, channels) in definitions.items():
        for datum in dataset.data(measure):
            assert datum.metadata["window_ms"] == window
            assert datum.metadata["channels"] == channels
            assert datum.metadata["source_column"] == column
            source = frame[
                (frame.article == datum.stimulus.metadata["article"])
                & (frame.sent_id == datum.stimulus.metadata["sent_id"])
            ].sort_values("tokenN_in_sent")
            np.testing.assert_equal(datum.values, source[column].to_numpy())


@pytest.mark.needs_data
def test_critical_word_corpora_load_per_subject():
    """S_N400 / M_N400 default to per-subject trials (real subjects on
    participant_id)."""
    for name in ("szewczyk2022_n400", "michaelov_n400"):
        ds = _factory(name)
        assert len(ds.participants) > 1
        assert "averaged" not in ds.participants


@pytest.mark.needs_data
def test_michaelov_per_subject_cluster_mean_and_condition():
    """M_N400 loads its 50 subjects; each subject's N400 is the centroparietal
    cluster mean (electrodes averaged), and the
    experimental condition rides on the stimulus. Values match the TSV."""
    ds = _factory("michaelov_n400")
    assert len(ds.participants) == 50
    assert set(s.metadata.get("condition") for s in ds.stimuli()) == {
        "BC",
        "Implaus",
        "Rel",
        "Unrel",
    }
    tsv = pd.read_csv(_release_dir("michaelov_n400") / "michaelov_2024.tsv", sep="\t")
    cluster = tsv.groupby(["Subject", "FullSentence"])["N400"].mean()
    got = np.sort(
        np.concatenate([d.values[~np.isnan(d.values)] for d in ds.data(measure="n400")])
    )
    np.testing.assert_allclose(got, np.sort(cluster.to_numpy()))


@pytest.mark.needs_data
def test_szewczyk_per_subject_matches_tsv():
    """A sub-study's per-subject N400 and baseline match the release TSV rows."""
    ds = _factory("federmeier2007_n400")
    tsv = pd.read_csv(
        _release_dir("federmeier2007_n400") / "szewczyk_2022.tsv", sep="\t"
    )
    tsv = tsv[tsv["dataset"] == "Federmeier et al. (2007)"]
    for measure, col in [("n400", "n400"), ("n400_baseline", "bline")]:
        got = np.sort(
            np.concatenate(
                [d.values[~np.isnan(d.values)] for d in ds.data(measure=measure)]
            )
        )
        np.testing.assert_allclose(got, np.sort(tsv[col].to_numpy()))


@pytest.mark.needs_data
@pytest.mark.parametrize(
    "name,expected",
    [
        ("szewczyk2022_n400", {"cloze", "concreteness", "orth_neighborhood"}),
        ("michaelov_n400", {"cloze", "plausibility", "orth_neighborhood"}),
        ("zuco_n400", {"n400_baseline", "p600_baseline", "ant_baseline"}),
    ],
)
def test_n400_predictors_exposed_as_covariates(name, expected):
    """The classical item-level N400 predictors are reachable through
    covariates(), so a researcher can model N400 ~ surprisal + cloze + ..."""
    ds = _factory(name)
    cov = ds.covariates(ds.stimuli()[0])
    assert expected <= set(cov), f"missing covariates: {expected - set(cov)}"
    assert not np.isnan(cov["log_frequency"]).all()


@pytest.mark.needs_data
def test_michaelov_electrode_mode_exposes_per_electrode():
    """aggregate='electrode' keeps each of the nine centroparietal electrodes as
    its own datum (electrode on Datum.metadata), matching the release TSV, so the
    electrode random effect in the M_N400 model is reproducible."""
    ds = MichaelovN400(aggregate="electrode")
    ds.load(str(R.get("michaelov_n400").default_path()))
    data = ds.data(measure="n400")
    assert {d.metadata.get("electrode") for d in data} == {
        "C3",
        "C4",
        "CP3",
        "CP4",
        "CPz",
        "Cz",
        "P3",
        "P4",
        "Pz",
    }
    tsv = pd.read_csv(_release_dir("michaelov_n400") / "michaelov_2024.tsv", sep="\t")
    got = np.sort(np.concatenate([d.values[~np.isnan(d.values)] for d in data]))
    np.testing.assert_allclose(got, np.sort(tsv["N400"].to_numpy()))


@pytest.mark.needs_data
def test_aggregate_mean_gives_subject_averaged_view():
    """aggregate='mean' collapses to the item-level averaged grain (the one the
    reproduction models), with a single 'averaged' participant."""
    ds = MichaelovN400(aggregate="mean")
    ds.load(str(R.get("michaelov_n400").default_path()))
    assert ds.participants == ["averaged"]
    assert len(ds.data(measure="n400")) == len(ds.stimuli())


@pytest.mark.needs_data
def test_szewczyk_combined_pools_all_source_studies():
    """The mega-N400 loads all five source studies together (as the release
    packages them): its per-subject N400 amplitudes are exactly the union of the
    five individually-registered studies, over all 138 pooled subjects."""
    combined = _factory("szewczyk_n400")
    assert len(combined.participants) == 138

    def amps(ds):
        return np.sort(
            np.concatenate(
                [d.values[~np.isnan(d.values)] for d in ds.data(measure="n400")]
            )
        )

    union = np.sort(np.concatenate([amps(_factory(s)) for s in _SZEWCZYK_STUDIES]))
    np.testing.assert_allclose(amps(combined), union)


@pytest.mark.needs_data
def test_szewczyk_study_filter_selects_one_source_study():
    """SzewczykN400(study=...) is a strict subset of the pooled mega-N400: it
    carries only that study's subjects and tags them on the release ``dataset``."""
    study = "Federmeier et al. (2007)"
    ds = SzewczykN400(study=study)
    ds.load(str(R.get("federmeier2007_n400").default_path()))
    combined = _factory("szewczyk_n400")
    assert set(ds.participants) < set(combined.participants)
    assert all(s.metadata.get("study") == study for s in ds.stimuli())


@pytest.mark.needs_data
def test_szewczyk_combined_exposes_study_per_datum():
    """An item recurs across studies, so the pooled mega-N400 carries study
    membership on each Datum, and the mega-analysis can fit a study random
    effect."""
    combined = _factory("szewczyk_n400")
    studies = {d.metadata.get("study") for d in combined.data(measure="n400")}
    assert studies == {
        "Federmeier et al. (2007)",
        "Hubbard et al. (2019)",
        "Wlotko & Federmeier (2012)",
        "Szewczyk & Federmeier (2022)",
        "Szewczyk et al. (2022)",
    }


def test_szewczyk_pooled_mean_is_rejected():
    """Mean aggregation requires a study to identify the subject pool."""
    with pytest.raises(ValueError, match="pooled mega-N400"):
        SzewczykN400(aggregate="mean")


def test_critical_word_exclusion_keeps_sentence_initial_target():
    """A critical-word amplitude on sentence position 0 must survive exclusion.
    The base EEG mask drops position 0 (no-spillover-baseline), which for a
    single-token amplitude would delete the whole datum."""
    import types

    ds = MichaelovN400()  # unloaded; exclusion_mask only reads datum.values
    datum = types.SimpleNamespace(values=np.array([-3.0, np.nan, np.nan, np.nan]))
    mask = ds.exclusion_mask(datum)
    assert bool(mask[0]) and int(mask.sum()) == 1, "sentence-initial target dropped"


@pytest.mark.needs_data
@pytest.mark.parametrize("name", ["szewczyk2022_n400", "michaelov_n400"])
def test_critical_word_corpora_are_full_sentences(name):
    """The critical-word corpora carry the full sentence (so surprisal has
    context), with the amplitude on exactly one token that survives exclusion."""
    ds = _factory(name)
    assert max(len(s.unit_strings("words")) for s in ds.stimuli()) > 1
    for d in ds.data(measure="n400")[:50]:
        assert np.sum(~np.isnan(d.values)) == 1, "one critical-word amplitude per item"
        assert int(ds.exclusion_mask(d).sum()) == 1, "exclusion leaves the target word"


@pytest.mark.needs_data
def test_pooled_covariates_match_each_source_study():
    dataset = _factory("szewczyk_n400")
    source = pd.read_csv(_release_dir("szewczyk_n400") / "szewczyk_2022.tsv", sep="\t")
    columns = dataset._TSV_COVARIATES
    variation = source.groupby("FullSentence")[list(columns.values())].nunique()
    [text] = variation.index[variation.max(axis=1) > 1]
    checked = 0
    variants = set()
    for datum in dataset.data("n400"):
        if datum.stimulus.text != text:
            continue
        variants.add(datum.stimulus.stimulus_id)
        rows = source[
            (source.FullSentence == text) & (source.dataset == datum.metadata["study"])
        ]
        covariates = dataset.covariates(datum.stimulus)
        for name, column in columns.items():
            [actual] = covariates[name][np.isfinite(covariates[name])]
            assert rows[column].nunique() == 1
            assert actual == rows[column].iloc[0]
        checked += 1
    assert checked > 30
    assert len(variants) == 2
