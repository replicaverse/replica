"""Compare conditional go-past to the releases, including zeros and missing cells."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replica.dataset.eye_tracking.meco.l1 import MECO_L1
from replica.dataset.eye_tracking.meco.waves import MECOL1W2, MECOL2W1, MECOL2W2
from replica.dataset.measures import FIRST_PASS_GO_PAST

pytestmark = pytest.mark.needs_data


def _check_views(ds, datum, expected):
    np.testing.assert_array_equal(datum.values, expected)
    for drop in (False, True):
        ds.drop_skips = drop
        viewed = next(
            d
            for d in ds.data(FIRST_PASS_GO_PAST, datum.participant_id)
            if d.stimulus is datum.stimulus
        )
        np.testing.assert_array_equal(
            viewed.values,
            np.where(expected == 0, np.nan, expected) if drop else expected,
        )
    # Reading either view must not change the release values in storage.
    np.testing.assert_array_equal(datum.values, expected)


@pytest.mark.parametrize(
    "loader,folder,site",
    [
        (MECO_L1, "meco", "en"),
        (MECOL1W2, "meco_l1_w2", "da"),
        (MECOL2W1, "meco_l2_w1", None),
        (MECOL2W2, "meco_l2_w2", None),
    ],
)
def test_meco_go_past_matches_release(
    data_root, monkeypatch, tmp_path, loader, folder, site
):
    import pyreadr

    from replica.dataset.eye_tracking.meco import l1, waves

    base = Path(data_root) / "eye-tracking/original" / folder
    paths = list(base.glob("*.rda"))
    if not paths:
        pytest.skip(f"MECO release missing: {base}")
    filename = (
        "joint_l1_data_trimmed_version2.1.rda"
        if loader is MECO_L1
        else loader.release.filename
    )
    frame = pyreadr.read_r(str(base / filename))["joint.data"]
    if site:
        frame = frame[frame.lang == site]
    # Use one complete participant, keeping all 12 source texts and areas.
    complete = frame.groupby("uniform_id").itemid.nunique()
    participant = complete[complete == 12].index[0]
    frame = frame[frame.uniform_id == participant].drop_duplicates()
    monkeypatch.setattr(
        l1 if loader is MECO_L1 else waves, "read_r_dataframe", lambda _: frame
    )
    ds = loader(site) if site else loader()
    # The suite optionally caches loads by path. Keep this sampled release
    # isolated from the complete release used by other tests.
    (tmp_path / filename).touch()
    ds.load(str(tmp_path))
    zeros = compared = 0
    for datum in [d for d in ds._data if d.measure_name == FIRST_PASS_GO_PAST]:
        item = datum.metadata.get(
            "source_item_id", datum.stimulus.metadata.get("source_item_id")
        )
        source = frame[frame.itemid.astype(int) == item].set_index("wordnum")
        numbers = datum.stimulus.metadata["source_word_numbers"]
        expected = source.loc[numbers, "firstrun.gopast"].to_numpy(dtype=float)
        _check_views(ds, datum, expected)
        arrival = next(
            d
            for d in ds._data
            if d.measure_name == "go_past" and d.stimulus is datum.stimulus
        )
        late = (source.loc[numbers, "firstrun.skip"].to_numpy() == 1) & (
            source.loc[numbers, "firstrun.dur"].to_numpy() > 0
        )
        assert np.isnan(arrival.values[late]).all()
        zeros += int((expected == 0).sum())
        compared += len(expected)
    assert compared > 1000 and zeros > 0


@pytest.mark.parametrize("corpus", ["emtec", "potec"])
def test_pymovements_release_go_past(data_root, tmp_path, corpus):
    from replica.dataset.eye_tracking.emtec import EMTeC
    from replica.dataset.eye_tracking.potec import PoTeC

    base = Path(data_root) / "eye-tracking/original" / corpus
    if corpus == "emtec":
        path = base / "reading_measures.csv"
        if not path.exists():
            pytest.skip(f"Missing {path}")
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        keys = ["subject_id", "item_id", "model", "decoding_strategy"]
        first = frame.iloc[0]
        frame = frame[(frame[keys] == first[keys]).all(axis=1)].sort_values("word_id")
        ds = EMTeC(measurement_source="released")
    else:
        paths = sorted(base.glob("reader*_merged.tsv"))
        if not paths:
            pytest.skip(f"Missing {base}")
        path = paths[0]
        frame = pd.read_csv(path, sep="\t", keep_default_na=False).sort_values(
            "word_index_in_text"
        )
        ds = PoTeC(measurement_source="released")
    frame.to_csv(tmp_path / path.name, sep="\t", index=False)
    ds.load(str(tmp_path))
    [datum] = [d for d in ds._data if d.measure_name == FIRST_PASS_GO_PAST]
    expected = pd.to_numeric(frame.RPD_inc, errors="raise").to_numpy(dtype=float)
    _check_views(ds, datum, expected)
    [arrival] = [d for d in ds._data if d.measure_name == "go_past"]
    late = (frame.FPRT.to_numpy(dtype=float) == 0) & (
        frame.TFT.to_numpy(dtype=float) > 0
    )
    assert late.any()
    assert np.isnan(arrival.values[late]).all()


def test_ucl_preserves_all_released_go_past_values(data_root):
    from replica.dataset.eye_tracking.ucl import UCL_ET

    base = Path(data_root) / "eye-tracking/original/ucl"
    path = base / "eyetracking.RT.txt"
    if not path.exists():
        pytest.skip(f"Missing {path}")
    source = pd.read_csv(path, sep="\t")
    groups = source.groupby(["subj_nr", "sent_nr"])
    ds = UCL_ET()
    ds.load(str(base))
    sentence_numbers = {
        id(stimulus): number for number, stimulus in ds._sent_map.items()
    }
    views = {}
    for drop in (False, True):
        ds.drop_skips = drop
        views[drop] = {
            (d.participant_id, id(d.stimulus)): d.values
            for d in ds.data(FIRST_PASS_GO_PAST)
        }
    compared = 0
    for datum in [d for d in ds._data if d.measure_name == FIRST_PASS_GO_PAST]:
        rows = groups.get_group(
            (int(datum.participant_id), sentence_numbers[id(datum.stimulus)])
        )
        expected = np.full(len(datum.values), np.nan)
        expected[rows.word_pos.to_numpy(dtype=int) - 1] = rows.RTgopast.to_numpy()
        np.testing.assert_array_equal(datum.values, expected)
        for drop in (False, True):
            np.testing.assert_array_equal(
                views[drop][datum.participant_id, id(datum.stimulus)],
                np.where(expected == 0, np.nan, expected) if drop else expected,
            )
        compared += len(rows)
    assert compared == len(source)
