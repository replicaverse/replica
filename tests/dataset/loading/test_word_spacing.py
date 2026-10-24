"""Corpora whose text joins words without spaces declare it in their metadata."""

from pathlib import Path

import replica.dataset
from replica.dataset.contracts import DatasetInfo
from replica.dataset.eye_tracking.meco.waves import MECOL1W2, MECOL2W1
from replica.dataset.metadata.io import from_dict, load_yaml, to_dict


def test_default_is_spaced_and_the_flag_round_trips():
    assert DatasetInfo(name="x").word_spacing is True
    assert "word_spacing" not in to_dict(DatasetInfo(name="x"))
    info = from_dict({"name": "x", "word_spacing": False})
    assert info.word_spacing is False
    assert to_dict(info)["word_spacing"] is False


def test_chinese_meco_sites_and_bsc_declare_no_word_spacing():
    assert MECOL1W2("ch_s").info.word_spacing is False
    assert MECOL1W2("ch_t").info.word_spacing is False
    assert MECOL1W2("en_uk").info.word_spacing is True
    assert MECOL2W1().info.word_spacing is True
    yaml_dir = (
        Path(replica.dataset.__file__).parent / "metadata" / "yaml" / "eye-tracking"
    )
    assert load_yaml(yaml_dir / "bsc.yaml").word_spacing is False
    assert load_yaml(yaml_dir / "provo.yaml").word_spacing is True
