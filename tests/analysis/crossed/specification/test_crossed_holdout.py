"""Identity independence and conservative crossed-test resampling checks."""

import json

import numpy as np
import pandas as pd
import pytest

from replica.analysis.crossed.inference.losses import compare_losses
from replica.analysis.crossed.specification.holdout import split


def frame(nr=20, nm=24):
    return pd.DataFrame(
        [
            {"participant": str(r), "material": str(m), "item": f"i{m}"}
            for r in range(nr)
            for m in range(nm)
        ]
    )


def test_split_is_order_invariant_and_separates_both_axes():
    data = frame()
    role, receipt = split(data)
    assert sum(receipt["rows"].values()) == len(data)
    reordered = data.sample(frac=1, random_state=8)
    other, _ = split(reordered)
    assert (
        pd.Series(other, index=reordered.index).sort_index().tolist() == role.tolist()
    )
    for group in ("participant", "material", "item"):
        assert not set(data.loc[role == "train", group]) & set(
            data.loc[role == "test", group]
        )
    data["item"] = "shared-source-item"
    with pytest.raises(ValueError, match="leaks"):
        split(data)


def test_prepare_cli_preserves_declared_split_settings(tmp_path, monkeypatch):
    from replica.analysis.crossed.__main__ import main

    data = frame()
    observations, specification = tmp_path / "rows.tsv", tmp_path / "spec.json"
    data.to_csv(observations, sep="\t", index=False)
    specification.write_text(
        json.dumps(
            {
                "random_effect_audits": [
                    {"group": group, "requested": []}
                    for group in ("participant", "material")
                ]
            }
        )
    )
    out = tmp_path / "holdout"
    monkeypatch.setattr(
        "sys.argv",
        [
            "crossed",
            "prepare",
            "--observations",
            str(observations),
            "--specification",
            str(specification),
            "--out",
            str(out),
            "--fraction",
            "0.4",
            "--seed",
            "17",
        ],
    )
    main()
    expected, source = split(data, fraction=0.4, seed=17)
    recorded = pd.read_csv(out / "observations.tsv.gz", sep="\t")
    np.testing.assert_array_equal(recorded.role, expected)
    receipt = json.loads((out / "source.json").read_text())
    assert receipt["seed"] == 17 and receipt["fraction_per_axis"] == 0.4
    assert receipt["test_identities"] == source["test_identities"]


def test_loss_cli_preserves_seed_budget_and_alpha(tmp_path, monkeypatch):
    from replica.analysis.crossed.__main__ import main

    data = frame(5, 6)
    data["base_loss"] = np.random.default_rng(12).normal(size=len(data))
    data["target_loss"] = 0
    predictions, comparisons = tmp_path / "losses.tsv", tmp_path / "pairs.json"
    data.to_csv(predictions, sep="\t", index=False)
    comparisons.write_text(
        json.dumps([{"name": "gain", "baseline": "base", "target": "target"}])
    )
    samples = []
    for seed in (3, 4):
        out = tmp_path / str(seed)
        monkeypatch.setattr(
            "sys.argv",
            [
                "crossed",
                "compare",
                "--predictions",
                str(predictions),
                "--comparisons",
                str(comparisons),
                "--out",
                str(out),
                "--draws",
                "99",
                "--seed",
                str(seed),
                "--alpha",
                "0.1",
            ],
        )
        main()
        receipt = json.loads((out / "source.json").read_text())
        assert (
            receipt["bootstrap_draws"],
            receipt["bootstrap_seed"],
            receipt["alpha"],
        ) == (99, seed, 0.1)
        with np.load(out / "bootstrap.npz") as archive:
            samples.append(archive["gain"])
        row = pd.read_csv(out / "tests.tsv", sep="\t").iloc[0]
        assert row.seed == seed and row.draws == 99
        assert bool(row.holds) == bool(row.effect > 0 and row.p_holm < 0.1)
    assert not np.array_equal(*samples)


def test_identical_losses_have_no_advantage_and_constant_advantage_is_positive():
    data = frame()
    equal, draws = compare_losses(
        data, np.ones(len(data)), np.ones(len(data)), draws=99
    )
    assert equal["p_value"] == 1
    assert np.all(draws == 0)
    advantage, _ = compare_losses(
        data, np.ones(len(data)), np.zeros(len(data)), draws=99
    )
    assert advantage["effect"] == advantage["ci_low"] == advantage["ci_high"] == 1
    assert advantage["p_value"] == 0.01


def test_duplicating_all_observations_does_not_create_independent_information():
    data = frame()
    rng = np.random.default_rng(19)
    loss = rng.normal(size=len(data))
    first, draws = compare_losses(data, loss, np.zeros(len(data)), draws=199)
    repeated = pd.concat([data] * 4, ignore_index=True)
    second, other = compare_losses(
        repeated, np.tile(loss, 4), np.zeros(len(repeated)), draws=199
    )
    np.testing.assert_allclose(draws, other)
    assert first["p_value"] == second["p_value"]
    assert first["ci_high"] == pytest.approx(second["ci_high"])


def test_complete_comparison_family_is_holm_adjusted(tmp_path):
    from replica.analysis.crossed.inference.losses import evaluate

    data = frame(5, 6)
    data["one_loss"], data["two_loss"], data["three_loss"] = 1.0, 1.0, 0.0
    path = tmp_path / "predictions.tsv"
    data.to_csv(path, sep="\t", index=False)
    pairs = [
        {"name": "equal", "baseline": "one", "target": "two"},
        {"name": "better", "baseline": "one", "target": "three"},
    ]
    result = evaluate(
        path,
        pairs,
        tmp_path / "tests",
        draws=np.int64(9999),
        seed=np.int64(17),
        alpha=np.float32(0.05),
    ).set_index("name")
    assert result.loc["equal", "p_holm"] == 1
    assert result.loc["better", "p_holm"] == 0.0002
    assert result.loc["better", "holds"]


def test_large_family_can_reject_a_uniform_improvement_with_declared_budget(tmp_path):
    from statsmodels.stats.multitest import multipletests

    from replica.analysis.crossed.evaluation import validate_manifest
    from replica.analysis.crossed.inference.losses import evaluate

    comparison = {"name": "better", "baseline": "baseline", "target": "target"}
    manifest = {
        "alpha": 0.05,
        "families": {"prediction": 630},
        "cells": [
            {
                "id": str(i),
                "family": "prediction",
                "kind": "prediction",
                "bootstrap_draws": 9999,
                "bootstrap_seed": 20260927,
                "comparisons": [comparison],
            }
            for i in range(630)
        ],
    }
    with pytest.raises(ValueError, match="resolution precludes"):
        validate_manifest(manifest)
    for cell in manifest["cells"]:
        cell["bootstrap_draws"] = 99999
    validate_manifest(manifest)
    data = frame(4, 4)
    data["baseline_loss"], data["target_loss"] = 1.0, 0.0
    path = tmp_path / "predictions.tsv"
    data.to_csv(path, sep="\t", index=False)
    result = evaluate(
        path,
        [comparison],
        tmp_path / "inference",
        draws=manifest["cells"][0]["bootstrap_draws"],
    )
    adjusted = multipletests([result.p_value.iloc[0], *([1.0] * 629)], method="holm")[1]
    assert adjusted[0] < 0.05
    assert result.effect.iloc[0] == 1.0


@pytest.mark.parametrize("budget", [0, 1, 2.5, True])
def test_bootstrap_requires_a_variance_estimable_integer_budget(budget):
    data = frame()
    with pytest.raises(ValueError, match="budget"):
        compare_losses(data, np.ones(len(data)), np.zeros(len(data)), draws=budget)


@pytest.mark.parametrize("seed", [None, True, -1, 1.5, "17"])
def test_loss_receipt_requires_reproducible_randomness(tmp_path, seed):
    from replica.analysis.crossed.inference.losses import evaluate

    data = frame(2, 2)
    data["base_loss"], data["target_loss"] = 1.0, 0.0
    path = tmp_path / "predictions.tsv"
    data.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="bootstrap_seed"):
        evaluate(
            path,
            [{"name": "gain", "baseline": "base", "target": "target"}],
            tmp_path / "out",
            draws=99,
            seed=seed,
        )
    assert not (tmp_path / "out").exists()
