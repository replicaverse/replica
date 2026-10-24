"""Table aggregation for the unigram-surprisal experiment."""

from pathlib import Path

import pandas as pd

from ..records.resources import resource_file


def runtime_summary(path: Path | str | None = None) -> pd.DataFrame:
    """Average repeats within each word, then report means and sample SDs."""
    if path is None:
        path = resource_file("hopton2026unigram", "data/runtime_timings.parquet")
    timings = pd.read_parquet(path)
    times = ["mc_time_s", "mmc_nc_time_s"]
    words = timings.groupby(["n_tokens", "word"])[times].mean()
    summary = words.groupby("n_tokens")[times].agg(["mean", "std"])
    summary.columns = ["nmc_mean", "nmc_sd", "mmc_mean", "mmc_sd"]
    return summary.reset_index()
