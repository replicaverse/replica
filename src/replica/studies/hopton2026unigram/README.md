# Revisiting the Estimation of Unigram Surprisal in Computational Psycholinguistics

- [BibTeX](#bibtex)
- [Abstract](#abstract)
- [Scope of reproduction](#scope-of-reproduction)
- [Notes](#notes)

## BibTeX

```bibtex
@misc{hopton2026unigram,
  title = {Revisiting the Estimation of Unigram Surprisal in Computational Psycholinguistics},
  author = {Hopton, Zachary William and Re, Francesco Ignazio and Kiegeland, Samuel and Opedal, Andreas and Chodroff, Eleanor and Cotterell, Ryan},
  year = {2026},
  archivePrefix = {arXiv},
  eprint = {2610.18414},
  url = {https://arxiv.org/abs/2610.18414}
}
```

## Abstract

[Paper abstract on arXiv](https://arxiv.org/abs/2610.18414)

## Scope of reproduction

This implementation maps Zach Hopton's study code at commit `b3e0715e936df2a0ba9f7f75882fcf6cc41def7e` onto the feature and analysis interfaces.
The manuscript reference is revision `c9bf92edfa18327b8d379f38636c06ed2f01fab1`.
[Input provenance](data/provenance.json) records the source paths and checksums.
Full input tables are fetched from the [study-bundle store](https://huggingface.co/datasets/replicaverse/replica-study-bundles) into the user cache. The [index](results/index.json) pins the archive revision and each file's SHA-256; the loader verifies them before use.
[Reference fixtures](../../../../tests/studies/hopton2026unigram/fixtures/README.md) distinguish manuscript transcriptions from retained fit outputs.

The author branch and retained predictor-store revision contain these inputs:

| `estimator` | Inputs and reference |
| --- | --- |
| `reference` (default) | Corrected August MMC estimates and the branch's independent R fixtures; includes three reading measures and both whitespace variants |
| `nmc` | Naive Monte Carlo estimates for the manuscript's gaze-duration analysis |
| `mmc_pathwise` | Pathwise-exact MMC estimates for the manuscript's gaze-duration analysis |
| `mmc_expectation_equivalent` | Expectation-equivalent MMC estimates for the manuscript's gaze-duration analysis |
| `priority_mmc_pathwise` | Priority pathwise MMC recovered from predictor-store revision `2508a73d` |
| `priority_mmc_expectation_equivalent` | Priority expectation-equivalent MMC from the same store revision |

Each estimator uses its own complete-case intersection with corpus predictors and its own archived folds. The paper variants currently support `gaze_rt` and the space-prefixed inputs only. Unsupported combinations raise an error.

The GAMM recipe uses raw reading times, training-fold means and sample standard deviations, cubic regression splines with six basis functions, and participant/trial random intercepts. Both baseline and full fits include those random effects in prediction. Ten folds replay R's `sample()` assignment with seed 250. Word-level scores average participant likelihood differences within each trial/word; manuscript macro summaries then weight languages equally.

BAM uses discretization for fitting and for prediction on each full held-out fold, matching the author's R code. Spillover predictors retain each participant's preceding observation. The independent English reference frequency fit reports nonconvergence in fold 9; its retained scores reproduce that execution.

```python
from replica import evaluate
from replica.studies.hopton2026unigram import Hopton2026Unigram

study = Hopton2026Unigram()
options = dict(language="en", estimator="nmc", measure="gaze_rt")
dataset = study.dataset(**options)
spec = study.evaluation_spec(dataset=dataset, comp="freq", **options)
result, = evaluate(dataset, spec, study.analysis_config(**options))
words = study.word_scores(result, **options)
print(words.dll.mean())
```

`result.delta_ll_per_unit` weights observations equally. Use `word_scores()` for the author's word-level summaries. It checks the recorded fold-assignment digest before aligning retained scores to words.

The retained runtime measurements reproduce the current manuscript's 16 printed values. Data preparation reproduces all five support statistics for 11 languages and all five paper estimators. The retained NMC and priority-MMC fit summaries match their 20 printed macro values each; these comparisons do not replace a fresh fit. The retained pathwise-MMC PMI mean is `0.0122`, while the manuscript prints `0.0121`.

Complete current-manuscript reproduction still requires the full fit comparison, the three-estimator variance dumps, and the figure inference settings. The branch's approximate-estimator summaries are not substitutes for expectation-equivalent results.

## Notes

For the MMC producer, install the checkout with `python -m pip install -e ".[models,mmc]"`. The `mmc` extra supplies this study's language classifier;
core frequency estimators accept a caller-provided partition without it.

The package joins participant reading times and estimator tables from the author branch to per-unit MECO tables pinned at context-reading-time commit `b2b573e858c5cfc77b2f4babe502a70d9371d9a4`. The reference fold table is imported unchanged; the paper-estimator maps use the same R sampling function on their respective complete-case sets.

The `replica.studies.hopton2026unigram.mmc` package produces the marginal Monte Carlo unigram-surprisal pickles of the `mmc`, `mmc_pool` and `mmc_pool_ee` frequency sources. It runs one stage per call as `python -m replica.studies.hopton2026unigram.mmc <stage>`; [mmc/__init__.py](mmc/__init__.py) lists the stages.

The pooled sampler accepts an explicit `Stratification` and `AllocationSpec` from `replica.feature.frequency`. The library does not choose language groups or map dataset names to weights. For example, another application can partition prefixes by length:

```python
from replica.feature.frequency import AllocationSpec, Stratification

partition = Stratification(
    name="prefix-length/1",
    labels=("short", "long"),
    window_chars=64,
    classify=lambda text: "short" if len(text) < 32 else "long",
)
allocation = AllocationSpec(strata=partition.labels, k_eval=256, replicates=2)
```

Pass `stratification=partition` to `boundary_index` or `build_pool_selection`, size the buffer's stratum axis to `len(partition.labels)`, and pass `allocation` at estimation. Version the partition name whenever the classifier changes; its fingerprint covers that declared version, the ordered labels, and the window size. Classification must be deterministic. Optional per-word `home_weights` use these labels directly.

This study supplies its fixed 21-group MECO profile from [mmc/profile.py](mmc/profile.py), retaining the 40-character window, Latin backstop, allocation defaults, and buffer-sizing rule. New indices, partial sweeps, and selections store their partition identity; merges reject inconsistent identities. Selections use the new `mmc_pps_selection/1` layout so older estimators reject them. Loading a legacy `mmc_pps/2` selection upgrades its layout metadata without changing its numerical arrays. Legacy NPZ artifacts have no such metadata, so loading them requires the explicit profile, for example `PoolSelection.load(path, stratification=profile.STRATIFICATION)`. The study commands provide it and authenticate legacy allocation plans with their original digest. Unknown MECO dataset names now raise an error.

The source significance analysis drops language from its final grouping keys. Because MECO reuses `(trialid, ianum)` across languages, unrelated words at matching positions are averaged together. The port preserves this behavior for code fidelity; its reported `n` counts distinct positions, so words from different languages at one position count once. For setup, see the [installation guide](../../../../docs/guides/environment.md).
