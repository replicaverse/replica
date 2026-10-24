# Revisiting the Estimation of Unigram Surprisal in Computational Psycholinguistics

- [Abstract](#abstract)
- [Predictors](#predictors)
- [Reproduction](#reproduction)
- [BibTeX](#bibtex)

## Abstract

Contextual and unigram surprisal are standard predictors of reading time in computational psycholinguistics. While contextual surprisal is typically derived from neural language models, unigram surprisal is usually obtained from independent corpus counts. Although both quantities are deterministic functionals of a single underlying distribution---capturing conditional and marginal structure respectively---this practice breaks their probabilistic link and may introduce estimation error from the mismatch. We propose three principles for estimating unigram surprisal in psycholinguistic research, including the requirement that both quantities be derived from the same language model. We operationalize this principle by introducing a Monte Carlo estimator of unigram surprisal from a language model, along with a marginalized variant that has lower variance empirically. When unigram surprisal is re-estimated under this unified framework, we find evidence that predictors change in their predictive contribution to reading time. We also confirm prior findings that frequency can be a stronger predictor of processing effort than contextual surprisal under this unified framework.

## Predictors

The study compares corpus-based and model-based unigram surprisal, contextual
surprisal, and pointwise mutual information as predictors of reading time in
MECO's 11 languages. The model-based estimates use mGPT.

| `estimator` | Unigram estimator |
|---|---|
| `nmc` | Naive Monte Carlo |
| `mmc_pathwise` | Pathwise-exact marginalized Monte Carlo |
| `mmc_expectation_equivalent` | Expectation-equivalent marginalized Monte Carlo |
| `priority_mmc_pathwise` | Priority pathwise marginalized Monte Carlo |
| `priority_mmc_expectation_equivalent` | Priority expectation-equivalent marginalized Monte Carlo |

These estimators support gaze duration (`gaze_rt`) with space-prefixed words.
The default `reference` configuration also supports total fixation time
(`total_rt`), first-fixation duration (`firstfix_rt`), and `whitespace=False`.

## Reproduction

Install the [R backend and its dependencies](../../../../docs/guides/environment.md#optional-capabilities),
then download the study inputs:

```bash
replica results fetch hopton2026unigram --verification
```

Run one language and predictor comparison:

```python
from replica import evaluate
from replica.studies.hopton2026unigram import Hopton2026Unigram

study = Hopton2026Unigram()
options = dict(language="en", estimator="nmc", measure="gaze_rt")
dataset = study.dataset(**options)
spec = study.evaluation_spec(dataset=dataset, comp="freq", **options)
result, = evaluate(dataset, spec, study.analysis_config(**options))
print(study.word_scores(result, **options).dll.mean())
```

The analysis uses ten-fold cross-validation with generalized additive mixed
models. `word_scores()` averages participant contributions within each word;
language summaries receive equal weight in the manuscript's aggregate results.

See the [result catalog](results/README.md) for inputs and the
[reference fixtures](../../../../tests/studies/hopton2026unigram/fixtures/README.md)
for numerical checks and remaining reproduction limitations.

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
