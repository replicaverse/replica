# Testing the Predictions of Surprisal Theory in 11 Languages

- [Abstract](#abstract)
- [Predictors](#predictors)
- [Usage](#usage)
- [BibTeX](#bibtex)

## Abstract

Surprisal theory posits that less-predictable words should take more time to process, with word predictability quantified as surprisal, i.e., negative log probability in context. While evidence supporting the predictions of surprisal theory has been replicated widely, much of it has focused on a very narrow slice of data: native English speakers reading English texts. Indeed, no comprehensive multilingual analysis exists. We address this gap in the current literature by investigating the relationship between surprisal and reading times in eleven different languages, distributed across five language families. Deriving estimates from language models trained on monolingual and multilingual corpora, we test three predictions associated with surprisal theory: (i) whether surprisal is predictive of reading times, (ii) whether expected surprisal, i.e., contextual entropy, is predictive of reading times, and (iii) whether the linking function between surprisal and reading times is linear. We find that all three predictions are borne out crosslinguistically. By focusing on a more diverse set of languages, we argue that these results offer the most robust link to date between information theory and incremental language processing across languages.

## Predictors

The recipe reads the authors' released surprisal, contextual entropy, frequency
and word-length columns from the
[merged MECO tables](https://github.com/wilcoxeg/xlang-processing/tree/73a2cd845ccc69f5706953b02b3f94ff6fcdb139/data/merged_data).
Each predictor includes two preceding-word lags. Choose the released model with
`lm`: `mgpt_lc` uses passage context, `mgpt_sc` uses sentence context, and
`monot_30m` and `monot_all` select the monolingual models.

The released entropy column sums subtoken entropies within a word. For a
first-subtoken entropy analysis, pass a live model exposing `token_entropy` and
set `token_entropy=True` in `evaluation_spec()`. Live models require a dataset
loaded with `full_text=True` so excluded observations remain available as context.

## Usage

The default comparison fits reading times from surprisal, controlling for
frequency and length, using ordinary least squares and ten sequential folds.
`dataset()` downloads the pinned upstream repository when it is missing.

```python
from replica import evaluate
from replica.studies import wilcox2023surprisal as study

options = dict(language="en", lm="mgpt_lc", dv="gaze_rt")
dataset = study.dataset(**options)
spec = study.evaluation_spec(dataset=dataset, **options)
result, = evaluate(dataset, spec, study.analysis_config())
print(result.delta_ll_per_unit)
```

Select `dv="total_rt"` or `dv="firstfix_rt"` for the other reading measures.
`contrast="ent_add"` tests entropy in addition to a surprisal baseline;
`evaluate_entropy_replace_dll()` compares entropy with surprisal directly.
The study also supplies `link_dataset()`, `link_evaluation_spec()` and
`link_analysis_config()` for its linear-versus-nonlinear comparison, which
requires the [R backend](../../../../docs/guides/environment.md#optional-capabilities).

## BibTeX

```bibtex
@article{wilcox-etal-2023-testing,
  title = {Testing the Predictions of Surprisal Theory in 11 Languages},
  author = {Wilcox, Ethan G. and
      Pimentel, Tiago and
      Meister, Clara and
      Cotterell, Ryan and
      Levy, Roger P.},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {11},
  year = {2023},
  url = {https://aclanthology.org/2023.tacl-1.82/},
  doi = {10.1162/tacl_a_00612},
  pages = {1451--1470}
}
```
