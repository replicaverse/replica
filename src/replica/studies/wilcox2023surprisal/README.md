# Testing the Predictions of Surprisal Theory in 11 Languages

- [Abstract](#abstract)
- [Predictors](#predictors)
- [BibTeX](#bibtex)

## Abstract

Surprisal theory posits that less-predictable words should take more time to process, with word predictability quantified as surprisal, i.e., negative log probability in context. While evidence supporting the predictions of surprisal theory has been replicated widely, much of it has focused on a very narrow slice of data: native English speakers reading English texts. Indeed, no comprehensive multilingual analysis exists. We address this gap in the current literature by investigating the relationship between surprisal and reading times in eleven different languages, distributed across five language families. Deriving estimates from language models trained on monolingual and multilingual corpora, we test three predictions associated with surprisal theory: (i) whether surprisal is predictive of reading times, (ii) whether expected surprisal, i.e., contextual entropy, is predictive of reading times, and (iii) whether the linking function between surprisal and reading times is linear. We find that all three predictions are borne out crosslinguistically. By focusing on a more diverse set of languages, we argue that these results offer the most robust link to date between information theory and incremental language processing across languages.

## Predictors

The authors' convention, the REPLICA code that reproduces it, and the published
predictor cell that is faithful to it.

Convention: **no BOS**, the first word of a unit is set to zero, values in
**bits**. The study ships two context scopes and compares them, so context scope
is an axis the paper itself defines.

| predictor | authors' file | REPLICA | faithful cell | agreement |
|---|---|---|---|---|
| surprisal, long context | [`lm_results/mgpt_lc/{lang}_preds.csv`](https://github.com/wilcoxeg/xlang-processing/tree/73a2cd845ccc69f5706953b02b3f94ff6fcdb139/data/lm_results/mgpt_lc) | [`HuggingFaceLM(add_bos=False)`](../../feature/backends/huggingface.py) over the whole passage | **none** | comparison output is not included |
| surprisal, sentence context | [`lm_results/mgpt_sc/{lang}_preds.csv`](https://github.com/wilcoxeg/xlang-processing/tree/73a2cd845ccc69f5706953b02b3f94ff6fcdb139/data/lm_results/mgpt_sc) | same with `context_scope="sentence"`, boundaries from MECO's own per-word `sentnum` ([`replica.dataset.catalog.sentences`](../../dataset/catalog/sentences.py)) | `ai-forever_mGPT_surprisal_nobos_sent_meco_*` | agrees to float noise over the partition-identical trials in most languages; du/gr/he carry a few per-word outliers from the authors' `retokenize` fallback (zero-filled sentences, space-marker misgrouping); the probe output is not tracked |
| contextual entropy | `ent` in the same files, summed over a word's subword tokens | [`HuggingFaceLM(token_level_entropy=True)`](../../feature/backends/huggingface.py) read by `TokenEntropy`, the first subword token's entropy, which is the paper's definition and the corrected reproduction's predictor | none published; scored live with `--corrected-entropy` | the released column is the summed variant, so the two are not expected to agree; the corrected run's decisions are tabulated above |
| unigram frequency, length | `freq`, `len` | dataset columns | n/a | read from the released CSV |

`mgpt_sc` and `mgpt_lc` differ on most rows, because `sc` conditions only
within a sentence. The two agree only inside each passage's first sentence.

Both scopes are storable under their own names: `add_bos` reaches the cache
tag as the `_nobos` suffix and `context_scope="sentence"` as `_sent`.
Sentence boundaries come from MECO's per-word `sentnum`, the same column the
authors' [`clean_data.Rmd`](https://github.com/wilcoxeg/xlang-processing/blob/73a2cd845ccc69f5706953b02b3f94ff6fcdb139/scripts/clean_data.Rmd#L132-L137)
grouped contexts by. Their released `mgpt_sc` files
carry that column pair swapped (which their pipeline renames back), diverge
from the MECO text mid-trial for English trials 2-3 and Russian trials 4-11,
and merge sentences at some semicolons where `sentnum` splits; the anchor
above therefore compares only trials whose sentence partition matches.

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
