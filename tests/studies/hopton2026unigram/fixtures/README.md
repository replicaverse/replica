# Hopton unigram references

- `paper_reference.json`: transcriptions of the twelve table environments in
  the indexed `paper_tables.tex` resource. Labels and line offsets locate each
  table in that excerpt; the values are reference targets, not recomputed results.
- `author_gamm_means.tsv`: per-language gaze-duration means extracted from the
  indexed `author_code/caseviz_meco.js`, grid `sp1_ou0_lg0`. The grid contains
  word-level summaries. Its NMC and priority-MMC macro summaries match the
  manuscript; the pathwise-MMC PMI mean differs by one printed decimal unit.
- `../../fixtures/hopton2026_{r,sig}_reference*.tsv`: retained R summary tables
  for the `reference` estimator and both whitespace variants. Their paths and
  byte hashes are recorded in the study's `data/provenance.json`.
- `author_r_words.parquet`: an R replay's 2,081 English word scores for each
  reference surprisal and frequency comparison. `author_r_words.json` records
  the indexed original and selected R scripts, inputs, settings and output hashes.

Indexed resources resolve through the study's
[result index](../../../../src/replica/studies/hopton2026unigram/results/index.json),
which records their archive locations, immutable revisions and checksums.
They identify retained evidence; original-generation details remain unknown
where the accompanying metadata says so.

The R replay test compares every reference word at `rtol=1e-6`, `atol=1e-8`.
The retained summary checks use the author's tolerance, `rtol=1e-6` and
`atol=3e-5`, for ΔLL mean, SD and SE, and check the reference word count.
The original R frequency fit reports nonconvergence in fold 9; the fixture
preserves that output and records the warning.
Runtime tests aggregate raw measurements before comparing printed values;
support tests prepare participant observations independently of the manuscript
transcription.

The manuscript tables and retained R/grid summaries report word-level ΔLL.
Participant contributions are averaged within words before averaging words.
Final figure confidence intervals and significance settings still need
provenance reconciliation.
