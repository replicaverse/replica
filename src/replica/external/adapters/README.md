# Upstream adapters

This directory contains input adapters and patches for pinned upstream code.
[`repositories.py`](../repositories.py) declares the sources and patch paths;
[`dependency.py`](../dependency.py) acquires and patches the checkouts. Each study
declares the repositories it uses through `repo_dependencies()`.

[`context_reading.py`](context_reading.py) locates the processed tables from
[Opedal et al. (2024)](https://aclanthology.org/2024.emnlp-main.179/), used by
the Hopton study. It prefers `merged_data_no_zero`
in [context-reading-time at b2b573e](https://github.com/rycolab/context-reading-time/tree/b2b573e858c5cfc77b2f4babe502a70d9371d9a4),
then the processed MECO directory under the configured data root.

## Compatibility patches

- [`cdr-emnlp18.patch`](cdr-emnlp18.patch) adapts
  [CDR at 0b8f9cf](https://github.com/coryshain/cdr/tree/0b8f9cf45395fe0bb34cae7bd6c0fcc572d28704).
  It preserves empty fields when reading space-delimited tables, converts only
  formula/grouping columns to R, evaluates transformed response names when
  scoring baselines, guards an exhausted history index, and removes an
  unsupported Keras progress-bar argument. These changes address input parsing,
  allocation failures and runtime errors in that release.
- [`anticipation-on-reading-times.patch`](anticipation-on-reading-times.patch)
  adapts [anticipation-on-reading-times at 14291ca](https://github.com/rycolab/anticipation-on-reading-times/tree/14291ca18467d4dd1c4955a141de326558b52e1b).
  It adapts Dundee preprocessing to current pandas APIs and missing values,
  drops fixation rows absent from the text word list, and makes the unused
  `tidymv` import optional in the linear-model path. The word-list filter is
  triggered when pandas reads the literal word `None` as missing; with the
  upstream missing-value settings it removes no rows.
- [`modelblocks-release.patch`](modelblocks-release.patch) supports the GECO
  and PROVO table producers in
  [ModelBlocks at 3ae5120](https://github.com/modelblocks/modelblocks-release/tree/3ae5120327005cde1c02576eda24876997e20550).
  It passes CSV separators by keyword and uses `.loc` for the PROVO offscreen
  label assignment. ModelBlocks-based studies consume these response tables.
