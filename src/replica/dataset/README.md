# Datasets

`replica.dataset` exports the dataset classes and the discovery and loading API.
Start with `available_datasets()`, `describe_dataset(name)` and
`load_dataset(name, **options)`.

| Location | Responsibility |
| --- | --- |
| `base.py`, `contracts.py` | Dataset interfaces, shared records, fixation calculations and in-memory tabular datasets. |
| `catalog/` | Dataset registration, discovery, source descriptions, sentence context, corpus fingerprints and recorded counts. |
| `eeg/`, `eye_tracking/`, `fmri/`, `maze/`, `spr/` | Loaders grouped by measurement paradigm; related releases share a subpackage. |
| `loading/` | Source-table readers, fixation processing and response reaggregation onto spans. |
| `modelblocks.py` | Adapter for ModelBlocks-format corpus tables. |
| `validation.py` | Stimulus-content checks on loaded datasets. |
| `measures/` | Measurement names, quantities and released source-column declarations. |
| `metadata/` | Metadata I/O, checksum verification and the shipped YAML descriptions. |
| `fetch/` | Dataset retrieval using the source catalog. |

ZuCo's EEG loaders live in `eeg/zuco/`, alongside its shared EEG view. The
predictability-norm reader lives with the de Varda release in
`eye_tracking/devarda2023/`; the UCL and Frank loaders reuse it explicitly.

`python -m replica.dataset.catalog.facts` produces dataset inventory information.
`python -m replica.dataset.metadata.generate` updates metadata checksums and
therefore writes the shipped YAML files. Ordinary loading reads those files.
