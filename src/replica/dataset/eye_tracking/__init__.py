"""Eye-tracking corpora.

One module per corpus, plus :mod:`replica.dataset.eye_tracking.meco` for the
multi-release MECO family. The shared per-word reading-measure contract lives
on ``EyeTrackingDataset`` in :mod:`replica.dataset.base`, and the registry
imports each loader by module path, so this package exports nothing itself.
"""
