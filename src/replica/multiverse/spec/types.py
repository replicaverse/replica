"""Coordinates for a dataset and dependent measure."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetCell:
    """One (dataset, dv) coordinate the driver iterates over."""

    name: str
    dv: str
    paradigm: str  # 'et' | 'spr' | 'erp' | 'judgement'
    feature_name: str | None = None

    @property
    def pickle_name(self) -> str:
        """Corpus name used for REPLICA predictor pickles."""
        return self.feature_name or self.name
