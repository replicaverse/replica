"""Dataset package: base classes and concrete dataset implementations."""

from replica.dataset.base import (
    Dataset,
    EEGDataset,
    EyeTrackingDataset,
    ReadingTimeDataset,
    SPRDataset,
    TabularDataset,
    TabularDatasetConfig,
)
from replica.dataset.catalog.description import (
    DatasetDescription,
    DatasetParameter,
    DatasetVariant,
    available_datasets,
    describe_dataset,
    format_dataset_description,
    load_dataset,
)
from replica.dataset.contracts import (
    DatasetInfo,
    FileEntry,
    FixationCalculation,
    IndexMapping,
)
from replica.dataset.eeg.zuco.zuco1 import ZuCo1EEG
from replica.dataset.eeg.zuco.zuco2 import ZuCo2EEG
from replica.dataset.features import fixation_feature
from replica.dataset.metadata.io import (
    ChecksumMismatchError,
    compute_checksums,
    compute_sha256,
    find_dataset_yaml,
    from_dict,
    load_yaml,
    save_yaml,
    to_dict,
    verify_checksums,
)

# EEG datasets
from .eeg.frank2015_erp import Frank2015ERP
from .eye_tracking.bsc import BSC
from .eye_tracking.celer import CELER
from .eye_tracking.copco import CopCo
from .eye_tracking.devarda2023 import DeVarda2023

# Eye-tracking datasets
from .eye_tracking.dundee.shain2024 import Dundee
from .eye_tracking.emtec import EMTeC
from .eye_tracking.geco import GECO
from .eye_tracking.meco import (
    L1W2_SITE_LANGUAGES,
    LANGUAGES,
    MECO,
    MECO_RELEASES,
    MECODataset,
    MECORelease,
    MECORepresentation,
)
from .eye_tracking.onestop import OneStop
from .eye_tracking.potec import PoTeC
from .eye_tracking.provo import PROVO
from .eye_tracking.rastros import RastrOS
from .eye_tracking.rsc import RSC
from .eye_tracking.sbsat import SBSAT
from .eye_tracking.ucl import UCL_ET
from .eye_tracking.zuco.zuco1 import ZuCo1
from .eye_tracking.zuco.zuco2 import ZuCo2

# fMRI datasets
from .fmri.narratives import Narratives

# Maze datasets
from .maze.base import MazeDataset
from .maze.natural_stories_maze import NaturalStoriesMaze
from .spr.brothers_kuperberg import BrothersKuperberg
from .spr.brown_spr import BrownSPR

# Self-paced reading datasets
from .spr.natural_stories import NaturalStories
from .spr.ucl import UCL_SPR

__all__ = [
    "BSC",
    "CELER",
    "GECO",
    "LANGUAGES",
    "MECO",
    "MECODataset",
    "MECORelease",
    "MECORepresentation",
    "MECO_RELEASES",
    "L1W2_SITE_LANGUAGES",
    "PROVO",
    "RSC",
    "RastrOS",
    "SBSAT",
    "UCL_ET",
    "UCL_SPR",
    "BrothersKuperberg",
    "BrownSPR",
    "ChecksumMismatchError",
    "CopCo",
    "Dataset",
    "DatasetDescription",
    "DatasetInfo",
    "DatasetParameter",
    "DatasetVariant",
    "DeVarda2023",
    "Dundee",
    "EEGDataset",
    "EMTeC",
    "EyeTrackingDataset",
    "FixationCalculation",
    "FileEntry",
    "Frank2015ERP",
    "IndexMapping",
    "Narratives",
    "MazeDataset",
    "NaturalStoriesMaze",
    "NaturalStories",
    "OneStop",
    "PoTeC",
    "ReadingTimeDataset",
    "SPRDataset",
    "TabularDataset",
    "TabularDatasetConfig",
    "ZuCo1",
    "ZuCo1EEG",
    "ZuCo2",
    "ZuCo2EEG",
    "available_datasets",
    "compute_checksums",
    "compute_sha256",
    "describe_dataset",
    "find_dataset_yaml",
    "fixation_feature",
    "format_dataset_description",
    "from_dict",
    "load_yaml",
    "load_dataset",
    "save_yaml",
    "to_dict",
    "verify_checksums",
]
