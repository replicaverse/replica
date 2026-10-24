"""EEG dataset loaders."""

from replica.dataset.eeg.zuco.n400 import ZuCoN400
from replica.dataset.eeg.zuco.zuco1 import ZuCo1EEG
from replica.dataset.eeg.zuco.zuco2 import ZuCo2EEG

from ._n400 import N400Dataset
from .frank2015_erp import Frank2015ERP
from .michaelov_n400 import MichaelovN400
from .szewczyk_n400 import SzewczykN400

__all__ = [
    "N400Dataset",
    "Frank2015ERP",
    "MichaelovN400",
    "SzewczykN400",
    "ZuCoN400",
    "ZuCo1EEG",
    "ZuCo2EEG",
]
