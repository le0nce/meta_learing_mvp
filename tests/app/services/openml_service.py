"""Service to access openML data"""

from dataclasses import dataclass
from typing import Tuple, Union
from scipy.sparse import csr_matrix # type: ignore
import numpy as np
import openml
import pandas as pd

DataMatrix = Union[np.ndarray, pd.DataFrame, csr_matrix]
ColumnMatrix = Union[np.ndarray, pd.DataFrame, None]

@dataclass
class OpenMLService:
    """Class OpenMLService for all OpenML related actions"""

    def load_dataset(self, dataset_id: int) -> Tuple[
        DataMatrix,
        ColumnMatrix
    ]:
        """Loads a dataset by the given openML dataset_id"""
        data: openml.OpenMLDataset = openml.datasets.get_dataset(
            dataset_id=dataset_id
        )
        dataset, target_column, _, _ = data.get_data(target=data.default_target_attribute)
        return dataset, target_column
