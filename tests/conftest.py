import json
import tempfile
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest

from config.data import DataConfig
from data.base import BaseDataManager


class DummyDataManager(BaseDataManager):

    def __init__(self, new_data: pd.DataFrame, **kwargs):
        # Initiate superclass stuff
        super().__init__(**kwargs)

        # Set the DataFrame managed by this object to be the dataframe provided
        self._data = new_data

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def _replace_data(self, new_df: pd.DataFrame):
        self._data = new_df

    def shallow_copy(self) -> Self:
        return DummyDataManager(self._data)

    @classmethod
    def from_config(cls, config: dict) -> Self:
        raise NotImplementedError(
            f"This is a dummy data manager meant for automated testing; it cannot be built from a config file."
        )

    def pre_split(self, is_cross: bool, targets: Self = None) -> Self:
        raise NotImplementedError(
            f"This is a dummy data manager meant for automated testing; it should not be split."
        )

    def split(self, train_idx: np.ndarray, test_idx: np.ndarray, train_target: Self, test_target: Self,
              is_cross: bool = True) -> (Self, Self):
        raise NotImplementedError(
            f"This is a dummy data manager meant for automated testing; it should not be split."
        )

    def __len__(self):
        return self._data.shape[0]


@pytest.fixture
def iris_data_config():
    """
    Fixture that returns a DataConfig instance for the Iris dataset.
    The tsv file is loaded as a TabularDataManager instance: data_config.data_manager
    Loaded data is stored as a pandas DataFrame: data_config.data_manager.data
    """
    config_dict = {
        "label": "IrisTesting",
        "format": "tabular",
        "data_source": str(Path(__file__).parent.resolve() / 'testing_files' / 'iris_data' / 'iris_testing.tsv'),
        "separator": "\t",
        "index": "id",
    }

    # Create a temporary file for the config
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_file:
        json.dump(config_dict, tmp_file)
        tmp_config_path = Path(tmp_file.name)  # Convert string path to Path object

    # Ensure the file exists before using it
    assert tmp_config_path.exists(), f"Temporary config file {tmp_config_path} was not created."

    data_config = DataConfig.from_json_file(tmp_config_path)

    return data_config


