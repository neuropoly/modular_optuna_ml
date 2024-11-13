from logging import Logger
from pathlib import Path
from typing import Iterable, Optional, Self

import pandas as pd

from config.utils import as_str, default_as, is_file, parse_data_config_entry
from data.base import BaseDataManager, registered_datamanager
from data.mixins import MultiFeatureMixin


@registered_datamanager("tabular")
class TabularDataManager(BaseDataManager, MultiFeatureMixin):
    """
    Data manager for data stored in tabular data formats (i.e. 'csv')

    Uses a Panda's dataframe as a backend to read the files and manage the majority of data queries and modifications.
    """
    def __init__(self, logger: Logger = Logger.root):
        # Use 'from_config' below, rather than using this constructor directly
        self.logger = logger

        # Default parameters
        self._data: Optional[pd.DataFrame] = None
        self._sep: str = ','

    def __len__(self):
        return self.data.shape[0]

    @property
    def data(self) -> pd.DataFrame:
        """
        Managed as a pseudo-cached property to allow for lazy loading
        """
        # If our data is still just a reference to a file, load it with pandas
        if isinstance(self._data, Path):
            self._data = pd.read_csv(self._data, sep=self._sep)
        # Return the result
        return self._data

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root):
        # Initiate a new instance of this class
        new_instance = cls(logger)

        # Track the data source for, making sure it's properly formatted
        new_instance._data = parse_data_config_entry(
            "data_source", config, as_str(logger), is_file(logger)
        )
        # Track the data separator for later
        default_sep = default_as(new_instance._sep, logger)
        new_instance._sep = parse_data_config_entry(
            "separator", config, default_sep, as_str(logger)
        )

        return new_instance

    def get_samples(self, idx) -> Self:
        sub_df = self.data.iloc[idx, :]
        if isinstance(sub_df, pd.Series):
            # This BS is needed because Pandas auto-casts single index queries to Series
            sub_df = pd.DataFrame(data=sub_df).T
        new_instance = self.shallow_copy()
        new_instance._data = sub_df
        return new_instance

    def get_features(self, idx) -> Self:
        # Creates a new TabularDataManager with only the queried features
        sub_df = self.data.loc[:, idx]
        if isinstance(sub_df, pd.Series):
            # This BS is needed because Pandas auto-casts single index queries to Series
            sub_df = pd.DataFrame(data=sub_df)
        new_instance = self.shallow_copy()
        new_instance._data = sub_df
        return new_instance

    def set_features(self, idx, new_data) -> Self:
        # Creates a new TabularDataManager with the modified features in place
        new_df = self.data.copy()
        new_df.loc[:, idx] = new_data
        new_instance = self.shallow_copy()
        new_instance._data = new_df
        return new_instance

    def drop_features(self, idx) -> Self:
        new_df = self.data.copy()
        new_df = new_df.drop(columns=idx)
        new_instance = self.shallow_copy()
        new_instance._data = new_df
        return new_instance

    def n_features(self) -> int:
        return self.data.shape[1]

    def features(self) -> Iterable[str]:
        return self.data.columns

    def as_array(self):
        return self.data.to_numpy()

    def pre_split(self) -> Self:
        # TODO: Implement this
        new_instance = self
        return new_instance

    def split(self, train_idx, test_idx) -> (Self, Self):
        # Initial split and instance setup
        train_instance = self.shallow_copy()
        train_instance._data = self.data.iloc[train_idx, :]
        test_instance = self.shallow_copy()
        test_instance._data = self.data.iloc[test_idx, :]

        # TODO; Implement post-split hook application

        # Return the resulting split
        return train_instance, test_instance

    def shallow_copy(self) -> Self:
        # Simple utility to reduce the amount of boilerplate
        new_instance = self.__class__()
        new_instance._data = self._data

        return new_instance
