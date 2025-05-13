from abc import ABC, abstractmethod
from itertools import chain
from logging import Logger
from typing import Iterable, Self, Sequence, Type

import numpy as np
import pandas as pd
from optuna import Trial

from tuning.utils import Tunable, TunableParam


# Denotes the type of data this class manages (float, filepaths etc.)
class BaseDataManager(Sequence, Tunable, ABC):
    """
    Base data manager class; you should subclass this and extend it with mixins to implement functionality!
    """
    def __init__(self, logger: Logger = Logger.root, **kwargs):
        super().__init__(**kwargs)

        # Track the logger for this object
        self.logger = logger

        # Keep tabs on the list of tunable parameters
        self.tunable_hooks: list[Tunable] = []

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """
        A pandas dataframe which will mediate the samples and their features for us
        """
        ...

    @abstractmethod
    def _replace_data(self, new_df: pd.DataFrame):
        """
        Protected method which directly replaces the dataframe inside this data manager.

        Should only be used by the class internally, as to retain consistency with any other
        attributes it may track.
        """
        ...

    def tune(self, trial: Trial):
        for h in self.tunable_hooks:
            h.tune(trial)

    def tunable_params(self) -> Iterable[TunableParam]:
        new_vals = [x.tunable_params() for x in self.tunable_hooks]
        return chain(*new_vals)

    @abstractmethod
    def shallow_copy(self) -> Self:
        """
        Generate a shallow copy of this object (that is, without entirely duplicating the data contained within it.
        Used in place of a full copy to greatly reduce memory overhead and runtime.
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> Self:
        """
        Generate an instance of this data manager based on the contents of a
        configuration file (provided in the form of dictionary)
        :param config: The contents of the configuration file relevant to this manager
        :return: The new data manager
        """
        ...

    def get_index(self) -> np.array:
        """
        Returns the index of the sample, as a np array.
        Usually is the positional index in dataset, but can be something else in some cases!
        (i.e. pandas dataframes with a set non-int index)
        """
        return self.data.index.to_numpy()

    def features(self) -> Iterable[str]:
        # List all features available in the dataset
        return self.data.columns

    def n_features(self) -> int:
        # Just returns the number of features in this dataset; required for certain checks
        return self.data.shape[1]

    def get_samples(self, idx) -> Self:
        """
        Get a set of samples based on the index provided.
        At minimum, this should allow selecting samples by their position
        :param idx: An index dictating what samples to return
        :return: The requested set of samples, as a DataManager
        """
        sub_df = self.data.iloc[idx, :]
        if isinstance(sub_df, pd.Series):
            # This BS is needed because Pandas auto-casts single index queries to Series
            sub_df = pd.DataFrame(data=sub_df).T
        new_instance = self.shallow_copy()
        new_instance._replace_data(sub_df)
        return new_instance

    def get_features(self, idx) -> Self:
        """
        Explicitly query for some features within this DataManager
        :param idx: The feature(s) to get from this class.
        :return: A subset of the DataManager's data with only the requested features.
            This should *always* be an instance of the same class to allow for function chaining!
        """
        sub_df = self.data.loc[:, idx]
        if isinstance(sub_df, pd.Series):
            # This BS is needed because Pandas auto-casts single index queries to Series
            sub_df = pd.DataFrame(data=sub_df)
        new_instance = self.shallow_copy()
        new_instance._replace_data(sub_df)
        return new_instance

    def set_features(self, idx, new_data) -> Self:
        """
        Set the values of some feature(s), overwriting them if they already exist
        :param idx: The feature(s) ot overwrite or set
        :param new_data: The data to use
        :return: An instance of the data manager w/ the new features
        """
        new_df = self.data.copy()
        new_df.loc[:, idx] = new_data
        new_instance = self.shallow_copy()
        new_instance._replace_data(new_df)
        return new_instance

    def drop_features(self, idx) -> Self:
        """
        Drop some subset of features from the dataset
        :param idx: The feature(s) to drop
        :return: A modified version of this instance
        """
        new_df = self.data.copy()
        new_df = new_df.drop(columns=idx)
        new_instance = self.shallow_copy()
        new_instance._replace_data(new_df)
        return new_instance

    def as_array(self) -> np.ndarray:
        """
        Return a numpy array representation of this Data Manager. Needed as some tools insist on using this type
        """
        return self.data.to_numpy()

    @abstractmethod
    def pre_split(self, is_cross: bool, targets: Self = None) -> Self:
        """
        Run anything that needs to be run prior to the data being train-test split.
        Returns an instance with these modifications applied
        """
        ...

    @abstractmethod
    def split(
            self,
            train_idx: np.ndarray,
            test_idx: np.ndarray,
            train_target: Self,
            test_target: Self,
            is_cross: bool = True
    ) -> (Self, Self):
        """
        Split the data into two subsets. Any post-split modifications should be done here
        :param train_idx: The sample indices for the training set
        :param test_idx: The sample indices for the testing set
        :param train_target: A dataset of values which will be used as a "target" for supervised data hooks
            during training data processing
        :param test_target: A dataset of values which will be used as a "target" for supervised data hooks
            during testing data processing
        :param is_cross: Whether this split is being run during cross-validation (v.s. during replicate setup)
        :return: Two sub-instances of the same type of datamanager, being the training and testing data, respectively
        """
        ...

    @abstractmethod
    def __len__(self):
        # How this is done will depend on the backing data structure
        ...

    def __getitem__(self, idx):
        # By default, delegate square bracket indexing to sample-wise querying
        return self.get_samples(idx)

DATA_MANAGERS: dict[str, Type[BaseDataManager]] = {}

# Decorator to allow for registry key to be kept alongside the class of interest
def registered_datamanager(key: str):
    def _decorator(cls: Type[BaseDataManager]):
        # Decorator which registers a data manager under a specific key automatically
        if key in DATA_MANAGERS.keys():
            Logger.root.warning(f"Overwriting data manager '{key}' which already existed. "
                                f"Are you sure you wanted to do this?")
        DATA_MANAGERS[key] = cls
    return _decorator