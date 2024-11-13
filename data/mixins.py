from abc import ABC, abstractmethod
from typing import Self, Iterable


class MultiFeatureMixin(ABC):
    @abstractmethod
    def features(self) -> Iterable[str]:
        # List all features available in the dataset
        ...

    @abstractmethod
    def get_features(self, idx) -> Self:
        """
        Explicitly query for some features within this DataManager
        :param idx: The feature(s) to get from this class.
        :return: A subset of the DataManager's data with only the requested features.
            This should *always* be an instance of the same class to allow for function chaining!
        """
        ...

    @abstractmethod
    def set_features(self, idx, new_data) -> Self:
        """
        Set the values of some feature(s), overwriting them if they already exist
        :param idx: The feature(s) ot overwrite or set
        :param new_data: The data to use
        :return: An instance of the data manager w/ the new features
        """
        ...

    @abstractmethod
    def drop_features(self, idx) -> Self:
        """
        Drop some subset of features from the dataset
        :param idx: The feature(s) to drop
        :return: A modified version of this instance
        """
        ...


    @abstractmethod
    def n_features(self) -> int:
        # Just returns the number of features in this dataset; required for certain checks
        ...

    def __getitem__(self, idx):
        # By default, try to delegate to superclass querying (usually query-by-sample)
        try:
            return self.__getitem__(idx)
        # If that fails, try and get features instead
        except IndexError:
            return self.get_features(idx)
