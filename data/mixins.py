from abc import ABC, abstractmethod
from typing import Self, Iterable


class MultiFeatureMixin(ABC):
    @abstractmethod
    def get_features(self, idx) -> Self:
        """
        Explicitly query for some features within this DataManager
        :param idx: The feature(s) to get from this class.
        :return: A subset of the DataManager's data with only the requested features.
            This should *always* be an instance of the same class to allow for function chaining!
        """
        ...

    def __getitem__(self, idx):
        # By default, try to delegate to superclass querying (usually query-by-sample)
        try:
            return self.__getitem__(idx)
        # If that fails, try and get features instead
        except IndexError:
            return self.get_features(idx)
