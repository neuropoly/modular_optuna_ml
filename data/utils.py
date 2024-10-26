from abc import ABC, abstractmethod
from logging import Logger
from typing import Self


class DataManager(ABC):
    """
    Abstract class for an object which manages data for our study
    """
    @staticmethod
    @abstractmethod
    def build_from_config_dict(config_dict: dict, logger: Logger):
        """
        Configure this datamanager using attributes from a passed dictionary
        """
        pass

    @abstractmethod
    def get_by_idx(self, idx):
        """
        Get a value within the dataset managed by this class with [] query notation
        :return: The value for the query, or (if it's a slice) a DataManager which manages that subset of samples
        """
        pass

    @abstractmethod
    def get_len(self):
        """
        :return: The number of entries managed by this dataset
        """
        pass

    @abstractmethod
    def array(self):
        """
        :return: The object's contents as an array-like object
        """
        pass

    @abstractmethod
    def process_pre_analysis(self):
        """
        Run any pre-processing that needs to be done before a train-test split
        """
        pass

    @abstractmethod
    def train_test_split(self, train_idx, test_idx):
        """
        Split the data into a train and test dataset, running any processing required (such as standardization) while doing so
        :param train_idx: Indices of the data which should be placed into the training dataset
        :param test_idx: Indices of the data which should be placed into the testing dataset
        :return: Two DataManagers, which contain the training and testing data respectively
        """
        pass

    # When indexing this object, use the 'get_by_idx' method
    def __getitem__(self, idx):
        return self.get_by_idx(idx)

    # When getting the length of this object, use the 'get_len' method
    def __len__(self):
        return self.get_len()

    # Numpy is a special snowflake and made this custom for itself!
    def __array__(self, dtype = None):
        return self.array()



class FeatureSplittableManager(DataManager, ABC):
    """
    Extension of DataManager which denotes that features within the DataManager can be split off
    without impacting the integrity of the DataManager
    """
    @abstractmethod
    def pop_features(self, features) -> Self:
        """
        Isolate a set of values (such as the target for a machine learning algorithm)
        :param features: What features to query from the data managed by this DataManager
        :return: The data for the feature(s) requested
        """
        pass
