from abc import ABC, abstractmethod
from itertools import chain
from logging import Logger
from typing import Iterable, Self, Sequence, Type

import numpy as np
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

    def tune(self, trial: Trial):
        for h in self.tunable_hooks:
            h.tune(trial)

    def tunable_params(self) -> Iterable[TunableParam]:
        new_vals = [x.tunable_params() for x in self.tunable_hooks]
        return chain(*new_vals)

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

    @abstractmethod
    def get_samples(self, idx) -> Self:
        """
        Get a set of samples based on the index provided.
        At minimum, this should allow selecting samples by their position
        :param idx: An index dictating what samples to return
        :return:
        """

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """
        Return a numpy array representation of this Data Manager. Needed as some tools insist on using this type
        """
        ...

    @abstractmethod
    def pre_split(self, is_cross: bool, targets: Self = None) -> Self:
        """
        Run anything that needs to be run prior to the data being train-test split.
        Returns and instance with these modifications applied
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