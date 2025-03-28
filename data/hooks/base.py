from abc import ABC, abstractmethod
from logging import Logger
from typing import Self, Optional

from config.utils import default_as, is_bool, parse_data_config_entry
from data.base import BaseDataManager


class DataHook(ABC):
    """
    Basic implementation for functions which can be called as data hooks.

    Data hook use is split into two stages: initialization and application.

    During initialization, any configuration settings the user provided are parsed and used to initialize the data hook
        instance. This is done before any ML analyses are run (to follow the fail-faster paradigm), and is where you
        should parse any configuration options and check their validity. Anything that needs to be run once at the
        beginning, before the data hook is applied to any data, should be done here as well. This code is usually placed
        into the `from_config` function.

    During application, the hook is applied to the data provided to it (in `BaseDataManager` form). At minimum, the
        data hook will receive a data manager `x` containing feature/regressor values `x`. If the analysis is a
        supervised one, can also receive a data manager containing the target values `y` as well. See `run` for further
        details.
    """
    def __init__(self, config: dict, logger: Logger = Logger.root):
        # Basic init implementation which tracks attributes shared with all data hooks
        self.logger = logger

        self.run_per_replicate = parse_data_config_entry(
            "run_per_replicate", config,
            default_as(True, logger), is_bool(logger)
        )
        self.run_per_cross = parse_data_config_entry(
            "run_per_cross", config,
            default_as(False, logger), is_bool(logger)
        )

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        """
        Generate a new instance of this DataHook based on the contents of a configuration file
        :param config: The contents of a data configuration file which are relevant to this hook
        :param logger: A logger to log any notifications with.
        :return: A new instance of this hook type
        """
        ...

    @abstractmethod
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        """
        Run this hook's process on a given DataManager in its entirety.

        To avoid potentially propagating data modifications across replicates and/or cross-validation splits, this
            function should return a modified copy of the original input data, rather than modifying the input
            DataManager(s) directly!

        :param x: The data to process
        :param y: The target metric to use, if the data hook needs it.
        :return: A copy of the original data manager `x`, with this data hook applied to it.
        """
        ...


class FittedDataHook(DataHook, ABC):
    """
    An extended data hook which allows for the hook to be "fit" to a training dataset, then applied to said training
        dataset AND another testing dataset. This should be used for data hooks which manage data transformations which,
        if they were applied to the entire dataset indiscriminately, would result in data leakage.

    Like `run` before it, this function should return modified copies of its input data, rather than modifying the input
        DataManager(s) directly!
    """
    @abstractmethod
    def run_fitted(self,
            x_train: BaseDataManager,
            x_test: Optional[BaseDataManager],
            y_train: Optional[BaseDataManager] = None,
            y_test: Optional[BaseDataManager] = None
        ) -> (BaseDataManager, BaseDataManager):
        """
        Run this hook's process on a pair of DataManagers, fitting on the training input applying to both

        :param x_train: A dataset which will be used to "train" the hook. Post-training, the hook is applied to it as well
        :param x_test: A dataset which will have the hook applied to it only, but will not affect how the hook is "trained".
        :param y_train: The target metric associated with `x_train` for each of its samples, should the analyses be a supervised one.
        :param y_test: The target metric associated with `x_test` for each of its samples, should the analyses be a supervised one.
            NOTE: 'y_test' is here solely for standardization, and should probably never be used to avoid overfitting!
        :return: The modified copies of x_train and x_test, after the fit has been applied to them.
        """
        ...
