from abc import ABC, abstractmethod
from logging import Logger
from typing import Self, Optional

from config.utils import default_as, is_bool, parse_data_config_entry
from data.base import BaseDataManager


class DataHook(ABC):
    """
    Basic implementation for functions which can be called as data hooks.

    These should be configured during program init (to allow for fail-faster checking),
    after which they will be run at the user-specified points within the dataset
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
        Run this hook's process on a given DataManager in its entirely.
        :param x: The data to process
        :param y: The target metric to use, if the data hook needs it.
        :return: The data manager, post-processing. For safety, it should generally be its own (copied) instance
        """
        ...


class FittedDataHook(DataHook, ABC):
    """
    Data hook which "fits" itself to a set of training data, and uses that fit to
    inform how it will be applied to other datasets
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
        :param x_train: The data which should be used to "fit" the hook to, before it is applied to both
        :param x_test: A dataset which will have the hook applied to it, but not fit to it.
        :param y_train: The target metric to use during fitting, if the data hook needs it.
        :param y_test: The target metric to use during application to testing, if the data hook needs it.
        :return: The modified versions of x_train and x_test, after the fit has been applied to them
        """
        ...
