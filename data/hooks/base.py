from abc import ABC, abstractmethod
from logging import Logger
from typing import Self

from data.base import BaseDataManager


class BaseDataHook(ABC):
    """
    Basic implementation for functions which can be called as data hooks.

    These should be configured during program init (to allow for fail-faster checking),
    after which they will be run at the user-specified points within the dataset
    """
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


class StatelessHook(BaseDataHook, ABC):
    """
    A Data hook which does need to fit to a defined set of data to work on the data.
    That is, regardless of what data it had been shown prior, its behaviour will not change.
    """
    @abstractmethod
    def run(self, data_in: BaseDataManager) -> BaseDataManager:
        """
        Run this hook's process on a given DataManager in its entirely.
        :param data_in: The data to process
        :return: The data manager, post-processing. For safety, it should generally be its own (copied) instance
        """
        ...


class FittedHook(BaseDataHook, ABC):
    """
    Data hook which "fits" itself to a set of training data, and uses that fit to
    inform how it will be applied to other datasets
    """
    @abstractmethod
    def run(self, train_in: BaseDataManager, test_in: BaseDataManager = None):
        """
        Run this hook's process on a pair of DataManagers, fitting on the training input applying to both
        :param train_in: The data which should be used to "fit" the hook to, before it is applied to both
        :param test_in: A dataset which will have the hook applied to it, but not fit to it.
                    If left blank, only the `train_in` data will be used and returned
        :return: Each of the datasets used
        """
        ...
