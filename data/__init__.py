""" Registry to allow for new managers to be added by external modules """
from logging import Logger
from typing import Type

from data.base import BaseDataManager

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

# TODO: Find a more elegant way to do this
from data.tabular import TabularDataManager
