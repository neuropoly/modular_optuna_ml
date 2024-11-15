""" Registry to allow for new managers to be added by external modules """
from logging import Logger
from typing import Type

from data.hooks.base import BaseDataHook

DATA_HOOKS: dict[str, Type[BaseDataHook]] = {}

# Decorator to allow for registry key to be kept alongside the class of interest
def registered_data_hook(key: str):
    def _decorator(cls: Type[BaseDataHook]):
        # Decorator which registers a data manager under a specific key automatically
        if key in DATA_HOOKS.keys():
            Logger.root.warning(f"Overwriting data hook '{key}' which already existed. "
                                f"Are you sure you wanted to do this?")
        DATA_HOOKS[key] = cls
    return _decorator


# TODO: Find a more elegant way to do this
from data.hooks.feature_selection import SampleNullityDrop, FeatureNullityDrop, ExplicitDrop, ExplicitKeep
from data.hooks.imputation import SimpleImputation
from data.hooks.encoding import OneHotEncoding
from data.hooks.standardization import StandardScaling