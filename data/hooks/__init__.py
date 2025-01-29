""" Registry to allow for new managers to be added by external modules """
from logging import Logger
from typing import Type

from data.hooks.base import DataHook

DATA_HOOKS: dict[str, Type[DataHook]] = {}

# Decorator to allow for registry key to be kept alongside the class of interest
def registered_data_hook(key: str):
    """
    Decorator for registering a data hook.

    :param key: The label that this data hook will be registered as. This will be the string that the user provides
        in the 'type' argument within the data configuration file to request the corresponding data hook be used.

    NOTE: Currently, for a data hook to be registered, the package it is part of must be imported in this file
    specifically. We are looking into a more elegant solution to this, but for now, add new imports to the set
    placed below this class to ensure the data hooks in that module are registered correctly.
    """
    def _decorator(cls: Type[DataHook]):
        # Decorator which registers a data manager under a specific key automatically
        if key in DATA_HOOKS.keys():
            Logger.root.warning(f"Overwriting data hook '{key}' which already existed. "
                                f"Are you sure you wanted to do this?")
        DATA_HOOKS[key] = cls
    return _decorator


# TODO: Find a more elegant way to do this
from data.hooks.feature_selection import (
    SampleNullityDrop, FeatureNullityDrop, ExplicitDrop, ExplicitKeep, PrincipalComponentAnalysis,
    RecursiveFeatureElimination
)
from data.hooks.imputation import SimpleImputation
from data.hooks.encoding import OneHotEncoding
from data.hooks.standardization import StandardScaling