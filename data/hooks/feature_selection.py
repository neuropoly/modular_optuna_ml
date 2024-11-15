from abc import ABC
from logging import Logger
from typing import Self

import numpy as np
import pandas as pd

from config.utils import default_as, is_float, is_list, parse_data_config_entry
from data.base import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import StatelessHook
from data.mixins import MultiFeatureMixin


### Explicit Feature Selection ###
class ExplicitFeatures(StatelessHook, ABC):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Get the list of features to parse
        features = parse_data_config_entry(
            "features", config,
            is_list(self.logger)
        )
        self.features = features

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        new_instance = cls(config, logger=logger)
        return new_instance

@registered_data_hook("drop_features_explicit")
class ExplicitDrop(ExplicitFeatures):
    def run(self, data_in: BaseDataManager) -> BaseDataManager | MultiFeatureMixin:
        # We can only drop features in a dataset if they have more than 1
        if not isinstance(data_in, MultiFeatureMixin):
            raise TypeError(f"DataManager of type '{self.__class__}' only has one feature! Cannot modify feature set.")
        data_in: BaseDataManager | MultiFeatureMixin
        return data_in.drop_features(self.features)

@registered_data_hook("keep_features_explicit")
class ExplicitKeep(ExplicitFeatures):
    def run(self, data_in: BaseDataManager) -> BaseDataManager | MultiFeatureMixin:
        # We can only drop features in a dataset if they have more than 1
        if not isinstance(data_in, MultiFeatureMixin):
            raise TypeError(f"DataManager of type '{self.__class__}' only has one feature! Cannot modify feature set.")
        data_in: BaseDataManager | MultiFeatureMixin
        return data_in.get_features(self.features)


### Feature Selection by Null Content ###
class NullityDrop(StatelessHook, ABC):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Get the threshold for how to generate this instance
        self.threshold = parse_data_config_entry(
            "threshold", config,
            # TODO: Add min and max checks here as well
            default_as(0.5, self.logger), is_float(self.logger)
        )

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config=config, logger=logger)

@registered_data_hook("sample_drop_null")
class SampleNullityDrop(NullityDrop):
    def run(self, data_in: BaseDataManager) -> BaseDataManager:
        # Iterate through all samples and calculate the null content for each, dropping
        # any which surpass the configured threshold
        n_samples = len(data_in)
        null_tolerance = int(n_samples * self.threshold)
        keep_idx = [i for i, k in enumerate(data_in)
                    if np.sum(pd.isnull(k.as_array())) < null_tolerance]
        return data_in[keep_idx]

@registered_data_hook("feature_drop_null")
class FeatureNullityDrop(NullityDrop):
    def run(self, data_in: BaseDataManager) -> BaseDataManager:
        # Make sure this DataManager has more than one feature!
        if not isinstance(data_in, MultiFeatureMixin):
            raise TypeError(f"DataManager of type '{self.__class__.__name__}' only has one feature which cannot be dropped!")
        data_in: BaseDataManager | MultiFeatureMixin
        # Otherwise, calculate the null content for each feature dropping any which surpass the configured threshold
        n_samples = len(data_in)
        null_tolerance = int(n_samples * self.threshold)
        drop_idx = [k for k in data_in.features()
                   if np.sum(pd.isnull(data_in.get_features(k).as_array())) > null_tolerance]
        return data_in.drop_features(drop_idx)
