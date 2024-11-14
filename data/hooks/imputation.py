from logging import Logger
from typing import Self

import pandas as pd
from sklearn.impute import SimpleImputer

from config.utils import default_as, is_list, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import FittedHook
from data.mixins import MultiFeatureMixin


@registered_data_hook("imputation_simple")
class SimpleImputation(FittedHook):
    def __init__(self, config: dict, logger: Logger = Logger.root):
        # Grab an explicit list of columns, if they were defined
        self.features = parse_data_config_entry(
            "features", config,
            default_as([], logger), is_list(logger)
        )

        # Pass the rest of the arguments through to SimpleEncoder directly; THIS IS TEMPORARY
        self.backing_encoder = SimpleImputer(**config)

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger)

    def run(self, train_in: BaseDataManager, test_in: BaseDataManager = None):
        # If this is multi-feature dataset, sub-features can be selected
        if isinstance(train_in, MultiFeatureMixin):
            selected_features = self.features
            if not selected_features:
                selected_features = train_in.features()
                # The following check is done because Pandas won't allow loc-based assignment if the query is an index
                if isinstance(selected_features, pd.Index):
                    selected_features = selected_features.to_numpy()
            # Fit to and transform the training data first
            train_in: BaseDataManager | MultiFeatureMixin
            tmp_train: BaseDataManager | MultiFeatureMixin = train_in.get_features(selected_features)
            tmp_train = self.backing_encoder.fit_transform(tmp_train.as_array())
            train_out = train_in.set_features(selected_features, tmp_train)

            # Then ONLY transform the testing data
            test_in: BaseDataManager | MultiFeatureMixin
            tmp_test: BaseDataManager | MultiFeatureMixin = test_in.get_features(selected_features)
            tmp_test = self.backing_encoder.transform(tmp_test.as_array())
            test_out = test_in.set_features(selected_features, tmp_test)
        # Otherwise, just fit and transform everything
        # TODO: Implement a method of converting back to the original DataManager type
        else:
            train_out = self.backing_encoder.fit_transform(train_in.as_array())
            test_out = self.backing_encoder.fit_transform(test_in.as_array())
        return train_out, test_out