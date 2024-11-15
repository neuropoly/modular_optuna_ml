from logging import Logger
from typing import Self

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.utils import default_as, is_list, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import FittedHook
from data.mixins import MultiFeatureMixin


@registered_data_hook("standard_scaling")
class StandardScaling(FittedHook):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Grab an explicit list of columns, if they were defined
        self.features = parse_data_config_entry(
            "features", config,
            default_as([], self.logger), is_list(self.logger)
        )

        # Pass the rest of the arguments through to the OneHotEncoder directly; THIS IS TEMPORARY
        self.backing_encoder = StandardScaler(**config)

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, train_in: BaseDataManager, test_in: BaseDataManager = None):
        # If this is multi-feature dataset, sub-features can be selected
        test_out = None
        if isinstance(train_in, MultiFeatureMixin):
            # Initialize with the features explicitly defined by the user
            selected_features = self.features
            if not selected_features:
                selected_features = train_in.features()
                # The following check is done because Pandas won't allow loc-based assignment if the query is an index
                if isinstance(selected_features, pd.Index):
                    selected_features = selected_features.to_numpy()

            # Fit to and transform the training data first
            train_in: BaseDataManager | MultiFeatureMixin
            tmp_train = train_in.get_features(selected_features)
            # noinspection PyUnresolvedReferences
            tmp_train = self.backing_encoder.fit_transform(tmp_train.as_array())
            # If we got something which can be "densified", do so
            if hasattr(tmp_train, "todense"):
                tmp_train = tmp_train.todense()

            # Generate the new feature names based on this transform, and delete the old ones!
            new_features = self.backing_encoder.get_feature_names_out(selected_features)
            train_out = train_in.drop_features(selected_features)
            train_out = train_out.set_features(new_features, tmp_train)

            # Then ONLY transform the testing data
            if test_in is not None:
                test_in: BaseDataManager | MultiFeatureMixin
                tmp_test: BaseDataManager | MultiFeatureMixin = test_in.get_features(selected_features)
                tmp_test = self.backing_encoder.transform(tmp_test.as_array())
                # If we got something which can be "densified", do so
                if hasattr(tmp_test, "todense"):
                    tmp_test = tmp_test.todense()
                test_out = test_in.drop_features(selected_features)
                test_out = test_out.set_features(new_features, tmp_test)

        # Otherwise, just fit and transform everything in bulk
        # TODO: Implement a method of converting back to the original DataManager type
        else:
            train_out = self.backing_encoder.fit_transform(train_in.as_array())
            if test_in is not None:
                test_out = self.backing_encoder.fit_transform(test_in.as_array())
        return train_out, test_out

