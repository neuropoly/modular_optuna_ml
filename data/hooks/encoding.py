from copy import copy as shallow_copy
from logging import Logger
from typing import Optional, Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config.utils import default_as, is_int, is_list, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import FittedDataHook
from data.mixins import MultiFeatureMixin


@registered_data_hook("one_hot_encode")
class OneHotEncoding(FittedDataHook):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Grab an explicit list of columns, if they were defined
        self.explicit_features = parse_data_config_entry(
            "features", config,
            default_as([], self.logger), is_list(self.logger)
        )
        # Grab the maximum number of unique values allowed before a column is treated as continuous
        self.cat_threshold = parse_data_config_entry(
            "max_unique_vals", config,
            default_as(-1, self.logger), is_int(self.logger)
        )

        # Pass the rest of the arguments through to the OneHotEncoder directly; THIS IS TEMPORARY
        self.backing_encoder = OneHotEncoder(**config)

        # Track the current feature labels being used to save some computation later
        self.tracked_features = None

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # If this is multi-feature dataset, sub-features can be selected
        if isinstance(x, MultiFeatureMixin):
            # Update the list of tracked features based on the dataset
            self.update_tracked_features(x)

            # Fit to and transform the training data first
            x: BaseDataManager | MultiFeatureMixin
            tmp_x = x.get_features(self.tracked_features)
            # noinspection PyUnresolvedReferences
            tmp_x = self.backing_encoder.fit_transform(tmp_x.as_array())
            # Densify result if it is in a sparse format
            if hasattr(tmp_x, "todense"):
                tmp_x = tmp_x.todense()

            # Generate the new feature names based on this transform, and delete the old ones!
            new_features = self.backing_encoder.get_feature_names_out(self.tracked_features)
            x_out = x.drop_features(self.tracked_features)
            x_out = x_out.set_features(new_features, tmp_x)
        # Otherwise, just fit and transform everything in bulk
        # TODO: Implement a method of converting back to the original DataManager type
        else:
            x_out = self.backing_encoder.fit_transform(x.as_array())
        return x_out

    def run_fitted(self,
            x_train: BaseDataManager,
            x_test: Optional[BaseDataManager],
            y_train: Optional[BaseDataManager] = None,
            y_test: Optional[BaseDataManager] = None
        ) -> (BaseDataManager, BaseDataManager):
        # If this is multi-feature dataset, sub-features can be selected
        if isinstance(x_train, MultiFeatureMixin):
            # Fit to and transform the training data first
            x_train: BaseDataManager | MultiFeatureMixin
            train_out = self.run(x_train, y_train)

            # Generate the new feature names based on this transform, and delete the old ones!
            new_features = self.backing_encoder.get_feature_names_out(self.tracked_features)

            # Then ONLY transform the testing data
            x_test: BaseDataManager | MultiFeatureMixin
            tmp_test: BaseDataManager | MultiFeatureMixin = x_test.get_features(self.tracked_features)
            tmp_test = self.backing_encoder.transform(tmp_test.as_array())
            if hasattr(tmp_test, "todense"):
                tmp_test = tmp_test.todense()
            test_out = x_test.drop_features(self.tracked_features)
            test_out = test_out.set_features(new_features, tmp_test)

        # Otherwise, just fit and transform everything in bulk
        # TODO: Implement a method of converting back to the original DataManager type
        else:
            train_out = self.backing_encoder.fit_transform(x_train.as_array())
            test_out = self.backing_encoder.fit_transform(x_test.as_array())
        return train_out, test_out

    def update_tracked_features(self, x):
        """
        Identifies and generates the list of features which should be OneHotEncoded
        :param x: The data to use for automated feature detection via unique value counting
        :return: The list of features which need to OneHotEncoded by this data hook for the given dataset
        """
        # Initialize with the features explicitly defined by the user
        tracked_features = shallow_copy(self.explicit_features)
        # If the user requested automatic detection as well, try to determine other categorical columns implicitly
        if self.cat_threshold != -1:
            to_test = [f for f in x.features() if f not in set(tracked_features)]
            for f in to_test:
                # noinspection PyUnresolvedReferences
                train_data = x.get_features(f).as_array()
                f_vals = np.ravel(train_data[pd.notnull(train_data)])
                f_unique = len(pd.unique(f_vals))
                if f_unique < self.cat_threshold:
                    tracked_features.append(f)
        # Return the result for easy use
        self.tracked_features = tracked_features

