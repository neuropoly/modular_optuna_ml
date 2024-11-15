from logging import Logger
from typing import Self
from copy import copy as shallow_copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config.utils import default_as, is_int, is_list, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import FittedHook
from data.mixins import MultiFeatureMixin


@registered_data_hook("one_hot_encode")
class OneHotEncoding(FittedHook):
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


    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, train_in: BaseDataManager, test_in: BaseDataManager = None):
        # If this is multi-feature dataset, sub-features can be selected
        test_out = None
        if isinstance(train_in, MultiFeatureMixin):
            # Initialize with the features explicitly defined by the user
            selected_features = shallow_copy(self.explicit_features)
            # If the user requested automatic detection as well, try to determine other categorical columns implicitly
            if self.cat_threshold != -1:
                to_test = [f for f in train_in.features() if f not in set(selected_features)]
                for f in to_test:
                    # noinspection PyUnresolvedReferences
                    train_data = train_in.get_features(f).as_array()
                    f_vals = np.ravel(train_data[pd.notnull(train_data)])
                    f_unique = len(pd.unique(f_vals))
                    if f_unique < self.cat_threshold:
                        selected_features.append(f)

            # Fit to and transform the training data first
            train_in: BaseDataManager | MultiFeatureMixin
            tmp_train = train_in.get_features(selected_features)
            # noinspection PyUnresolvedReferences
            tmp_train = self.backing_encoder.fit_transform(tmp_train.as_array())
            # Densify result if it is in a sparse format
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

