from logging import Logger
from typing import Self, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.utils import default_as, is_list, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import FittedDataHook
from data.mixins import MultiFeatureMixin


@registered_data_hook("standard_scaling")
class StandardScaling(FittedDataHook):
    """
    Standardizes each feature in the provided dataset to follow a unit norm (that is, have mean of 0 and a standard
        deviation of 1).

    Utilizes SciKit-Learn's StandardScaler implementation. As such, aside from the addition of a "features" argument
        to allow the user to specify a set of feature they want the scaling applied to, its implementation and
        use here is identical.

    If no features list is explicitly defined by the user (via the "features" argument), this hook will be applied to
        all features in the dataset it was provided.

    Example usage:
    {
      "type": "standard_scaling",
      "features": ["age", "height", "weight"],
      "with_mean": false
    }
    """
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

    def run(self,
            x: BaseDataManager,
            y: Optional[BaseDataManager] = None
        ) -> BaseDataManager:
        # If this is multi-feature dataset, sub-features can be selected
        if isinstance(x, MultiFeatureMixin):
            # Initialize with the features explicitly defined by the user
            selected_features = self.features
            if not selected_features:
                selected_features = x.features()
                # The following check is done because Pandas won't allow loc-based assignment if the query is an index
                if isinstance(selected_features, pd.Index):
                    selected_features = selected_features.to_numpy()

            # Fit to and transform the training data first
            x: BaseDataManager | MultiFeatureMixin
            tmp_x = x.get_features(selected_features)
            # noinspection PyUnresolvedReferences
            tmp_x = self.backing_encoder.fit_transform(tmp_x.as_array())
            # If we got something which can be "densified", do so
            if hasattr(tmp_x, "todense"):
                tmp_x = tmp_x.todense()

            # Generate the new feature names based on this transform, and delete the old ones!
            new_features = self.backing_encoder.get_feature_names_out(selected_features)
            x_out = x.drop_features(selected_features)
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
            x_train: BaseDataManager | MultiFeatureMixin
            train_out = self.run(x_train, y_train)
            # Initialize with the features explicitly defined by the user
            selected_features = self.features
            if not selected_features:
                selected_features = x_train.features()
                # The following check is done because Pandas won't allow loc-based assignment if the query is an index
                if isinstance(selected_features, pd.Index):
                    selected_features = selected_features.to_numpy()

            # Generate the new feature names based on this transform, and delete the old ones!
            new_features = self.backing_encoder.get_feature_names_out(selected_features)

            # Then ONLY transform the testing data
            x_test: BaseDataManager | MultiFeatureMixin
            tmp_test: BaseDataManager | MultiFeatureMixin = x_test.get_features(selected_features)
            tmp_test = self.backing_encoder.transform(tmp_test.as_array())
            # If we got something which can be "densified", do so
            if hasattr(tmp_test, "todense"):
                tmp_test = tmp_test.todense()
            test_out = x_test.drop_features(selected_features)
            test_out = test_out.set_features(new_features, tmp_test)

        # Otherwise, just fit and transform everything in bulk
        # TODO: Implement a method of converting back to the original DataManager type
        else:
            train_out = self.backing_encoder.fit_transform(x_train.as_array())
            test_out = self.backing_encoder.transform(x_test.as_array())
        return train_out, test_out

