from copy import copy as shallow_copy
from logging import DEBUG, Logger
from typing import Optional, Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from config.utils import default_as, is_int, is_list, is_not_null, parse_data_config_entry
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


@registered_data_hook("ordinal_encode")
class OrdinalEncoding(FittedDataHook):
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Grab an explicit list of columns, if they were defined
        self.explicit_features = parse_data_config_entry(
            "features", config,
            default_as([], self.logger), is_list(self.logger)
        )
        # Grab the list of categories to use for each feature, if they were defined
        categories = parse_data_config_entry(
            "categories", config,
            default_as(None, self.logger), is_list(self.logger)
        )
        if categories and not isinstance(categories[0], list):
            categories = [categories]

        # Create the underlying OrdinalEncoder
        self.backing_encoder = OrdinalEncoder()#categories=categories)

        # Keep track of which features we encode
        self.tracked_features = None

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # If this is a multi-feature dataset, select the relevant features
        if isinstance(x, MultiFeatureMixin):
            self.update_tracked_features(x)
            tmp_x = x.get_features(self.tracked_features).as_array()

            # Fit+transform (for training) the ordinal encoder
            tmp_x = self.backing_encoder.fit_transform(tmp_x)

            # Replace them with the new columns
            x_out = x.set_features(self.tracked_features, tmp_x)
        else:
            # Not a multi-feature dataset, so encode the entire array
            x_out = self.backing_encoder.fit_transform(x.as_array())

        return x_out

    def run_fitted(self,
                   x_train: BaseDataManager,
                   x_test: Optional[BaseDataManager],
                   y_train: Optional[BaseDataManager] = None,
                   y_test: Optional[BaseDataManager] = None
                   ) -> (BaseDataManager, BaseDataManager):
        if isinstance(x_train, MultiFeatureMixin):
            # Fit+transform on the training data
            train_out = self.run(x_train, y_train)

            # Transform (only) the test data
            tmp_test = x_test.get_features(self.tracked_features).as_array()
            tmp_test = self.backing_encoder.transform(tmp_test)
            test_out = x_test.drop_features(self.tracked_features)
            test_out = test_out.set_features(self.tracked_features, tmp_test)
        else:
            # Encode the entire arrays when not using multi-feature
            train_out = self.backing_encoder.fit_transform(x_train.as_array())
            test_out = self.backing_encoder.transform(x_test.as_array())

        return train_out, test_out

    def update_tracked_features(self, x: BaseDataManager) -> None:
        """
        TODO: implement autodetection of features to encode
        """
        self.tracked_features = shallow_copy(self.explicit_features)


@registered_data_hook("ladder_encode")
class LadderEncoding(FittedDataHook):
    """
    Extension of one-hot-encoding which allows for a specified order to be preserved
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Grab an explicit list of columns, if they were defined
        self.feature = parse_data_config_entry(
            "feature", config,
            is_not_null(self.logger)
        )
        # Grab the maximum number of unique values allowed before a column is treated as continuous
        self.order = parse_data_config_entry(
            "order", config,
            default_as([], self.logger), is_list(self.logger)
        )

        # Pass the rest of the arguments through to the OneHotEncoder directly; THIS IS TEMPORARY
        self.backing_ohe_encoder = OneHotEncoder(**config)

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # If this is multi-feature dataset, sub-features can be selected
        if isinstance(x, MultiFeatureMixin):
            # Fit to and transform the training data first
            x: BaseDataManager | MultiFeatureMixin

            # Setup
            tmp_x = x.get_features([self.feature])

            # Encode initially using One-Hot-Encoding; we'll build up from here
            # noinspection PyUnresolvedReferences
            tmp_x = self.backing_ohe_encoder.fit_transform(tmp_x.as_array())
            # Densify result if it is in a sparse format
            if hasattr(tmp_x, "todense"):
                tmp_x = tmp_x.todense()

            # Use the (now fit) OHE to generate our ladder encode
            x_df: pd.DataFrame = self.ladder_encode(tmp_x)

            # Update the dataset using these new feature names
            x_out = x.drop_features([self.feature])
            x_out = x_out.set_features(x_df.columns, x_df.to_numpy())
            x_out: MultiFeatureMixin | BaseDataManager

            # Return the result
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
            x_test: BaseDataManager | MultiFeatureMixin
            train_out = self.run(x_train, y_train)

            # Use the now-fit encoder to transform our testing input
            tmp_x: BaseDataManager | MultiFeatureMixin = x_test.get_features([self.feature])
            tmp_x = self.backing_ohe_encoder.transform(tmp_x.as_array())
            # Densify result if it is in a sparse format
            if hasattr(tmp_x, "todense"):
                tmp_x = tmp_x.todense()

            x_df = self.ladder_encode(tmp_x)

            # Update the testing dataset with these results
            x_test = x_test.drop_features([self.feature])
            test_out = x_test.set_features(x_df.columns, x_df.to_numpy())

        # Ladder Encoding only makes sense in the context of multiple features; as such, any other type will not work!
        else:
            raise NotImplementedError("Ladder Encoding only makes sense in the context of a multi-feature dataset!")
        return train_out, test_out

    def ladder_encode(self, tmp_x):
        """
        Uses the np.cumsum trick to convert a One-Hot-Encoded dataset into a Ladder encoded one
        """
        # Generate the OHE feature names to help with iteration
        ohe_feature_cols = self.backing_ohe_encoder.get_feature_names_out([self.feature])
        ordered_feature_cols = [f"{self.feature}_{c}" for c in self.order]

        # If the user is wanting debugging info, report the features which are present in only 'order' or the OHE
        if self.logger.isEnabledFor(DEBUG):
            ohe_set = set(ohe_feature_cols)
            order_set = set(ordered_feature_cols)

            # DEBUG ONLY: Warn the user if any columns exist in one set, but not the other
            for f in (ohe_set - order_set):
                self.logger.debug(f"Feature {f} exists in the OHE, but not the specified order list.")
            for f in (order_set - ohe_set):
                self.logger.debug(f"Feature {f} exists in the order list, but not the OHE.")

        # Convert our matrix into a dataframe for sanity’s sake
        x_df = pd.DataFrame(tmp_x, columns=ohe_feature_cols)

        # Filter and re-order the dataset to contain only the shared columns in both, preserving the 'order'
        x_df = x_df.loc[:, [f for f in ordered_feature_cols if f in ohe_feature_cols]]

        # Use the CumSum trick to format it into a proper "ladder" encode
        x_df = np.cumsum(x_df, axis=1)

        # Return the result
        return x_df
