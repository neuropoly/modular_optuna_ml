from copy import copy as shallow_copy
from logging import DEBUG, Logger
from typing import Optional, Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from config.utils import default_as, is_float, is_int, is_list, is_not_null, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import FittedDataHook


@registered_data_hook("one_hot_encode")
class OneHotEncoding(FittedDataHook):
    """
    Modified implementation of SciKit-Learn's OneHotEncoder to allow it to function in the context of data of mixed
        categorical/continuous features. Which features it will be applied to can be specified with two configuration
        parameters:

    * `features`: A list of features within the dataset which should explicitly be treated as categorical and encoded
        using this data hook
    * `max_unique_vals`: An integer threshold; if a feature within the dataset has fewer than this many unique
        values across all samples is assumed to be "categorical", and is encoded.

    How the resulting data is encoded is otherwise identical to SciKit's implementation; its parameters can be specified
    like any other config option (see "handle_unknown" in the example below).

    Example usage:
    {
        "type": "one_hot_encode",
        "features": ["color"],
        "max_unique_vals": 5,
        "handle_unknown: "ignore"
    }
    In the case of a binary variable:
        {
        "type": "one_hot_encode",
        "features": ["sex"],
        "drop": "if_binary",
        "handle_unknown": "warn"
    }
    """
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
        # Update the list of tracked features based on the dataset
        self.update_tracked_features(x)

        # Fit to and transform the training data first
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

        return x_out

    def run_fitted(self,
            x_train: BaseDataManager,
            x_test: Optional[BaseDataManager],
            y_train: Optional[BaseDataManager] = None,
            y_test: Optional[BaseDataManager] = None
        ) -> (BaseDataManager, BaseDataManager):
        # Fit to and transform the training data first
        train_out = self.run(x_train, y_train)

        # Generate the new feature names based on this transform, and delete the old ones!
        new_features = self.backing_encoder.get_feature_names_out(self.tracked_features)

        # Then ONLY transform the testing data
        tmp_test = x_test.get_features(self.tracked_features)
        tmp_test = self.backing_encoder.transform(tmp_test.as_array())
        if hasattr(tmp_test, "todense"):
            tmp_test = tmp_test.todense()
        test_out = x_test.drop_features(self.tracked_features)
        test_out = test_out.set_features(new_features, tmp_test)

        return train_out, test_out

    def update_tracked_features(self, x):
        """
        Identifies and generates the list of features which should be OneHotEncoded. This is tracked internally within
            the data hook, and simply merges the explicitly defined and implicitly detected categorical features into
            a single, easier to work with list.

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
    """
    Modified implementation of SciKit-Learn's OrdinalEncoder to allow it to function in the context of data of mixed
        categorical/continuous features. The following additional/modified parameters are available to accomplish
        this:

    * `features`: A list of features within the dataset which should explicitly be treated as categorical and encoded
        using this data hook
    * `categories`: A list, or list of lists (if multiple features were specified) designating the order the labels
        should be ordered in (i.e. setting it to ["A", "C", "B"] would result in the data ["A", "B", "C"] being encoded
        as [0, 2, 1], rather than the default [0, 1, 2]). Unlike the base implementation of OrdinalEncoder, this is
        required in this context, as otherwise the randomized splitting of the input data can result in some different
        replicates having the same label for different initial classes (i.e. in replicate one, class "A" is labelled
        as 1, but in replicate two it gets labelled as 2).

    How the resulting data is encoded is otherwise identical to SciKit's implementation.

    Example usage:
    {
        "type": "ordinal_encode",
        "features": ["size"],
        "categories": ["small", "medium", "large"],
        "handle_unknown: "ignore"
    }
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Grab an explicit list of features, if they were defined
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
        self.backing_encoder = OrdinalEncoder(categories=categories, **config)

        # Keep track of which features we encode
        self.tracked_features = None

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # If this is a multi-feature dataset, select the relevant features
        self.update_tracked_features(x)
        tmp_x = x.get_features(self.tracked_features).as_array()

        # Fit+transform (for training) the ordinal encoder
        tmp_x = self.backing_encoder.fit_transform(tmp_x)

        # Replace them with the new columns
        x_out = x.set_features(self.tracked_features, tmp_x)

        return x_out

    def run_fitted(self,
                   x_train: BaseDataManager,
                   x_test: Optional[BaseDataManager],
                   y_train: Optional[BaseDataManager] = None,
                   y_test: Optional[BaseDataManager] = None
                   ) -> (BaseDataManager, BaseDataManager):
        # Fit+transform on the training data
        train_out = self.run(x_train, y_train)

        # Transform (only) the test data
        tmp_test = x_test.get_features(self.tracked_features).as_array()
        tmp_test = self.backing_encoder.transform(tmp_test)
        test_out = x_test.drop_features(self.tracked_features)
        test_out = test_out.set_features(self.tracked_features, tmp_test)

        return train_out, test_out

    def update_tracked_features(self, x: BaseDataManager) -> None:
        """
        TODO: implement autodetection of features to encode
        """
        self.tracked_features = shallow_copy(self.explicit_features)


@registered_data_hook("ladder_encode")
class LadderEncoding(FittedDataHook):
    """
    A niche extension of one-hot-encoding which allows for a user specified order to be preserved without enforcing that
        the step size between each class is the same. For why this is valuable, consider the following example:

    ['small', 'medium', 'small', 'large', 'medium', 'small']

    By default, applying the One-Hot encoding to it would result in the following:

    --------------------------
    | small | medium | large |
    --------------------------
    |   1   |   0    |   0   |
    |   0   |   1    |   0   |
    |   1   |   0    |   0   |
    |   0   |   0    |   1   |
    |   0   |   1    |   0   |
    |   1   |   0    |   0   |
    --------------------------

    This does not preserve any order in the data; label encoding it creates a different problem

    --------
    | size |
    --------
    |   0  |
    |   1  |
    |   0  |
    |   2  |
    |   1  |
    |   0  |
    --------

    Namely, while order is now preserved, the "step" between each category is now required to be identical (as the
        difference between each class is 1). Ladder encoding attempts to act as a middle ground by preserving the
        column-wise stratification of one-hot encoding, while forcing the underlying ML model to make their effects
        additive to one another:

    --------------------------
    | small | medium | large |
    --------------------------
    |   1   |   1    |   1   |
    |   0   |   1    |   1   |
    |   1   |   1    |   1   |
    |   0   |   0    |   1   |
    |   0   |   1    |   1   |
    |   1   |   1    |   1   |
    --------------------------

    It requires two configuration parameters to be used:

    * `feature`: The feature you want to ladder encode
    * `order`: The order the features should be encoded in (that is, how the ladder will be formed).

    Aside from these params, it can accept any parameter that the SciKit-Learn implementation of OneHotEncoder accepts. For example:
    
    * `min_frequency`: Any value which is present in less than this number (if int) or proportion (if float) of samples in the dataset are treated as part of the 'infrequent' class during encoding, rather than receiving their own encoded feature.

    Example usage:
    {
        "type": "ladder_encode",
        "feature": "size",
        "order": ["small", "medium", "large"],
        "min_frequency": 0.3,
        "handle_unknown": "warn"
    }
    """
    def __init__(self, config: dict, **kwargs):
        # TODO: Allow this to handle multiple features simultaneously
        super().__init__(config, **kwargs)

        # Grab an explicit list of columns, if they were defined
        self.feature = parse_data_config_entry(
            "feature", config,
            is_not_null(self.logger)
        )
        # Have the user define the order of the ladder explicitly;
        #  this needs to provided for a LadderEncode to make sense!
        order = parse_data_config_entry(
            "order", config
        )
        if type(order) is not list:
            raise ValueError("Ladder encoding needs a list of ordinal values, in the desired order, to function. "
                             f"Value of the data config was of type '{type(order)}'.")
        if len(order) < 2:
            raise ValueError("To ladder encode data, at least 2 values must be provided in the 'order' argument. "
                             f"The provided list had {len(order)} value(s) instead.")
        self.order = order

        # We handle min-frequency detection ourselves, to avoid the creation of an "infrequent" column which has no
        #  position in the order
        self.min_frequency = parse_data_config_entry(
            "min_frequency", config,
            default_as(0., self.logger), is_float(self.logger)
        )

        # Slap the user for trying to specify whether the data should be spare or not
        sparse_output = parse_data_config_entry(
            "sparse_output", config
        )
        if sparse_output is not None:
            raise ValueError("Please don't define whether the data should be sparse of not; this is handled by the tool.")

        # Generate the backing OneHotEncoder; note that many config options are isolated explicitly to prevent headaches
        self.backing_ohe_encoder = OneHotEncoder(
            min_frequency=None,
            sparse_output=False,
            feature_name_combiner=lambda _, y: y,
            **config
        )

        # Track a list of values which will be concatenated, determined during fit
        self.order_groups = []

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # Setup
        sub_x = x.get_features([self.feature])

        # Fit this model to the provided feature subset
        x_df = self.fit(sub_x)

        # Use the (now fit) encoder to generate our encoded data
        x_df = self.ohe_to_ladder(x_df)

        # Update the dataset using these new feature names
        x_out = x.drop_features([self.feature])
        x_out = x_out.set_features(x_df.columns, x_df.to_numpy())

        # Return the result
        return x_out

    def fit(self, x):
        """
        Fits the OneHotEncoder to our data, and identifies groups of sequential columns which, when grouped together,
            are "frequent" enough to pass the frequency check given by the user.

        Doing so explicitly prevents issues down the line (i.e. a rare class which appears during training but not
            testing breaking the transform)
        """
        # Reset the groupings before proceeding
        self.order_groups = []

        x = self.backing_ohe_encoder.fit_transform(x.as_array())

        # Generate the OHE feature names to help with iteration
        ohe_feature_cols = self.backing_ohe_encoder.get_feature_names_out([self.feature])

        # If the user wants debugging info, report the features which are present in only 'order' or the OHE groups
        if self.logger.isEnabledFor(DEBUG):
            ohe_set = set(ohe_feature_cols)
            order_set = set(self.order)

            # DEBUG ONLY: Warn the user if any columns exist in one set, but not the other
            for f in (ohe_set - order_set):
                self.logger.debug(f"Feature {f} exists in the OHE, but not the specified order list.")
            for f in (order_set - ohe_set):
                self.logger.debug(f"Feature {f} exists in the order list, but not the OHE.")

        # Convert our OHE matrix into a dataframe for sanity’s sake
        ohe_df = pd.DataFrame(x, columns=ohe_feature_cols)

        # Generate the list of features shared between the OHE and the specified order, sorting them to abide by the latter
        ordered_and_shared_cols = [f for f in self.order if f in ohe_feature_cols]

        # Use the prior list to re-order and filter the OHE dataframe
        ohe_df = ohe_df.loc[:, ordered_and_shared_cols]

        # Iterate through the columns in the OHE in our specified order to identity clusters of infrequent groups
        col_group = []
        for c in self.order:
            # If this column doesn't exist in the OHE, treat it as frequency 0 and add a homogenous column to the ohe_df
            if c not in ohe_df.columns:
                col_group.append(c)
                ohe_df[c] = 0
                continue

            # Otherwise, append it to our current column group
            col_group.append(c)

            # Test whether this new column group results in a grouping which is no longer infrequent
            ladder_sample = np.cumsum(ohe_df.loc[:, col_group], axis=1).iloc[:, -1]

            # If not, proceed as normal
            if np.sum(ladder_sample) < self.min_frequency * ohe_df.shape[0]:
                continue

            # Otherwise, append the group to our order group list as-is
            self.order_groups.append(col_group)

            # Reset the state in preparation for the next column
            col_group = []

        # Edge-Case; if the tailing classes in the order are infrequent on their own, they would be "orphaned" by the
        #   prior loop. To avoid this, append them to the last group we tested
        if len(col_group) > 0:
            self.order_groups[-1].extend(col_group)

        # Return the OHE-encoded version of the data for re-use
        return ohe_df

    def run_fitted(self,
            x_train: BaseDataManager,
            x_test: Optional[BaseDataManager],
            y_train: Optional[BaseDataManager] = None,
            y_test: Optional[BaseDataManager] = None
        ) -> (BaseDataManager, BaseDataManager):
        # Fit to and transform the training data first
        train_out = self.run(x_train, y_train)

        # Use the now-fit encoder to transform our testing input
        sub_x_train = x_test.get_features([self.feature])
        sub_x_train = self.backing_ohe_encoder.transform(sub_x_train.as_array())

        ohe_feature_cols = self.backing_ohe_encoder.get_feature_names_out([self.feature])

        # Convert it to a dataframe for ease of use
        ohe_df = pd.DataFrame(sub_x_train, columns=ohe_feature_cols)

        x_df = self.ohe_to_ladder(ohe_df)

        # Update the testing dataset with these results
        x_test = x_test.drop_features([self.feature])
        test_out = x_test.set_features(x_df.columns, x_df.to_numpy())

        return train_out, test_out


    def ohe_to_ladder(self, x_df: pd.DataFrame):
        """
        Uses the CumSum trick to convert a One-Hot-Encoded dataset into a Ladder encoded one

        :param x_df: The one-hot-encoded form of the dataset you want to encode, encoded using the OneHotEncoder
            managed by an instance of this class and formatted as a dense pandas DataFrame
        """
        # Cache a set-formatted version of the columns in the provided dataframe for later re-use
        x_df_col_set = set(x_df.columns)

        # Iterate through the fitted groups for this model to pool them into "rungs" on the ladder
        ladder_dict = {}
        prior_group_str = None
        for g in self.order_groups:
            # Generate a union of the group's contents and the contents of the OHE columns to avoid invalid queries
            joint_cols = list(set(g).intersection(x_df_col_set))

            # Isolate these groups from the rest of the data
            rung_df = x_df.loc[:, joint_cols]

            # Use panda's "any" check to generate pool all columns into a single one
            rung_val = rung_df.any(axis="columns")

            # Generate a string representing the combination of columns which will be grouped
            group_str = "|".join([str(x) for x in g])

            # Generate the corresponding column's name, but only if we had a prior group (the first col will be dropped)
            col_name = ""
            if prior_group_str is not None:
                col_name = f"{self.feature} ({prior_group_str} <- {group_str})"

            # Save this result into the dictionary for later
            ladder_dict[col_name] = rung_val

            # Update the prior group's label
            prior_group_str = group_str

        # Convert this dictionary into a proper dataframe for ease of use
        ladder_df = pd.DataFrame.from_dict(ladder_dict)

        # Drop the first (base) column to avoid a homogenous final column
        ladder_df = ladder_df.iloc[:, 1:]

        # Reverse the column order before applying cumsum
        ladder_df = ladder_df.iloc[:, ::-1]

        # Apply CumSum again to finalize the ladder encoding
        ladder_df = np.cumsum(ladder_df, axis=1)

        # Reverse the column order back to the original
        ladder_df = ladder_df.iloc[:, ::-1]

        # Return the result
        return ladder_df
