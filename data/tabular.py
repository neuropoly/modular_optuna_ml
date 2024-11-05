from copy import deepcopy
from logging import Logger
from sys import maxsize as int_maxsize
from typing import Optional, Collection

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.utils import parse_data_config_entry, as_str, default_as, is_file, \
    is_list, is_float, is_int
from data.utils import FeatureSplittableManager


class TabularManager(FeatureSplittableManager):
    def __init__(self):
        """
        Use 'build_from_config_dict' below, rather than using this constructor directly
        """
        self.logger = Logger.root

        self.config : Optional[TabularManagerConfig] = None
        self._data : Optional[pd.DataFrame] = None

    @staticmethod
    def build_from_config_dict(config_dict: dict, logger: Logger = Logger.root):
        # Generate the manager and its configuration file
        manager = TabularManager()

        # Set the logger for the manager
        manager.logger = logger

        # Parse the configuration dictionary's contents and track it within the manager
        config = TabularManagerConfig(config_dict, logger)
        manager.config = config

        # Return the new manager
        return manager

    """
    Pseudo-caching property here, so we can defer File I/O until its needed
    """
    @property
    def data(self):
        # Raise an error if this tabular manager has not yet been configured
        if self.config is None:
            raise AttributeError("TabularManager is not configured yet, and therefore cannot handle data")
        # If there is data in the correct format, return it as is
        if self._data is not None:
            return self._data
        # If not, try to load the data with pandas and cache it
        else:
            self._data = pd.read_csv(
                self.config.data_source,
                sep=self.config.separator
            )
            return self._data

    @data.setter
    def data(self, new_data: pd.DataFrame):
        if type(new_data) is not pd.DataFrame:
            raise TypeError(
                f"Attempted to assign object of type '{type(new_data).__name__}' to TabularManager. "
                f"TabularManager can only handle '{pd.DataFrame.__name__}' objects."
            )
        self._data = new_data

    @data.deleter
    def data(self):
        self._data = None

    """
    Abstract method overrides
    """
    def get_by_idx(self, idx):
        # If the index is an integer or slice, assume they want a single sample
        if isinstance(idx, (int, np.integer)):
            return self.data.iloc[idx, :]

        # If it's a string, assume they want a single feature
        if isinstance(idx, str):
            return self.data.loc[:, idx]

        # All further options need some processing
        # Slices should be passed to Panda's iloc indexer
        if isinstance(idx, slice):
            result = self.data.iloc[idx, :]

        # For other collections, we need to sample
        elif isinstance(idx, (np.ndarray, Collection)):
            idx_sample = idx[0]
            # If the sample is an integer, use iloc
            if isinstance(idx_sample, (int, np.integer)):
                result = self.data.iloc[idx, :]
            # Otherwise, use loc
            else:
                result = self.data.loc[idx, :]

        # If everything else failed, pass to pandas and hope for the best
        else:
            result = self.data[idx]

        # If the result is a series, we need to convert it back to a DataFrame for consistency
        if isinstance(result, pd.Series):
            result = result.to_frame()

        # Build a new TabularDataManager using the result, to ensure consistency in chained operations
        return self._build_child_manager(result, f"subset[{idx}]")

    def get_len(self):
        return self.data.shape[0]

    def array(self):
        return self.data.to_numpy()

    def process_pre_analysis(self):
        # Drop any columns requested by the user
        drop_cols = self.config.drop_columns
        new_df = self.data.drop(columns=drop_cols)

        # Drop columns which fail to pass the nullity check
        column_nullity = self.config.column_nullity
        to_drop = []
        n = new_df.shape[0]
        for c in new_df.columns:
            v = np.sum(new_df.loc[:, c].isnull())
            if v / n > column_nullity:
                to_drop.append(c)
        new_df = new_df.drop(columns=to_drop)

        # Drop rows which fail to pass the nullity check
        row_nullity = self.config.row_nullity
        to_drop = []
        n = new_df.shape[1]
        for r in new_df.index:
            v = np.sum(new_df.loc[r, :].isnull())
            if v / n > row_nullity:
                to_drop.append(r)
        new_df = new_df.drop(index=to_drop)

        return self._build_child_manager(new_df, "preprocessed")

    def train_test_split(self, train_idx, test_idx):
        # Split the data using the indices provided
        train_df = self.data.loc[train_idx, :]
        test_df = self.data.loc[test_idx, :]

        # Identify any categorical columns in the dataset
        explicit_cats = self.config.categorical_cols
        detected_cats = []
        # Identify any other categorical column/s in the dataset automatically if the user requested it
        cat_threshold = self.config.categorical_threshold
        if cat_threshold is not None:
            # Identify the categorical columns in question
            nunique_vals = train_df.nunique(axis=0, dropna=True)
            detected_cats = list(test_df.loc[:, nunique_vals <= cat_threshold].columns)
            detected_cats = [x for x in detected_cats if x not in explicit_cats]
            if len(detected_cats) > 0:
                self.logger.warning(f"Auto-detected categorical columns: {detected_cats}")
        cat_columns = [*explicit_cats, *detected_cats]

        # Process the categorical columns
        train_df, test_df = self.process_categorical(cat_columns, train_df, test_df)

        # Run pre-processing on the continuous columns
        train_df, test_df = self.process_continuous(train_df, test_df)

        # Build new DataManagers based on these subsets
        train_data = self._build_child_manager(train_df, "train_split")
        test_data = self._build_child_manager(test_df, "test_split")

        # Return the result
        return train_data, test_data

    def pop_features(self, features):
        # Grab only the features requested
        new_df = self.data.loc[:, features].to_frame()
        new_manager = self._build_child_manager(new_df, "pop_result")

        # Delete the same features from this dataset
        self.data = self.data.drop(columns=features)

        # Update config label Data Source to denote it is modified version
        self.config.data_source += "pop_remainder"

        # Return a DataManager with only the requested features in it
        return new_manager

    """
    Utility methods
    """
    @staticmethod
    def process_categorical(columns: list, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Processes the categorical columns in a pair of dataframes, via imputation and OHE
        :param columns: The categorical columns in the dataframes
        :param train_df: The training data, used for fitting our processors
        :param test_df:  The testing data, which will only have processing applied to it
        :return: The modified versions of the original dataframes post-processing
        """
        # Isolate the categorical columns from the rest of the data
        train_cat_subdf = train_df.loc[:, columns]
        test_cat_subdf = test_df.loc[:, columns]

        # Apply imputation via mode to each column
        imp = SimpleImputer(strategy='most_frequent')
        train_cat_subdf = imp.fit_transform(train_cat_subdf)
        test_cat_subdf = imp.transform(test_cat_subdf)

        # Encode the categorical columns into OneHot form
        ohe = OneHotEncoder(drop='if_binary', handle_unknown='ignore')
        train_cat_subdf = ohe.fit_transform(train_cat_subdf)
        test_cat_subdf = ohe.transform(test_cat_subdf)

        # Overwrite the original dataframes with these new features
        new_cols = ohe.get_feature_names_out(columns)
        train_df = train_df.drop(columns=columns)
        train_df[new_cols] = train_cat_subdf.toarray()
        test_df = test_df.drop(columns=columns)
        test_df[new_cols] = test_cat_subdf.toarray()

        # Output the resulting dataframes post-processing if debug is on
        return train_df, test_df

    @staticmethod
    def process_continuous(train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Processes all columns in a pair of dataframes by standardizing and imputing missing values
        :param train_df: The training data, used for fitting our processors
        :param test_df:  The testing data, which will only have processing applied to it
        :return: The modified versions of the original dataframes post-processing
        """
        # Save the columns for later
        columns = train_df.columns

        # Normalize all values in the dataset pre-imputation
        standardizer = StandardScaler()
        train_df = standardizer.fit_transform(train_df)
        test_df = standardizer.transform(test_df)

        # Impute any remaining values via KNN imputation (5 neighbors)
        knn_imp = KNNImputer(n_neighbors=5)
        train_df = knn_imp.fit_transform(train_df)
        test_df = knn_imp.transform(test_df)

        # Re-normalize all values to control for any distributions swings caused by imputation
        train_df = standardizer.fit_transform(train_df)
        test_df = standardizer.transform(test_df)

        # Format everything as a dataframe again
        train_df = pd.DataFrame(data=train_df, columns=columns)
        test_df = pd.DataFrame(data=test_df, columns=columns)

        # Output the resulting dataframes post-processing if debug is on
        return train_df, test_df

    def _build_child_manager(self, sub_df: pd.DataFrame, label: str):
        """
        Builds a new DataManager using a DataFrame which is modified version of the one currently managed
        """
        # Generate the manager to handle the subset
        new_manager = TabularManager()
        new_manager.logger = self.logger
        new_manager.data = sub_df
        new_manager.config = deepcopy(self.config)

        # Update the config's source to denote it is a child
        new_manager.config.data_source += ":" + label

        # Return the result
        return new_manager

"""
Class to handle config options unique to this class
"""
class TabularManagerConfig(object):
    def __init__(self, config_data: dict, logger: Logger = Logger.root):
        # Track the logger and data for use later
        self.logger = logger
        self.config_data = config_data

        # Parse the JSON data immediately, so we fail before running anything else
        self.data_source = self.parse_data_source()
        self.separator = self.parse_separator()
        self.drop_columns = self.parse_drop_columns()
        self.column_nullity = self.parse_column_nullity()
        self.row_nullity = self.parse_row_nullity()
        self.categorical_cols = self.parse_categorical_cols()
        self.categorical_threshold = self.parse_categorical_threshold()

        # Report any remaining values in the config file to the user
        self.report_remaining_values()

    """ Content parsers for elements in the configuration file """
    def parse_data_source(self):
        return parse_data_config_entry(
            "data_source", self.config_data, as_str(self.logger), is_file(self.logger), as_str(self.logger)
        )

    def parse_separator(self):
        default_comma = default_as(',', self.logger)
        return parse_data_config_entry(
            "separator", self.config_data, default_comma, as_str(self.logger)
        )

    def parse_drop_columns(self):
        default_empty = default_as([], self.logger)
        return parse_data_config_entry(
            "drop_columns", self.config_data, default_empty, is_list(self.logger)
        )

    def parse_column_nullity(self):
        default_nullity = default_as(0.75, self.logger)
        return parse_data_config_entry(
            "column_nullity", self.config_data, default_nullity, is_float(self.logger)
        )

    def parse_row_nullity(self):
        default_nullity = default_as(0.75, self.logger)
        return parse_data_config_entry(
            "row_nullity", self.config_data, default_nullity, is_float(self.logger)
        )

    def parse_categorical_cols(self):
        default_empty = default_as([], self.logger)
        return parse_data_config_entry(
            "categorical_cols", self.config_data, default_empty, is_list(self.logger)
        )

    def parse_categorical_threshold(self):
        default_max_int = default_as(int_maxsize)
        return parse_data_config_entry(
            "categorical_threshold", self.config_data, default_max_int, is_int(self.logger)
        )

    """Misc."""
    def report_remaining_values(self):
        if len(self.config_data) == 0:
            return
        for k in self.config_data.keys():
            self.logger.warning(
                f"Entry '{k}' in tabular configuration file is not a valid option and was ignored"
            )
