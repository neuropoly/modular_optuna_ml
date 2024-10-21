import json
import logging
from argparse import ArgumentParser
from doctest import debug
from inspect import isclass
from pathlib import Path
from typing import List, Callable, Any, Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models import MANAGER_MAP
from models.utils import OptunaModelManager

LOGGER = logging.getLogger(__name__)

def parse_data_config_entry(config_key: str, json_dict: dict, config_dict: dict, checks: List[Callable[[Any, Dict], Any]]):
    """
    Automatically parses a key contained within the JSON file, running any checks requested by the user in the process
    :param config_key: The key to query for within the JSON file
    :param json_dict: The JSON file's contents, in dictionary format
    :param config_dict: The to-be-updated configuration dictionary
    :param checks: A list of functions to run on the value parsed from the JSON file. Run in the order they are provided
    :return: An updated version of the 'config_dict' with the new config value
    """
    # Pull the value, returning non if needed
    config_val = json_dict.pop(config_key, None)
    # Run any and all checks requested by the user
    for fn in checks:
        config_val = fn(config_val, config_dict)
    # Assign the resulting value to its associated key in the config dict
    new_dict = config_dict.copy()
    new_dict[config_key] = config_val
    return new_dict


def parse_data_config(config_path: Path) -> dict:
    # Load the JSON with some validation
    config_data = load_json_with_validation(config_path)

    # If the JSON is not coded as a dictionary, raise an error
    if type(config_data) is not dict:
        LOGGER.error(f"JSON should be formatted as a dictionary, was formatted as a {type(config_data)}; terminating")
        raise TypeError

    # Parsed config dictionary
    parsed_config = {}

    # Get the explicit random seed, generating one at random otherwise
    config_key = "random_seed"
    def default_seed(v, _):
        if v is None:
            default = 71554
            LOGGER.warning(f"No entry for {config_key} was found, defaulting to {default}")
            return default
        return v
    def check_int(v, _):
        if type(v) is not int:
            LOGGER.error(f"'{config_key}' specified in the configuration file was not an integer; terminating")
            raise TypeError
        return v
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [default_seed, check_int])

    # Get the list of columns to explicitly drop, if any
    config_key = "drop_columns"
    def default_empty_list(v, _):
        if v is None:
            default = []
            LOGGER.warning(f"No entry for {config_key} was found, defaulting to empty list")
            return default
        return v
    def check_list(v, _):
        if type(v) is not list:
            LOGGER.error(f"'{config_key}' specified in the configuration file was not a list; terminating")
            raise TypeError
        return v
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [default_empty_list, check_list])


    # Get the threshold of nullity required to automatically drop a column
    config_key = "column_nullity"
    def default_nullity(v, _):
        if v is None:
            default = 0.75
            LOGGER.warning(f"No entry for {config_key} was found, defaulting to {default}")
            return default
        return v
    def check_float(v, _):
        if type(v) is not float:
            LOGGER.error(f"'{config_key}' specified in the configuration file was not a float; terminating")
            raise TypeError
        return v
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [default_nullity, check_float])

    # Get the threshold of nullity required to automatically drop a row (sample)
    config_key = "row_nullity"
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [default_nullity, check_float])

    # Get the number of replicates desired
    config_key = "no_replicates"
    def default_repeats(v, _):
        if v is None:
            default = 10
            LOGGER.warning(
                f"'{config_key}' was not specified in the configuration file, defaulting to {default}.")
            return default
        return v
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [default_repeats, check_int])

    # Get the number of cross-validations desired
    config_key = "no_crosses"
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [default_repeats, check_int])

    # Get the desired size of validation replicates. Defaults to 1/n, where n is the number of replicates
    config_key = "target_column"
    def value_required(v, _):
        if v is None:
            LOGGER.error(f"Config value '{config_key}' must be specified by the user. Terminating.")
            raise ValueError()
        return v
    def as_str(v, _):
        return str(v)
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [value_required, as_str])

    config_key = "categorical_cols"
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [default_empty_list, check_list])

    config_key = "categorical_threshold"
    def int_or_none(v, _):
        if v is not None and type(v) is not int:
            LOGGER.error(f"Config value '{config_key}' must be an integer or left blank. Terminating.")
            raise ValueError()
        return v
    parsed_config = parse_data_config_entry(config_key, config_data, parsed_config, [int_or_none])

    # Warn the user of any unused entries in the config file
    if len(config_data) > 0:
        for k in config_data.keys():
            LOGGER.warning(
                f"Entry '{k}' in configuration file did not match a valid configuration option and was skipped"
            )

    return parsed_config


def parse_model_config(config_path: Path) -> dict:
    # Load the JSON with some validation
    config_data = load_json_with_validation(config_path)

    # If the JSON is not coded as a list, raise an error
    if type(config_data) is not list:
        LOGGER.error(f"JSON should be formatted as a list, was formatted as a {type(config_data)}; terminating")
        raise TypeError

    # Iterate through the list to parse the results
    optuna_managers = {}
    for i, entry in enumerate(config_data):
        # Get the label, if one is given
        label = entry.pop('label', f"Unnamed [{i}]")

        # Terminate if any entry does not reference a valid model type
        manager_class = dict(entry).pop('model', None)
        manager_class = MANAGER_MAP.get(manager_class, None)
        if manager_class is None:
            raise ValueError(f"Entry '{label}' did not designate a valid model, terminating!")

        # Terminate if the manager class is not a subclass of OptunaModelManager
        if not isclass(manager_class) or not issubclass(manager_class, OptunaModelManager):
            raise ValueError(
                f"Manager class for model entry '{label}' is not a subclass of OptunaModelManager.")

        # Confirm the parameters exist
        params = entry.pop('parameters', None)
        if params is None:
            raise ValueError(f"Entry '{label}' did not specify parameters")

        # Save the results
        optuna_managers[label] = manager_class(**params)

    # Return the result
    return optuna_managers


def load_json_with_validation(json_path):
    # Check to confirm the file exists and is a valid file
    if not json_path.exists():
        LOGGER.error("JSON configuration file designated was not found; terminating")
        raise FileNotFoundError()
    if not json_path.is_file():
        LOGGER.error("JSON configuration file specified was a directory, not a file; terminating")
        raise TypeError()
    # Attempt to load the files contents w/ JSON
    with open(json_path) as json_file:
        try:
            json_data = json.load(json_file)
        except Exception as e:
            LOGGER.error("Failed to load JSON file, see error below; terminating")
            raise e
    return json_data


def process_df_pre_analysis(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Runs any pre-processing which can be done before splitting into train-test subsets
    :param df: The dataframe to process
    :param config: The configuration passed to the main file earlier.
    :return: The modified train dataframes, post-processing
    """
    # Drop any columns requested by the user
    drop_cols = config.pop('drop_columns', [])
    df = df.drop(columns=drop_cols)

    # Drop columns which fail to pass the nullity check
    column_nullity = config.pop('column_nullity', 0)
    to_drop = []
    n = df.shape[0]
    for c in df.columns:
        v = np.sum(df.loc[:, c].isnull())
        if v / n > column_nullity:
            to_drop.append(c)
    df = df.drop(columns=to_drop)

    # Drop rows which fail to pass the nullity check
    row_nullity = config.pop('row_nullity', 0)
    to_drop = []
    n = df.shape[1]
    for r in df.index:
        v = np.sum(df.loc[r, :].isnull())
        if v / n > row_nullity:
            to_drop.append(r)
    df = df.drop(index=to_drop)

    return df


def process_df_post_split(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict):
    """
    Runs any remaining pre-processing to be done, after the initial data split
    :param train_df: The training data to fit any transforms on
    :param test_df: The testing data, on which transforms will only be applied to
    :param config: The configuration passed to the main file earlier.
    :return: The modified train and test dataframes, post-processing
    """
    # Identify any categorical columns in the dataset
    explicit_cats = config.get('categorical_cols')
    detected_cats = []
    # Identify any other categorical column/s in the dataset automatically if the user requested it
    cat_threshold = config.get("categorical_threshold")
    if cat_threshold is not None:
        # Identify the categorical columns in question
        nunique_vals = train_df.nunique(axis=0, dropna=True)
        detected_cats = train_df.loc[:, nunique_vals <= cat_threshold].columns
        LOGGER.debug(f"Auto-detected categorical columns: {detected_cats}")
    cat_columns = [*explicit_cats, *detected_cats]

    # Mark the rest as continuous for later
    con_columns = list(train_df.drop(columns=cat_columns).columns)

    # Run pre-processing on the categorical columns
    train_df, test_df = process_categorical(cat_columns, train_df, test_df)

    # Run pre-processing on the continuous columns
    train_df, test_df = process_continuous(train_df, test_df)

    # Return the result
    return train_df, test_df


def process_categorical(columns: list, test_df: pd.DataFrame, train_df: pd.DataFrame):
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
    if debug:
        train_df.to_csv('debug/train_explicit_cat_processed.tsv', sep='\t')
        test_df.to_csv('debug/test_explicit_cat_processed.tsv', sep='\t')
    return train_df, test_df


def process_continuous(test_df: pd.DataFrame, train_df: pd.DataFrame):
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
    if debug:
        train_df.to_csv('debug/train_explicit_con_processed.tsv', sep='\t')
        test_df.to_csv('debug/test_explicit_con_processed.tsv', sep='\t')
    return train_df, test_df


def main(in_path: Path, out_path: Path, data_config: Path, model_config: Path):
    # Parse the data configuration file
    data_config = parse_data_config(data_config)

    # Load the model configuration file
    model_config = parse_model_config(model_config)

    # Control for RNG before proceeding
    init_seed = data_config.pop('random_seed', 71554)
    np.random.seed(init_seed)

    # Attempt to load the data from the input file
    df = pd.read_csv(in_path, sep='\t')

    # Process the dataframe with any operations that should be done pre-split
    df = process_df_pre_analysis(df, data_config)

    if debug:
        presplit_out = "debug/presplit.tsv"
        LOGGER.debug(f"Saving pre-split dataset to {presplit_out}")
        df.to_csv(presplit_out, sep='\t')

    # Generate the requested number of different train-test split workspaces
    no_replicates = data_config.pop('no_replicates')
    replicate_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=no_replicates)
    skf_splitter = StratifiedKFold(n_splits=no_replicates, random_state=init_seed, shuffle=True)

    # Run the analysis n times with the specified replicates
    target_column = data_config.pop('target_column')
    x = df.drop(columns=[target_column])
    y = df.loc[:, target_column]
    for i, (train_idx, test_idx) in enumerate(skf_splitter.split(df, y)):
        # Set up the workspace for this replicate
        s = replicate_seeds[i]
        np.random.seed(s)
        train_x = x.loc[train_idx, :]
        test_x = x.loc[test_idx, :]
        train_y = y.loc[train_idx]
        test_y = y.loc[test_idx]

        # If debugging, report the sizes
        LOGGER.debug(f"Test/Train ration (split {i}): {len(test_idx)}/{len(train_idx)}")

        # Do post-split processing
        train_x, test_x = process_df_post_split(train_x, test_x, data_config)

        # Run an ML study using this data
        for label, manager in model_config.items():
            pass


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser(
        prog="Classic ML GridSearch",
        description="Runs a gridsearch of all potential parameters using a given datasets and model type"
    )

    parser.add_argument(
        '-i', '--in_path', required=True, type=Path,
        help="The dataset to use for the analysis"
    )
    parser.add_argument(
        '-o', '--out_path', default='out.tsv', type=Path,
        help="Where the results of the analysis should be stored (in tsv format)"
    )
    parser.add_argument(
        '-d', '--data_config', default='data_config.json', type=Path,
        help="Data processing configuration file in JSON format"
    )
    parser.add_argument(
        '-m', '--model_config', default='model_config.json', type=Path,
        help="Machine Learning Model configuration file in JSON format"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Whether to show debug statements'
    )

    argvs = parser.parse_args().__dict__

    # Generate a logger for this program
    logging.basicConfig(
        format="{asctime} {levelname}:  \t{message}",
        style='{',
        datefmt="%H:%M:%S"
    )
    debug = argvs.pop('debug', False)
    if debug:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    # Run the analysis
    main(**argvs)