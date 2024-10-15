import json
import logging
from argparse import ArgumentParser
from email.policy import default
from pathlib import Path
from typing import Tuple, List, Callable, Any, Dict, NoReturn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_config_entry(config_key: str, json_dict: dict, config_dict: dict, checks: List[Callable[[Any, Dict], Any]]):
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


def parse_config(config_path: Path) -> dict:
    # Check to confirm the file exists and is a valid file
    if not config_path.exists():
        logging.error("JSON configuration file designated was not found; terminating")
        raise FileNotFoundError()
    if not config_path.is_file():
        logging.error("JSON configuration file specified was a directory, not a file; terminating")
        raise TypeError()

    # Attempt to load the files contents w/ JSON
    with open(config_path) as json_file:
        try:
            config_data = json.load(json_file)
        except Exception as e:
            logging.error("Failed to load JSON file, see error below; terminating")
            raise e

    # If the JSON is not coded as a dictionary, raise an error
    if type(config_data) is not dict:
        logging.error(f"JSON should be formatted as a dictionary, was formatted as a {type(config_data)}; terminating")
        raise TypeError

    # Parsed config dictionary
    parsed_config = {}

    # Get the explicit random seed, generating one at random otherwise
    config_key = "random_seed"
    def default_seed(v, _):
        if v is None:
            default = 71554
            logging.warning(f"No entry for {config_key} was found, defaulting to {default}")
            return default
        return v
    def check_int(v, _):
        if type(v) is not int:
            logging.error(f"'{config_key}' specified in the configuration file was not an integer; terminating")
            raise TypeError
        return v
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_seed, check_int])

    # Get the list of columns to explicitly drop, if any
    config_key = "drop_columns"
    def default_empty_list(v, _):
        if v is None:
            default = []
            logging.warning(f"No entry for {config_key} was found, defaulting to empty list")
            return default
        return v
    def check_list(v, _):
        if type(v) is not list:
            logging.error(f"'{config_key}' specified in the configuration file was not a list; terminating")
            raise TypeError
        return v
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_empty_list, check_list])


    # Get the threshold of nullity required to automatically drop a column
    config_key = "column_nullity"
    def default_nullity(v, _):
        if v is None:
            default = 0.75
            logging.warning(f"No entry for {config_key} was found, defaulting to {default}")
            return default
        return v
    def check_float(v, _):
        if type(v) is not float:
            logging.error(f"'{config_key}' specified in the configuration file was not a float; terminating")
            raise TypeError
        return v
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_nullity, check_float])

    # Get the threshold of nullity required to automatically drop a row (sample)
    config_key = "row_nullity"
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_nullity, check_float])

    # Get the number of replicates desired
    config_key = "no_replicates"
    def default_repeats(v, _):
        if v is None:
            default = 10
            logging.warning(
                f"'{config_key}' was not specified in the configuration file, defaulting to {default}.")
            return default
        return v
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_repeats, check_int])

    # Get the number of cross-validations desired
    config_key = "no_crosses"
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_repeats, check_int])

    # Get the desired size of test replicates. Defaults to 1/n, where n is the number of replicates
    config_key = "test_size"
    def default_rep_size(v, c):
        if v is None:
            no_replicates = c['no_replicates']
            default = 1 / no_replicates
            logging.warning(
                f"'{config_key}' was not specified in the configuration file, defaulting to 1/{no_replicates}.")
            return default
        return v
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_rep_size, check_float])

    # Get the desired size of validation replicates. Defaults to 1/n, where n is the number of replicates
    config_key = "validation_size"
    def default_cross_size(v, c):
        if v is None:
            no_crosses = c['no_crosses']
            default = 1 / no_crosses
            logging.warning(
                f"'{config_key}' was not specified in the configuration file, defaulting to 1/{no_crosses}.")
            return default
        return v
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [default_cross_size, check_float])

    # Get the desired size of validation replicates. Defaults to 1/n, where n is the number of replicates
    config_key = "target_column"
    def value_required(v, _):
        if v is None:
            logging.error(f"Config value '{config_key}' must be specified by the user. Terminating.")
            raise ValueError()
        return v
    def as_str(v, _):
        return str(v)
    parsed_config = parse_config_entry(config_key, config_data, parsed_config, [value_required, as_str])

    # Warn the user of any unused entries in the config file
    if len(config_data) > 0:
        for k in config_data.keys():
            logging.warning(
                f"Entry '{k}' in configuration file did not match a valid configuration option and was skipped"
            )

    return parsed_config


def process_df_pre_analysis(df: pd.DataFrame, config: dict) -> pd.DataFrame:
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


def main(in_path: Path, out_path: Path, config: Path, debug: bool):
    # Parse the configuration file
    config = parse_config(config)

    # Control for RNG before proceeding
    init_seed = config.pop('random_seed', 71554)
    np.random.seed(init_seed)

    # Attempt to load the data from the input file
    df = pd.read_csv(in_path, sep='\t')

    # Process the dataframe with any operations that should be done pre-split
    df = process_df_pre_analysis(df, config)

    if debug:
        presplit_out = "presplit.tsv"
        logging.debug(f"Saving pre-split dataset to {presplit_out}")
        df.to_csv(presplit_out, sep='\t')

    # Generate 10 different train-test splits
    no_replicates = config.pop('no_replicates', 10)
    replicate_seeds = np.random.randint(
        0, np.iinfo(np.int32).max, size=no_replicates
    )
    replicate_idxs_train_test = []
    test_size = config.pop("test_size")
    for i in range(no_replicates):
        train_idx, test_idx = train_test_split(df.index, test_size=test_size)
        replicate_idxs_train_test.append((train_idx, test_idx))

    logging.debug(f"Training dataset sizes: {[len(x[0]) for x in replicate_idxs_train_test]}")
    logging.debug(f"Testing dataset sizes: {[len(x[1]) for x in replicate_idxs_train_test]}")

    # Run the analysis n time with the specified replicates
    for i, s in enumerate(replicate_seeds):
        # Set up the workspace for this replicate
        np.random.seed(s)
        train_idx, test_idx = replicate_idxs_train_test[i]
        train_df = df.loc[train_idx, :]
        test_df = df.loc[test_idx, :]


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
        '-c', '--config', default='config.json', type=Path,
        help="Configuration file in JSON format"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Whether to show debug statements'
    )

    argvs = parser.parse_args().__dict__

    # Enable debug logging if requested
    debug = argvs.get('debug', False)
    if debug:
        logging.root.setLevel(logging.DEBUG)

    # Run the analysis
    main(**argvs)