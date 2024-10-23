import logging
from argparse import ArgumentParser
from doctest import debug
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.data import DataConfig
from config.model import ModelConfig
from config.study import StudyConfig

LOGGER = logging.getLogger(__name__)


def process_df_pre_analysis(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """
    Runs any pre-processing which can be done before splitting into train-test subsets
    :param df: The dataframe to process
    :param config: The configuration passed to the main file earlier.
    :return: The modified train dataframes, post-processing
    """
    # Drop any columns requested by the user
    drop_cols = config.drop_columns
    df = df.drop(columns=drop_cols)

    # Drop columns which fail to pass the nullity check
    column_nullity = config.column_nullity
    to_drop = []
    n = df.shape[0]
    for c in df.columns:
        v = np.sum(df.loc[:, c].isnull())
        if v / n > column_nullity:
            to_drop.append(c)
    df = df.drop(columns=to_drop)

    # Drop rows which fail to pass the nullity check
    row_nullity = config.row_nullity
    to_drop = []
    n = df.shape[1]
    for r in df.index:
        v = np.sum(df.loc[r, :].isnull())
        if v / n > row_nullity:
            to_drop.append(r)
    df = df.drop(index=to_drop)

    return df


def process_df_post_split(train_df: pd.DataFrame, test_df: pd.DataFrame, config: DataConfig):
    """
    Runs any remaining pre-processing to be done, after the initial data split
    :param train_df: The training data to fit any transforms on
    :param test_df: The testing data, on which transforms will only be applied to
    :param config: The configuration passed to the main file earlier.
    :return: The modified train and test dataframes, post-processing
    """
    # Identify any categorical columns in the dataset
    explicit_cats = config.categorical_cols
    detected_cats = []
    # Identify any other categorical column/s in the dataset automatically if the user requested it
    cat_threshold = config.categorical_threshold
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


def main(in_path: Path, out_path: Path, data_config: Path, model_config: Path, study_config: Path):
    # Parse the configuration files
    data_config = DataConfig.from_json_file(data_config)
    model_config = ModelConfig.from_json_file(model_config)
    study_config = StudyConfig.from_json_file(study_config)

    # Control for RNG before proceeding
    init_seed = study_config.random_seed
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
    no_replicates = study_config.no_replicates
    replicate_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=no_replicates)
    skf_splitter = StratifiedKFold(n_splits=no_replicates, random_state=init_seed, shuffle=True)

    # Run the analysis n times with the specified replicates
    target_column = data_config.target_column
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

        for label, sub_config in model_config.sub_configs.items():
            model_factory = sub_config.model_factory
            # TODO: Move this to a proper configurable manager
            def opt_func(trial: optuna.Trial):
                model = model_factory.build_model(trial)
                model.fit(train_x, train_y)
                prob_y = model.predict_proba(test_x)
                return log_loss(test_y, prob_y)
            study = optuna.create_study()
            study.optimize(opt_func, n_trials=10)


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
        help="Data Processing configuration file in JSON format"
    )
    parser.add_argument(
        '-m', '--model_config', default='model_config.json', type=Path,
        help="Machine Learning Model configuration file in JSON format"
    )
    parser.add_argument(
        '-s', '--study_config', default='study_config.json', type=Path,
        help="Machine Learning Study configuration file in JSON format"
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