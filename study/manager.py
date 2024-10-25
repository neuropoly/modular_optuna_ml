import logging

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


class StudyManager(object):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, study_config: StudyConfig, debug: bool):
        # Track each of the configs for this analysis
        self.data_config = data_config
        self.model_config = model_config
        self.study_config = study_config

        # Track whether to run the model in debug mode or not
        self.debug = debug
        self.logger = self.init_logger()

    def init_logger(self):
        """Generates the logger for this study"""
        logger = logging.getLogger(self.study_config.label)

        # Define how messages for this logger are formatted
        msg_handler = logging.StreamHandler()
        msg_formatter = logging.Formatter(
            fmt=f"[{self.study_config.label}" + " {asctime} {levelname}] {message}",
            datefmt="%H:%M:%S",
            style='{'
        )
        msg_handler.setFormatter(msg_formatter)
        logger.addHandler(msg_handler)

        # Set the logger to debug mode if requested
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Return the result
        return logger

    def run(self):
        # Control for RNG before proceeding
        init_seed = self.study_config.random_seed
        np.random.seed(init_seed)

        # Attempt to load the data from the file
        # TODO: Move this to a Data Manager of some sort
        df = pd.read_csv(self.data_config.data_source, sep='\t')

        # Process the dataframe with any operations that should be done pre-split
        df = self.process_df_pre_analysis(df)

        # Split the data into features and target
        target_column = self.data_config.target_column
        x = df.drop(columns=[target_column])
        y = df.loc[:, target_column]

        # Generate the requested number of splits, so each replicate will have a unique validation group
        replicate_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=self.study_config.no_replicates)
        skf_splitter = StratifiedKFold(n_splits=self.study_config.no_replicates, random_state=init_seed, shuffle=True)

        # Run the study once for each replicate
        for i, (train_idx, test_idx) in enumerate(skf_splitter.split(x, y)):
            # Set up the workspace for this replicate
            s = replicate_seeds[i]
            np.random.seed(s)
            study_name = f"{self.study_config.label} [{i}]"

            # Split the data using the indices provided
            train_x = x.loc[train_idx, :]
            test_x = x.loc[test_idx, :]
            train_y = y.loc[train_idx]
            test_y = y.loc[test_idx]

            # If debugging, report the sizes
            self.logger.debug(f"Test/Train ratio (split {i}): {len(test_idx)}/{len(train_idx)}")

            # Do post-split processing
            train_x, test_x = self.process_df_post_split(train_x, test_x)

            # Run a sub-study using this data
            self.run_study(study_name, train_x, train_y, test_x, test_y)


    def process_df_pre_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs any pre-processing which can be done before splitting into train-test subsets
        :param df: The dataframe to process
        :return: The modified train dataframes, post-processing
        """
        # Drop any columns requested by the user
        drop_cols = self.data_config.drop_columns
        new_df = df.drop(columns=drop_cols)

        # Drop columns which fail to pass the nullity check
        column_nullity = self.data_config.column_nullity
        to_drop = []
        n = new_df.shape[0]
        for c in new_df.columns:
            v = np.sum(new_df.loc[:, c].isnull())
            if v / n > column_nullity:
                to_drop.append(c)
        new_df = new_df.drop(columns=to_drop)

        # Drop rows which fail to pass the nullity check
        row_nullity = self.data_config.row_nullity
        to_drop = []
        n = new_df.shape[1]
        for r in new_df.index:
            v = np.sum(new_df.loc[r, :].isnull())
            if v / n > row_nullity:
                to_drop.append(r)
        new_df = new_df.drop(index=to_drop)

        return new_df

    def process_df_post_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Runs any remaining pre-processing to be done, after the initial data split
        :param train_df: The training data to fit any transforms on
        :param test_df: The testing data, on which transforms will only be applied to
        :return: The modified train and test dataframes, post-processing
        """
        # Identify any categorical columns in the dataset
        explicit_cats = self.data_config.categorical_cols
        detected_cats = []
        # Identify any other categorical column/s in the dataset automatically if the user requested it
        cat_threshold = self.data_config.categorical_threshold
        if cat_threshold is not None:
            # Identify the categorical columns in question
            nunique_vals = train_df.nunique(axis=0, dropna=True)
            detected_cats = train_df.loc[:, nunique_vals <= cat_threshold].columns
            self.logger.debug(f"Auto-detected categorical columns: {detected_cats}")
        cat_columns = [*explicit_cats, *detected_cats]

        # Mark the rest as continuous for later
        con_columns = list(train_df.drop(columns=cat_columns).columns)

        # Run pre-processing on the categorical columns
        train_df, test_df = self.process_categorical(cat_columns, train_df, test_df)

        # Run pre-processing on the continuous columns
        train_df, test_df = self.process_continuous(train_df, test_df)

        # Return the result
        return train_df, test_df

    def process_categorical(self, columns: list, train_df: pd.DataFrame, test_df: pd.DataFrame):
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
        if self.debug:
            train_df.to_csv('debug/train_explicit_cat_processed.tsv', sep='\t')
            test_df.to_csv('debug/test_explicit_cat_processed.tsv', sep='\t')
        return train_df, test_df

    def process_continuous(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
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
        if self.debug:
            train_df.to_csv('debug/train_explicit_con_processed.tsv', sep='\t')
            test_df.to_csv('debug/test_explicit_con_processed.tsv', sep='\t')
        return train_df, test_df

    def run_study(self, study_name: str, train_x, train_y, test_x, test_y):
        # Run the model specified by the model config on the data
        model_factory = self.model_config.model_factory

        # Define the function which will utilize the trial's parameters to generate models to-be-tested
        def opt_func(trial: optuna.Trial):
            model = model_factory.build_model(trial)
            model.fit(train_x, train_y)
            prob_y = model.predict_proba(test_x)
            return log_loss(test_y, prob_y)

        # Run the study with these parameters
        study = optuna.create_study(study_name=study_name)
        study.optimize(opt_func, n_trials=1)
