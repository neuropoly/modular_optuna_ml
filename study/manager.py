import logging

import numpy as np
import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from config.data import DataConfig
from config.model import ModelConfig
from config.study import StudyConfig
from data.utils import FeatureSplittableManager


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

        # Get the DataManager from the config
        data_manager = self.data_config.data_manager

        # Generate the requested number of splits, so each replicate will have a unique validation group
        replicate_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=self.study_config.no_replicates)
        skf_splitter = StratifiedKFold(n_splits=self.study_config.no_replicates, random_state=init_seed, shuffle=True)

        # Process the dataframe with any operations that should be done pre-split
        data_manager = data_manager.process_pre_analysis()

        # Isolate the target column(s) from the dataset
        if isinstance(data_manager, FeatureSplittableManager):
            x = data_manager
            y = data_manager.pop_features(self.study_config.target)
        else:
            raise NotImplementedError("Unsupervised analyses are not currently supported")

        # Run the study once for each replicate
        for i, (train_idx, test_idx) in enumerate(skf_splitter.split(x.array(), y.array())):
            # Set up the workspace for this replicate
            s = replicate_seeds[i]
            np.random.seed(s)
            study_name = f"{self.study_config.label} [{i}]"

            # If debugging, report the sizes
            self.logger.debug(f"Test/Train ratio (split {i}): {len(test_idx)}/{len(train_idx)}")

            # Split the data using the indices provided
            train_x, test_x = x.train_test_split(train_idx, test_idx)
            train_y, test_y = y[train_idx], y[test_idx]

            # Run a sub-study using this data
            self.run_supervised(study_name, train_x, train_y, test_x, test_y)

    def run_supervised(self, study_name: str, train_x, train_y, test_x, test_y):
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
