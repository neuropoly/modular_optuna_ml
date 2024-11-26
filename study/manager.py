import logging
import sqlite3
from copy import copy as shallow_copy
from itertools import chain
from types import NoneType
from typing import Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

from config.data import DataConfig
from config.model import ModelConfig
from config.study import StudyConfig
from data import BaseDataManager
from data.mixins import MultiFeatureMixin
from study import METRIC_FUNCTIONS, MetricUpdater

UNIVERSAL_DB_ENTRIES = {
    "replicate": "INTEGER",
    "trial": "INTEGER",
    "objective": "REAL"
}


class StudyManager(object):
    """
    Manages an Optuna study, handling and co-ordinating it with the dataset requested to be parsed,
    and the model requested to be optimized by said study

    Always runs as though we are testing in a multi-replicate, cross-validated way; you can disable this by proxy
    by setting the `n_replicates` and `n_crosses` to be 1.
    """
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, study_config: StudyConfig,
                 overwrite: bool, debug: bool):
        # Track each of the configs for this analysis
        self.data_config = data_config
        self.model_config = model_config
        self.study_config = study_config

        # Track whether to run the model in overwrite and/or debug mode
        self.overwrite = overwrite
        self.debug = debug

        # Initiate the logger for this study
        self.logger = self.init_logger()

        # Generate a unique label for the combination of configs in this analysis
        self.study_label = f"{self.study_config.label}__{self.model_config.label}__{self.data_config.label}"

        # Pull the objective function for this study
        self.objective_func: MetricUpdater = METRIC_FUNCTIONS.get(self.study_config.objective)

        # Track each of the metric calculating functions which needs to be hook in to run throughout model assessment
        self.train_hooks = {f"{k} (train)": METRIC_FUNCTIONS[k] for k in self.study_config.train_hooks}
        self.validate_hooks = {f"{k} (validate)": METRIC_FUNCTIONS[k] for k in self.study_config.validate_hooks}
        self.test_hooks = {f"{k} (test)": METRIC_FUNCTIONS[k] for k in self.study_config.test_hooks}

        # Generate some null DB-related attributes to be filled during DB initialization
        self.db_cols: Optional[dict[str: str]] = None
        self.db_connection : Optional[sqlite3.Connection] = None
        self.db_cursor : Optional[sqlite3.Cursor] = None

    """ Utils """
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

    def train_hook_keys(self):
        """
        As there are (usually) multiple crosses per validation step, train hooks are run multiple times as well.
        This gives each cross a unique DB column to save within
        """
        train_cols = []
        for k, v in self.train_hooks.items():
            new_cols = [f"{k} [{i}]" for i in range(self.study_config.no_crosses)]
            train_cols.extend(new_cols)
        return train_cols

    """ DB Management """
    def init_db(self):
        # Create the requested database file if it does not already exist
        if not self.study_config.output_path.exists():
            self.study_config.output_path.touch()

        # Initiate a connection w/ a 'with' clause, to ensure the connection closes when the program does
        with sqlite3.connect(self.study_config.output_path) as con:
            # Initiate the cursor
            cur = con.cursor()

            # If we're enabling overwrites, delete any table with the same name before proceeding
            if self.overwrite:
                # Check if a table with the current study name exists
                current_tables = cur.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.study_label}'"
                ).fetchall()
                table_exists = len(current_tables) > 0
                # If it does, warn the user and reset it
                if table_exists:
                    self.logger.warning(f"DB table for '{self.study_label}' already existed and was overwritten")
                    cur.execute(
                        f"DROP TABLE IF EXISTS {self.study_label};"
                    )

            # Initiate list for maintaining order in our saved metrics, so they can be saved to the database easily
            self.db_cols = {k: v for k, v in UNIVERSAL_DB_ENTRIES.items()}

            # Track any tunable pre-processing parameters
            for p in self.data_config.data_manager.tunable_params():
                self.db_cols[p.label] = p.db_type

            # Track any tune-able model parameter as well
            for p in self.model_config.model_manager.tunable_params():
                self.db_cols[p.label] = p.db_type

            # Track our data hooks as well, in train->validate->test order
            for k in chain(self.train_hook_keys(), self.validate_hooks.keys(), self.test_hooks.keys()):
                # Metrics are always floating point numbers
                self.db_cols[k] = "REAL"
                # TODO: accomodate for non-floating types

            # Generate the column entries needed to define the SQL query
            sql_components = [f"'{label}' {db_type}" for label, db_type in self.db_cols.items()]

            # Create the table for this study
            try:
                cur.execute(
                    # Table should share its name with the study
                    f"CREATE TABLE {self.study_label} "
                    # Should contain all columns desired by the user
                    f"({', '.join(sql_components)})"
                )
            except sqlite3.OperationalError as err:
                if "already exists" in err.args[0]:
                    err.args = (f"The DB table for study '{self.study_label}' already exists; use the '--overwrite' flag if you want to overwrite it",)
                raise err

            # Return the result
            return con, cur

    def save_results(self, replicate_n, trial: optuna.Trial, objective_val, metrics):
        # Generate the list of values to be saved to the DB
        new_entry_components = shallow_copy(metrics)

        # Extend the dict with our universal metrics
        new_entry_components.update({
            "replicate": replicate_n,
            "trial": trial.number,
            "objective": objective_val
        })

        # Extend further with our trial-tuned parameter values
        tunable_param = [*self.data_config.data_manager.tunable_params(), *self.model_config.model_manager.tunable_params()]
        for p in tunable_param:
            p_label = p.label
            v = trial.params.get(p_label, None)
            # If the trial didn't have a value for the parameter, set it to null
            if v is None:
                new_entry_components[p_label] = "NULL"
            # For everything else, just leave it be
            else:
                new_entry_components[p_label] = v

        # Re-order the values so they can cleanly save into the dataset
        ordered_values = [new_entry_components[k] for k in self.db_cols]

        # Format them as valid strings, so that SQL doesn't have a fit
        ordered_values = [str(v) if isinstance(v, int | float | NoneType) else f"'{v}'" for v in ordered_values]
        new_entry = ", ".join(ordered_values)

        # Push the results to the db
        self.db_cursor.execute(f"INSERT INTO {self.study_label} VALUES ({new_entry})")
        self.db_connection.commit()

    """ Management """
    def run(self):
        init_seed, replicate_seeds, x, y = self.prepare_run()

        # Run the study once for each replicate
        skf_splitter = StratifiedKFold(n_splits=self.study_config.no_replicates, random_state=init_seed, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(skf_splitter.split(x.as_array(), y.as_array())):
            # Set up the workspace for this replicate
            s = int(replicate_seeds[i])
            np.random.seed(s)

            # If debugging, report the sizes
            self.logger.debug(f"Test/Train ratio (split {i}): {len(test_idx)}/{len(train_idx)}")

            # Run a sub-study using this data
            self.run_replicate(i, train_idx, test_idx, x, y, s)

    def prepare_run(self):
        # Control for RNG before proceeding
        init_seed = self.study_config.random_seed
        np.random.seed(init_seed)
        # Get the DataManager from the config
        data_manager = self.data_config.data_manager
        # Generate the requested number of splits, so each replicate will have a unique validation group
        replicate_seeds = np.random.randint(0, np.iinfo(np.int32).max, size=self.study_config.no_replicates)

        # Isolate the target column(s) from the dataset
        if self.study_config.target is not None:
            if not isinstance(data_manager, MultiFeatureMixin):
                raise TypeError("Tried to target a feature in a dataset which only has a single feature!")
            x = data_manager.get_features([c for c in data_manager.features() if c != self.study_config.target])
            y = data_manager.get_features(self.study_config.target)
            # Why is PyCharm's type hinting so dogshit?
            x: BaseDataManager | MultiFeatureMixin
            y: BaseDataManager | MultiFeatureMixin
        else:
            raise NotImplementedError("Unsupervised analyses are not currently supported")

        # Initiate the DB and create a table within it for the study's results
        self.db_connection, self.db_cursor = self.init_db()
        return init_seed, replicate_seeds, x, y

    @staticmethod
    def train_test_split(test_idx, train_idx, x, y):
        # Naive split is required for target values to avoid running post-split processing
        train_y, test_y = y[train_idx], y[test_idx]

        # Split the data using the indices provided
        train_x, test_x = x.split(train_idx, test_idx, train_y, test_y, is_cross=False)

        return test_x, test_y, train_x, train_y

    def run_replicate(
            self,
            rep: int,
            train_idx,
            test_idx,
            x,
            y,
            seed: int
    ):
        # Generate the name for this run
        study_name = f"{self.study_label} [{rep}]"

        # Grab the model manager specified by the model config
        model_manager = self.model_config.model_manager

        # Define the function which will utilize a trial's parameters to generate models to-be-tested
        def opt_func(trial: optuna.Trial):
            # Initiate a dictionary to track all metrics requested to be recorded by the user
            metric_dict = dict()

            # Tune the data and model managers
            model_manager.tune(trial)
            x.tune(trial)

            # Run any pre-split pre-processing
            prepped_x = x.pre_split(target=y, is_cross=False)

            # Split the data into the composite components
            test_x, test_y, train_x, train_y = self.train_test_split(test_idx, train_idx, prepped_x, y)

            # Run a subset analysis on the training data, split once for each cross requested
            objective_value = self.run_cv_trial(train_x, train_y, trial, metric_dict)

            # Generate and fit a model to the full dataset with the current replicate
            train_y_flat = np.ravel(train_y.as_array()) # Ravel prevents a warning log
            model_manager.fit(train_x.as_array(), train_y_flat)

            # Calculate and record any validation metrics
            for k, metric_func in self.validate_hooks.items():
                metric_dict[k] = metric_func(model_manager, train_x, train_y)

            # Calculate any metrics requested by the user, including the objective function
            for k, metric_func in self.test_hooks.items():
                metric_dict[k] = metric_func(model_manager, test_x, test_y)

            # Save the metric values to the DB
            self.save_results(rep, trial, objective_value, metric_dict)

            # Return the objective function so Optuna can run optimization based on it
            return objective_value

        # Run the study with these parameters
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler
        )
        study.optimize(opt_func, n_trials=self.study_config.no_trials)

    def run_cv_trial(self, x, y, trial, metric_dict):
        # Grab the model manager for the model we want to test
        model_manager = self.model_config.model_manager

        # Run any pre-split preparations the dataset has again
        prepped_x = x.pre_split(target=y, is_cross=True)

        # Track the objective values for each cross
        objective_cross_values = np.zeros(self.study_config.no_crosses)

        cross_splitter = StratifiedKFold(
            n_splits=self.study_config.no_crosses, random_state=self.study_config.random_seed, shuffle=True
        )
        for i, (ti, vi) in enumerate(cross_splitter.split(x.as_array(), y.as_array())):
            # Tune the model based on the trial's parameters
            model_manager.tune(trial)

            # Split the components along the desired axes
            ty, vy = y[ti], y[vi]
            tx, vx = prepped_x.split(ti, vi, ty, vy, is_cross=True)

            # Generate and fit a new instance of the model to the training subset
            rty = np.ravel(ty.as_array())
            model_manager.fit(tx.as_array(), rty)  # 'ravel' saves us a warning log

            # Calculate the objective metric for this function and store it
            objective_cross_values[i] = self.objective_func(model_manager, vx, vy)

            # Calculate the metrics requested by the user at the "train" hook
            for k, v in self.train_hooks.items():
                metric_dict[f"{k} [{i}]"] = v(model_manager, tx, ty)

        return np.mean(objective_cross_values)