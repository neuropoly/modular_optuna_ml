import logging
import sqlite3
from typing import Optional, TypeVar

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold

from config.data import DataConfig
from config.model import ModelConfig
from config.study import StudyConfig
from data.utils import FeatureSplittableManager
from models.utils import OptunaModelManager
from study import METRIC_FUNCTIONS, MetricUpdater

UNIVERSAL_DB_KEYS = [
    'replicate',
    'trial',
    'objective'
]


class StudyManager(object):
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
        self.objective_func = METRIC_FUNCTIONS.get(self.study_config.objective)

        # Track the list of other metrics to measure and track
        self.metric_funcs: dict[str, MetricUpdater] = {k: METRIC_FUNCTIONS[k] for k in self.study_config.metrics}
        self.metric_order = self.define_metric_order()

        # Generate some null attributes to be filled later
        self.db_connection : Optional[sqlite3.Connection] = None
        self.db_cursor : Optional[sqlite3.Cursor] = None

    """ Logger Management """
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

    """ DB Management """
    def define_metric_order(self):
        # Initiate with our basic metrics
        metric_order = [*UNIVERSAL_DB_KEYS]

        # Append the metrics requested by the user
        metric_order.extend(self.metric_funcs.keys())

        # If the config requested it, extend with the list of model parameters
        if self.study_config.track_params:
            metric_order.extend(self.model_config.parameters.keys())

        # Return the result
        return metric_order

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

            # Generate a list of all the columns to place in the table
            col_vals = [*UNIVERSAL_DB_KEYS, *self.metric_funcs.keys()]

            # If we're tracking the model parameters as well, add them to the DB columns as well
            if self.study_config.track_params:
                for k, v in self.model_config.parameters.items():
                    # Everything that is not a dictionary defining how it's should be samples must be text-like
                    if not isinstance(v, dict):
                        col_vals.append(f"{k} TEXT")
                    # If it is a dict, pull the type and check against it instead
                    elif v.get("type") == "float":
                        col_vals.append(f"{k} REAL")
                    elif v.get("type") == "int":
                        col_vals.append(f"{k} INTEGER")
                    else:
                        col_vals.append(f"{k} TEXT")

            # Create the table for this study
            try:
                cur.execute(
                    # Table should share its name with the study
                    f"CREATE TABLE {self.study_label} "
                    # Should contain all columns desired by the user
                    f"({', '.join(col_vals)})"
                )
            except sqlite3.OperationalError as err:
                if "already exists" in err.args[0]:
                    err.args = (f"The DB table for study '{self.study_label}' already exists; use the '--overwrite' flag if you want to overwrite it",)
                raise err

            # Return the result
            return con, cur

    def save_results(self, replicate_n, trial, objective_val, metrics):
        # Generate the list of values to be saved to the DB
        new_entry_components = metrics.copy()
        new_entry_components.update({
            "replicate": replicate_n,
            "trial": trial.number,
            "objective": objective_val
        })

        # If the user wants model parameters saved, add them too
        if self.study_config.track_params:
            for k, v in self.model_config.parameters.items():
                tv = trial.params.get(k, None)
                # If the trial didn't have a value for the parameter, set it to null
                if tv is None:
                    new_entry_components[k] = "NULL"
                # If the associated value is non-numeric, format it to play nicely with SQLite's queries
                elif not isinstance(v, dict):
                    new_entry_components[k] = f"'{tv}'"
                # For everything else, just leave it be
                else:
                    new_entry_components[k] = tv

        # Format them into clean strings so they play nice with the DB query formatting
        ordered_values = [str(new_entry_components[k]) for k in self.metric_order]
        new_entry = ", ".join(ordered_values)

        # Push the results to the db
        self.db_cursor.execute(f"INSERT INTO {self.study_label} VALUES ({new_entry})")
        self.db_connection.commit()

    """ ML Management """
    T = TypeVar('T')
    def calculate_metrics(self, manager: OptunaModelManager[T], model: T, context: dict):
        # Instantiate the set of values to be saved to the DB
        metric_dict = {}

        # Calculate the rest
        for k, metric_func in self.metric_funcs.items():
            metric_dict[k] = metric_func(manager, model, context)

        # Return the objective function's value for re-use
        return metric_dict

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

        # Initiate the DB and create a table within it for the study's results
        self.db_connection, self.db_cursor = self.init_db()

        # Run the study once for each replicate
        for i, (train_idx, test_idx) in enumerate(skf_splitter.split(x.array(), y.array())):
            # Set up the workspace for this replicate
            s = replicate_seeds[i]
            np.random.seed(s)

            # If debugging, report the sizes
            self.logger.debug(f"Test/Train ratio (split {i}): {len(test_idx)}/{len(train_idx)}")

            # Split the data using the indices provided
            train_x, test_x = x.train_test_split(train_idx, test_idx)
            train_y, test_y = y[train_idx], y[test_idx]

            # Run a sub-study using this data
            self.run_supervised(i, train_x, train_y, test_x, test_y, s)

    def run_supervised(self, rep: int, train_x, train_y, test_x, test_y, seed):
        # Generate the study name for this run
        study_name = f"{self.study_label} [{rep}]"

        # Run the model specified by the model config on the data
        model_manager = self.model_config.model_manager

        # Define the function which will utilize a trial's parameters to generate models to-be-tested
        def opt_func(trial: optuna.Trial):
            # Run a subset analysis on the training data, split once for each cross requested
            cross_splitter = StratifiedKFold(n_splits=self.study_config.no_crosses, random_state=seed, shuffle=True)
            objective_cross_values = np.zeros(self.study_config.no_crosses)
            for i, (ti, vi) in enumerate(cross_splitter.split(train_x.array(), train_y.array())):
                # Split the components along the desired axes
                tx, vx = train_x[ti], train_x[vi]
                ty, vy = train_y[ti], train_y[vi]

                # Generate and fit a new instance of the model to the training subset
                model = model_manager.build_model(trial)
                model.fit(tx, np.ravel(ty))  # 'ravel' saves us a warning log

                # Calculate the objective metric for this function and store it
                context = {
                    "train_x": tx,
                    "train_y": ty,
                    "test_x": vx,
                    "test_y": vy
                }
                objective_cross_values[i] = self.objective_func(model_manager, model, context)

            # Generate and fit the model to the full training set
            model = model_manager.build_model(trial)
            model.fit(train_x, np.ravel(train_y))  # Ravel prevents a warning log

            # Calculate the objective function's value on the test set as well
            objective_value = np.mean(objective_cross_values)

            # Calculate any metrics requested by the user, including the objective function
            context = {
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y
            }
            metric_vals = self.calculate_metrics(model_manager, model, context)
            # noinspection PyTypeChecker
            metric_vals['objective'] = objective_value

            # Save the metric values to the DB
            self.save_results(rep, trial, objective_value, metric_vals)

            # Return the objective function so Optuna can run optimization based on it
            return objective_value

        # Run the study with these parameters
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler
        )
        study.optimize(opt_func, n_trials=self.study_config.no_trials)
