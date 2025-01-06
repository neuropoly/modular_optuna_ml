from logging import Logger
from pathlib import Path

from config.utils import default_as, parse_data_config_entry, is_int, load_json_with_validation, is_not_null, as_str, \
    as_path, is_valid_option, all_valid_options, is_list, is_bool, is_dict
from study import METRIC_FUNCTIONS


class StudyConfig(object):
    """
    Configuration manager which handles the loading and parsing of study configuration JSON files
    """
    def __init__(self, json_data: dict, logger: Logger = Logger.root):
        # Track the logger and data for use later
        self.logger = logger
        self.json_data = json_data

        # Parse the JSON data immediately, so we fail before running anything else
        self.label = self.parse_label()
        self.target = self.parse_target()
        self.random_seed = self.parse_random_seed()
        self.no_replicates = self.parse_no_replicates()
        self.no_crosses = self.parse_no_crosses()
        self.no_trials = self.parse_no_trials()
        self.objective = self.parse_objective()
        self.output_path = self.parse_output_path()

        # Special case, as this is a dict of lists that gets split based on labels
        self.train_hooks, self.validate_hooks, self.test_hooks = self.parse_metric_hooks()

    @staticmethod
    def from_json_file(json_file: Path, logger: Logger = Logger.root):
        """Creates a StudyConfig using the contents of a JSON file"""
        json_data = load_json_with_validation(json_file)

        if type(json_data) is not dict:
            raise TypeError(f"JSON should be formatted as a dictionary, was formatted as a {type(json_data)}; terminating")

        return StudyConfig(json_data, logger)

    """ Content parsers for elements in the configuration file """
    def parse_label(self):
        return parse_data_config_entry(
            "label", self.json_data, is_not_null(self.logger), as_str(self.logger)
        )

    def parse_target(self):
        return parse_data_config_entry(
            "target", self.json_data, is_not_null(self.logger), as_str(self.logger)
        )

    def parse_random_seed(self):
        default_seed = default_as(71554, self.logger)
        return parse_data_config_entry(
            "random_seed", self.json_data, default_seed, is_int(self.logger)
        )

    def parse_no_replicates(self):
        default_reps = default_as(10, self.logger)
        return parse_data_config_entry(
            "no_replicates", self.json_data, default_reps, is_int(self.logger)
        )

    def parse_no_crosses(self):
        default_crosses = default_as(10, self.logger)
        return parse_data_config_entry(
            "no_crosses", self.json_data, default_crosses, is_int(self.logger)
        )

    def parse_no_trials(self):
        default_trials = default_as(100, self.logger)
        return parse_data_config_entry(
            "no_trials", self.json_data, default_trials, is_int(self.logger)
        )

    def parse_objective(self):
        # Pull the model name from the config
        valid_objective_choice = is_valid_option(set(METRIC_FUNCTIONS.keys()), self.logger)
        return parse_data_config_entry(
            "objective", self.json_data,
            is_not_null(self.logger), as_str(self.logger), valid_objective_choice
        )

    def parse_metric_hooks(self):
        """
        Pulls a set of (up to) 3 sets of metric requests, depending on what data the models were trained and tested on.
        :return: Dictionaries of the metrics the user wants tracked at the following stages of model analysis:
            - "train": Trained on training, evaluated on training
            - "validate": Trained on training, evaluated on validation
            - "test": Trained on training + validation, evaluated on testing
        """
        # Pull the dictionary which (should) contain the requested metrics
        metric_data = parse_data_config_entry(
            "metrics", self.json_data,
            default_as(dict(), self.logger), is_dict(self.logger)
        )

        metrics_are_valid = all_valid_options(set(METRIC_FUNCTIONS.keys()), self.logger)

        # Parse each of the hook config's contents, if they were specified
        train_hooks = parse_data_config_entry(
            "train", metric_data,
            default_as(list(), self.logger), is_list(self.logger), metrics_are_valid
        )
        validate_hooks = parse_data_config_entry(
            "validate", metric_data,
            default_as(list(), self.logger), is_list(self.logger), metrics_are_valid
        )
        test_hooks = parse_data_config_entry(
            "test", metric_data,
            default_as(list(), self.logger), is_list(self.logger), metrics_are_valid
        )

        # Return the results
        return train_hooks, validate_hooks, test_hooks


    def parse_output_path(self):
        # TODO: Allow for non-filepath based storage options
        return parse_data_config_entry(
            "output_path", self.json_data, is_not_null(self.logger), as_path()
        )
