from logging import Logger
from pathlib import Path

from config.utils import default_as, parse_data_config_entry, is_int, load_json_with_validation, is_not_null, as_str, \
    as_path, is_valid_option, all_valid_options, is_list, is_bool
from study import METRIC_FUNCTIONS


class StudyConfig(object):
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
        self.metrics = self.parse_metrics()
        self.output_path = self.parse_output_path()
        self.track_params = self.parse_track_params()


    @staticmethod
    def from_json_file(json_file: Path, logger: Logger = Logger.root):
        """Creates a StudyConfig using the contents of a JSON file"""
        json_data = load_json_with_validation(json_file)

        if type(json_data) is not dict:
            logger.error(
                f"JSON should be formatted as a dictionary, was formatted as a {type(json_data)}; terminating"
            )
            raise TypeError

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

    def parse_metrics(self):
        # Pull the model name from the config
        valid_metric_choice = all_valid_options(set(METRIC_FUNCTIONS.keys()), self.logger)
        return parse_data_config_entry(
            "metrics", self.json_data,
            is_not_null(self.logger), is_list(self.logger), valid_metric_choice
        )

    def parse_output_path(self):
        # TODO: Allow for non-filepath based storage options
        return parse_data_config_entry(
            "output_path", self.json_data, is_not_null(self.logger), as_path()
        )

    def parse_track_params(self):
        default_false = default_as(False, self.logger)
        return parse_data_config_entry(
            "track_params", self.json_data,default_false, is_bool(self.logger)
        )
