from logging import Logger
from pathlib import Path

from config.utils import default_as, parse_data_config_entry, is_int, load_json_with_validation


class StudyConfig(object):
    def __init__(self, json_data: dict, logger: Logger = Logger.root):
        # Track the logger and data for use later
        self.logger = logger
        self.json_data = json_data

        # Parse the JSON data immediately, so we fail before running anything else
        self.random_seed = self.parse_random_seed()
        self.no_replicates = self.parse_no_replicates()
        self.no_crosses = self.parse_no_crosses()


    @staticmethod
    def from_json_file(json_file: Path, logger: Logger = Logger.root):
        """Creates a DataConfigManager using the contents of a JSON file"""
        json_data = load_json_with_validation(json_file)

        if type(json_data) is not dict:
            logger.error(
                f"JSON should be formatted as a dictionary, was formatted as a {type(json_data)}; terminating"
            )
            raise TypeError

        return StudyConfig(json_data, logger)

    """ Content parsers for elements in the configuration file """
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