from logging import Logger
from pathlib import Path
from sys import maxsize as int_maxsize

from config.utils import default_as, as_str, is_not_null, is_int, is_float, is_list, parse_data_config_entry, load_json_with_validation


class DataConfig(object):
    def __init__(self, json_data: dict, logger: Logger = Logger.root):
        # Track the logger and data for use later
        self.logger = logger
        self.json_data = json_data

        # Parse the JSON data immediately, so we fail before running anything else
        self.drop_columns = self.parse_drop_columns()
        self.column_nullity = self.parse_column_nullity()
        self.row_nullity = self.parse_row_nullity()
        self.target_column = self.parse_target_column()
        self.categorical_cols = self.parse_categorical_cols()
        self.categorical_threshold = self.parse_categorical_threshold()

        # Report any remaining values in the config file to the user
        self.report_remaining_values()


    @staticmethod
    def from_json_file(json_file: Path, logger: Logger = Logger.root):
        """Creates a DataConfig using the contents of a JSON file"""
        json_data = load_json_with_validation(json_file)

        if type(json_data) is not dict:
            logger.error(
                f"JSON should be formatted as a dictionary, was formatted as a {type(json_data)}; terminating"
            )
            raise TypeError

        return DataConfig(json_data, logger)

    """ Content parsers for elements in the configuration file """
    def parse_drop_columns(self):
        default_empty = default_as([], self.logger)
        return parse_data_config_entry(
            "drop_columns", self.json_data, default_empty, is_list(self.logger)
        )

    def parse_column_nullity(self):
        default_nullity = default_as(0.75, self.logger)
        return parse_data_config_entry(
            "column_nullity", self.json_data, default_nullity, is_float(self.logger)
        )

    def parse_row_nullity(self):
        default_nullity = default_as(0.75, self.logger)
        return parse_data_config_entry(
            "row_nullity", self.json_data, default_nullity, is_float(self.logger)
        )

    def parse_target_column(self):
        return parse_data_config_entry(
            "target_column", self.json_data, is_not_null(self.logger), as_str(self.logger)
        )

    def parse_categorical_cols(self):
        default_empty = default_as([], self.logger)
        return parse_data_config_entry(
            "categorical_cols", self.json_data, default_empty, is_list(self.logger)
        )

    def parse_categorical_threshold(self):
        default_max_int = default_as(int_maxsize)
        return parse_data_config_entry(
            "categorical_threshold", self.json_data, default_max_int, is_int(self.logger)
        )

    def report_remaining_values(self):
        if len(self.json_data) == 0:
            return
        for k in self.json_data.keys():
            self.logger.warning(
                f"Entry '{k}' in configuration file is not a valid configuration option and was ignored"
            )

