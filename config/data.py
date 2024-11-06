from logging import Logger
from pathlib import Path

from config.utils import as_str, is_not_null, parse_data_config_entry, \
    load_json_with_validation
from data import MANAGER_MAP
from data.utils import DataManager


class DataConfig(object):
    """
    Configuration manager which handles the loading and parsing of data configuration JSON files
    """
    def __init__(self, json_data: dict, logger: Logger = Logger.root):
        # Track the logger and data for use later
        self.logger = logger
        self.json_data = json_data

        # Attempt to grab the type of data which should be managed
        self.format = self.parse_format()
        self.label = self.parse_label()

        # Parse the remaining config using the config manager associated with the format
        self.manager_cls = MANAGER_MAP.get(self.format)
        self.data_manager : DataManager = self.manager_cls.build_from_config_dict(self.json_data)

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
    def parse_format(self):
        return parse_data_config_entry(
            "format", self.json_data, is_not_null(self.logger), as_str(self.logger)
        )

    def parse_label(self):
        return parse_data_config_entry(
            "label", self.json_data, is_not_null(self.logger), as_str(self.logger)
        )

    def report_remaining_values(self):
        if len(self.json_data) == 0:
            return
        for k in self.json_data.keys():
            self.logger.warning(
                f"Entry '{k}' in configuration file is not a valid configuration option and was ignored"
            )

