from inspect import isclass
from logging import Logger
from pathlib import Path

from config.utils import parse_data_config_entry, is_not_null, as_str, is_valid_option, default_as, is_dict, \
    load_json_with_validation
from models import MANAGER_MAP
from models.utils import OptunaModelManager


class ModelConfig(object):
    """
    Configuration manager which handles the loading and parsing of model configuration JSON files
    """
    def __init__(self, json_data: dict, logger: Logger = Logger.root):
        # Save the logger and entry data for later
        self.logger = logger
        self.json_data = json_data

        # Parse the JSON data immediately, so we fail before running anything else
        self.label = self.parse_label()
        self.model_name = self.parse_model()
        self.parameters = self.parse_parameters()

        # Pre-build the model manager using the values prior
        self.model_manager = self.generate_model_manager()

    @staticmethod
    def from_json_file(json_file: Path, logger: Logger = Logger.root):
        """Creates a ModelConfig using the contents of a JSON file"""
        json_data = load_json_with_validation(json_file)

        if type(json_data) is not dict:
            logger.error(
                f"JSON should be formatted as a dictionary, was formatted as a {type(json_data)}; terminating"
            )
            raise TypeError

        return ModelConfig(json_data, logger)

    """ Content parsers for elements in the configuration file """
    def parse_label(self):
        index_label_default = default_as("OptunaModel", self.logger)
        return parse_data_config_entry(
            "label", self.json_data,index_label_default, as_str(self.logger)
        )

    def parse_model(self):
        # Pull the model name from the config
        valid_model_choice = is_valid_option(MANAGER_MAP.keys(), self.logger)
        return parse_data_config_entry(
            "model", self.json_data,
            is_not_null(self.logger), as_str(self.logger), valid_model_choice
        )

    def parse_parameters(self):
        return parse_data_config_entry(
            "parameters", self.json_data, is_not_null(self.logger), is_dict(self.logger)
        )

    def report_remaining_values(self):
        if len(self.json_data) == 0:
            return
        for k in self.json_data.keys():
            self.logger.warning(
                f"Entry '{k}' for model '{self.label}' in configuration file is not a valid configuration option and was ignored"
            )

    """ Miscellaneous """
    def generate_model_manager(self):
        # Confirm that it is a valid model type, in case any invalid post-hoc modification occurred
        manager_class = MANAGER_MAP.get(self.model_name)
        if not isclass(manager_class) or not issubclass(manager_class, OptunaModelManager):
            raise ValueError(
                f"Manager class for model entry '{self.label}' is not a subclass of OptunaModelManager; terminating.")
        # Generate the model manager using the parameters specified by the user
        model_manager = manager_class(**self.parameters)

        return model_manager