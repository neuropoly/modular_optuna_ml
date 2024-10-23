from inspect import isclass
from logging import Logger
from pathlib import Path

from typing import Dict

from config.utils import parse_data_config_entry, is_not_null, as_str, is_valid_option, default_as, is_dict, \
    load_json_with_validation
from models import FACTORY_MAP
from models.utils import OptunaModelFactory


class ModelConfig(object):
    def __init__(self, json_data: list, logger: Logger = Logger.root):
        # Track the logger and data for use later
        self.logger = logger

        # Parse the JSON data immediately, so we fail before running anything else
        self.sub_configs: Dict[str, ModelSubConfig] = {}
        for i, entry in enumerate(json_data):
            new_subconfig = ModelSubConfig(i, entry, self.logger)
            self.sub_configs[new_subconfig.label] = new_subconfig

    @staticmethod
    def from_json_file(json_file: Path, logger: Logger = Logger.root):
        """Creates a DataConfigManager using the contents of a JSON file"""
        json_data = load_json_with_validation(json_file)

        if type(json_data) is dict:
            logger.warning(
                f"Model configuration JSON was formatted as a dictionary; assuming a single model entry"
            )
            json_data = [json_data]

        return ModelConfig(json_data, logger)

class ModelSubConfig(object):
    def __init__(self, index: int, entry_data: dict, logger: Logger = Logger.root):
        # Save the logger and entry data for later
        self.index = index
        self.logger = logger
        self.entry_data = entry_data

        # Parse the JSON data immediately, so we fail before running anything else
        self.label = self.parse_label()
        self.model_name = self.parse_model()
        self.parameters = self.parse_parameters()

        # Pre-build the model factory using the values prior
        self.model_factory = self.generate_model_factory()

    """ Content parsers for elements in the configuration file """
    def parse_label(self):
        index_label_default = default_as(f"Unnamed [{self.index}]", self.logger)
        return parse_data_config_entry(
            "label", self.entry_data,index_label_default, as_str(self.logger)
        )

    def parse_model(self):
        # Pull the model name from the config
        valid_model_choice = is_valid_option(FACTORY_MAP.keys(), self.logger)
        return parse_data_config_entry(
            "model", self.entry_data,
            is_not_null(self.logger), as_str(self.logger), valid_model_choice
        )

    def parse_parameters(self):
        return parse_data_config_entry(
            "parameters", self.entry_data, is_not_null(self.logger), is_dict(self.logger)
        )

    def report_remaining_values(self):
        if len(self.entry_data) == 0:
            return
        for k in self.entry_data.keys():
            self.logger.warning(
                f"Entry '{k}' for model '{self.label}' in configuration file is not a valid configuration option and was ignored"
            )

    """ Miscellaneous """
    def generate_model_factory(self):
        # Confirm that it is a valid model type, in case any invalid post-hoc modification occurred
        factory_class = FACTORY_MAP.get(self.model_name)
        if not isclass(factory_class) or not issubclass(factory_class, OptunaModelFactory):
            raise ValueError(
                f"Manager class for model entry '{self.label}' is not a subclass of OptunaModelManager; terminating.")
        # Generate the model factory using the parameters specified by the user
        model_factory = factory_class(**self.parameters)

        return model_factory