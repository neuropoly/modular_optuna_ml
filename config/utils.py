"""
Collection of utilities for managing configuration files dynamically
"""
import json
from logging import Logger
from pathlib import Path
from typing import Callable, Any, Union, TypeVar

""" JSON parsing helpers """
def load_json_with_validation(json_path: Path, logger: Logger = Logger.root) -> Union[list, dict]:
    """
    Attempts to load a JSON file, doing some checks in the process to avoid file IO issues
    :param json_path: The file path to the JSON file to be loaded
    :param logger: A logger to log results to. Defaults to the root logger is one is not specified
    :return: The parsed contents of the JSON file, which can be a list or dictionary
    """
    # Check to confirm the file exists and is a valid file
    if not json_path.exists():
        logger.error("JSON configuration file designated was not found; terminating")
        raise FileNotFoundError()
    if not json_path.is_file():
        logger.error("JSON configuration file specified was a directory, not a file; terminating")
        raise TypeError()
    # Attempt to load the files contents w/ JSON
    with open(json_path) as json_file:
        try:
            json_data = json.load(json_file)
        except Exception as e:
            logger.error("Failed to load JSON file, see error below; terminating")
            raise e
    return json_data

T = TypeVar("T")
Check = Callable[[str, T], T]

def parse_data_config_entry(config_key: str, json_dict: dict, *checks: Check) -> Any:
    """
    Automatically parses a key contained within the JSON file, running any checks requested by the user in the process
    :param config_key: The key to query for within the JSON file
    :param json_dict: The JSON file's contents, in dictionary format
    :param checks: A sequence of functions to run on the value parsed from the JSON file. Run in the order they are provided
    :return: An updated version of the 'config_dict' with the new config value
    """
    # Pull the value, returning non if needed
    config_val = json_dict.pop(config_key, None)
    # Run any and all checks requested by the user
    for fn in checks:
        config_val = fn(config_key, config_val)
    # Return the processed and check value if all checks passed
    return config_val

""" Transforming functions """
def default_as(default_val, logger: Logger = Logger.root) -> Check:
    """Returns a default value if a null value is observed"""
    def check(k: str, v):
        if v is None:
            logger.warning(f"No value for '{k}' was found, defaulting to {default_val}")
            return default_val
        return v
    return check

def as_str(logger: Logger) -> Check:
    def check(k, v):
        if not type(v) is str:
            logger.warning(f"Value for '{k}' was not a native string, and was converted automatically")
            return str(v)
        return v
    return check

def as_path() -> Check:
    def check(_, v):
        p = Path(v)
        return p
    return check

""" Value-checking functions"""
def is_not_null(logger: Logger) -> Check:
    def check(k: str, v):
        if v is None:
            logger.error(f"Config value '{k}' must be specified by the user. Terminating.")
            raise ValueError()
        return v
    return check

# noinspection DuplicatedCode
def is_bool(logger: Logger) -> Check:
    """Confirms the values is a boolean"""
    def check(k: str, v):
        if type(v) is not bool:
            logger.error(f"'{k}' specified in the configuration file was not a boolean; terminating")
            raise TypeError
        return v
    return check

# noinspection DuplicatedCode
def is_int(logger: Logger) -> Check:
    """Confirms the value is an integer"""
    def check(k: str, v):
        if type(v) is not int:
            logger.error(f"'{k}' specified in the configuration file was not an integer; terminating")
            raise TypeError
        return v
    return check

# noinspection DuplicatedCode
def is_float(logger: Logger) -> Check:
    """Confirms the value is a float"""
    def check(k: str, v):
        if type(v) is not float:
            logger.error(f"'{k}' specified in the configuration file was not a float; terminating")
            raise TypeError
        return v
    return check

# noinspection DuplicatedCode
def is_list(logger: Logger) -> Check:
    """Confirms the value is a list"""
    def check(k: str, v):
        if type(v) is not list:
            logger.error(f"'{k}' specified in the configuration file was not a list; terminating")
            raise TypeError
        return v

    return check

def is_dict(logger: Logger) -> Check:
    """Confirms the value is a dictionary"""
    def check(k: str, v):
        if type(v) is not dict:
            logger.error(f"'{k}' specified in the configuration file was not a dictionary; terminating")
            raise TypeError
        return v

    return check

def is_file(logger: Logger) -> Check:
    """Confirms the value is a file which exists"""
    def check(k: str, v):
        file_path = Path(v)
        if not file_path.exists():
            logger.error(f"The path specified in '{k}' does not exist; terminating")
            raise TypeError
        elif not file_path.is_file():
            logger.error(f"The path specified in '{k}' is not a file; terminating")
            raise TypeError
        return file_path

    return check

def is_valid_option(check_set: set, logger: Logger) -> Check:
    """Confirms a value is on of a set of options"""
    def check(k: str, v):
        if v not in check_set:
            logger.error(f"Value of '{k}' must be one of the following: {check_set}. Terminating")
            raise TypeError
        return v
    return check

def all_valid_options(check_set: set, logger: Logger) -> Check:
    """Confirms all values within a list are a valid choice"""
    def check(k: str, v):
        v = list(v)
        invalid_choices = [x for x in v if x not in check_set]
        if len(invalid_choices) > 0:
            logger.error(f"All values contained in '{k}' must be one of the following: {check_set}. "
                         f"Invalid values were {invalid_choices}. Terminating")
            raise ValueError
        return v

    return check
