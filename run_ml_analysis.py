import logging
from argparse import ArgumentParser
from doctest import debug
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.data import DataConfig
from config.model import ModelConfig
from config.study import StudyConfig
from study.manager import StudyManager

LOGGER = logging.getLogger(__name__)


def main(data_config: Path, model_config: Path, study_config: Path):
    # Parse the configuration files
    data_config = DataConfig.from_json_file(data_config)
    model_config = ModelConfig.from_json_file(model_config)
    study_config = StudyConfig.from_json_file(study_config)

    # Generate a StudyManager to run the actual study
    study_manager = StudyManager(
        data_config,
        model_config,
        study_config,
        debug=debug
    )

    # Run the study
    study_manager.run()


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser(
        prog="Classic ML GridSearch",
        description="Runs a gridsearch of all potential parameters using a given datasets and model type"
    )

    parser.add_argument(
        '-d', '--data_config', default='data_config.json', type=Path,
        help="Data Processing configuration file in JSON format"
    )
    parser.add_argument(
        '-m', '--model_config', default='model_config.json', type=Path,
        help="Machine Learning Model configuration file in JSON format"
    )
    parser.add_argument(
        '-s', '--study_config', default='study_config.json', type=Path,
        help="Machine Learning Study configuration file in JSON format"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Whether to show debug statements'
    )

    argvs = parser.parse_args().__dict__

    # Generate a logger for this program
    logging.basicConfig(
        format="{asctime} {levelname}:  \t{message}",
        style='{',
        datefmt="%H:%M:%S"
    )
    debug = argvs.pop('debug', False)
    if debug:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    # Run the analysis
    main(**argvs)