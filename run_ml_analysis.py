"""
Main entrypoint for analyses using this framework.

Running this program requires 3 configuration files:
    - A data configuration file, which describes which data to analyse, and how to process it before ML analysis
    - A model configuration file, which describes which model to test, as well as which hyperparameters to explore
    - A study configuration file, which dictates how to optimize the model's hyperparameters, as well as what metrics
        to record and where to save them

Example usage (using the testing data provided in this repository):
> python run_ml_analysis.py -d iris_config.json -m log_reg.json -s testing_study_config.json

Author: Kalum Ost
"""

from argparse import ArgumentParser
from pathlib import Path

from config.data import DataConfig
from config.model import ModelConfig
from config.study import StudyConfig
from study.manager import StudyManager


def main(data_config: Path, model_config: Path, study_config: Path, overwrite: bool, debug: bool):
    # Parse the configuration files
    data_config = DataConfig.from_json_file(data_config)
    model_config = ModelConfig.from_json_file(model_config)
    study_config = StudyConfig.from_json_file(study_config)

    # Generate a StudyManager to run the actual study
    study_manager = StudyManager(
        data_config,
        model_config,
        study_config,
        overwrite=overwrite,
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
        '--overwrite', action='store_true',
        help="Whether the program should be allowed to overwrite existing database tables to save its output"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Whether to show debug statements'
    )

    argvs = parser.parse_args().__dict__

    # Run the analysis
    main(**argvs)