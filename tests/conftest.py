import json
import tempfile
from pathlib import Path

import pytest

from config.data import DataConfig


@pytest.fixture
def iris_data_config():
    """
    Fixture that returns a DataConfig instance for the Iris dataset.
    The tsv file is loaded as a TabularDataManager instance: data_config.data_manager
    Loaded data is stored as a pandas DataFrame: data_config.data_manager.data
    """
    config_dict = {
        "label": "IrisTesting",
        "format": "tabular",
        "data_source": str(Path(__file__).parent.resolve() / 'testing_files' / 'iris_data' / 'iris_testing.tsv'),
        "separator": "\t",
        "index": "id",
    }

    # Create a temporary file for the config
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_file:
        json.dump(config_dict, tmp_file)
        tmp_config_path = Path(tmp_file.name)  # Convert string path to Path object

    # Ensure the file exists before using it
    assert tmp_config_path.exists(), f"Temporary config file {tmp_config_path} was not created."

    data_config = DataConfig.from_json_file(tmp_config_path)

    return data_config


