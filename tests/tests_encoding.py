#######################################################################
#
# Tests for data/hooks/encoding.py
#
# Usage:
#   python -m pytest -v tests/tests_encoding.py
#######################################################################


import json
import tempfile
import pytest

import numpy as np
import pandas as pd

from pathlib import Path

from data.hooks import DATA_HOOKS
from config.data import DataConfig


@pytest.fixture
def iris_manager():
    """
    Fixture that returns a DataConfig instance for the Iris dataset.
    The tsv file is loaded as a TabularDataManager instance: data_config.data_manager
    Loaded data is stored as a pandas DataFrame: data_config.data_manager.data
    """
    config_dict = {
        "label": "IrisTesting",
        "format": "tabular",
        "data_source": str(Path(__file__).resolve().parent.parent / 'testing' / 'iris_data' / 'iris_testing.tsv'),
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

def test_one_hot_encoding(iris_manager):
    """
    Tests OneHotEncoding on the 'color' column with three unique values ('white', 'purple', 'pink') and 'nan'.
    """
    hook_cls = DATA_HOOKS.get('one_hot_encode', None)
    ohe = hook_cls.from_config(config={'features': ['color']})
    encoded = ohe.run(iris_manager.data_manager)

    cols = list(encoded.data.columns)
    assert "color" not in cols  # 'color' column should be removed after encoding

    COLORS = [col.split('_')[-1] for col in cols if 'color' in col]     # There might be also 'nan'
    # Expect one-hot columns for each color ('white', 'purple', and 'pink'; plus 'nan')
    for color in COLORS:
        assert any(color in col for col in cols)

    # Get one row for each color from input_data
    cols_color = [col for col in cols if 'color' in col]    # Keep only columns starting with 'color'
    input_data = iris_manager.data_manager.data['color']
    encoded_data = encoded.data[cols_color]

    # Combine input_data and encoded_data into a single DataFrame for easier comparison
    combined = pd.concat([input_data, encoded_data], axis=1)
    # Check that the encoding is correct
    for row in combined.iterrows():
        color = str(row[1]['color'])    # str() is needed to handle NaN values
        for col in cols_color:
            if color in col:
                assert row[1][col] == 1
            else:
                assert row[1][col] == 0

def test_one_hot_encoding_binary(iris_manager):
    """
    Tests OneHotEncoding on the 'flower_category' column with two unique values ('small-flower', 'large-flower').
    'drop' is set to 'if_binary' and 'handle_unknown' is set to 'warn'.
    """
    hook_cls = DATA_HOOKS.get('one_hot_encode', None)
    ohe = hook_cls.from_config(config={'features': ['flower_category'],
                                       "drop": "if_binary",
                                       "handle_unknown": "warn"})
    encoded = ohe.run(iris_manager.data_manager)

    cols = list(encoded.data.columns)
    assert "flower_category" not in cols  # 'flower_category' column should be removed after encoding

    input_data = iris_manager.data_manager.data['flower_category'].rename('flower_category')
    encoded_data = encoded.data['flower_category_small-flower'].rename('flower_category_small-flower')

    # Combine input_data and encoded_data into a single DataFrame for easier comparison
    combined = pd.concat([input_data, encoded_data], axis=1)
    # Check that the encoding is correct
    for row in combined.iterrows():
        flower_category = row[1]['flower_category']
        small_flower = row[1]['flower_category_small-flower']
        if flower_category == 'small-flower':
            assert small_flower == 1
        else:
            assert small_flower == 0

def test_ordinal_encoding(iris_manager):
    """
    Tests OrdinalEncoding on the 'color' column.
    """

    ordinal_encoding = {'white': 0, 'purple': 1, 'pink': 2}

    hook_cls = DATA_HOOKS.get('ordinal_encode', None)
    ordinal = hook_cls.from_config(config={'features': ['color'],
                                           'categories': list(ordinal_encoding.keys()),
                                           'unknown_value': np.nan,
                                           'handle_unknown': 'use_encoded_value'})
    encoded = ordinal.run(iris_manager.data_manager)

    input_data = iris_manager.data_manager.data['color'].rename('color')
    encoded_data = encoded.data['color'].rename('color_encoded')

    # Combine input_data and encoded_data into a single DataFrame for easier comparison
    combined = pd.concat([input_data, encoded_data], axis=1)
    # Check that the encoding is correct
    for row in combined.iterrows():
        color = row[1]['color']
        color_encoded = row[1]['color_encoded']
        if color in ordinal_encoding:
            assert color_encoded == ordinal_encoding[color]
        else:
            assert pd.isna(color_encoded)

