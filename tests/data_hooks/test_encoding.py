#######################################################################
#
# Tests for data/hooks/encoding.py
#
# Usage:
#   python -m pytest -v tests/data_hooks/tests_encoding.py
#######################################################################

import pytest

import numpy as np
import pandas as pd

from conftest import DummyDataManager
from data.hooks import DATA_HOOKS


@pytest.fixture(scope='module')
def categorical_data_manager():
    testing_df = pd.DataFrame.from_dict({
        "binary_categorical": ['male', 'male', 'female', 'female', 'male', 'female'],
        "trinary_categorical": ['red', 'blue', 'green', 'red', 'green', 'blue'],
        "ordinal": ['low', 'medium', 'high', 'high', 'medium', 'low'],
        "binary_categorical_with_nan": ['male', 'male', 'female', 'female', np.nan, 'female'],
        "trinary_categorical_with_nan": ['red', np.nan, 'green', 'red', 'green', 'blue'],
        "ordinal_with_nan": ['low', 'medium', 'high', np.nan, 'medium', 'low'],
        "dummy_continuous": [1, 2, 3, 4, 5, 6],
        "dummy_continuous_with_nan": [np.nan, 2, 3, 4, 5, 6],
        "oops_half_nans": ['low', np.nan, 'high', np.nan, 'medium', np.nan],
        "oops_all_bob": ['bob', 'bob', 'bob', 'bob', 'bob', 'bob'],
        "oops_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    })

    yield DummyDataManager(testing_df)


"""
One Hot Encoder
"""

@pytest.mark.parametrize("feature_cols,expected_newcols", [
    (["binary_categorical"], {'binary_categorical_male', 'binary_categorical_female'}),
    (["trinary_categorical"], {'trinary_categorical_red', 'trinary_categorical_blue', 'trinary_categorical_green'}),
    (["ordinal"], {'ordinal_low', 'ordinal_medium', 'ordinal_high'}),
    (["binary_categorical_with_nan"], {'binary_categorical_with_nan_male', 'binary_categorical_with_nan_female', 'binary_categorical_with_nan_nan'}),
    (["trinary_categorical_with_nan"], {'trinary_categorical_with_nan_red', 'trinary_categorical_with_nan_blue', 'trinary_categorical_with_nan_green', 'trinary_categorical_with_nan_nan'}),
    (["ordinal_with_nan"], {'ordinal_with_nan_low', 'ordinal_with_nan_medium', 'ordinal_with_nan_high', 'ordinal_with_nan_nan'}),
    (["oops_half_nans"], {'oops_half_nans_low', 'oops_half_nans_medium', 'oops_half_nans_high', 'oops_half_nans_nan'}),
    (["oops_all_bob"], {'oops_all_bob_bob'}),
    (["oops_all_nan"], {'oops_all_nan_nan'}),
    (["binary_categorical", "trinary_categorical"], {'binary_categorical_male', 'binary_categorical_female', 'trinary_categorical_red', 'trinary_categorical_blue', 'trinary_categorical_green'}),
])
def test_one_hot_encoding__default(feature_cols, expected_newcols, categorical_data_manager):
    # Identify the set of columns which should be left over post-run
    too_keep_cols = set(categorical_data_manager.features())
    too_keep_cols -= set(feature_cols)

    # Run the OneHotEncoder hook
    hook_cls = DATA_HOOKS.get('one_hot_encode', None)
    ohe = hook_cls.from_config(config={'features': feature_cols})
    result = ohe.run(categorical_data_manager)
    result_cols = set(result.features())

    # Confirm all the new features we expected to appear did appear
    missing_expected_cols = expected_newcols - result_cols
    if len(missing_expected_cols) > 0:
        pytest.fail(f"Expected new features '{expected_newcols}', missing '{missing_expected_cols}'")

    # Confirm all the features we expected to remain are still there
    missing_kept_cols = too_keep_cols - result_cols
    if len(missing_kept_cols) > 0:
        pytest.fail(f"Expected conserved features '{too_keep_cols}', missing '{missing_kept_cols}'")

    # Confirm that the original "encoded" features are no longer present
    persistent_cols = set(feature_cols).intersection(result_cols)
    if len(persistent_cols) > 0:
        pytest.fail(f"Features '{feature_cols}' are supposed to be removed, yet '{persistent_cols}' remained")

    # Report any columns which were found, but should not be
    invlaid_cols = result_cols - (expected_newcols.union(too_keep_cols))
    if len(invlaid_cols) > 0:
        pytest.fail(f"Found features '{invlaid_cols}' which should not exist.")

@pytest.mark.parametrize("feature_cols,expected_newcols", [
    (["binary_categorical"], {'binary_categorical_male'}),
    (["trinary_categorical"], {'trinary_categorical_red', 'trinary_categorical_blue', 'trinary_categorical_green'}),
    (["ordinal"], {'ordinal_low', 'ordinal_medium', 'ordinal_high'}),
    (["binary_categorical_with_nan"], {'binary_categorical_with_nan_male', 'binary_categorical_with_nan_female', 'binary_categorical_with_nan_nan'}),
    (["trinary_categorical_with_nan"], {'trinary_categorical_with_nan_red', 'trinary_categorical_with_nan_blue', 'trinary_categorical_with_nan_green', 'trinary_categorical_with_nan_nan'}),
    (["ordinal_with_nan"], {'ordinal_with_nan_low', 'ordinal_with_nan_medium', 'ordinal_with_nan_high', 'ordinal_with_nan_nan'}),
    (["oops_half_nans"], {'oops_half_nans_low', 'oops_half_nans_medium', 'oops_half_nans_high', 'oops_half_nans_nan'}),
    (["oops_all_bob"], {'oops_all_bob_bob'}),
    (["oops_all_nan"], {'oops_all_nan_nan'}),
    (["binary_categorical", "trinary_categorical"], {'binary_categorical_male', 'trinary_categorical_red', 'trinary_categorical_blue', 'trinary_categorical_green'}),
])
def test_one_hot_encoding__if_binary(feature_cols, expected_newcols, categorical_data_manager):
    # Identify the set of columns which should be left over post-run
    too_keep_cols = set(categorical_data_manager.features())
    too_keep_cols -= set(feature_cols)

    # Run the OneHotEncoder hook
    hook_cls = DATA_HOOKS.get('one_hot_encode', None)
    ohe = hook_cls.from_config(config={
        'features': feature_cols,
        'drop': "if_binary"
    })
    result = ohe.run(categorical_data_manager)
    result_cols = set(result.features())

    # Confirm all the new features we expected to appear did appear
    missing_expected_cols = expected_newcols - result_cols
    if len(missing_expected_cols) > 0:
        pytest.fail(f"Expected new features '{expected_newcols}', missing '{missing_expected_cols}'")

    # Confirm all the features we expected to remain are still there
    missing_kept_cols = too_keep_cols - result_cols
    if len(missing_kept_cols) > 0:
        pytest.fail(f"Expected conserved features '{too_keep_cols}', missing '{missing_kept_cols}'")

    # Confirm that the original "encoded" features are no longer present
    persistent_cols = set(feature_cols).intersection(result_cols)
    if len(persistent_cols) > 0:
        pytest.fail(f"Features '{feature_cols}' are supposed to be removed, yet '{persistent_cols}' remained")

    # Report any columns which were found, but should not be
    invlaid_cols = result_cols - (expected_newcols.union(too_keep_cols))
    if len(invlaid_cols) > 0:
        pytest.fail(f"Found features '{invlaid_cols}' which should not exist.")

def test_ordinal_encoding(iris_data_config):
    """
    Tests OrdinalEncoding on the 'color' column.
    """

    ordinal_encoding = {'white': 0, 'purple': 1, 'pink': 2}

    hook_cls = DATA_HOOKS.get('ordinal_encode', None)
    ordinal = hook_cls.from_config(config={'features': ['color'],
                                           'categories': list(ordinal_encoding.keys()),
                                           'unknown_value': np.nan,
                                           'handle_unknown': 'use_encoded_value'})
    encoded = ordinal.run(iris_data_config.data_manager)

    input_data = iris_data_config.data_manager.data['color'].rename('color')
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

