#######################################################################
#
# Tests for data/hooks/encoding.py
#
# Usage:
#   python -m pytest -v tests/data_hooks/tests_encoding.py
#######################################################################

import numpy as np
import pandas as pd
import pytest

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
        "homogenous_with_nans": [1, 1, np.nan, 1, np.nan, np.nan],
        "oops_half_nans": ['low', np.nan, 'high', np.nan, 'medium', np.nan],
        "oops_all_bob": ['bob', 'bob', 'bob', 'bob', 'bob', 'bob'],
        "oops_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    })

    yield DummyDataManager(testing_df)


"""
=== OneHotEncoding ===
"""

### COMMON APPLICATIONS ###
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

    # Confirm the contents of each feature contain only 0s and 1s
    for c in expected_newcols:
        value_set = set(result.data.loc[:, c])
        assert value_set - {0, 1} == set()

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

    # Confirm the contents of each feature contain only 0s and 1s
    for c in expected_newcols:
        value_set = set(result.data.loc[:, c])
        assert value_set - {0, 1} == set()

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

### EDGE CASE TESTS ###
@pytest.mark.xfail(raises=ValueError, strict=True)
def test_one_hot_encoding__unknown_in_test__fail_by_default(categorical_data_manager):
    """
    Confirm that, by default, trying to encode a category which didn't exist in training will fail
    """
    # Badly formed dataset with a color which didn't exist in the dataset
    bad_test_df = pd.DataFrame.from_dict({
        "binary_categorical": ['male', 'male', 'female', 'female', 'male', 'female'],
        # Note the "yellow" color that is now here
        "trinary_categorical": ['red', 'blue', 'green', 'yellow', 'green', 'blue'],
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
    bad_data_manager = DummyDataManager(bad_test_df)

    # Run the OneHotEncoder hook, focusing on the problematic code
    hook_cls = DATA_HOOKS.get('one_hot_encode', None)
    ohe = hook_cls.from_config(config={'features': ['trinary_categorical']})
    # This should fail
    ohe.run_fitted(categorical_data_manager, bad_data_manager)

def test_one_hot_encoding__unknown_in_test__handle_unknown_warning(categorical_data_manager):
    """
    Confirm that the user can configure the underlying SKLearn impl. to no longer fail
    """
    # Badly formed dataset with a color which didn't exist in the dataset
    bad_test_df = pd.DataFrame.from_dict({
        "binary_categorical": ['male', 'male', 'female', 'female', 'male', 'female'],
        # Note the "yellow" color that is now here, but isn't in the original `categorical_data_manager`
        "trinary_categorical": ['red', 'blue', 'green', 'yellow', 'green', 'blue'],
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
    bad_data_manager = DummyDataManager(bad_test_df)

    # Run the OneHotEncoder hook, focusing on the problematic code
    hook_cls = DATA_HOOKS.get('one_hot_encode', None)
    ohe = hook_cls.from_config(config={
        'features': ['trinary_categorical'],
        'handle_unknown': 'warn'
    })
    # This should produce a warning
    with pytest.warns(UserWarning):
        ohe.run_fitted(categorical_data_manager, bad_data_manager)


"""
=== OrdinalEncoding ===
"""

### COMMON APPLICATIONS ###
@pytest.mark.parametrize("feature_cols,cat_order", [
    (["binary_categorical"], ['male', 'female']),
    (["trinary_categorical"], ['red', 'green', 'blue']),
    (["ordinal"], ['low', 'medium', 'high']),
    # I hope no one actually does this, but it's here just in case...
    (["oops_all_bob"], ['bob']),
    (["binary_categorical", "trinary_categorical"], [['male', 'female'], ['red', 'green', 'blue']]),
])
def test_ordinal_encoding__default(feature_cols, cat_order, categorical_data_manager):
    # Run the OrdinalEncoder hook
    hook_cls = DATA_HOOKS.get('ordinal_encode', None)
    ore = hook_cls.from_config(config={
        'features': feature_cols,
        'categories': cat_order
    })
    result = ore.run(categorical_data_manager)

    for i, c in enumerate(feature_cols):
        # Confirm that the column now only contains integers from 0 to n
        value_set = set(result.data.loc[:, c])
        if isinstance(cat_order[i], list):
            cat_vals = cat_order[i]
        else:
            cat_vals = cat_order

        # Generate a mapping of values to ordinal positions
        ordinal_map = {v: i for i, v in enumerate(cat_vals)}
        expected_set = {i for i in ordinal_map.values()}

        # Confirm no unusual values were generated
        delta_set = value_set - expected_set
        if len(delta_set) > 0:
            pytest.fail(
                f"Ordinal Encoder created unknown values {delta_set} when only {expected_set} should have been present"
            )

        # Confirm that these features were mapped correctly
        for v in cat_vals:
            in_df = categorical_data_manager.data
            v_df = in_df.loc[in_df[c] == v, c]
            v_idx = v_df.index
            v_int = ordinal_map[v]
            bad_vals = set(result.data.loc[v_idx, c]) - {v_int}
            if len(bad_vals) > 0:
                pytest.fail(
                    f"Values in column '{c}' had unknown values {bad_vals} after Ordinal Encoding."
                )

### EDGE CASE TESTS ###
@pytest.mark.xfail(raises=ValueError, strict=True)
def test_ordinal_encoding__fail_on_nan(categorical_data_manager):
    # Run the OrdinalEncoder hook
    hook_cls = DATA_HOOKS.get('ordinal_encode', None)
    ore = hook_cls.from_config(config={
        'features': ['binary_categorical_with_nan'],
        # As we're matching SKLearn's implementation, specifying None as part of the list should still fail
        'categories': ['male', 'female', None]
    })
    result = ore.run(categorical_data_manager)

def test_ordinal_encoding__handle_nan_as_unknown(categorical_data_manager):
    # Setup
    target_col = 'binary_categorical_with_nan'
    categories = ['male', 'female']
    unknown_value = -1

    # Run the OrdinalEncoder hook
    hook_cls = DATA_HOOKS.get('ordinal_encode', None)
    ore = hook_cls.from_config(config={
        'features': [target_col],
        # As we're matching SKLearn's implementation, specifying None as part of the list should still fail
        'categories': categories,
        'handle_unknown': 'use_encoded_value',
        'unknown_value': unknown_value
    })
    result = ore.run(categorical_data_manager)

    # Confirm that the column now only contains integers from -1 to n
    value_set = set(result.data.loc[:, target_col])

    # Generate a mapping of values to ordinal positions
    ordinal_map = {v: i for i, v in enumerate(categories)}
    expected_set = {i for i in ordinal_map.values()}

    # Confirm no unusual values were generated
    delta_set = value_set - expected_set - {unknown_value}
    if len(delta_set) > 0:
        pytest.fail(
            f"Ordinal Encoder created unknown values {delta_set} when only {expected_set} should have been present"
        )

    # Confirm that non-NaN features were mapped correctly
    for v in categories:
        in_df = categorical_data_manager.data
        v_df = in_df.loc[in_df[target_col] == v, target_col]
        v_idx = v_df.index
        v_int = ordinal_map[v]
        bad_vals = set(result.data.loc[v_idx, target_col]) - {v_int}
        if len(bad_vals) > 0:
            pytest.fail(
                f"Values in column '{target_col}' had unknown values {bad_vals} after Ordinal Encoding."
            )

    # Confirm that NaN features were mapped correctly
    in_df = categorical_data_manager.data
    v_df = in_df.loc[in_df[target_col].isna(), target_col]
    v_idx = v_df.index
    v_int = unknown_value
    bad_vals = set(result.data.loc[v_idx, target_col]) - {v_int}
    if len(bad_vals) > 0:
        pytest.fail(
            f"NaN values in column '{target_col}' should all be { {unknown_value} }; "
            f"values {bad_vals} were erroneously present."
        )

"""
=== LadderEncoding ===
"""

@pytest.mark.parametrize("feature_col,cat_order", [
    ("binary_categorical", ['male', 'female']),
    ("trinary_categorical", ['red', 'green', 'blue']),
    ("ordinal", ['low', 'medium', 'high']),
    # NaNs are ignored by default
    ("binary_categorical_with_nan", ['male', 'female']),
    ("trinary_categorical_with_nan", ['red', 'green', 'blue']),
])
def test_ladder_encoding__default(feature_col, cat_order, categorical_data_manager):
    # Run the Ladder Encoder hook
    hook_cls = DATA_HOOKS.get('ladder_encode', None)
    ladder = hook_cls.from_config(config={
        'feature': feature_col,
        'order': cat_order
    })
    result = ladder.run(categorical_data_manager)

    # Confirm that the old column was removed
    assert feature_col not in result.features()

    # Confirm that all columns which were supposed to be generated were
    expected_cols = [f"{feature_col} ({cat_order[i]} <- {cat_order[i+1]})" for i in range(len(cat_order)-1)]
    missing_cols = set(expected_cols) - set(result.features())
    if len(missing_cols) != 0:
        pytest.fail(f'Output of ladder encoder was missing columns "{missing_cols}"; '
                    f'produced columns "{set(result.features())}" instead')

    # Confirm that all other columns remained untouched
    preserved_cols = set(categorical_data_manager.features()) - {feature_col}
    lost_cols = set(preserved_cols) - set(result.features())
    if len(lost_cols) > 0:
        pytest.fail(f'Output of ladder encoder deleted columns "{missing_cols}" which should have bee preserved.')


@pytest.mark.xfail(
    reason="Ladder Encoding makes no sense in the context of a single feature",
    raises=ValueError
)
def test_ladder_encoding__fail_on_single_feature(categorical_data_manager):
    # Run the Ladder Encoder hook
    hook_cls = DATA_HOOKS.get('ladder_encode', None)
    ladder = hook_cls.from_config(config={
        'feature': ["oops_all_bob"],
        'order': ["bob"]
    })
    ladder.run(categorical_data_manager)


def test_ladder_encoding__work_with_missing_steps(categorical_data_manager):
    """
    # Ladder encoding should still work if a "step" isn't seen in the data;
    # it will just be grouped in with an adjacent step instead
    """
    # Run the Ladder Encoder hook
    hook_cls = DATA_HOOKS.get('ladder_encode', None)
    ladder = hook_cls.from_config(config={
        'feature': "trinary_categorical",
        'order': ['red', 'green', 'yellow', 'blue']  # Yellow doesn't exist
    })
    result = ladder.run(categorical_data_manager)
    print(result)
