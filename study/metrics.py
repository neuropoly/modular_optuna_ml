"""
Metric-reporting closures for use in this framework.
"""
import sys

import numpy as np
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score

from data import BaseDataManager
from models.base import OptunaModelManager


""" Utilities """
def clean_val_for_db(val):
    return str(val).replace("'", "").replace('"', '')


""" Supervised """
def sk_log_loss(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Log Loss
    py = manager.predict_proba(x.as_array())
    y_labels = [i for i in range(py.shape[1])]
    return log_loss(y.as_array(), py, labels=y_labels)

def sk_balanced_accuracy(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Balanced Accuracy
    py = manager.predict(x.as_array())
    return balanced_accuracy_score(y.as_array(), py)

def sk_roc_auc(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # ROC AUC
    py = manager.predict_proba(x.as_array())
    if py.shape[1] != 2:
        raise ValueError(f"ROC AUC can only be calculated for binary classification tasks; found {py.shape[1]} classes")
    py = py[:, 1]  # No idea why it's always the second class' value, but it is
    return roc_auc_score(y.as_array(), py)

def sk_precision_weighted_avg(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Precision, weighted average
    py = manager.predict(x.as_array())
    return precision_score(y.as_array(), py, average='weighted')

def sk_precision_perclass(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Precision, measured per class
    py = manager.predict(x.as_array())
    y_flat = y.as_array().flatten()
    scores = precision_score(y_flat, py, average=None)
    cat_set = list(set(y_flat))
    score_dict = dict()
    for i, v in enumerate(scores):
        score_dict[str(cat_set[i])] = str(v)
    score_dict = clean_val_for_db(score_dict)
    return score_dict

def sk_recall_weighted_avg(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Recall, weighted average
    py = manager.predict(x.as_array())
    return recall_score(y.as_array(), py, average='weighted')

def sk_recall_perclass(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Recall, measured per class
    py = manager.predict(x.as_array())
    y_flat = y.as_array().flatten()
    scores = recall_score(y_flat, py, average=None)
    cat_set = list(set(y_flat))
    score_dict = dict()
    for i, v in enumerate(scores):
        score_dict[str(cat_set[i])] = str(v)
    score_dict = clean_val_for_db(score_dict)
    return score_dict

def sk_f1_weighted_avg(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # F1-score, weighted average
    py = manager.predict(x.as_array())
    return f1_score(y.as_array(), py, average='weighted')

def sk_f1_perclass(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Recall, measured per class
    py = manager.predict(x.as_array())
    y_flat = y.as_array().flatten()
    scores = f1_score(y_flat, py, average=None)
    cat_set = list(set(y_flat))
    score_dict = dict()
    for i, v in enumerate(scores):
        score_dict[str(cat_set[i])] = str(v)
    score_dict = clean_val_for_db(score_dict)
    return score_dict


""" Feature Importance """
def importance_by_permutation(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Get the mean importance values
    importance_vals = permutation_importance(manager.get_model(), x.as_array(), y.as_array()).importances_mean
    # Pair them with their feature labels
    importance_vals = {k: importance_vals[i] for i, k in enumerate(x.features())}
    # Sort the results from most to least ABSOLUTE importance. TODO: Make this configurable
    importance_vals = dict(sorted(
        importance_vals.items(), key=lambda v: np.abs(v[1]), reverse=True
    ))
    # Convert it to a string-formatted dictionary, in 'feature_name: feature_importance' form
    importance_vals = [f'{k}: {v}' for k, v in importance_vals.items()]
    # Convert it to a cleaned string so the SQLite backend doesn't explode
    importance_vals = clean_val_for_db(importance_vals)
    return importance_vals

def shap_additive(manager: OptunaModelManager, x: BaseDataManager, _: BaseDataManager):
    """
    To restore the (raw) values for a given run, run the following snippet:

    ```
    from io import StringIO

    # You can omit the Numpy import if you manually
    # parse the inner string in the list comp
    import numpy as np

    # This is the SHAP value within the database you want to parse
    val = ...

    # Strip the brackets first
    val = val.strip("{").strip("}")
    # Split by commas
    entry_strs = val.split(", ")
    # Split the dataset by feature name
    shap_map = dict()
    for entry_str in entry_strs:
        # Split the text along the colon to get the feature label back
        feature_label, shap_value_str = entry_str.split(": ")
        # Parse the shap value string back into numeric form
        shap_vals = [list(np.fromstring(x, sep=" ")) for x in shap_value_str.split("\n")]
        # Add it to the map
        shap_map[feature_label] = shap_vals
    ```

    Each entry in `shap_map` will be Numpy array of with the following dimensions:
        * n is the number of samples in the input dataset (train, validate, or test), and
        * c is the number of categorical classes used during training;
            If this is binary classification, or a continuous target, c=1.

    For categorical targets with more than 2 classes, each class is treated as
    unique feature by SHAP for the purpose of calculating SHAP values.

    TODO: Save the `shap_values` directly via pickle into a SQLite blob
    """
    # Initialize the explainer, using the x data as both the mask and feature list
    x_arr = x.as_array()
    model = manager.get_model()
    try:
        # Default to the "generic" explainer
        explainer = shap.Explainer(
            model, x_arr, feature_names=x.features()
        )
        # Calculate the Shapley values from this dataset
        shap_values = explainer(x_arr)
    except TypeError as err:
        # If that failed, try to use the model's "predict" function instead
        if hasattr(model, "predict"):
            explainer = shap.Explainer(
                model.predict, x_arr, feature_names=x.features()
            )
            # Calculate the Shapley values from this dataset
            shap_values = explainer(x_arr)
        else:
            raise err

    shap_list = list()
    for i, v in enumerate(shap_values.feature_names):
        # SHAP auto-reduces the shape of its features if it is targeting
        # a binary classification OR a continuous metric
        if len(shap_values.values.shape) < 3:
            val_str = np.array2string(shap_values.values[:, i], max_line_width=sys.maxsize, threshold=sys.maxsize)
        else:
            val_str = np.array2string(shap_values.values[:, i, :], max_line_width=sys.maxsize, threshold=sys.maxsize)
        # Remove the brackets; despite Numpy adding them, it cannot parse them after...
        val_str = val_str.replace("[", "").replace("]", "")
        val_str = f"{v}: {val_str}"
        shap_list.append(val_str)

    # This nonsense is required because Python maps
    # "/n" to "//n" if you string convert a dict;
    # why the hell does it do that?!?!?
    full_str = "{"
    full_str += ", ".join(shap_list)
    full_str += "}"

    # Return the result to be saved
    return full_str


""" Sample Reporting """
def correct_samples(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Get the model's predicted values
    y_pred = manager.predict(x.as_array())

    # Generate a "mask" of the values which match across the predicted and true y values
    y_flat = y.as_array().flatten()
    correct_mask = y_pred == y_flat

    # Pull out the sample labels which are valid for this metric hook
    good_samples = x[correct_mask].get_index()

    # Strip quotation marks from the result so the DB backend doesn't explode
    good_samples = clean_val_for_db(good_samples)

    return good_samples

def incorrect_samples(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Get the model's predicted values
    y_pred = manager.predict(x.as_array())

    # Generate a "mask" of the values which match across the predicted and true y values
    y_flat = y.as_array().flatten()
    incorrect_mask = y_pred != y_flat

    # Pull out the sample labels which are valid for this metric hook
    bad_samples = x[incorrect_mask].get_index()

    # Strip quotation marks from the result so the DB backend doesn't explode
    bad_samples = clean_val_for_db(bad_samples)

    return bad_samples

""" ROC Curve """
def y_true_collector(_: OptunaModelManager, __: BaseDataManager, y: BaseDataManager):
    """ Collects the true binary labels for ROC curve generation. """
    return clean_val_for_db(list(y.as_array().flatten()))

def y_pred_proba_collector(manager: OptunaModelManager, x: BaseDataManager, _: BaseDataManager):
    """ Collects predicted probabilities for the positive class. """
    py = manager.predict_proba(x.as_array())
    if py.shape[1] != 2:
        raise ValueError(f"Expected binary classification with two probability columns; found {py.shape[1]}.")
    return clean_val_for_db(list(py[:, 1]))  # Probabilities for the positive class