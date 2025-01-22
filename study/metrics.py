"""
Metric-reporting closures for use in this framework.
"""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score

from data import BaseDataManager
from data.mixins import MultiFeatureMixin
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
    # Sanity check that the user is requesting this on a data manager with multiple features
    if not issubclass(type(x), MultiFeatureMixin):
        # TODO: Move this check to pre-run, so it will fail before any (potentially slow) analyses are run
        raise TypeError(
            f"DataManager of type '{type(x).__name__}' only has one feature, feature importance is irrelevant"
        )
    x: BaseDataManager | MultiFeatureMixin
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
