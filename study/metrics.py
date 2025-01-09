"""
Metric-reporting closures for use in this framework.
"""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score

from data import BaseDataManager
from data.mixins import MultiFeatureMixin
from models.base import OptunaModelManager

""" Supervised """
def sk_log_loss(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Log Loss
    py = manager.predict_proba(x.as_array())
    return log_loss(y.as_array(), py)

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

def sk_recall_weighted_avg(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # Recall, weighted average
    py = manager.predict(x.as_array())
    return recall_score(y.as_array(), py, average='weighted')

def sk_f1_weighted_avg(manager: OptunaModelManager, x: BaseDataManager, y: BaseDataManager):
    # F1-score, weighted average
    py = manager.predict(x.as_array())
    return f1_score(y.as_array(), py, average='weighted')


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
    importance_vals = str(importance_vals).replace("'", "").replace('"', '')
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
    good_samples = str(good_samples).replace("'", "").replace('"', '')

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
    bad_samples = str(bad_samples).replace("'", "").replace('"', '')

    return bad_samples