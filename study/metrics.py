"""
Metric-reporting closures for use in this framework.
"""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

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
    # Convert it to a quote string so the SQLite backend doesn't explode
    importance_vals = str(importance_vals).replace("'", "").replace('"', '')
    importance_vals = f"'{importance_vals}'"
    return importance_vals
