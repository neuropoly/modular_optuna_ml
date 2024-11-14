"""
Metric-reporting closures for use in this framework.
"""
from typing import TypeVar

from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

from data import BaseDataManager
from models.base import OptunaModelManager

T = TypeVar('T')

""" Supervised """
def sk_log_loss(manager: OptunaModelManager, model, x: BaseDataManager, y: BaseDataManager):
    # Log Loss
    py = manager.predict_proba(model, x.as_array())
    return log_loss(y.as_array(), py)

def sk_balanced_accuracy(manager: OptunaModelManager, model, x: BaseDataManager, y: BaseDataManager):
    # Balanced Accuracy
    py = manager.predict(model, x.as_array())
    return balanced_accuracy_score(y.as_array(), py)

def sk_roc_auc(manager: OptunaModelManager, model, x: BaseDataManager, y: BaseDataManager):
    # ROC AUC
    py = manager.predict_proba(model, x.as_array())
    if py.shape[1] != 2:
        raise ValueError(f"ROC AUC can only be calculated for binary classification tasks; found {py.shape[1]} classes")
    py = py[:, 1]  # No idea why it's always the second class' value, but it is
    return roc_auc_score(y.as_array(), py)
