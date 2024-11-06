"""
Metric-reporting closures for use in this framework.
"""

from typing import TypeVar

from sklearn.metrics import log_loss, balanced_accuracy_score, roc_auc_score

T = TypeVar('T')

""" Supervised """
def sk_log_loss(manager, model, x, y):
    # Log Loss
    py = manager.predict_proba(model, x)
    return log_loss(y, py)

def sk_balanced_accuracy(manager, model, x, y):
    # Balanced Accuracy
    py = manager.predict(model, x)
    return balanced_accuracy_score(y, py)

def sk_roc_auc(manager, model, x, y):
    # ROC AUC
    py = manager.predict_proba(model, x)
    if py.shape[1] != 2:
        raise ValueError(f"ROC AUC can only be calculated for binary classification tasks; found {py.shape[1]} classes")
    py = py[:, 1]  # No idea why it's always the second class' value, but it is
    return roc_auc_score(y, py)
