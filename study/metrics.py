from typing import TypeVar

import numpy as np

from sklearn.metrics import log_loss, balanced_accuracy_score, roc_auc_score

from models.utils import OptunaModelManager

T = TypeVar('T')

""" Supervised """
### Log Loss
def _log_loss(manager, model, x, y):
    py = manager.predict_proba(model, x)
    return log_loss(y, py)

def training_log_loss(manager: OptunaModelManager[T], model: T, context: dict) -> float:
    # Pull the testing data out of the context
    x = context.get("train_x")
    y = context.get("train_y")

    # Calculate the log loss from it
    return _log_loss(manager, model, x, y)

def testing_log_loss(manager: OptunaModelManager[T], model: T, context: dict) -> float:
    # Pull the testing data out of the context
    x = context.get("test_x")
    y = context.get("test_y")

    # Calculate the log loss from it
    return _log_loss(manager, model, x, y)

### Balanced Accuracy
def _balanced_accuracy(manager, model, x, y):
    py = manager.predict(model, x)
    return balanced_accuracy_score(y, py)

def training_bacc(manager: OptunaModelManager[T], model: T, context: dict) -> float:
    # Pull the testing data out of the context
    x = context.get("train_x")
    y = context.get("train_y")

    # Calculate the log loss from it
    return _balanced_accuracy(manager, model, x, y)

def testing_bacc(manager: OptunaModelManager[T], model: T, context: dict) -> float:
    # Pull the testing data out of the context
    x = context.get("test_x")
    y = context.get("test_y")

    # Calculate the log loss from it
    return _balanced_accuracy(manager, model, x, y)

### ROC AUC
def _roc_auc(manager, model, x, y):
    py = manager.predict_proba(model, x)
    if py.shape[1] != 2:
        raise ValueError(f"ROC AUC can only be calculate for binary classification tasks; found {py.shape[1]} classes")
    py = py[:, 1]  # No idea why it's always the second class' value, but it is
    return roc_auc_score(y, py)

def training_roc_auc(manager: OptunaModelManager[T], model: T, context: dict) -> float:
    # Pull the testing data out of the context
    x = context.get("train_x")
    y = context.get("train_y")

    # Calculate the log loss from it
    return _roc_auc(manager, model, x, y)

def testing_roc_auc(manager: OptunaModelManager[T], model: T, context: dict) -> float:
    # Pull the testing data out of the context
    x = context.get("test_x")
    y = context.get("test_y")

    # Calculate the log loss from it
    return _roc_auc(manager, model, x, y)
