from typing import Callable, Any

from models.utils import OptunaModelManager
from study.metrics import *

# Generic type for functions which can generate some metric(s) to be recorded by the user
T = TypeVar('T')
MetricUpdater = Callable[[OptunaModelManager[T], T, Any, Any], Any]

# Dictionary of available-by-default metric functions
METRIC_FUNCTIONS: dict[str, MetricUpdater] = {
    "log_loss": sk_log_loss,
    "balanced_accuracy": sk_balanced_accuracy,
    "roc_auc": sk_roc_auc
}
