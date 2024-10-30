from typing import Callable

from study.metrics import *

# Generic type for functions which can generate some metric(s) to be recorded by the user
T = TypeVar('T')
MetricUpdater = Callable[[OptunaModelManager[T], T, dict], float]

# Dictionary of available-by-default metric functions
METRIC_FUNCTIONS: dict[str, MetricUpdater] = {
    "training_log_loss": training_log_loss,
    "testing_log_loss": testing_log_loss,
    "training_bacc": training_bacc,
    "testing_bacc": testing_bacc
}
