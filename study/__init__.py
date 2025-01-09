from typing import Any, Callable

from study.metrics import *

# Generic type for functions which can generate some metric(s) to be recorded by the user
MetricUpdater = Callable[[OptunaModelManager, BaseDataManager, BaseDataManager], Any]

# Dictionary of available-by-default metric functions
METRIC_FUNCTIONS: dict[str, MetricUpdater] = {
    "log_loss": sk_log_loss,
    "balanced_accuracy": sk_balanced_accuracy,
    "roc_auc": sk_roc_auc,
    "sk_precision_weighted_avg": sk_precision_weighted_avg,
    "sk_precision_perclass": sk_precision_perclass,
    "sk_recall_weighted_avg": sk_recall_weighted_avg,
    "sk_recall_perclass": sk_recall_perclass,
    "sk_f1_weighted_avg": sk_f1_weighted_avg,
    "sk_f1_perclass": sk_f1_perclass,
    "importance_by_permutation": importance_by_permutation,
    "correct_samples": correct_samples,
    "incorrect_samples": incorrect_samples
}
