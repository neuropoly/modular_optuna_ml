"""
Contains default implementations for a number of ensemble models within SciKitLearn for use in this framework
"""
from optuna import Trial
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from models.sklearn.base import SciKitLearnModelManager


class AdaBoostClassifierManager(SciKitLearnModelManager[AdaBoostClassifier]):
    """
    Optuna model manager for the AdaBoostClassifier class in SciKit-Learn.

    Will always use the SAMME algorithm, both as it is the standard default of SciKit-Learn, and because it is able
        to function implicitly with Optuna parameter tuning of various. Once the algorithm parameter is fully
        deprecated in SciKit-Learn v1.8, this will be removed as well

    Example Usage:
    {
        "label": "adaboost",
        "model": "AdaBoostClassifier",
        "parameters": {
            "n_estimators": {
                "label": "ada_n_estimators",
                "type": "int",
                "low": 10,
                "high": 100
            },
            "learning_rate": {
                "label": "ada_learning_rate",
                "type": "float",
                "log": true,
                "low": 1e-3,
                "high": 1e3
            }
        }
    }
    """
    def tune(self, trial: Trial):
        # Tune this model based on the trial parameters
        super().tune(trial)

        # Rebuild the model using the newly tuned parameters
        model_kwargs = {k: self.evaluate_param(k) for k in self.params.keys()}
        self._model = AdaBoostClassifier(**model_kwargs, algorithm="SAMME")

class RandomForestClassifierManager(SciKitLearnModelManager[RandomForestClassifier]):
    """
    Optuna model manager for the RandomForestClassifier class in SciKit-Learn

    Example usage:
    {
        "label": "rf_classifier",
        "model": "RFC",
        "parameters": {
            "criterion": {
                "label": "rfc_criterion",
                "type": "categorical",
                "choices": ["gini", "entropy", "log_loss"]
            },
            "min_samples_split": {
                "label": "rfc_min_samples_split",
                "type": "int",
                "low": 2,
                "high": 12
            },
            "max_features": {
                "label": "rfc_max_features",
                "type": "categorical",
                "choices": [10, "sqrt", "log2"]
            }
        }
    }
    """
    def tune(self, trial: Trial):
        # Tune this model based on the trial parameters
        super().tune(trial)

        # Rebuild the model using the newly tuned parameters
        model_kwargs = {k: self.evaluate_param(k) for k in self.params.keys()}
        self._model = RandomForestClassifier(**model_kwargs)
