"""
Contains default implementations for a number of support vector models within SciKitLearn for use in this framework
"""
from optuna import Trial
from sklearn.svm import SVC

from models.sklearn.base import SciKitLearnModelManager


class SVCManager(SciKitLearnModelManager[SVC]):
    """
    Optuna model manager for the SVC class in SciKit-Learn.

    Example Usage:
    {
        "label": "svm_classifier",
        "model": "SVC",
        "parameters": {
            "kernel": {
                "label": "svc_kernel",
                "type": "categorical",
                "choices": ["linear", "poly", "rbf", "sigmoid"]
            },
            "C": {
                "label": "svc_C",
                "type": "float",
                "log": true,
                "low": 1e-3,
                "high": 1e3
            }
        }
    }
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tune(self, trial: Trial):
        # Tune this model based on the trial parameters
        super().tune(trial)

        # Rebuild the model using the newly tuned parameters
        model_kwargs = {k: self.evaluate_param(k) for k in self.params.keys()}
        self._model = SVC(**model_kwargs, probability=True)

    def predict_proba(self, x):
        if self._model.probability:
            return self._model.predict_proba(x)
        else:
            # Local import to avoid a global one potentially polluting the namespace during "healthy" runs
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False"
            )
