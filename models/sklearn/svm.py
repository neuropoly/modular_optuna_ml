"""
Contains default implementations for a number of support vector models within SciKitLearn for use in this framework
"""
from optuna import Trial
from sklearn.svm import SVC

from models.sklearn.base import SciKitLearnModelManager


class SVCManager(SciKitLearnModelManager[SVC]):
    """
    Optuna model manager for the SVC class in SciKit-Learn
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tune(self, trial: Trial):
        # Get the kernel for this trial
        kernel = self.trial_closures['kernel'](trial)

        # Get the C value for this trial
        c = self.trial_closures['c'](trial)

        # Return the resulting model
        self._model = SVC(C=c, kernel=kernel, probability=True)

        # TODO: Extend this with more parameters

    def predict_proba(self, x):
        if self._model.probability:
            return self._model.predict_proba(x)
        else:
            # Local import to avoid a global one potentially polluting the namespace during "healthy" runs
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False"
            )
