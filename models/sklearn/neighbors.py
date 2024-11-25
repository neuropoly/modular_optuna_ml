"""
Contains default implementations for a number of nearest-neighbor models within SciKitLearn for use in this framework
"""

from optuna import Trial
from sklearn.neighbors import KNeighborsClassifier

from models.sklearn.base import SciKitLearnModelManager


class KNeighborsClassifierManager(SciKitLearnModelManager[KNeighborsClassifier]):
    """
    Optuna model manager for the KNeighborsClassifier class in SciKit-Learn
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tune(self, trial: Trial):
        # Tune this
        super().tune(trial)

        # Rebuild the model using the newly tuned parameters
        model_kwargs = {k: self.evaluate_param(k) for k in self.params.keys()}
        self._model = KNeighborsClassifier(**model_kwargs)