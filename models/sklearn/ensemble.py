"""
Contains default implementations for a number of ensemble models within SciKitLearn for use in this framework
"""
from optuna import Trial
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from models.sklearn.base import SciKitLearnModelManager


class AdaBoostClassifierManager(SciKitLearnModelManager[AdaBoostClassifier]):
    """
    Optuna model manager for the AdaBoostClassifier class in SciKit-Learn
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
    """

    def tune(self, trial: Trial):
        # Tune this model based on the trial parameters
        super().tune(trial)

        # Rebuild the model using the newly tuned parameters
        model_kwargs = {k: self.evaluate_param(k) for k in self.params.keys()}
        self._model = RandomForestClassifier(**model_kwargs)
