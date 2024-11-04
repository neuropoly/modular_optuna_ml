from optuna import Trial
from sklearn.ensemble import AdaBoostClassifier

from models.sklearn.base import SciKitLearnModelManager


class AdaBoostClassifierManager(SciKitLearnModelManager[AdaBoostClassifier]):
    """
    Optuna model manager for the KNeighborsClassifier class in SciKit-Learn
    """
    def build_model(self, trial: Trial):
        # Get the number of neighbors which should be used
        n_estimators = self.trial_closures['n_estimators'](trial)

        # Get the weighting scheme to use for this model
        learning_rate = self.trial_closures['learning_rate'](trial)

        # Return the resulting model
        return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm="SAMME")
