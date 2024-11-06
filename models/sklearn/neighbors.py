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
    def build_model(self, trial: Trial):
        # Get the number of neighbors which should be used
        n_neighbors = self.trial_closures['n_neighbors'](trial)

        # Get the weighting scheme to use for this model
        weights = self.trial_closures['weights'](trial)

        # Get the p value for this trial
        p = self.trial_closures['p'](trial)

        # Return the resulting model
        return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
