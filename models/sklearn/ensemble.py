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
        # Get the number of neighbors which should be used
        n_estimators = self.trial_closures['n_estimators'](trial)

        # Get the weighting scheme to use for this model
        learning_rate = self.trial_closures['learning_rate'](trial)

        # Return the resulting model
        self._model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm="SAMME")

class RandomForestClassifierManager(SciKitLearnModelManager[RandomForestClassifier]):
    """
    Optuna model manager for the RandomForestClassifier class in SciKit-Learn
    """
    def tune_model(self, trial: Trial):
        # Get the criterion to use decision tree splits
        criterion = self.trial_closures['criterion'](trial)

        # Get the minimum number of samples that need to be in each split in a decision tree
        min_samples_split = self.trial_closures['min_samples_split'](trial)

        # Get the maximum number of features that can be used by a tree at each step
        max_features = self.trial_closures['max_features'](trial)

        # Return the resulting model
        self._model = RandomForestClassifier(criterion=criterion, min_samples_split=min_samples_split, max_features=max_features)
