"""
Contains default implementations for a number of linear models within SciKitLearn for use in this framework
"""

from optuna import Trial
from sklearn.linear_model import LogisticRegression

from models.sklearn.base import SciKitLearnModelManager


class LogisticRegressionManager(SciKitLearnModelManager[LogisticRegression]):
    """
    Optuna model manager for the LogisticRegression class in SciKit-Learn
    """
    def tune(self, trial: Trial):
        # Get the solver to use for this model
        solver = self.trial_closures['solver'](trial)

        # Get the currently requested model penalty, as it determines how to proceed
        penalty = self.trial_closures['penalty'](trial)

        # If the penalty is none, just return the model without further params
        if penalty is None:
            self._model = LogisticRegression(penalty=penalty, solver=solver)
        # If the penalty is l1 or l2, assign the C of the model to the respective value
        elif penalty == 'l1':
            l1_c = self.trial_closures['l1_c'](trial)
            self._model = LogisticRegression(penalty=penalty, C=l1_c, solver=solver)
        # If the penalty is l1, use the corresponding weight param
        elif penalty == 'l2':
            l2_c = self.trial_closures['l2_c'](trial)
            self._model = LogisticRegression(penalty=penalty, C=l2_c, solver=solver)
        # ElasticNet needs a bit more management to convert the pair of l1 and l2 C values
        elif penalty == 'elasticnet':
            # Get the respective C values
            l1_c = self.trial_closures['l1_c'](trial)
            l2_c = self.trial_closures['l2_c'](trial)
            # Calculate the l1 ratio...
            l1_ratio = l1_c / (l1_c + l2_c)
            # ... and use it to calculate the effective C value
            c = l1_c / l1_ratio
            # Return the resulting model
            self._model = LogisticRegression(penalty=penalty, C=c, l1_ratio=l1_ratio, solver=solver)
        else:
            # If the above checks fail, raise an error
            raise ValueError(
                "Invalid penalty term used for LogisticRegression!\n",
                "Please make sure all values for the 'penalty' value in your ML config are one of the following:",
                "['l1', 'l2', 'elasticnet', null]"
            )
