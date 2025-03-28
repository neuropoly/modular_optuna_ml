"""
Contains default implementations for a number of linear models within SciKitLearn for use in this framework
"""

from optuna import Trial
from sklearn.linear_model import LogisticRegression

from models.sklearn.base import SciKitLearnModelManager
from tuning.utils import TunableParam


class LogisticRegressionManager(SciKitLearnModelManager[LogisticRegression]):
    """
    Optuna model manager for the LogisticRegression class in SciKit-Learn.

    NOTE: Due to the need to avoid chaining tunable parameters (as configuring such a use case would be infeasible),
        this implementation is a more involved on than most SciKit-Learn models. This is most apparent in the case where
        both L1 and L2 regularization are used (an 'elasticnet' regression). In this case, L1 and L2 are still both
        tuned by Optuna, but rather than being fed to the model directly (which would result in crash) are instead used
        to construct the parameters which are actually required by SciKit-Learn's ElasticNet implementation (being the
        scaler 'C' and ratio of l1 and l2 'l1_ratio'). Mathematically, these are equivalent, but should be noted if you
        wish to utilize the hyperparameters of the "best" model identified by this tool within SciKit-Learn!

    Example Usage:
    {
        "label": "LogisticRegression",
        "model": "LogisticRegression",
        "parameters": {
            "penalty": {
                  "label": "penalty",
                  "type": "categorical",
                  "choices": ["l1", "l2", "elasticnet", null]
            },
            "solver": "saga",
            "l1_c": {
                  "label": "l1",
                  "type": "float",
                  "log": true,
                  "low": 1e-3,
                  "high": 1e3
            },
            "l2_c": {
                  "label": "l2",
                  "type": "float",
                  "log": true,
                  "low": 1e-3,
                  "high": 1e3
            }
        }
    }
    """
    def tune(self, trial: Trial):
        def tune_and_get(key: str):
            param = self.params[key]
            if isinstance(param, TunableParam):
                return param.trial_closure(trial)
            else:
                return param

        # Get the solver to use for this model
        solver = tune_and_get('solver')

        # Get the currently requested model penalty, as it determines how to proceed
        penalty = tune_and_get('penalty')

        # If the penalty is none, just return the model without further params
        if penalty is None:
            self._model = LogisticRegression(penalty=penalty, solver=solver)
        # If the penalty is l1 or l2, assign the C of the model to the respective value
        elif penalty == 'l1':
            l1_c = tune_and_get('l1_c')
            self._model = LogisticRegression(penalty=penalty, C=l1_c, solver=solver)
        # If the penalty is l1, use the corresponding weight param
        elif penalty == 'l2':
            l2_c = tune_and_get('l2_c')
            self._model = LogisticRegression(penalty=penalty, C=l2_c, solver=solver)
        # ElasticNet needs a bit more management to convert the pair of l1 and l2 C values
        elif penalty == 'elasticnet':
            # Get the respective C values
            l1_c = tune_and_get('l1_c')
            l2_c = tune_and_get('l2_c')
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
