from optuna import Trial, Study
from sklearn.linear_model import LogisticRegression

from models.utils import OptunaModelFactory

class LogisticRegressionFactory(OptunaModelFactory):
    """
    Optuna model factory for the LogisticRegression class in SciKit-Learn
    """
    def get_model_type(self):
        return LogisticRegression

    def build_model(self, trial: Trial):
        # Get the currently requested model penalty, as it determines how to proceed
        penalty = self.trial_closures['penalty'](trial)

        # If the penalty is none, just return the model without further params
        if penalty is None:
            return LogisticRegression(penalty=penalty)
        # If the penalty is l1 or l2, assign the C of the model to the respective value
        elif penalty is 'l1':
            l1_c = self.trial_closures['l1_c'](trial)
            return LogisticRegressionFactory(penalty=penalty, C=l1_c)
        # If the penalty is l1, use the corresponding weight param
        elif penalty is 'l2':
            l2_c = self.trial_closures['l2_c'](trial)
            return LogisticRegressionFactory(penalty=penalty, C=l2_c)
        # ElasticNet needs a bit more management to convert the pair of l1 and l2 C values
        elif penalty is 'elasticnet':
            # Get the respective C values
            l1_c = self.trial_closures['l1_c'](trial)
            l2_c = self.trial_closures['l2_c'](trial)
            # Calculate the l1 ratio...
            l1_ratio = l1_c / (l1_c + l2_c)
            # ... and use it to calculate the effective C value
            c = l1_c / l1_ratio
            # Return the resulting model
            return LogisticRegressionFactory(penalty=penalty, C=c, l1_ratio=l1_ratio)

        # If the above checks fail, raise an error
        raise ValueError("""
        Invalid penalty term used for LogisticRegression! 
        Please make sure all values for the 'penalty' value in your ML config are one of the following:
        ['l1', 'l2', 'elasticnet', null]
        """)
