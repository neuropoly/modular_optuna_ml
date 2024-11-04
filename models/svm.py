from optuna import Trial
from sklearn.svm import SVC

from models.utils import OptunaModelManager


class SVCManager(OptunaModelManager[SVC]):
    """
    Optuna model manager for the SVC class in SciKit-Learn
    """
    def build_model(self, trial: Trial):
        # Get the kernel for this trial
        kernel = self.trial_closures['kernel'](trial)

        # Get the C value for this trial
        c = self.trial_closures['c'](trial)

        # Return the resulting model
        return SVC(C=c, kernel=kernel, probability=True)

        # TODO: Extend this with more parameters

    def predict(self, model: SVC, x):
        return model.predict(x)

    def predict_proba(self, model: SVC, x):
        return model.predict_proba(x)
