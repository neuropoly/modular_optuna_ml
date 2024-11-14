from abc import ABC
from typing import TypeVar

import numpy as np

from models.base import OptunaModelManager

T = TypeVar('T')
class SciKitLearnModelManager(OptunaModelManager[T], ABC):
    """
    Simple wrapper for SciKit-Learn based model managers
    """

    def predict(self, model: T, x: np.ndarray):
        """
        Use SciKit-Learn's standardized 'predict' function by default
        """
        return model.predict(x)

    def predict_proba(self, model: T, x: np.ndarray):
        """
        Use SciKit-Learn's standardized 'predict_proba' function by default
        """
        return model.predict_proba(x)