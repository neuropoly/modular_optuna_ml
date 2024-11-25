from abc import ABC
from typing import TypeVar

import numpy as np

from models.base import OptunaModelManager

T = TypeVar('T')
class SciKitLearnModelManager(OptunaModelManager[T], ABC):
    """
    Simple wrapper for SciKit-Learn based model managers
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None

    def get_model(self) -> T:
        return self._model

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Use SciKit-Learn's standardized 'fit' function by default
        """
        self.get_model().fit(x, y)

    def predict(self, x: np.ndarray):
        """
        Use SciKit-Learn's standardized 'predict' function by default
        """
        return self.get_model().predict(x)

    def predict_proba(self, x: np.ndarray):
        """
        Use SciKit-Learn's standardized 'predict_proba' function by default
        """
        return self.get_model().predict_proba(x)