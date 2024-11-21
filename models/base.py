from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from tuning.utils import Tunable, parse_tunable

T = TypeVar('T')

class OptunaModelManager(Tunable, Generic[T], ABC):
    """
    An abstract class which should be subclassed and implemented for all machine learning models which want automated
    hyperparameter tuning within this framework.
    """
    _type_T: type[T]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Check if this is a properly instantiated class w/ a declared generic type
        if orig_bases := cls.__dict__.get('__orig_bases__', False):
            # If it is, and there is a type declared, track it; ignore the warnings, PyCharm is just being stupid here
            if len(orig_bases) > 0:
                cls._type_T = orig_bases[0]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # By default, just parse all key-word arguments into a 'trial_closures' parameter to manage later
        self.trial_closures = {}
        for k, v in kwargs.items():
            self.trial_closures[k] = parse_tunable(k, v)

    @abstractmethod
    def get_model(self) -> T:
        """
        Returns current model instance managed by this manager, if one exists
        """
        ...

    def get_type(self):
        return self._type_T

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the model managed by this model manager to the data provided
        :param x: The data features to fit
        :param y: The data target to fit
        """
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate the predictions from a model of the type managed by this class
        :param model: The model to generate predictions from
        :param x: The data for said model to use to generate the predictions, in NP-array-like format
        :return: The generated predictions, in a np-like array
        """
        ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the (pseudo-) probability of each class the model has been tasked with. Optional implementation, as
        not all OptunaModelManagers manage categorical models
        :param model: The model to use to generate the probability estimates
        :param x: The features the model should use to calculate said probabilities.
        :return: The generated probability estimates
        """
        raise NotImplementedError(f"'{type(self)}' has not implemented the 'predict_proba' function")
