from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar

import numpy as np
from optuna import Trial

from tuning.utils import Tunable, TunableParam

T = TypeVar('T')

class OptunaModelManager(Tunable, Generic[T], ABC):
    """
    An abstract class which should be subclassed and implemented for all machine learning models which want automated
    hyperparameter tuning within this framework.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # By default, just parse all key-word arguments into a 'params' dict to manage later
        self.params: dict[str, Any] = {}
        self._tunable_params: list[TunableParam] = []
        for k, v in kwargs.items():
            if isinstance(v, dict):
                new_param = TunableParam.from_config_entry(v)
                self.params[k] = new_param
                self._tunable_params.append(new_param)
            else:
                self.params[k] = v

    @abstractmethod
    def get_model(self) -> T:
        """
        Returns current model instance managed by this manager, if one exists
        """
        ...

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the model managed by this model manager to the data provided
        :param x: The data features to fit
        :param y: The data target to fit
        """
        ...

    def evaluate_param(self, key: str):
        """
        Evaluates a given parameter explicitly, usually right before the model is instantiated and about to be trained.

        This function mainly exists to provide a way for use to define parameters which should be tuned by Optuna,
            and request they be tuned at specific points of the run-time.

        :param key: The key of the parameter we want to evaluate.
        """
        param = self.params[key]
        if isinstance(param, TunableParam):
            return param.value
        else:
            return param

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

    def tune(self, trial: Trial):
        # Tune all tunable parameters
        for p in self.tunable_params():
            p.tune(trial)

    def tunable_params(self) -> Iterable[TunableParam]:
        # By default, just report the labels for all tracked tunable params
        return self._tunable_params
