from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from optuna import Trial

T = TypeVar('T')

class OptunaModelManager(Generic[T], ABC):
    """
    An abstract class which should be subclassed and implemented for all machine learning models which want automated
    hyperparameter tuning within this framework.
    """
    @staticmethod
    def optuna_trial_param_parser(key: str, params: Any):
        """
        Generates a closure function capable of generating Optuna-compatible values when provided an Optuna trial
        :param key: The name/label this closure should have, used by Optuna when logging results
        :param params: The parameters used to determine how new model hyperparameters will be generated for a given trial
        :return: A closure function which takes an Optuna trial and returns a hyperparameter values samples from it
        """
        # If the parameter is a float or int range, it will be in the config as a dictionary
        param_type = type(params)
        if param_type is dict:
            param_type = params.get('type')
            if param_type == 'float':
                low_val = params.get('low')
                high_val = params.get('high')
                return lambda t: t.suggest_float(
                    name=key,
                    low=low_val,
                    high=high_val,
                    step=params.get('step', None),
                    log=params.get('log', False)
                )
            elif param_type == 'int':
                low_val = params.get('low')
                high_val = params.get('high')
                return lambda t: t.suggest_int(
                    name=key,
                    low=low_val,
                    high=high_val,
                    step=params.get('step', 1),
                    log=params.get('log', False)
                )
        # If it's a list of values, its categorical
        elif param_type is list:
            return lambda t: t.suggest_categorical(
                name=key,
                choices=params
            )
        # If it's a string, it's a single-choice and which should always be returned
        elif param_type is str:
            return lambda t: t.suggest_categorical(
                name=key,
                choices=[params]
            )
        # Otherwise warn the user and return None
        print(f"WARNING: parameter '{key}' within the ML configuration was an invalid type!")
        return None

    _type_T: type[T]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Check if this is a properly instantiated class w/ a declared generic type
        if orig_bases := cls.__dict__.get('__orig_bases__', False):
            # If it is, and there is a type declared, track it; ignore the warnings, PyCharm is just being stupid here
            if len(orig_bases) > 0:
                cls._type_T = orig_bases[0]

    def __init__(self, **kwargs):
        # By default, just parse all key-word arguments into a 'trial_closures' parameter to manage later
        self.trial_closures = {}
        for k, v in kwargs.items():
            self.trial_closures[k] = OptunaModelManager.optuna_trial_param_parser(k, v)

    @abstractmethod
    def get_model(self) -> T:
        """
        Returns current model instance managed by this manager, if one exists
        """
        ...

    def get_type(self):
        return self._type_T

    @abstractmethod
    def tune_model(self, trial: Trial):
        """
        Tunes the model managed by this manages using parameters derived from an Optuna trial.
        :param trial: The Optuna trial to generate params from
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
