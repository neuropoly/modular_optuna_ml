from abc import ABC, abstractmethod
from typing import Any

from optuna import Trial, Study


class OptunaModelManager(ABC):
    """
    A factory for producing Optuna-tunable models
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

    def __init__(self, **kwargs):
        # By default, just parse all key-word arguments into a 'trial_closures' parameter to manage later
        self.trial_closures = {}
        for k, v in kwargs.items():
            self.trial_closures[k] = OptunaModelManager.optuna_trial_param_parser(k, v)

    @abstractmethod
    def get_model_type(self):
        """
        Return the model type associate with this class, for validation’s sake
        """
        return None

    @abstractmethod
    def build_model(self, trial: Trial):
        """
        Build a corresponding model using parameters derived from an Optuna trial.
        :param trial: The Optuna trial to generate params from
        :return: A model generated using the trial
        """
        return None
