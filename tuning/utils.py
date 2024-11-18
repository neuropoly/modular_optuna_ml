from abc import ABC, abstractmethod

from optuna import Trial


class Tunable(ABC):
    """
    Simple mixing which denotes that an object can be tuned via an Optuna Study using
    parameters derived from an Optuna Trial.
    """
    @abstractmethod
    def tune(self, trial: Trial):
        """
        Updates the state of this object using parameters from an Optuna trial
        """
        ...

def parse_tunable(key: str, params: dict | list | int | float | str):
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
    # If it is a list, assume categorical by default
    elif param_type is list:
        return lambda t: t.suggest_categorical(
            name=key,
            choices=params
        )
    # If it is anything else, treat it as a constant
    else:
        return lambda t: t.suggest_categorical(
            name=key,
            choices=[params]
        )
    # Otherwise warn the user and return None
    print(f"WARNING: parameter '{key}' within the ML configuration was an invalid type!")
    return None