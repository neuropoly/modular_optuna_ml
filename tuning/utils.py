from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

from optuna import Trial


class TunableParam:
    TrialClosure = Callable[[Trial], Any]
    """
    Defines a parameter which can be tuned by Optuna during model optimization.

    In essence, this is just a wrapper for a float, int, or string which is not evaluated until requested by this
        framework (usually during Optuna parameter tuning). Once it is requested to be evaluated, it will request the
        provided Optuna Trial object provide a value, which will then be used to as this parameter's value until 
        another tuning request is made.
        
    This class is meant to be used in conjunction with the "Tunable" mixin, which provides standardized utility 
        function for (re-)evaluating these parameters when an Optuna study needs it to be.
        
    This class also managed the following for itself:
        * Configuration parsing (including the type and range of values which Optuna can sample for each parameter)
        * How the parameter's value should be tracked and managed by the backing SQLite database. Due to how SQL updates
            are coded, however, saving the parameter's values is managed the StudyManager instead.
    """

    unlabelled_idx = 0

    def __init__(self, label: str, trial_closure: TrialClosure, db_type: str):
        # Track our own label
        self.label = label

        # Track the tuning closure
        self.trial_closure = trial_closure

        # Track the DB type of the parameter
        self.db_type = db_type

        # The current value of this parameter
        self.value = None

    @classmethod
    def from_config_entry(cls, config_dict: dict):
        label = config_dict.pop('label', None)
        if label is None:
            label = f"Unlabeled {cls.unlabelled_idx}"
            cls.unlabelled_idx += 1
        trial_closure, db_type = cls.parse_tunable(label, config_dict)
        return cls(label=label, trial_closure=trial_closure, db_type=db_type)

    @classmethod
    def parse_tunable(cls, label: Optional[str], params: dict | list | int | float | str) -> (TrialClosure, str):
        """
        Generates a closure function capable of generating Optuna-compatible values when provided an Optuna trial
        :param label: The label this closure should have, used by Optuna when logging results
        :param params: The parameters used to determine how new model hyperparameters will be generated for a given trial
        :return: The trial closure derived from the provided parameter data, along with the corresponding DB type
        """
        # If the parameter is a float or int range, it will be in the config as a dictionary
        param_type = type(params)
        if param_type is dict:
            param_type = params.get('type')
            if param_type == 'float':
                low_val = params.get('low')
                high_val = params.get('high')
                return lambda t: t.suggest_float(
                    name=label,
                    low=low_val,
                    high=high_val,
                    step=params.get('step', None),
                    log=params.get('log', False)
                ), 'REAL'
            elif param_type == 'int':
                low_val = params.get('low')
                high_val = params.get('high')
                return lambda t: t.suggest_int(
                    name=label,
                    low=low_val,
                    high=high_val,
                    step=params.get('step', 1),
                    log=params.get('log', False)
                ), 'INTEGER'
            elif param_type == 'categorical':
                choices = params.get('choices')
                return lambda t: t.suggest_categorical(
                    name=label,
                    choices=choices
                ), 'TEXT'
            elif param_type == 'constant':
                value = params.get('value')
                return lambda t: t.suggest_categorical(
                    name=label,
                    choices=[value]
                ), 'TEXT'
        # Otherwise warn the user and return None
        print(f"WARNING: parameter '{label}' an invalid type for a tunable parameter!")
        return None, None

    def tune(self, trial: Trial):
        self.value = self.trial_closure(trial)


class Tunable(ABC):
    """
    Mixin which denotes that an object can be tuned via an Optuna Study using parameters derived from an Optuna Trial.

    Classes which utilize this mixin should generally contain (or have the potential to contain) at least one
        instance of a `TunableParameter` class, whose values will be (re-)evaluated during the `tune` function.
        Likewise, this class must be able to return a list of the TunableParameters it manages, so that other elements
        of this framework (such as the Study) can track them for use in its own utilities (such as database management)
    """
    def __init__(self, label: str = '', **kwargs):
        self.label = label

    @abstractmethod
    def tune(self, trial: Trial):
        """
        Updates the state of this object using parameters from an Optuna trial
        """
        ...

    @abstractmethod
    def tunable_params(self) -> Iterable[TunableParam]:
        ...
