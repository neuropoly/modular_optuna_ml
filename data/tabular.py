from itertools import chain
from logging import Logger
from pathlib import Path
from typing import Iterable, Optional, Self

import numpy as np
import pandas as pd

from config.utils import as_str, default_as, is_file, is_list, parse_data_config_entry
from data.base import BaseDataManager, registered_datamanager
from data.hooks import DATA_HOOKS
from data.hooks.base import DataHook, FittedDataHook
from tuning.utils import Tunable


@registered_datamanager("tabular")
class TabularDataManager(BaseDataManager):
    """
    Data manager for data stored in tabular data formats (i.e. 'csv')

    Uses a Panda's dataframe as a backend to read the files and manage the majority of data queries and modifications.
    """
    def __init__(self, **kwargs):
        # Don't use this constructor directly; use 'from_config' instead
        super().__init__(**kwargs)

        # Default parameters
        self._data: Optional[pd.DataFrame] = None
        self._sep: str = ','
        self._index: Optional[str] = None

        # Hook storing variables for later
        self.pre_split_hooks: list[DataHook] = []
        self.post_split_hooks: list[DataHook] = []
        self.tunable_hooks: list[Tunable | DataHook] = []

    def __len__(self):
        return self.data.shape[0]

    @property
    def data(self) -> pd.DataFrame:
        """
        Managed as a pseudo-cached property to allow for lazy loading
        """
        # If our data is still just a reference to a file, load it with pandas
        if isinstance(self._data, Path):
            if self._index:
                self._data = pd.read_csv(self._data, sep=self._sep, index_col=self._index)
            else:
                self._data = pd.read_csv(self._data, sep=self._sep)
        # Return the result
        return self._data

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root):
        # Initiate a new instance of this class
        new_instance = cls(logger=logger)

        # Track the data source for, making sure it's properly formatted
        new_instance._data = parse_data_config_entry(
            "data_source", config, as_str(logger), is_file(logger)
        )
        # Track the data separator for later
        default_sep = default_as(new_instance._sep, logger)
        new_instance._sep = parse_data_config_entry(
            "separator", config, default_sep, as_str(logger)
        )
        # Track the index, if one was provided
        new_instance._index = parse_data_config_entry(
            "index", config, default_as(None, logger)
        )

        # Retrieve and parse the pre-split data hooks
        pre_split_hooks = parse_data_config_entry(
            "pre_split_hooks", config, default_as([], logger), is_list(logger)
        )
        for hook_config in pre_split_hooks:
            # Interpret the configuration file for this hook to ensure it is a valid type
            hook_label = hook_config.pop('type')
            hook_cls = DATA_HOOKS.get(hook_label, None)
            # If no hook of the type queried was found, raise an error and return
            if hook_cls is None:
                raise ValueError(f"Could not find a registered data hook of type '{hook_label}'; terminating.")

            # If the hook was not stateless, warn the user
            if issubclass(hook_cls, FittedDataHook):
                logger.warning(
                    f"Pre-split hook '{hook_cls.__name__}' is designed to fit to one dataset, then apply to another; "
                    f"this is likely to result in data overfitting! Are you sure this was intended?"
                )

            # Attempt to instantiate the hook type based on the configs contents
            new_hook = hook_cls.from_config(config=hook_config, logger=logger)

            # Save the results
            new_instance.pre_split_hooks.append(new_hook)

        # Retrieve and parse the post-split data hooks
        post_split_hooks = parse_data_config_entry(
            "post_split_hooks", config, default_as([], logger), is_list(logger)
        )
        for hook_config in post_split_hooks:
            hook_label = hook_config.pop('type')
            hook_cls = DATA_HOOKS.get(hook_label, None)

            # If no hook of the type queried was found, raise an error and return
            if hook_cls is None:
                raise ValueError(f"Could not find data hook of type '{hook_label}'; terminating.")

            # If the hook was not stateless, warn the user
            if not issubclass(hook_cls, FittedDataHook):
                logger.warning(f"Post-split hook '{hook_cls.__name__}' is not fitted; "
                               f"are you sure you wanted to run it post-split?")

            # Attempt to instantiate the hook type based on the configs contents
            new_hook = hook_cls.from_config(config=hook_config, logger=logger)

            # Save the results
            new_instance.post_split_hooks.append(new_hook)

        # Identify any hooks which can be tuned, so they can be tuned upon request
        for h in chain(new_instance.pre_split_hooks, new_instance.post_split_hooks):
            if isinstance(h, Tunable):
                new_instance.tunable_hooks.append(h)

        return new_instance

    def get_index(self) -> np.array:
        return self.data.index.to_numpy()

    def get_samples(self, idx) -> Self:
        sub_df = self.data.iloc[idx, :]
        if isinstance(sub_df, pd.Series):
            # This BS is needed because Pandas auto-casts single index queries to Series
            sub_df = pd.DataFrame(data=sub_df).T
        new_instance = self.shallow_copy()
        new_instance._data = sub_df
        return new_instance

    def get_features(self, idx) -> Self:
        # Creates a new TabularDataManager with only the queried features
        sub_df = self.data.loc[:, idx]
        if isinstance(sub_df, pd.Series):
            # This BS is needed because Pandas auto-casts single index queries to Series
            sub_df = pd.DataFrame(data=sub_df)
        new_instance = self.shallow_copy()
        new_instance._data = sub_df
        return new_instance

    def set_features(self, idx, new_data) -> Self:
        """Creates a new TabularDataManager with the modified features in place"""
        new_df = self.data.copy()
        new_df.loc[:, idx] = new_data
        new_instance = self.shallow_copy()
        new_instance._data = new_df
        return new_instance

    def drop_features(self, idx) -> Self:
        new_df = self.data.copy()
        new_df = new_df.drop(columns=idx)
        new_instance = self.shallow_copy()
        new_instance._data = new_df
        return new_instance

    def n_features(self) -> int:
        return self.data.shape[1]

    def features(self) -> Iterable[str]:
        return self.data.columns

    def as_array(self) -> np.ndarray:
        return self.data.to_numpy()

    def pre_split(self, is_cross: bool, target: Self = None) -> Self:
        new_instance = self
        for hook in self.pre_split_hooks:
            # Skip a hook if it has specified it should only be run during a hook point we are not in
            if not is_cross and hook.run_per_cross:
                continue
            elif is_cross and hook.run_per_replicate:
                continue

            # If not, apply the hook to the data
            new_instance = hook.run(new_instance, target)

        # Return the results
        return new_instance

    def split(
            self,
            train_idx: np.ndarray,
            test_idx: np.ndarray,
            train_target: Self,
            test_target: Self,
            is_cross: bool = True
    ) -> (Self, Self):
        # Initial split and instance setup
        train_instance = self.shallow_copy()
        train_instance._data = self.data.iloc[train_idx, :]
        test_instance = self.shallow_copy()
        test_instance._data = self.data.iloc[test_idx, :]

        # Run at data processing hooks the user requested
        for hook in self.post_split_hooks:
            # Skip a hook if it has specified it should only be run during a hook point we are not in
            if is_cross and not hook.run_per_cross:
                continue
            elif not is_cross and not hook.run_per_replicate:
                continue

            # If the hook needs to be fit, use the training set to do so
            if isinstance(hook, FittedDataHook):
                train_instance, test_instance = hook.run_fitted(
                    train_instance, test_instance, train_target, test_target
                )
            # Otherwise, just apply the hook to both instances independently
            else:
                train_instance = hook.run(train_instance, train_target)
                test_instance = hook.run(test_instance, test_target)

        # Return the resulting split
        return train_instance, test_instance

    def shallow_copy(self) -> Self:
        # Simple utility to reduce the amount of boilerplate
        new_instance = self.__class__()
        new_instance._data = self._data
        new_instance._sep = self._sep
        new_instance.pre_split_hooks = self.pre_split_hooks
        new_instance.post_split_hooks = self.post_split_hooks
        new_instance.tunable_hooks = self.tunable_hooks

        return new_instance
