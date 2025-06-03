from logging import Logger
from typing import Self, Optional

import pandas as pd
from sklearn.impute import SimpleImputer

from config.utils import default_as, is_list, parse_data_config_entry
from data import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import FittedDataHook


@registered_data_hook("imputation_simple")
class SimpleImputation(FittedDataHook):
    """
    Imputes missing data through "simple" strategies, including mean, median, most common, and constant fill.

    Utilizes SciKit-Learn's SimpleImputer implementation. As such, aside from the addition of a "features" argument
        to allow the user to specify a set of feature they want the imputation applied to, its implementation and
        use here is identical.

    Example usage:
    {
      "type": "imputation_simple",
      "features": ["color", "species"],
      "strategy": "most_frequent"
    }
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config=config, **kwargs)
        # Grab an explicit list of columns, if they were defined
        self.features = parse_data_config_entry(
            "features", config,
            default_as([], self.logger), is_list(self.logger)
        )

        # Pass the rest of the arguments through to SimpleEncoder directly; THIS IS TEMPORARY
        self.backing_encoder = SimpleImputer(**config)

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config, logger=logger)

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # If this is multi-feature dataset, sub-features can be selected
        selected_features = self.features
        if not selected_features:
            selected_features = x.features()
            # The following check is done because Pandas won't allow loc-based assignment if the query is an index
            if isinstance(selected_features, pd.Index):
                selected_features = selected_features.to_numpy()
        # Fit to and transform the training data first
        tmp_train = x.get_features(selected_features)
        tmp_train = self.backing_encoder.fit_transform(tmp_train.as_array())
        if hasattr(tmp_train, "todense"):
            tmp_train = tmp_train.todense()
        x_out = x.set_features(selected_features, tmp_train)

        return x_out

    def run_fitted(
            self,
            x_train: BaseDataManager,
            x_test: Optional[BaseDataManager],
            y_train: Optional[BaseDataManager] = None,
            y_test: Optional[BaseDataManager] = None
    ) -> (BaseDataManager, BaseDataManager):
        # If this is multi-feature dataset, sub-features can be selected
        selected_features = self.features
        if not selected_features:
            selected_features = x_train.features()
            # The following check is done because Pandas won't allow loc-based assignment if the query is an index
            if isinstance(selected_features, pd.Index):
                selected_features = selected_features.to_numpy()
        # Fit to and transform the training data first
        tmp_train = x_train.get_features(selected_features)
        tmp_train = self.backing_encoder.fit_transform(tmp_train.as_array())
        if hasattr(tmp_train, "todense"):
            tmp_train = tmp_train.todense()
        train_out = x_train.set_features(selected_features, tmp_train)

        # Then ONLY transform the testing data
        tmp_test = x_test.get_features(selected_features)
        tmp_test = self.backing_encoder.transform(tmp_test.as_array())
        if hasattr(tmp_test, "todense"):
            tmp_test = tmp_test.todense()
        test_out = x_test.set_features(selected_features, tmp_test)

        return train_out, test_out