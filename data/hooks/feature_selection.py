from abc import ABC
from logging import Logger
from typing import Optional, Self

import numpy as np
import pandas as pd
from optuna import Trial
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from config.utils import default_as, is_float, is_list, parse_data_config_entry
from data.base import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import DataHook, FittedDataHook
from tuning.utils import Tunable, TunableParam


### Explicit Feature Selection ###
class ExplicitFeatures(DataHook, ABC):
    """
    Abstract DataHook which allows the user to explicitly define a set of features they want the hook applied too,
        via the "features" argument.
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Get the list of features to parse
        features = parse_data_config_entry(
            "features", config,
            is_list(self.logger)
        )
        self.features = features

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        new_instance = cls(config, logger=logger)
        return new_instance

@registered_data_hook("drop_features_explicit")
class ExplicitDrop(ExplicitFeatures):
    """
    Data Hook which drops any features the user specifies from the dataset outright.

    Example usage:
    {
      "type": "drop_features_explicit",
      "features": ["foo", "bar"]
    }
    """
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # We can only drop features in a dataset if they have more than 1
        if x.n_features() == 1:
            raise TypeError(f"DataManager of type '{self.__class__}' only has one feature! Refused to make an empty dataset.")
        return x.drop_features(self.features)

@registered_data_hook("keep_features_explicit")
class ExplicitKeep(ExplicitFeatures):
    """
    Data Hook which drops any features NOT specified by the user outright.

    Example usage:
    {
      "type": "keep_features_explicit",
      "features": ["baz", "bing"]
    }
    """
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # We can only drop features in a dataset if they have more than 1
        if x.n_features() == 1:
            raise TypeError(f"DataManager of type '{self.__class__}' only has one feature! Refused to make an empty dataset.")
        return x.get_features(self.features)


### Feature Selection by Null Content ###
class NullityDrop(DataHook, ABC):
    """
    Abstract data hook for dropping some component of the data based on a null "threshold"
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)

        # Get the threshold for how to generate this instance
        self.threshold = parse_data_config_entry(
            "threshold", config,
            # TODO: Add min and max checks here as well
            default_as(0.5, self.logger), is_float(self.logger)
        )

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config=config, logger=logger)

@registered_data_hook("sample_drop_null")
class SampleNullityDrop(NullityDrop):
    """
    Data hook which will automatically remove samples in the dataset which contain more than some threshold amount of
        NA values. For example, with a threshold of 0.5, any samples which are missing more than half of their
        features are dropped from the dataset.

    Example usage:
    {
      "type": "sample_drop_null",
      "threshold": 0.5
    }
    """
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # Iterate through all samples to calculate their null content, dropping any above the configured threshold
        null_tolerance = int(x.n_features() * self.threshold)
        keep_idx = [i for i, k in enumerate(x) if np.sum(pd.isnull(k.as_array())) < null_tolerance]
        return x[keep_idx]

@registered_data_hook("feature_drop_null")
class FeatureNullityDrop(NullityDrop):
    """
    Data hook which will automatically remove features in the dataset which contain more than some threshold amount of
        null values. For example, with a threshold of 0.5, any feature whose sample's values are missing more than half
        of the time are dropped from the dataset.

    Example usage:
    {
      "type": "feature_drop_null",
      "threshold": 0.5
    }
    """
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # Otherwise, calculate the null content for each feature dropping any which surpass the configured threshold
        n_samples = len(x)
        null_tolerance = int(n_samples * self.threshold)
        drop_idx = [k for k in x.features()
                    if np.sum(pd.isnull(x.get_features(k).as_array())) > null_tolerance]
        if len(drop_idx) == x.n_features():
            raise IndexError(f"Data Hook of class `feature_drop_null` tried to drop all remaining features in a dataset! "
                             f"Consider reducing the threshold somewhat, or doing imputation instead.")
        return x.drop_features(drop_idx)


### Principal Component Analysis ###
@registered_data_hook("principal_component_analysis")
class PrincipalComponentAnalysis(Tunable, FittedDataHook):
    """
    This data hook will transform the set of features provided to it into their representative Principal Components,
        replacing the original set of features in the process. This is an extended implementation of SciKit-Learn's
        implementation, which allows for the proportion of components being preserve to be tuned by Optuna dynamically;
        it otherwise works identically to said implementation with default parameters.

    Example usage:
    {
      "type": "principal_component_analysis",
      "proportion": {
        "label": "pca_component_prop",
        "type": "float",
        "low": 0.1,
        "high": 0.9
      }
    }
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config=config, **kwargs)
        super(FittedDataHook, self).__init__(config=config, **kwargs)

        # Grab the proportion of features to select; defaults to 70%
        select_prop = config.get("proportion", {
            "label": "proportion",
            "type": "constant",
            "value": 0.7
        })
        self.prop_tuner: TunableParam = TunableParam.from_config_entry(select_prop)

        # Keep tabs on a backing instance for later user
        # TODO: Preserve the remaining config to allow other PCA-related params to be specified by the user
        self.backing_pca: PCA | None = None


    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config=config, logger=logger)

    def tune(self, trial: Trial):
        self.prop_tuner.tune(trial)
        # Generate the new backing model based on this setup
        self.backing_pca = PCA(n_components=self.prop_tuner.value)

    def tunable_params(self) -> list[TunableParam]:
        return [self.prop_tuner]

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # Calculate and regenerate the features in the training set
        tmp_train = self.backing_pca.fit_transform(x.as_array())
        feature_labels = [f'PC{i}' for i in range(tmp_train.shape[1])]

        # Drop all features and replace them with the new components
        x_out = x.drop_features(x.features())
        x_out = x_out.set_features(feature_labels, tmp_train)

        return x_out

    def run_fitted(self,
               x_train: BaseDataManager,
               x_test: Optional[BaseDataManager],
               y_train: Optional[BaseDataManager] = None,
               y_test: Optional[BaseDataManager] = None
           ) -> (BaseDataManager, BaseDataManager):
        # Run the fitted analysis first
        tmp_train = self.backing_pca.fit_transform(x_train.as_array())
        feature_labels = [f'PC{i}' for i in range(tmp_train.shape[1])]

        # Drop all features and replace them with the new components
        train_out = x_train.drop_features(x_train.features())
        train_out = train_out.set_features(feature_labels, tmp_train)

        # Do the same to our testing data, but without re-fitting
        tmp_test = self.backing_pca.transform(x_test.as_array())
        test_out = x_test.drop_features(x_test.features())
        test_out = test_out.set_features(feature_labels, tmp_test)

        return train_out, test_out


### Recursive Feature Elimination ###
@registered_data_hook("recursive_feature_elimination")
class RecursiveFeatureElimination(Tunable, FittedDataHook):
    """
    This data hook will remove the features which have the least impact on a LogisticRegression's performance,
        identified by recursively removing them until the specified threshold of features is reached. This is an
        extended implementation of SciKit-Learn's implementation, which allows for the proportion of components being
        kept to be tuned by Optuna dynamically; it otherwise works identically to said implementation with default
        parameters.

    Example usage:
    {
      "type": "recursive_feature_elimination",
      "proportion": {
        "label": "rfe_component_prop",
        "type": "float",
        "low": 0.1,
        "high": 0.9
      }
    }
    """
    def __init__(self, config: dict, **kwargs):
        super().__init__(config=config, **kwargs)
        super(FittedDataHook, self).__init__(config=config, **kwargs)

        # Grab the proportion of features to select; defaults to 70%
        select_prop = config.get("proportion", {
            "label": "proportion",
            "type": "constant",
            "value": 0.7
        })
        self.prop_tuner: TunableParam = TunableParam.from_config_entry(select_prop)

        # Keep tabs on a backing instance for later user
        self.backing_rfe: RFE | None = None

        # Keep track of the backing features independent of the RFE instance to avoid some code smells later
        self.selected_features = None

    @classmethod
    def from_config(cls, config: dict, logger: Logger = Logger.root) -> Self:
        return cls(config=config, logger=logger)

    def tune(self, trial: Trial):
        self.prop_tuner.tune(trial)
        # Generate the new backing model based on this setup
        # TODO: Generalize this to work with continuous targets as well
        new_lor = LogisticRegression()
        self.backing_rfe = RFE(estimator=new_lor, n_features_to_select=self.prop_tuner.value)

    def tunable_params(self) -> list[TunableParam]:
        return [self.prop_tuner]

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        # If x contains only one feature already, just return that feature, as RFE has a stroke otherwise
        if x.n_features() == 1:
            self.logger.warning("Only one feature in the dataset was found, making RFE redundant. "
                                "Original (unmodified) dataset returned instead.")
            self.selected_features = x.features()
            return x

        # If the number of features selected would be 0, make it 1 instead
        if self.prop_tuner.value * x.n_features() < 1:
            self.backing_rfe.n_features_to_select = 1

        # Calculate and regenerate the features in the training set
        self.backing_rfe.fit(x.as_array(), np.ravel(y.as_array())) # Ravel prevents some warning spam
        self.selected_features = self.backing_rfe.get_feature_names_out(x.features())

        # Isolate these selected features from the rest of the data
        x_out = x.get_features(self.selected_features)

        # Otherwise, raise an error, as it makes no sense to feature select a single feature!
        return x_out

    def run_fitted(self,
               x_train: BaseDataManager,
               x_test: Optional[BaseDataManager],
               y_train: Optional[BaseDataManager] = None,
               y_test: Optional[BaseDataManager] = None
           ) -> (BaseDataManager, BaseDataManager):
        # Run the fitted analysis first
        train_out = self.run(x_train, y_train)

        # Use the same set of features to filter the x_test set
        test_out = x_test.get_features(self.selected_features)

        return train_out, test_out
