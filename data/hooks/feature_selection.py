from abc import ABC
from logging import Logger
from typing import Self, Optional

import numpy as np
import pandas as pd
from optuna import Trial
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from config.utils import default_as, is_float, is_list, parse_data_config_entry
from data.base import BaseDataManager
from data.hooks import registered_data_hook
from data.hooks.base import DataHook, FittedDataHook
from data.mixins import MultiFeatureMixin
from tuning.utils import Tunable, TunableParam


### Explicit Feature Selection ###
class ExplicitFeatures(DataHook, ABC):
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
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager | MultiFeatureMixin:
        # We can only drop features in a dataset if they have more than 1
        if not isinstance(x, MultiFeatureMixin):
            raise TypeError(f"DataManager of type '{self.__class__}' only has one feature! Cannot modify feature set.")
        x: BaseDataManager | MultiFeatureMixin
        return x.drop_features(self.features)

@registered_data_hook("keep_features_explicit")
class ExplicitKeep(ExplicitFeatures):
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager | MultiFeatureMixin:
        # We can only drop features in a dataset if they have more than 1
        if not isinstance(x, MultiFeatureMixin):
            raise TypeError(f"DataManager of type '{self.__class__}' only has one feature! Cannot modify feature set.")
        x: BaseDataManager | MultiFeatureMixin
        return x.get_features(self.features)


### Feature Selection by Null Content ###
class NullityDrop(DataHook, ABC):
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
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager | MultiFeatureMixin:
        # Iterate through all samples and calculate the null content for each, dropping
        # any which surpass the configured threshold
        n_samples = len(x)
        null_tolerance = int(n_samples * self.threshold)
        keep_idx = [i for i, k in enumerate(x)
                    if np.sum(pd.isnull(k.as_array())) < null_tolerance]
        return x[keep_idx]

@registered_data_hook("feature_drop_null")
class FeatureNullityDrop(NullityDrop):
    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager | MultiFeatureMixin:
        # Make sure this DataManager has more than one feature!
        if not isinstance(x, MultiFeatureMixin):
            raise TypeError(f"DataManager of type '{self.__class__.__name__}' only has one feature which cannot be dropped!")
        x: BaseDataManager | MultiFeatureMixin
        # Otherwise, calculate the null content for each feature dropping any which surpass the configured threshold
        n_samples = len(x)
        null_tolerance = int(n_samples * self.threshold)
        drop_idx = [k for k in x.features()
                    if np.sum(pd.isnull(x.get_features(k).as_array())) > null_tolerance]
        return x.drop_features(drop_idx)


### Principal Component Analysis ###
@registered_data_hook("principal_component_analysis")
class PrincipalComponentAnalysis(Tunable, FittedDataHook):
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
        if isinstance(x, MultiFeatureMixin):
            # Denote the type of our inputs so type hinting doesn't suck
            x: MultiFeatureMixin | BaseDataManager

            # Calculate and regenerate the features in the training set
            tmp_train = self.backing_pca.fit_transform(x.as_array())
            feature_labels = [f'PC{i}' for i in range(tmp_train.shape[1])]

            # Drop all features and replace them with the new components
            x_out = x.drop_features(x.features())
            x_out: BaseDataManager | MultiFeatureMixin = x_out.set_features(feature_labels, tmp_train)

        # Otherwise, report an error, as we can't dimensionally reduce a single dimension!
        else:
            raise ValueError("Dimensionality reduction cannot be run on a dataset with only 1 dimension!")
        return x_out

    def run_fitted(self,
               x_train: BaseDataManager,
               x_test: Optional[BaseDataManager],
               y_train: Optional[BaseDataManager] = None,
               y_test: Optional[BaseDataManager] = None
           ) -> (BaseDataManager, BaseDataManager):
        if isinstance(x_train, MultiFeatureMixin):
            # Denote the type of our inputs so type hinting doesn't suck
            x_train: MultiFeatureMixin | BaseDataManager
            x_test: MultiFeatureMixin | BaseDataManager

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

        # Otherwise, report an error, as we can't dimensionally reduce a single dimension!
        else:
            raise ValueError("Dimensionality reduction cannot be run on a dataset with only 1 dimension!")
        return train_out, test_out


### Recursive Feature Elimination ###
@registered_data_hook("recursive_feature_elimination")
class RecursiveFeatureElimination(Tunable, FittedDataHook):
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
        new_lor = LogisticRegression()
        self.backing_rfe = RFE(estimator=new_lor, n_features_to_select=self.prop_tuner.value)

    def tunable_params(self) -> list[TunableParam]:
        return [self.prop_tuner]

    def run(self, x: BaseDataManager, y: Optional[BaseDataManager] = None) -> BaseDataManager:
        if isinstance(x, MultiFeatureMixin):
            # Denote the type of our inputs so type hinting doesn't suck
            x: MultiFeatureMixin | BaseDataManager

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

            # Select only the features
            x_out: BaseDataManager | MultiFeatureMixin = x.get_features(self.selected_features)

        # Otherwise, raise an error, as it makes no sense to feature select a single feature!
        else:
            raise ValueError(f"The RFE data hook currently only supports multi-feature datasets; "
                             f"received {type(x)} instead")
        return x_out

    def run_fitted(self,
               x_train: BaseDataManager,
               x_test: Optional[BaseDataManager],
               y_train: Optional[BaseDataManager] = None,
               y_test: Optional[BaseDataManager] = None
           ) -> (BaseDataManager, BaseDataManager):
        if isinstance(x_train, MultiFeatureMixin):
            # Denote the type of our inputs so type hinting doesn't suck
            x_train: MultiFeatureMixin | BaseDataManager
            x_test: MultiFeatureMixin | BaseDataManager

            # Run the fitted analysis first
            train_out = self.run(x_train, y_train)

            # Use the same set of features to filter the x_test set
            test_out = x_test.get_features(self.selected_features)


        # Otherwise, report an error, as we can't dimensionally reduce a single dimension!
        else:
            raise ValueError("Dimensionality reduction cannot be run on a dataset with only 1 dimension!")
        return train_out, test_out
