from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.svm import SVC


# The gridsearch parameters to test
model_choices = {
    "lr": {
        "estimator": LogisticRegression(),
        "param_grid": {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']
        }
    },
    "svm": {
        "estimator": SVC(),
        "param_grid": {
            'clf__C': [0.1, 1, 10, 100],
            'clf__kernel': ['rbf', 'sigmoid', 'linear'],
            'clf__probability': [True]
        }
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            'clf__n_neighbors': list(range(3, 16, 2)),
            'clf__weights': ['uniform', 'distance'],
            'clf__p': [1, 2]
        }
    },
    "ada": {
        "estimator": AdaBoostClassifier(),
        "param_grid": {
            'clf__n_estimators': list(range(50, 300, 50)),
            'clf__learning_rate': [0.1, 1, 10],
            'clf__random_state': [214324]
        }
    },
    "rf": {
        "estimator": RandomForestClassifier(),
        "param_grid": {
            'clf__criterion': ['gini', 'entropy', 'log_loss'],
            'clf__min_samples_split': list(range(2, 12, 2)),
            'clf__max_features': [10, 'sqrt', 'log2'],
            'clf__random_state': [214324]
        }
    }
}

def add_pca_features(grid_params: dict):
    # Extend a set of grid-params with variables related to feature elimination/transformation
    new_grid_params = grid_params.copy()
    new_grid_params.update({
        'pca__n_components': np.arange(0.1, 1, 0.1)
    })
    return new_grid_params


def add_rfe_features(grid_params: dict):
    # Extend a set of grid-params with variables related to feature elimination/transformation
    new_grid_params = grid_params.copy()
    new_grid_params.update({
        'rfe__n_features_to_select': np.arange(0.1, 1, 0.1),
        'rfe__estimator': [LogisticRegression()]
    })
    return new_grid_params


def split_data(df):
    # Split the data into meta, target, and feature
    meta_df = df.loc[:, [
        'Slice (I->S) [V2]',  # Descriptive
        'Slice (I->S) [V3]',  # Descriptive
        'Slice (I->S) [V4]',  # Descriptive
        'Slice (I->S) [V5]',  # Descriptive
        'Slice (I->S) [V6]',  # Descriptive
        'mJOA [1 Year]',  # Redundant
        'Followup: 12 month',  # Meaningless
        'Followup: 24 month',  # Meaningless
        'Followup: 60 month',  # Meaningless
        'Date of Assessment',  # Potentially misleading
        'Site',  # Mono-class
        'Number of Surgeries',  # Redundant
        'Treatment Plan',  # Redundant
        'CSM Duration',  # Redundant
        'mJOA [1 Year]'  # Promotes overfitting
    ]]
    target_df = df.loc[:, ['Recovery Class']]
    feature_df = df.drop(target_df.columns, axis=1)
    feature_df = feature_df.drop(meta_df.columns, axis=1)

    return feature_df, target_df, meta_df


def prep_feature_data(df):
    # Define the categorical columns expected in this dataset
    cat_cols = [
        'Work Status (Category)',
        'Comorbidities: Nicotine (Smoking)',
        'Comorbidities: Nicotine (Smokeless)',
        'Comorbidities: Nicotine (Patches)',
        'Comorbidities: Nicotine (Recent Quit)',
        'Sex',
        'Symptom Duration'
    ]

    # Surgical may or may not exist, depending on whether the dataset was filtered or not
    if 'Surgical' in df.columns:
        cat_cols.append('Surgical')

    # TODO Swap back to KNN with immediate (single) neighbor for categorical data

    # Impute the categorical data with immediate neighbor imputation (KNN with 1 neighbor)
    cat_imp = SimpleImputer(strategy='most_frequent', missing_values=pd.NA)
    cat_data = pd.DataFrame(cat_imp.fit_transform(df), index=df.index, columns=df.columns)
    cat_data = cat_data.loc[:, cat_cols]

    # OneHotEncode the resulting categories
    ohe = OneHotEncoder(drop='if_binary')
    ohe_results = ohe.fit_transform(cat_data).toarray()
    cat_data = pd.DataFrame(ohe_results, index=cat_data.index, columns=ohe.get_feature_names_out(cat_data.columns))

    # Impute the continuous data using 5-nearest-neighbor average
    con_imp = KNNImputer(n_neighbors=5)
    con_data = df.drop(cat_cols, axis=1)
    con_data = pd.DataFrame(con_imp.fit_transform(con_data), index=con_data.index, columns=con_data.columns)

    # Join the data back together and return it
    new_df = con_data
    new_df[cat_data.columns] = cat_data

    return new_df


def score_model(X_validate, y_validate, i_validate, model, result_dict):
    # Initialize a copy of the results to avoid overwriting the original if testing is needed
    new_results = result_dict.copy()

    # Calculate the predictions for the validation set
    y_pred = model.predict(X_validate)

    # Evaluate the corresponding metrics
    validate_bal_acc = balanced_accuracy_score(y_validate, y_pred)
    new_results['balanced accuracy'] = [validate_bal_acc]

    p, r, f, _ = precision_recall_fscore_support(y_validate, y_pred, average='binary', pos_label='fair')
    new_results['precision (fair)'] = [p]
    new_results['recall (fair)'] = [r]
    new_results['f-score (fair)'] = [f]

    p, r, f, _ = precision_recall_fscore_support(y_validate, y_pred, average='binary', pos_label='good')
    new_results['precision (good)'] = [p]
    new_results['recall (good)'] = [r]
    new_results['f-score (good)'] = [f]

    # Track the MRI sequences which were guessed correctly and incorrectly for later analysis
    new_results['good_predictions'] = [[i_validate[i] for i in range(len(y_pred)) if y_pred[i] == y_validate[i]]]
    new_results['bad_predictions'] = [[i_validate[i] for i in range(len(y_pred)) if y_pred[i] != y_validate[i]]]

    # Return the result
    return new_results


def run_gridsearch(X_train, X_validate, y_train, y_validate, i_validate,
                   acq_label, weight_label, model_label,
                   param_grid, pipe, random_seed, out_path,
                   feature_list, replicate):
    splits = StratifiedKFold(random_state=random_seed, shuffle=True)
    clf = GridSearchCV(pipe, param_grid, cv=splits, n_jobs=-1, scoring='balanced_accuracy')
    # Run the grid-search
    clf.fit(X_train, y_train)

    # Grab the results we want
    result = {
        'acq': [acq_label],
        'weight': [weight_label],
        'model': [model_label],
        'replicate': [replicate],
        'is_rfe': ["rfe__n_features_to_select" in param_grid.keys()],
        'is_pca': ["pca__n_components" in param_grid.keys()],
        'best_params': [clf.best_params_],
        'best_score': [clf.best_score_]
    }

    # Update them with feature-related metrics
    result.update(report_feature_values(feature_list, clf))

    # Update them with validation metrics
    result = score_model(X_validate, y_validate, i_validate, clf, result)

    df = pd.DataFrame.from_dict(result)
    if not out_path.exists():
        df.to_csv(out_path, header=True, sep='\t')
    else:
        df.to_csv(out_path, mode='a', header=False, sep='\t')
    return result


def report_feature_values(init_features: list, gscv: GridSearchCV):
    # Extract the best estimator found by the gridsearch
    pipe = gscv.best_estimator_

    selected_features = init_features
    # If RFE is used, grab the subset of features used
    if "rfe" in pipe.named_steps.keys():
        rfe = pipe.named_steps['rfe']
        selected_features = init_features[rfe.get_support(indices=True)]
    # If PCA is used, grab how many components were selected
    feature_list = np.array(selected_features)
    if "pca" in pipe.named_steps.keys():
        pca = pipe.named_steps['pca']
        feature_list = np.array([f"pc{i}" for i in range(pca.n_components_)])

    # Sort he features based on the model type and their importance to the model's prediction
    feature_importance = None
    importance_order = None
    clf = pipe.named_steps['clf']
    if isinstance(clf, LogisticRegression):
        feature_importance = clf.coef_[0]
        importance_order = np.argsort(np.abs(feature_importance))
    elif isinstance(clf, SVC):
        if clf.kernel == 'linear':
            feature_importance = clf.coef_[0]
            importance_order = np.argsort(np.abs(feature_importance))
    elif isinstance(clf, AdaBoostClassifier) or isinstance(clf, RandomForestClassifier):
        feature_importance = clf.feature_importances_
        importance_order = np.argsort(feature_importance)

    # Parse the result
    if importance_order is not None:
        # Reverse so that we are in descending order
        importance_order = list(importance_order[::-1])
        # Sort the other two lists using this order
        feature_list = list(feature_list[importance_order])
        feature_importance = list(feature_importance[importance_order])
    else:
        feature_list = list()
        feature_importance = list()

    # Return the results
    return {
        "selected_features": [feature_list],
        "feature_importance": [feature_importance]
    }


def evaluate_models(df, random_seed, out_path, replicate):
    # For each possible combination of acq and weight, run a gridsearch
    acqs = set(df.index.get_level_values('acq'))
    weights = set(df.index.get_level_values('weight'))
    for a, w in product(acqs, weights):
        # Filter down to only include relevant entries
        sub_df = df.query(f'acq == "{a}" and weight == "{w}"')

        # If the number of records is less than 50, skip it
        if sub_df.shape[0] < 50:
            print(f"Method {a}-{w} contained too few entries, skipping")
            continue

        # Grab only the last valid run in the dataset
        sub_df = sub_df.groupby(level=["GRP"]).last()

        # Split the data into our desired subsets
        feature_df, target_df, meta_df = split_data(sub_df)

        # Prepare the features for final ML analysis
        feature_df = prep_feature_data(feature_df)

        # Unpack the target so that it is a 1D-array-like
        target_df = target_df['Recovery Class']

        # Grab the column labels of our features for later use
        feature_labels = feature_df.columns

        # Split the features and targets into train and validation sets
        X_train, X_validate, y_train, y_validate, i_train, i_validate \
            = train_test_split(feature_df, target_df, feature_df.index.values,
                               test_size=0.2, random_state=random_seed)

        # Normalize the training data, and apply it to the validation set
        normalizer = Normalizer()
        X_train = normalizer.fit_transform(X_train)
        X_validate = normalizer.transform(X_validate)

        # Run the gridsearch for each model type
        for m, v in model_choices.items():
            # == Standalone Model == #
            estimator = v['estimator']
            pipe = Pipeline([
                ('clf', estimator)
            ])
            param_grid = v['param_grid']
            run_gridsearch(X_train, X_validate, y_train, y_validate, i_validate, a, w, m,
                           param_grid, pipe, random_seed, out_path, feature_labels, replicate)

            # == PCA-Only == #
            estimator = v['estimator']
            pipe = Pipeline([
                ('pca', PCA()),
                ('clf', estimator)
            ])
            param_grid = v['param_grid']
            param_grid = add_pca_features(param_grid)
            run_gridsearch(X_train, X_validate, y_train, y_validate, i_validate, a, w, m,
                           param_grid, pipe, random_seed, out_path, feature_labels, replicate)

            # == RFE-Only == #
            estimator = v['estimator']
            pipe = Pipeline([
                ('rfe', RFE(estimator=LogisticRegression())),
                ('clf', estimator)
            ])
            param_grid = v['param_grid']
            param_grid = add_rfe_features(param_grid)
            run_gridsearch(X_train, X_validate, y_train, y_validate, i_validate, a, w, m,
                           param_grid, pipe, random_seed, out_path, feature_labels, replicate)

            # == RFE and PCA == #
            estimator = v['estimator']
            pipe = Pipeline([
                ('rfe', RFE(estimator=LogisticRegression())),
                ('pca', PCA()),
                ('clf', estimator)
            ])
            param_grid = v['param_grid']
            param_grid = add_rfe_features(param_grid)
            param_grid = add_pca_features(param_grid)
            run_gridsearch(X_train, X_validate, y_train, y_validate, i_validate, a, w, m,
                           param_grid, pipe, random_seed, out_path, feature_labels, replicate)


def main(in_path: Path, out_path: Path, random_seed_init: int, no_replicates: int):
    # Load data
    df = pd.read_csv(in_path, sep='\t', index_col=[0, 1, 2, 3])

    # Reset the output file
    if out_path.exists():
        if out_path.is_dir():
            raise TypeError(f"out_path '{out_path}' is a directory, not a file")
        else:
            out_path.unlink()

    # Generate a set of seeds
    np.random.seed(random_seed_init)
    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=no_replicates)

    # Run evaluation for each replicated
    for i, s in enumerate(random_seeds):
        # Set our random seed
        np.random.seed(s)

        # Run the gridsearch
        evaluate_models(df, s, out_path, i)


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser(
        prog="Classic ML GridSearch",
        description="Runs a gridsearch of all potential parameters using a given datasets and model type"
    )

    parser.add_argument(
        '-i', '--in_path', required=True, type=Path,
        help="The dataset to use for the analysis"
    )
    parser.add_argument(
        '-o', '--out_path', default='out.tsv', type=Path,
        help="Where the results of the analysis should be stored (in tsv format)"
    )
    parser.add_argument(
        '-r', '--random_seed_init', type=int, default=71544,   # Randomly selected
        help="The random seed to use for this analysis (to preserve stability across the different subsets)"
    )
    parser.add_argument(
        '-n', '--no_replicates', type=int, default=10,
        help="Number of replicates that should be run for each model"
    )

    argvs = parser.parse_args().__dict__
    main(**argvs)
