"""
Script for inspecting and analyzing the output database.

This script:
- Reads tables from an SQLite database containing ML trial results.
- Extracts feature importance values and model performance metrics.
- Computes weighted statistics (mean and standard deviation) using model performance as weights.
- Identifies the best models based on a specified metric.
- Saves feature importance and model performance data as CSV files.
- Generates plots to visualize model performance across replicates and trials.

Author: Jan Valosek, Kalum Ost
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlite3 import connect

from sqlalchemy.dialects.mssql.information_schema import columns

#target='AIS_change_bin'
target='UEMS_change_bin'
#target='LEMS_change_bin'
#target='AIS_change_bin_gt0'
#target='SNL_Class_initial_bin'
num_of_trials = 100


def read_db(target):
    """
    Read tables from the database as dataframes
    :param target: target variable
    :return: dictionary with the tables (dataframes) from the database
    """
    con = connect(f'testing/output/output_{target}_{num_of_trials}_trials.db')
    tables = pd.read_sql(
        "SELECT * FROM sqlite_master",
        con=con
    ).loc[:, 'name']
    tables_dict = {}
    for t in tables:
        # Pull the dataframe from the database
        try:
            df = pd.read_sql(
                f"SELECT * FROM {t}",
                con=con
            )
            tables_dict[t] = df
        except:
            print(f"Failed to read table {t}, ignoring it")
            continue
    con.close()

    return tables_dict

def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted standard deviation.
    :param values: Array of feature importance values.
    :param weights: Array of weights (e.g., model performance scores).
    :return: Weighted standard deviation.
    """
    # Compute the weighted mean
    weighted_mean = np.average(values, weights=weights)
    # Compute the weighted variance
    weighted_variance = np.average((values - weighted_mean) ** 2, weights=weights)

    # Take the square root to obtain the weighted standard deviation
    return np.sqrt(weighted_variance)

def compute_weighted_feature_importance(best_models, metric):
    """
    Compute the weighted average feature importance using `importance_by_permutation (test)`
    with `balanced_accuracy (test)` as the weight.

    :param best_models: DataFrame with best models selected for each replicate
    :param metric: Performance metric used as weight (e.g., 'balanced_accuracy (test)')
    :return: DataFrame with weighted feature importance
    """

    # Extract importance and performance metric
    best_models = best_models[['model_name', 'replicate', 'trial', metric, 'importance_by_permutation (test)']]

    # Convert 'importance_by_permutation (test)' from str to dict using re
    pattern = r'([\w\s\(\)<\-]+): ([\d\.]+)'    # Works with names containing spaces, (), <-, and _
    best_models['importance_by_permutation (test)'] = best_models['importance_by_permutation (test)'].apply(
        lambda x: {match[0].strip(): float(match[1]) for match in re.findall(pattern, x)})

    # Convert the dictionaries contained with the feature_col dicts into dataframes which can be stacked
    raw_dfs = []
    weighted_dfs = []
    for r in best_models.iterrows():
        rvals = r[1]
        tmp_df = pd.DataFrame.from_dict({k: [v] for k, v in rvals['importance_by_permutation (test)'].items()})
        raw_dfs.append(tmp_df)

    # Stack the dataframes
    raw_feature_imps = pd.concat(raw_dfs).fillna(0)

    # Query the weights list
    weights = best_models[metric].astype('float64')

    # For each feature, calculate our desired statistics
    return_cols = ['Mean', 'STD', 'Weighted Mean', 'Weighted STD']
    return_df_dict = {}
    for c in raw_feature_imps.columns:
        # Single query of the dataframe, as pandas can be slow w/ repeated queries
        samples = raw_feature_imps[c]
        # Raw Mean
        c_mean = np.mean(samples)
        # Raw STD
        c_std = np.std(samples)
        # Weighted mean
        c_mean_weighted = np.average(samples, weights=weights)
        # Weighted STD
        c_std_weighted = weighted_std(samples, weights)
        # Stack them into a list and store it in the dictionary
        return_df_dict[c] = [c_mean, c_std, c_mean_weighted, c_std_weighted]

    weighted_importance_df = pd.DataFrame.from_dict(return_df_dict, columns=return_cols, orient='index')
    # Sort by 'Weighted Mean'
    weighted_importance_df = weighted_importance_df.sort_values('Weighted Mean', ascending=False)

    return weighted_importance_df

def get_best_replicate(tables_dict, metric) -> None:
    """
    Get the best replicate (i.e., best performing model) for each trail based on the specified metric.
    Also, compute the weighted average of `importance_by_permutation (test)` features, with the weight being
    the model's performance (e.g., `balanced_accuracy (test)`).
    Save the best models and weighted feature importance to CSV files.
    :param tables_dict: dictionary with the tables (dataframes) from the database
    :param metric: metric to use for selecting the best models; e.g., 'balanced_accuracy (test)'
    """

    os.makedirs('testing/output/csv', exist_ok=True)
    fname_out = f'testing/output/csv/{target}_{metric}_best_models'

    # Loop over individual models
    for model_name, df in tables_dict.items():
        # Get the best model (trial) for each replicate
        best_models = df.sort_values(metric, ascending=True).groupby('replicate').tail(1)
        # Sort by best_models by replicate
        best_models = best_models.sort_values('replicate')

        # Save metric and 'importance_by_permutation (test)' into a XLSX file; append models to the same file
        # include model name as the first column
        best_models.insert(0, 'model_name', model_name)
        # Save the best models to a CSV file
        best_models[['model_name', 'replicate', 'trial', metric, 'importance_by_permutation (test)']].to_csv(
            f'{fname_out}.csv', mode='a', index=False, header=True)

        # Compute weighted average of `importance_by_permutation (test)` features, with the weight being the model's
        # performance (e.g., `balanced_accuracy (test)`)
        weighted_importance_df = compute_weighted_feature_importance(best_models, metric)
        weighted_importance_df.insert(0, 'model_name', model_name)
        # Save the weighted feature importance to a CSV file
        weighted_importance_df.to_csv(f'{fname_out}_weighted_feature_importance.csv',
                                      mode='a', index=True, header=True)

    print(f"Saved best models to {fname_out}.csv")
    print(f"Saved weighted feature importance to {fname_out}_weighted_feature_importance.csv")


def get_df_for_plotting(tables_dict, metric) -> pd.DataFrame:
    """
    Iterate over the dataframes in tables_dict and merge them into a single dataframe for plotting
    :param tables_dict: dictionary with the tables (dataframes) from the database
    :param metric: metric to plot; e.g., 'balanced_accuracy (test)'
    :return: dataframe for plotting
    """

    df_plotting = pd.DataFrame(columns=['replicate', 'trial'])

    # Loop over individual models
    for model_name, df in tables_dict.items():
        df_temp = df[['replicate', 'trial', metric]]
        # Rename balanced_accuracy to model_name
        df_temp = df_temp.rename(columns={metric: model_name})
        # Add df_temp to df_plotting based on 'replicate' and 'trial'; do not replicate the 'replicate' and 'trial' columns
        df_plotting = pd.merge(df_plotting, df_temp, on=['replicate', 'trial'], how='outer')

    # Some additional cleaning for plotting
    # Sort by 'replicate' and 'trial'
    df_plotting = df_plotting.sort_values(['replicate', 'trial'])
    # Shorten column names (first two columns are 'replicate' and 'trial')
    for column in df_plotting.columns[2:]:
        df_plotting.rename(columns={column: column.replace(f'{target}__LogisticRegression__', '')}, inplace=True)

    return df_plotting

def plotting(df_plotting, metric):
    """
    Plot the balanced accuracy across replicates and trials for each model
    :param df_plotting: dataframe for plotting
    :param metric: metric to plot; e.g., 'balanced_accuracy (test)'
    """

    metric_title = metric.replace('_', ' ').title()   # e.g., Balanced Accuracy (Test)
    metric_fname = metric.replace(' ', '_').replace('(', '').replace(')', '')   # e.g., balanced_accuracy_test

    # Melt the dataframe to a long format for easier plotting
    df_long = df_plotting.melt(
        id_vars=['replicate', 'trial'],
        value_vars=df_plotting.columns[2:],  # Skip 'replicate' and 'trial'
        var_name='model_name',
        value_name=metric
    )

    df_long[metric] = pd.to_numeric(df_long[metric])
    # agg_df = df_long.groupby(['replicate', 'model_name'])[metric].agg(['mean', 'std']).reset_index()

    os.makedirs('testing/output/plots', exist_ok=True)

    # # x-axis: replicate
    # plt.figure(figsize=(10, 6))
    # #sns.lineplot(data=agg_df, x='replicate', y='mean', hue='model_name')
    # sns.lineplot(data=df_long, x='replicate', y=metric, hue='model_name', errorbar='sd')
    # # Customize the plot
    # plt.title(f'{target} -- Mean and Std of {metric_title} Across Trials for Each Replicate')
    # plt.xlabel('Replicate')
    # plt.ylabel(metric_title)
    # plt.legend(title='Model Name')
    # # Show horizontal gridlines
    # plt.grid(axis='y')
    # # Show all x-ticks
    # plt.xticks(df_long['replicate'].unique())
    # plt.tight_layout()
    # #plt.show()
    # # Save with 300 dpi
    # plt.savefig(f'testing/output/plots/{target}_{metric_fname}_replicates_num_of_trials_{num_of_trials}.png', dpi=300)
    # plt.close()

    # x-axis: trial
    plt.figure(figsize=(10, 6))
    # sns.lineplot(data=agg_df, x='replicate', y='mean', hue='model_name')
    sns.lineplot(data=df_long, x='trial', y=metric, hue='model_name', errorbar='sd')
    # Customize the plot
    plt.title(f'{target} -- Mean and Std of {metric_title} Across Replicates for Each Trial')
    plt.xlabel('Trial')
    plt.ylabel(metric_title)
    plt.legend(title='Model Name')
    # Show horizontal gridlines
    plt.grid(axis='y')
    # Make legend smaller
    plt.legend(title='Model Name', fontsize='small')
    # Show all x-ticks
    # plt.xticks(df_long['trial'].unique())
    plt.tight_layout()
    # plt.show()
    # Save with 300 dpi
    plt.savefig(f'testing/output/plots/{target}_{metric_fname}_trials_num_of_trials_{num_of_trials}.png', dpi=300)
    print(f"Saved plots to 'plots' directory")
    plt.close()


def main():
    # Read tables from the database as dataframes
    tables_dict = read_db(target)

    #for metric in ['balanced_accuracy (test)', 'balanced_accuracy (validate)']:
    for metric in ['balanced_accuracy (test)']:
        # Get the best replicate (i.e., best performing model) for each trial (train/test split)
        get_best_replicate(tables_dict, metric)
        # Prepare the dataframe for plotting
        df_plotting = get_df_for_plotting(tables_dict, metric)
        # Plot the metric across trials for each model
        plotting(df_plotting, metric)

if __name__ == '__main__':
    main()




