import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import read_features
import methods.ltr as ltr
import TREC

TEST_SET_SIZE = 20


def run_ltr(approach: ltr.Approach):
    """
    Run the given LTR approach.
    :param approach: The LTR approach to run.
    """
    features_df = read_features()

    if approach == ltr.Approach.RFR:
        run_rfr_experiment(features_df, 10)
    if approach == ltr.Approach.SVR:
        run_svr_experiment(features_df, 1)
    if approach == ltr.Approach.AdaRank:
        run_adarank_experiment(features_df, 20)


def split_data(features_df):
    """
    Splits the features DataFrame in train and test sets and separates query/table information for NDCG calculation.
    :param features_df: The raw features DataFrame.
    :return: x_train, y_train, x_test, y_test, train_info (query/table info), test_info (query/table info),
    train_features (DataFrame with feature labels)
    """
    # Randomly sample queries for the test set and divide the data in training/test sets
    random_test_queries = np.random.choice(features_df['query_id'].unique(), TEST_SET_SIZE, replace=False)
    test = features_df[features_df['query_id'].isin(random_test_queries)]
    train = features_df[~features_df['query_id'].isin(random_test_queries)]

    # Separate the query and table information
    test_info = test[['query_id', 'query', 'table_id']].reset_index()
    train_info = train[['query_id', 'query', 'table_id']].reset_index()

    # Create plain arrays for the test and train labels
    y_test = np.array(test['rel'])
    y_train = np.array(train['rel'])

    # Drop all the columns that aren't features. This leaves a dataframe of features (and column names).
    test_features = test.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)
    train_features = train.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)

    print(f"Training set:\n\n{train}")
    print(f'Training labels shape: {y_train.shape}\n')
    print(f"Testing set:\n\n{test}")
    print(f'Testing labels shape: {y_test.shape}\n')

    # Create plain arrays from the feature dataframes
    x_test = np.array(test_features)
    x_train = np.array(train_features)

    # The train_features are only necessary for feature importances since we need the feature labels.
    return x_train, y_train, x_test, y_test, train_info, test_info, train_features


def run_rfr_experiment(features_df, runs):
    """
    Runs an experiment using Random Forest Regression.
    Prints the average NDCG values over the given amount of runs.
    Prints the average feature importances over the given amount of runs.
    Saves a plot of the average feature importances to results directory.
    :param features_df: A dataframe of raw feature data and its attributes.
    :param runs: The number of runs that should be executed, of which the results will be averaged.
    """
    results = pd.DataFrame()
    importances = pd.DataFrame()

    for i in range(runs):
        x_train, y_train, x_test, y_test, train_info, test_info, train_features = split_data(features_df)
        rfr = ltr.RFR(x_train, y_train, x_test, y_test, train_info, test_info, train_features)

        predictions, scores = rfr.run()
        results = results.append(scores, ignore_index=True)
        TREC.write_results(predictions, f'LTR_RFR_{i}_{TEST_SET_SIZE}')
        res_imp = rfr.feature_importance().add_suffix(f'_{i}')
        importances = pd.merge(importances, res_imp, how='outer', left_index=True, right_index=True)

    print(f"---\nAverage NDCG over {runs} runs at cutoff points:\n{results.mean(axis=0)}\n")

    importances['importance_mean'] = importances.loc[:, importances.columns.str.contains('importance')].mean(axis=1)
    importances['std_mean'] = importances.loc[:, importances.columns.str.contains('std')].mean(axis=1)
    importances = importances.sort_values('importance_mean', ascending=False)

    print("Feature importances: feature (mean, sd)")
    for index, row in importances.iterrows():
        print(f"{index} ({row['importance_mean']}, {row['std_mean']})")

    plt.figure(figsize=(15, 10))
    plt.title(f"Average feature importances over {runs} runs")
    plt.bar(range(len(importances.index)), importances['importance_mean'], color="b", yerr=importances['std_mean'],
            align="center")
    plt.xticks(range(len(importances.index)), importances.index, rotation=45, ha='right')
    plt.xlim([-1, len(importances.index)])
    plt.savefig('results/avg_feature_importances.pdf')


def run_svr_experiment(features_df, runs):
    """
    Runs an experiment using Support Vector Regression.
    Prints the average NDCG values over the given amount of runs.
    :param features_df: A dataframe of raw feature data and its attributes.
    :param runs: The number of runs that should be executed, of which the results will be averaged.
    """
    results = pd.DataFrame()

    for i in range(runs):
        x_train, y_train, x_test, y_test, train_info, test_info, _ = split_data(features_df)
        svr = ltr.SVR(x_train, y_train, x_test, y_test, train_info, test_info)

        predictions, scores = svr.run()
        results = results.append(scores, ignore_index=True)
        TREC.write_results(predictions, f'LTR_SVR_{i}_{TEST_SET_SIZE}')

    print(f"---\nAverage NDCG over {runs} runs at cutoff points:\n{results.mean(axis=0)}\n")


def run_adarank_experiment(features_df, runs):
    results = pd.DataFrame()

    for i in range(runs):
        x_train, y_train, x_test, y_test, train_info, test_info, _ = split_data(features_df)
        adr = ltr.AdaRank(x_train, y_train, x_test, y_test, train_info, test_info)

        predictions, scores = adr.run()
        results = results.append(scores, ignore_index=True)
        TREC.write_results(predictions, f'LTR_SVR_{i}_{TEST_SET_SIZE}')

    print(f"---\nAverage NDCG over {runs} runs at cutoff points:\n{results.mean(axis=0)}\n")
