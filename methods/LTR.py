import pandas as pd
import matplotlib.pyplot as plt
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
        rfr = ltr.RFR(features_df, TEST_SET_SIZE)
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
