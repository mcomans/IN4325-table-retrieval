import pandas as pd
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
    :param features_df: A dataframe of raw feature data and its attributes.
    :param runs: The number of runs that should be executed, of which the results will be averaged.
    """
    results = pd.DataFrame()

    for i in range(runs):
        rfr = ltr.RFR(features_df, TEST_SET_SIZE)
        predictions, scores = rfr.run()
        results = results.append(scores, ignore_index=True)
        TREC.write_results(predictions, f'LTR_RFR_{i}_{TEST_SET_SIZE}')

    print(f"Average NDCG over {runs} runs at cutoff points:\n{results.mean(axis=0)}")
