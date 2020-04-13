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
        run_rfr_experiment(features_df)


def run_rfr_experiment(features_df):
    """
    Runs an experiment using Random Forest Regression.
    :param features_df: A dataframe of raw feature data and its attributes.
    """
    total = 0
    no_runs = 10
    for i in range(no_runs):
        rfr = ltr.RFR(features_df, TEST_SET_SIZE)
        results, score = rfr.run()
        total += score
        TREC.write_results(results, f'LTR_RFR_{i}_{TEST_SET_SIZE}')
    print(f"Average NDCG@20 over {no_runs} runs: {total / no_runs}")
