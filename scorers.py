import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def ndcg_scorer(y_true, y_pred, feature_data, train_info, test_info, k=20):
    """
    Custom scorer for NDCG. Needs the feature data and query information for properly grouping the queries in order to
    calculate the NDCG per query.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param feature_data: The raw feature dataframe.
    :param train_info: The query information for the training dataset.
    :param test_info: The query information for the test dataset.
    :param k: The top results cutoff length (default 20).
    :return: Returns the NDCG score.
    """
    # Join all the data back to a dataframe
    df = pd.DataFrame({'prediction': y_pred})
    if len(y_pred) > .5 * len(feature_data):
        df = df.join(train_info)
    else:
        df = df.join(test_info)
    df = df.join(pd.DataFrame({'rel': y_true}))

    # Group the predictions and the true labels by query
    predictions_grouped = []
    actual_grouped = []

    queries = df['query'].unique()
    queryDict = {elem: pd.DataFrame for elem in queries}

    for key in queryDict.keys():
        queryDict[key] = df[:][df['query'] == key]
        predictions_grouped.append(np.array(queryDict[key]['prediction']))
        actual_grouped.append(np.array(queryDict[key]['rel']))

    # Fill the data grouped by query with zeros to become equal length (a rectangle 2D matrix)
    # due to scikit-learn only accepting rectangular matrices for some reason.
    predictions_grouped_rect = np.zeros(
        [len(predictions_grouped), len(max(predictions_grouped, key=lambda a: len(a)))])
    for i, j in enumerate(predictions_grouped):
        predictions_grouped_rect[i][0:len(j)] = j

    actual_grouped_rect = np.zeros([len(actual_grouped), len(max(actual_grouped, key=lambda a: len(a)))])
    for i, j in enumerate(actual_grouped):
        actual_grouped_rect[i][0:len(j)] = j

    return ndcg_score(actual_grouped_rect, predictions_grouped_rect, k)
