import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def ndcg_scorer(y_true, y_pred, info, k=20):
    """
    Custom scorer for NDCG. Needs the feature data and query information for properly grouping the queries in order to
    calculate the NDCG per query.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param info: The query information belong to the labels that are being scored. This should be the appropriate
    info for the labels being scored (i.e. the test or training set).
    :param k: The top results cutoff length (default 20).
    :return: Returns the NDCG score.
    """
    # Join all the data back to a dataframe
    df = pd.DataFrame({'prediction': y_pred})

    if not len(info) == len(y_pred) == len(y_true):
        raise Exception("Lengths are not equal")

    df = df.join(info.reset_index())
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
