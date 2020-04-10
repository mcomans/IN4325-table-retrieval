from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd


def ndcg_scorer(estimator, x, y):
    """
    Custom scorer for NDCG.
    :param estimator: The estimator to score.
    :param x: The data.
    :param y: The true labels.
    :return: Returns the NDCG score.
    """
    predictions = estimator.predict(x)

    # Join all the data back to a dataframe
    df = pd.DataFrame({'prediction': predictions})
    if len(y) > .5 * len(feature_data):
        df = df.join(train_info)
    else:
        df = df.join(test_info)
    df = df.join(pd.DataFrame({'rel': y}))

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
    predictions_grouped_rect = np.zeros([len(predictions_grouped), len(max(predictions_grouped, key=lambda a: len(a)))])
    for i, j in enumerate(predictions_grouped):
        predictions_grouped_rect[i][0:len(j)] = j

    actual_grouped_rect = np.zeros([len(actual_grouped), len(max(actual_grouped, key=lambda a: len(a)))])
    for i, j in enumerate(actual_grouped):
        actual_grouped_rect[i][0:len(j)] = j

    return ndcg_score(actual_grouped_rect, predictions_grouped_rect, k=20)


def rfr_model(x, y, x_test, y_test):
    """
    Run a random forest regression model.
    :param x: The training data
    :param y: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    """
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(2, 5),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring=ndcg_scorer, verbose=0, n_jobs=-1)

    grid_result = gsc.fit(x, y)
    best_params = grid_result.best_params_
    print(best_params)

    rfr = RandomForestRegressor(max_depth=best_params['max_depth'], n_estimators=best_params['n_estimators'],
                                random_state=0, verbose=False)
    rfr.fit(x, y)
    score = ndcg_scorer(rfr, x_test, y_test)
    print(score)

    write_trec_results(rfr, x)


def write_trec_results(rfr, x):
    predictions = rfr.predict(x)
    df = test_info.join(pd.DataFrame({'score': predictions}))
    with open('results/trec_scores_ltr_testset.txt', 'w') as file:
        for index, row in df.iterrows():
            file.write(f"{row['query_id']} Q0 {row['table_id']} 1 {row['score']} ltr\n")


feature_data = pd.read_csv('data/features.csv')

# Group the feature data by queries in order to create a train/test split
queries = feature_data['query'].unique()
queryDict = {elem: pd.DataFrame for elem in queries}

for key in queryDict.keys():
    queryDict[key] = feature_data[:][feature_data['query'] == key]

# Group the queries (currently an arbitrary amount)
test = pd.concat([v for (k,v) in list(queryDict.items())[:15]])
train = pd.concat([v for (k,v) in list(queryDict.items())[15:]])

# Separate the query information
test_info = test[['query_id', 'query', 'table_id']]
train_info = train[['query_id', 'query', 'table_id']]

test_labels = np.array(test['rel'])
train_labels = np.array(train['rel'])

# Drop all the columns that aren't features
test = test.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)
train = train.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)

print(test)
print(train)

test_features = np.array(test)
train_features = np.array(train)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rfr_model(train_features, train_labels, test_features, test_labels)