from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from scorers import ndcg_scorer


class LTR:
    def __init__(self, features_df):
        self.feature_data = features_df

        # Randomly sample queries for the test set and divide the data in training/test sets
        random_test_queries = np.random.choice(self.feature_data['query_id'].unique(), 20, replace=False)
        test = self.feature_data[self.feature_data['query_id'].isin(random_test_queries)]
        train = self.feature_data[~self.feature_data['query_id'].isin(random_test_queries)]

        # Separate the query information
        self.test_info = test[['query_id', 'query', 'table_id']].reset_index()
        self.train_info = train[['query_id', 'query', 'table_id']].reset_index()

        test_labels = np.array(test['rel'])
        train_labels = np.array(train['rel'])

        # Drop all the columns that aren't features
        test = test.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)
        train = train.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)

        print(f"Training set:\n\n{train}")
        print(f'Training labels shape: {train_labels.shape}\n')
        print(f"Testing set:\n\n{test}")
        print(f'Testing labels shape: {test_labels.shape}\n')

        test_features = np.array(test)
        train_features = np.array(train)

        self.rfr_model(train_features, train_labels, test_features, test_labels)

    def rfr_model(self, x_train, y_train, x_test, y_test):
        """
        Run a random forest regression model.
        :param x_train: The training data
        :param y_train: The training labels
        :param x_test: The test data
        :param y_test: The test labels
        """
        scorer = make_scorer(ndcg_scorer, feature_data=self.feature_data, train_info=self.train_info,
                             test_info=self.test_info)

        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'max_depth': range(2, 5),
                'n_estimators': (10, 50, 100, 1000),
            },
            cv=5, scoring=scorer, verbose=0, n_jobs=-1)

        grid_result = gsc.fit(x_train, y_train)
        best_params = grid_result.best_params_
        print(best_params)

        rfr = RandomForestRegressor(max_depth=best_params['max_depth'], n_estimators=best_params['n_estimators'],
                                    random_state=0, verbose=False)
        rfr.fit(x_train, y_train)
        y_pred = rfr.predict(x_test)
        score = ndcg_scorer(y_test, y_pred, feature_data=self.feature_data, train_info=self.train_info,
                            test_info=self.test_info)
        print(score)

        self.write_trec_results(rfr, x_test)

    def write_trec_results(self, rfr, x):
        predictions = rfr.predict(x)
        df = self.test_info.join(pd.DataFrame({'score': predictions}))
        df = df.groupby('query_id').apply(lambda x: x.sort_values(['score'], ascending=False)).reset_index(drop=True)
        with open('results/trec_scores_ltr_testset.txt', 'w') as file:
            for index, row in df.iterrows():
                file.write(f"{row['query_id']} Q0 {row['table_id']} 1 {row['score']} ltr\n")


features_df = pd.read_csv('data/features.csv')
LTR(features_df)
