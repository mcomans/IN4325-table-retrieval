from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from scorers import ndcg_scorer


class RFR:
    def __init__(self, features_df, test_set_size):
        """
        Initializes the Random Forest Regression class by preprocessing the dataset and initializing the features.
        :param features_df: A dataframe containing the raw features and their attributes.
        :param test_set_size: The size that the test set should be.
        """
        self.feature_data = features_df

        # Randomly sample queries for the test set and divide the data in training/test sets
        random_test_queries = np.random.choice(self.feature_data['query_id'].unique(), test_set_size, replace=False)
        test = self.feature_data[self.feature_data['query_id'].isin(random_test_queries)]
        train = self.feature_data[~self.feature_data['query_id'].isin(random_test_queries)]

        # Separate the query and table information
        self.test_info = test[['query_id', 'query', 'table_id']].reset_index()
        self.train_info = train[['query_id', 'query', 'table_id']].reset_index()

        # Create plain arrays for the test and train labels
        self.y_test = np.array(test['rel'])
        self.y_train = np.array(train['rel'])

        # Drop all the columns that aren't features. This leaves a dataframe of features (and column names).
        self.test_features = test.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)
        self.train_features = train.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)

        print(f"Training set:\n\n{train}")
        print(f'Training labels shape: {self.y_train.shape}\n')
        print(f"Testing set:\n\n{test}")
        print(f'Testing labels shape: {self.y_test.shape}\n')

        # Create plain arrays from the feature dataframes
        self.x_test = np.array(self.test_features)
        self.x_train = np.array(self.train_features)

        self.model = None

    def find_best_params(self):
        """
        Finds the best parameters for the Random Forest Regression model given the training data using a grid search
        with respect to the NDCG@20 metric.
        :return: Returns the best parameters.
        """
        best = {'score': 0, 'n_estimators': 0, 'max_depth': 0}
        for max_depth in range(2, 4):
            for n_estimators in [50, 100, 1000]:
                rfr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=0)
                rfr.fit(self.x_train, self.y_train)
                y_pred = rfr.predict(self.x_train)
                score = ndcg_scorer(self.y_train, y_pred, self.train_info)
                if score > best['score']:
                    best['score'] = score
                    best['n_estimators'] = n_estimators
                    best['max_depth'] = max_depth

        print(f"Best params: n_estimators {best['n_estimators']}, max_depth {best['max_depth']}")
        return best

    def run(self, max_depth=-1, n_estimators=-1):
        """
        Run a random forest regression model.
        :param max_depth: The maximum depth of a tree in the model. Best value is estimated by default.
        :param n_estimators: The number of trees in the forest. Best value is estimated by default.
        :return: A dataframe of predictions and their respective query information, and the NDCG@20 score for the run.
        """
        if max_depth == -1 or n_estimators == -1:
            best_params = self.find_best_params()
            max_depth = best_params['max_depth']
            n_estimators = best_params['n_estimators']

        self.model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=0)
        self.model.fit(self.x_train, self.y_train)
        y_pred = self.model.predict(self.x_test)

        scores = dict.fromkeys([5, 10, 15, 20])
        scores[5] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=5)
        scores[10] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=10)
        scores[15] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=15)
        scores[20] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=20)

        return self.test_info.join(pd.DataFrame({'score': y_pred})), scores

    def feature_importance(self):
        """
        Returns a dataframe containing the importances of the different features in the model and their standard
        deviations.
        :return: Dataframe containing importance and std columns.
        """
        if self.model is None:
            print("Model has not been defined yet. Skipping feature importances...")
            return

        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

        return pd.DataFrame({'importance': importances, 'std': std}, index=self.train_features.columns)
