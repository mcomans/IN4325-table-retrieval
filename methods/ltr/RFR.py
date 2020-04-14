from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from scorers import ndcg_scorer


class RFR:
    def __init__(self, x_train, y_train, x_test, y_test, train_info, test_info, train_features):
        """
        Initializes the Random Forest Regression class by initializing the data.
        :param x_train: An array containing the training features.
        :param y_train: An array containing the training labels.
        :param x_test: An array containing the testing features.
        :param y_test: An array containing the testing labels.
        :param train_info: A DataFrame containing the query/table information for the training data.
        :param test_info: A DataFrame containing the query/table information for the testing data.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.train_info = train_info
        self.test_info = test_info
        self.train_features = train_features

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
