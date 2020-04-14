from sklearn.svm import SVR as SKSVR
import numpy as np
import pandas as pd
from scorers import ndcg_scorer


class SVR:
    def __init__(self, features_df, test_set_size):
        """
        Initializes the Support Vector Regression class by preprocessing the dataset and initializing the features.
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
        Finds the best parameters for the Support Vector Regression model given the training data using a grid search
        with respect to the NDCG@20 metric.
        :return: Returns the best parameters.
        """
        best = {'score': 0, 'kernel': '', 'epsilon': 0, 'c': 0}
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            for epsilon in [0.1, 0.5, 1, 5]:
                for c in [0.1, 1, 10, 100]:
                    print(f"Running {kernel} with epsilon {epsilon} and c {c}")
                    svr = SKSVR(kernel=kernel, epsilon=epsilon, C=c)
                    svr.fit(self.x_train, self.y_train)
                    y_pred = svr.predict(self.x_train)
                    score = ndcg_scorer(self.y_train, y_pred, self.train_info)
                    if score > best['score']:
                        best['score'] = score
                        best['kernel'] = kernel
                        best['epsilon'] = epsilon
                        best['c'] = c

        print(f"Best params: kernel {best['kernel']}, epsilon {best['epsilon']}, c {best['c']}")
        return best

    def run(self, kernel='', epsilon=-1, c=-1):
        """
        Run a Support Vector Regression model.
        :return: A dataframe of predictions and their respective query information, and the NDCG@20 score for the run.
        """
        if not kernel or epsilon == -1 or c == -1:
            best_params = self.find_best_params()
            kernel = best_params['kernel']
            epsilon = best_params['epsilon']
            c = best_params['c']

        self.model = SKSVR(kernel=kernel, epsilon=epsilon, C=c)
        self.model.fit(self.x_train, self.y_train)
        y_pred = self.model.predict(self.x_test)

        scores = dict.fromkeys([5, 10, 15, 20])
        scores[5] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=5)
        scores[10] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=10)
        scores[15] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=15)
        scores[20] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=20)

        return self.test_info.join(pd.DataFrame({'score': y_pred})), scores
