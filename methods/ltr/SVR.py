from sklearn.svm import SVR as SKSVR
import pandas as pd
from scorers import ndcg_scorer


class SVR:
    def __init__(self, x_train, y_train, x_test, y_test, train_info, test_info):
        """
        Initializes the Support Vector Regression class by initializing the data.
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
        :param kernel: The kernel to be used in the model. Best value is estimated by default.
        :param epsilon: The epsilon parameter to be used in the model. Best value is estimated by default.
        :param c: The C regularization parameter. Best value is estimated by default.
        :return: A dataframe of predictions and their respective query information, and the NDCG scores for the run.
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
