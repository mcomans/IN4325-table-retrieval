from AdaRank.adarank import AdaRank as ExtAdaRank
from AdaRank.metrics import NDCGScorer
from scorers import ndcg_scorer
import pandas as pd


class AdaRank:
    def __init__(self, x_train, y_train, x_test, y_test, train_info, test_info):
        """
        Initializes the AdaRank class by initializing the data.
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

        self.qid_train = train_info['query_id'].to_numpy()
        self.qid_test = test_info['query_id'].to_numpy()

        self.model = None

    def run(self):
        scorer = NDCGScorer(k=20)
        model = ExtAdaRank(max_iter=500, estop=10, scorer=scorer).fit(self.x_train, self.y_train, self.qid_train)
        y_pred = model.predict(self.x_test, self.qid_test)
        print(scorer(self.y_test, y_pred, self.qid_test).mean())

        scores = dict.fromkeys([5, 10, 15, 20])
        scores[5] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=5)
        scores[10] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=10)
        scores[15] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=15)
        scores[20] = ndcg_scorer(self.y_test, y_pred, info=self.test_info, k=20)

        return self.test_info.join(pd.DataFrame({'score': y_pred})), scores
