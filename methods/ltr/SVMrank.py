from load_data import get_write_file


def _write(features, filename):
    """
    Separates the query information from the features and writes the features to a SVMrank format input file.
    :param features: A DataFrame containing the features to write.
    :param filename: The filename to write to (in the data/svm_rank dir).
    """
    stripped_features = features.drop(['query_id', 'query', 'table_id', 'rel'], axis=1)

    file = get_write_file('data/svm_rank', filename)
    print(f"Writing SVMrank input to data/svm_rank/{filename}")

    for index, row in features.iterrows():
        file.write(f"{row['rel']} qid:{row['query_id']} ")
        col_idx = 1
        for col in stripped_features:
            file.write(f"{col_idx}:{stripped_features.iloc[index][col]} ")
            col_idx += 1
        file.write(f"# {row['table_id']}\n")

    file.close()


class SVMrank:
    def __init__(self, train, test):
        """
        Initialize an SVMrank object with the train and test features as DataFrames.
        :param train: A DataFrame containing the training features.
        :param test: A DataFrame containing the test features.
        """
        self.train = train
        self.test = test

    def write_input_files(self):
        """
        Writes the train and test sets to separate SVMrank input files.
        :return:
        """
        _write(self.train, 'train.dat')
        _write(self.test, 'test.dat')
