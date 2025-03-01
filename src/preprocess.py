import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data_path):
        """
        data_path — the path to the folder containing BankNote_Authentication.csv
        """
        self.data_path = data_path

    def load_data(self):
        """
        Loads the CSV from self.data_path.
        """
        file_path = os.path.join(self.data_path, 'BankNote_Authentication.csv')
        data = pd.read_csv(file_path)
        return data

    def split_data(self, data, test_size=0.1, random_state=36):
        """
        Splits the DataFrame into X_train, X_test, y_train, y_test.
        """
        X = data[['variance', 'skewness', 'curtosis', 'entropy']]
        y = data['class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        return X_train, X_test, y_train, y_test