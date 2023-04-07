import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load(file_path: str):
    """
    Handles opening the specific CSV file when provided the file path.

    :param file_path: a string, the file path for the CSV to be opened
    :return: returns a Pandas DataFrame object
    """
    data = pd.read_csv(file_path, low_memory=False)

    return data

# file_path = "/Users/at/My Drive/Files/School/Term4/COMP4983/Labs/w6_11-10-22_RidgeRegressionLasso/4983_lab6/data_lab6.csv"
# file_path = "/Users/at/My Drive/Files/School/Term4/COMP4983/Labs/w6_11-10-22_RidgeRegressionLasso/4983_lab6/data_lab6_expanded.csv"
# data = load(file_path)
# print(data.head())

# ohe = OneHotEncoder(sparse=False)
# df = data.loc[:, "OSOURCE"]
# print(df.head())
#
# # data.head()
# print("debug")
