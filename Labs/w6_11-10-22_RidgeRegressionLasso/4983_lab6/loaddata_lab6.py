import pandas as pd


def load(file_path: str):
    """
    Handles opening the specific CSV file when provided the file path.

    :param file_path: a string, the file path for the CSV to be opened
    :return: returns a Pandas DataFrame object
    """
    data = pd.read_csv(file_path, low_memory=False)

    return data
