import pandas as pd
import numpy as np
import random
import itertools

# suppress the FutureWarnings produced by pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Part 2: Linear Regression using Python
"""

# load automobile price data
# ENSURE "AutomobilePrice.csv" is under your working directory
data = pd.read_csv("AutomobilePrice_Lab3.csv")
# data should contain 205 rows and 26 columns
print('\n\ndata.info():')
print(data.info())

# inserting column of 1's into the matrix for beta-hat
data.insert(0, 'beta-hat', [1 for x in range(0, 195)])

# split the data into training set and test set
# use 75 percent of the data to train the model and hold back 25 percent
# for testing
train_ratio = 0.75
# number of samples in the data_subset
num_rows = data.shape[0]
# shuffle the indices
shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

# calculate the number of rows for training
train_set_size = int(num_rows * train_ratio)

# training set: take the first 'train_set_size' rows
train_indices = shuffled_indices[:train_set_size]
# test set: take the remaining rows
test_indices = shuffled_indices[train_set_size:num_rows]

# create training set and test set
train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]
print(len(train_data), "training samples + ", len(test_data), "test samples")

# prepare training features and training labels
# features: all columns except 'price'
# labels: 'price' column
train_features = train_data.drop(columns="price", inplace=False)
train_labels = train_data.loc[:, ["price"]]

# prepare test features and test labels
test_features = test_data.drop(columns="price", inplace=False)
test_labels = test_data.loc[:, ["price"]]
# print(len(train_input), "training samples + ", len(test_input), "test samples")

"""
Part 3: Linear Regression Model Implementation
"""


def comp4983_lin_reg_fit(X, y):
    """
    Fit linear regression model.
    :param X: a DataFrame; training input
    :param y: a DataFrame, training output
    :return: beta-hat; estimated weights of the linear regression model
    """
    w = np.matmul((np.linalg.inv(np.matmul(np.transpose(X), X))), (np.matmul(np.transpose(X), y)))

    return w


def comp4983_lin_reg_predict(X, w):
    """
    Predict using the linear regression model
    :param X: a DataFrame; test input
    :param w: a DataFrame; estimated weights from comp4983_lin_reg_fit()
    :return: a DataFrame; predicted output
    """
    y = np.dot(X, w)

    return y


w = comp4983_lin_reg_fit(train_features, train_labels)
y = comp4983_lin_reg_predict(test_features, w)

# compute mean absolute error
# formula: MAE = mean of | y_i - y_i_pred |
# where y_i is the i-th element of test_labels
#       y_i_pred is the i-th element of the price_pred
# Ref: https://en.wikipedia.org/wiki/Mean_absolute_error
mae = np.mean(abs(test_labels - y))
print('Mean Absolute Error = ', mae)

# compute root means square error
# formula: RMSE = square root of mean of ( y_i - y_i_pred )^2
# Ref: https://en.wikipedia.org/wiki/Root-mean-square_deviation
rmse = np.sqrt(np.mean((test_labels - y) ** 2))
print('Root Mean Squared Error = ', rmse)

# compute coefficient of determination (aka R squared)
# formula: CoD = 1 - SSres/SStot, where
# SSres = sum of squares of ( y_i - y_i_pred )
# SStot = sum of squares of ( y_i - mean of y_i )
# Ref: https://en.wikipedia.org/wiki/Coefficient_of_determination
total_sum_sq = ((test_labels - y)).sum()
res_sum_sq = ((test_labels - np.mean(test_labels))).sum()
CoD = 1 - (res_sum_sq / total_sum_sq)
print('Coefficient of Determination = ', CoD)

"""
Part 4: Feature Selection
price pred = line_reg_predic(new test feat, line_reg_fit(new train feat, train label))
"""

# create all possible combinations of the features
feature_combinations = list(itertools.combinations(data.columns.values, 2))
mae_features_dict = {}  # dictionary containing each of their features and their MAE

for feature_combination in feature_combinations:

    # removes the beta-hat and price column from the dataset
    if feature_combination[0] not in ["beta-hat", "price"] and feature_combination[1] not in ["beta-hat", "price"]:
        # pulls the data for the 2 features for the training and test data set
        temp_train_feature = train_data.loc[:, [feature_combination[0], feature_combination[1]]]
        temp_test_features = test_data.loc[:, [feature_combination[0], feature_combination[1]]]

        # trains the model of the 2 features and then predicts the price of the 2 features
        w = comp4983_lin_reg_fit(temp_train_feature, train_labels)
        y = comp4983_lin_reg_predict(temp_test_features, w)

        # compute mean absolute error
        # formula: MAE = mean of | y_i - y_i_pred |
        # where y_i is the i-th element of test_labels
        #       y_i_pred is the i-th element of the price_pred
        # Ref: https://en.wikipedia.org/wiki/Mean_absolute_error
        mae = np.mean(abs(test_labels - y))

        print("\n")
        print("=" * 10 + f" MAE for: {feature_combination[0].title()} and {feature_combination[1].title()} " + "=" * 10)
        print('Mean Absolute Error = ', mae)
        key = f"{feature_combination[0]}_{feature_combination[1]}"
        mae_features_dict[key] = mae[0]

# gets the smallest MAE, and then gets the features with the smallest MAE, and outputs those values
smallest_mae = min(mae_features_dict.values())
features = None
for key, value in mae_features_dict.items():
    if value == smallest_mae:
        features = key
print(f"\n\n{features} have the smallest MAE at {smallest_mae}")
