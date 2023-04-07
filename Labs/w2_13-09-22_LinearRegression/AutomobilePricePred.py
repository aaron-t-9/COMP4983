import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# load automobile price data
# ENSURE "AutomobilePrice.csv" is under your working directory
data = pd.read_csv("AutomobilePrice.csv")
# data should contain 205 rows and 26 columns
print('\n\ndata.info():')
print(data.info())

# data preprocessing
# remove the 'normalized-losses' column
data.drop(???, axis=???, inplace=True)

# remove rows that contain missing values
# first replace '?' in the data with special values 'nan'
# 'nan' (not a number) is defined in the numpy module
data.replace(to_replace='?', value=np.nan, inplace=True)
# remove any rows that contain 'nan'
data.dropna(axis=???, how=???, inplace=True)

# select a subset of features for regression:
# 'make', 'body-style', 'wheel-base', 'engine-size', 'horsepower',
# 'peak-rpm', 'highway-mpg', 'price'
data_subset = data.loc[:, ???]

# ENSURE 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm',
# 'highway-mpg', and 'price' are handled as 'float64' type
data_subset[['wheel-base', 'engine-size', 'horsepower', 'peak-rpm','highway-mpg', 'price']] \
= data_subset[['wheel-base', 'engine-size', 'horsepower', 'peak-rpm','highway-mpg', 'price']].astype(???)

# encode categorical values in 'make' and 'body-style' using one hot encoding
# which convert each category value into a new column and assigns a 1 or 0
# (True/False) value to the column
data_subset = pd.get_dummies(data_subset, columns = ['make', 'body-style'], \
                             prefix=['make', 'body-style'])

# data_subset now should contain 193 rows and 32 columns
print('\n\ndata_subset.info():')
print(???)

# split the data into training set and test set
# use 75 percent of the data to train the model and hold back 25 percent
# for testing
train_ratio = ???
# number of samples in the data_subset
num_rows = data_subset.shape[0]
# shuffle the indices
shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

# calculate the number of rows for training 
train_set_size = int(???)

# training set: take the first 'train_set_size' rows
train_indices = shuffled_indices[:train_set_size]
# test set: take the remaining rows
test_indices = shuffled_indices[???]

# create training set and test set
train_data = data_subset.iloc[train_indices, :]
test_data = data_subset.iloc[test_indices, ???]
print(len(train_data), "training samples + ", len(test_data), "test samples")

# prepare training features and training labels
# features: all columns except 'price'
# labels: 'price' column
train_features = train_data.drop(???, axis=???, inplace=False)
train_labels = train_data.loc[???,???]

# prepare test features and test labels
test_features = ???
test_labels = ???

# train a linear regression model using training data
lin_reg = LinearRegression()
# Note: you can ingore warning messages regarding gelsd driver
lin_reg.fit(???, ???)

# predict new prices on test data
price_pred = lin_reg.predict(???)

# compute mean absolute error
# formula: MAE = mean of | y_i - y_i_pred |
# where y_i is the i-th element of test_labels
#       y_i_pred is the i-th element of the price_pred
# Ref: https://en.wikipedia.org/wiki/Mean_absolute_error
mae = np.mean(???)
print('Mean Absolute Error = ', mae)

# compute root means square error
# formula: RMSE = square root of mean of ( y_i - y_i_pred )^2
# Ref: https://en.wikipedia.org/wiki/Root-mean-square_deviation
rmse = np.sqrt(???)
print('Root Mean Squared Error = ', rmse)

# compute coefficient of determination (aka R squared)
# formula: CoD = 1 - SSres/SStot, where
# SSres = sum of squares of ( y_i - y_i_pred )
# SStot = sum of squares of ( y_i - mean of y_i )
# Ref: https://en.wikipedia.org/wiki/Coefficient_of_determination
total_sum_sq = ???
res_sum_sq = ???
CoD = 1 - (res_sum_sq/total_sum_sq)
print('Coefficient of Determination = ', CoD)

