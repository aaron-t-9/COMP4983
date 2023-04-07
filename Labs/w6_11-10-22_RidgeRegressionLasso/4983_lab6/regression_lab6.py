import loaddata_lab6 as loader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import random

# suppress the FutureWarnings produced by pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = loader.load("/Users/at/My Drive/Files/School/Term4/COMP4983/Labs/w6_11-10-22_RidgeRegressionLasso/4983_lab6/data_lab6_expanded.csv")

num_rows = data.shape[0]

shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

train_ratio = 0.3
train_set_size = int(num_rows * train_ratio)

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:num_rows]

train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]

train_features = train_data.drop(columns="TARGET_D", inplace=False)
train_labels = train_data.loc[:, ["TARGET_D"]]
test_features = test_data.drop(columns="TARGET_D", inplace=False)
test_labels = test_data.loc[:, ["TARGET_D"]]

lin_reg = LinearRegression()
lin_reg.fit(train_features, train_labels)

lin_pred = lin_reg.predict(test_features)

print("\n" + "="*30 + " Part 1 - Step 5 " + "="*30)
mae = np.mean(abs(test_labels - lin_pred))
print('Mean Absolute Error = ', mae)
print("="*77 + "\n\n")


print("==================== Part 2: Ridge Regression ====================\n\n")


def ridge_kfoldCV(x, y, lam, K):
    """
    Finds the training sample_error and cross-validation sample_error for the specified regularization parameter using
    Ridge Regression.

    :param x: training input
    :param y: training output
    :param lam: the regularization/tuning parameter
    :param K: an integer, the number of folds
    :return: a tuple, containing the MAE of the Training dataset, and the MAE of the Cross Validation dataset
    """
    ZERO_INDEX_OFFSET = 1

    train_error_arr = []
    cv_error_arr = []

    ratio = int(len(x) / K)
    for fold in range(1, K + ZERO_INDEX_OFFSET):
        lower_range = fold - 1

        # slices the dataset for the validation dataset
        training_set_lower_half = [x[:lower_range * ratio].tolist(), y[:lower_range * ratio].tolist()]
        training_set_upper_half = [x[fold * ratio:].tolist(), y[fold * ratio:].tolist()]

        # combines the sliced training dataset
        training_set = [training_set_lower_half[0] + training_set_upper_half[0], training_set_lower_half[1] + training_set_upper_half[1]]

        # trains the model and gets the prediction for the training data
        ridge = Ridge(lam)
        ridge.fit(training_set[0], training_set[1])
        training_predict = ridge.predict(training_set[0])
        train_error_arr.append(np.mean(abs(training_set[1] - training_predict)))

        # combines the validation dataset
        validation_set = [x[lower_range * ratio:fold * ratio], y[lower_range * ratio:fold * ratio]]

        # gets the prediction for the validation dataset
        validation_predict = ridge.predict(validation_set[0])
        cv_error_arr.append(np.mean(abs(validation_set[1] - validation_predict)))

    train_error = np.mean(train_error_arr)
    cv_error = np.mean(cv_error_arr)
    return train_error, cv_error

# initializes Scaler obj
scaler = StandardScaler()
labels = data.loc[:, ["TARGET_D"]].values  # extract labels from dataset
processed_data = scaler.fit_transform(data.iloc[:, :-1])  # standardizes the dataset

# initialize train & test features & labels
train_features = processed_data[train_indices]
train_labels = labels[train_indices]
test_labels = labels[test_indices]
test_features = processed_data[test_indices]

# initialize tuning parameters for Ridge Regression
tuning_params = [float(10 ** x) for x in range(-3, 11)]

# initialize empty arrays
train_errors = []
cv_errors = []

# iterate through each tuning parameter to train the Ridge Regression Model & to get the train and CV errors
for param in tuning_params:

    errors = ridge_kfoldCV(train_features, train_labels, param, 5)
    train_errors.append(errors[0])
    cv_errors.append(errors[1])

    print("-"*25)
    print(f"Ridge Regression Tuning Parameter: {param}")
    print(f"Training MAE: {train_errors[-1]}")
    print(f"CV MAE: {cv_errors[-1]}")
    print("-"*25 + "\n")

plt.plot(tuning_params, train_errors, label="Training Error")
plt.plot(tuning_params, cv_errors, label="CV Error")
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontsize=10
)
plt.xscale("log")
plt.title(f"Train Errors & CV Errors vs Tuning Parameter For Ridge Regression")
plt.xlabel("Tuning Parameter")
plt.ylabel("Training Error")
plt.ylim(0, 20)
plt.grid(True)
plt.legend(loc=9)
plt.tight_layout()
plt.show()
plt.close()

print("It seems the tuning parameter 10^4 would be the best compromise between bias and "
      "variance for the model.")

print("\n" + "="*30 + " Part 2 - Step 6 " + "="*30)
ridge = Ridge(10**4)
ridge.fit(train_features, train_labels)
test_pred = ridge.predict(test_features)
mae = np.mean(abs(test_labels - test_pred))
print('Mean Absolute Error = ', mae)

print("\nThe MAE on the test set is significantly lower than the MAE produces from the Linear Regression\n"
      "model from Part 1. The MAE from the Ridge Regression is lower by over a factor of 10^5 than from the\n"
      "Linear Regression model.")
print("="*77 + "\n\n")


print("==================== Part 3: The Lasso ====================\n\n")


def lasso_kfoldCV(x, y, lam, K):
    """
    Finds the training sample_error and cross-validation sample_error for the specified regularization parameter using
    applying Lasso.

    :param x: training input
    :param y: training output
    :param lam: the regularization/tuning parameter
    :param K: an integer, the number of folds
    :return: a tuple, containing the MAE of the Training dataset, and the MAE of the Cross Validation dataset
    """
    ZERO_INDEX_OFFSET = 1

    train_error_arr = []
    cv_error_arr = []

    ratio = int(len(x) / K)
    for fold in range(1, K + ZERO_INDEX_OFFSET):
        lower_range = fold - 1

        # slices the dataset for the validation dataset
        training_set_lower_half = [x[:lower_range * ratio].tolist(), y[:lower_range * ratio].tolist()]
        training_set_upper_half = [x[fold * ratio:].tolist(), y[fold * ratio:].tolist()]

        # combines the sliced training dataset
        training_set = [training_set_lower_half[0] + training_set_upper_half[0], training_set_lower_half[1] + training_set_upper_half[1]]

        # trains the model and gets the prediction for the training data
        lasso = Lasso(alpha=lam)
        lasso.fit(training_set[0], training_set[1])
        training_predict = lasso.predict(training_set[0])
        train_error_arr.append(np.mean(abs(training_set[1] - training_predict)))

        # combines the validation dataset
        validation_set = [x[lower_range * ratio:fold * ratio], y[lower_range * ratio:fold * ratio]]

        # gets the prediction for the validation dataset
        validation_predict = lasso.predict(validation_set[0])
        cv_error_arr.append(np.mean(abs(validation_set[1] - validation_predict)))

    train_error = np.mean(train_error_arr)
    cv_error = np.mean(cv_error_arr)
    return train_error, cv_error

# initialize empty arrays
train_errors_lasso = []
cv_errors_lasso = []

# initialize the array containing the tuning parameter values
temp_arr = np.arange(-2, 2.25, 0.25)
tuning_params_lasso = []
for coefficient in temp_arr:
    tuning_params_lasso.append(10**coefficient)

# iterate through each tuning parameter to train the Lasso Model & to get the train and CV errors
for param in tuning_params_lasso:
    errors = lasso_kfoldCV(train_features, train_labels, param, 5)
    train_errors_lasso.append(errors[0])
    cv_errors_lasso.append(errors[1])

    print("-"*25)
    print(f"LASSO Tuning Parameter: {param}")
    print(f"Training MAE: {train_errors_lasso[-1]}")
    print(f"CV MAE: {cv_errors_lasso[-1]}")
    print("-"*25 + "\n")

plt.plot(tuning_params_lasso, train_errors_lasso, label="Training Error")
plt.plot(tuning_params_lasso, cv_errors_lasso, label="CV Error")
plt.xscale("log")
plt.title(f"Train Errors & CV Errors vs Tuning Parameter For Lasso")
plt.xlabel("Tuning Parameter")
plt.ylabel("Training Error")
plt.grid(True)
plt.legend(loc=9)
plt.tight_layout()
plt.show()
plt.close()

print("A tuning parameter of 10**0 may be the best tuning parameter to use.")
print("\n" + "="*30 + " Part 3 - Step 4 " + "="*30)
lasso = Lasso(10**0)
lasso.fit(train_features, train_labels)
test_pred_lasso = lasso.predict(test_features)
mae = np.mean(abs(test_labels - test_pred_lasso))
print('Mean Absolute Error = ', mae)

print("\nThe MAE from using Lasso is slightly lower than the MAE using Ridge Regression, and is significantly\n"
      "lower than the MAE from using Linear Regression.")
print("="*77 + "\n\n")

print("\n" + "="*30 + " Part 3 - Step 5 " + "="*30)
print("The most suitable model to predict the amount a donor would donate would be the Ridge Regression model.\n"
      "This is because I feel the Ridge Regression model produces learning curves that aren't as unpredictable.")
print("="*77 + "\n\n")

top_features_arr = []  # initialize empty arr
lasso_coeffs = lasso.coef_  # get the coefficients calculated through Lasso

for _ in range(0, 3):
    # gets the index of th feature with the largest coefficient
    index = lasso_coeffs.argmax()

    # extracts the name of the feature with the largest coefficient
    top_features_arr.append(data.iloc[:, index].name)

    # removes the feature with the largest coefficient in order to find the next feature with the largest coefficient
    lasso_coeffs = np.delete(lasso_coeffs, np.where(lasso_coeffs == lasso_coeffs[index]))

print("\n" + "="*30 + " Part 3 - Step 6 " + "="*30)
print(f"The features that exhibit the strongest effect for Lasso are: {top_features_arr}")
print("="*77 + "\n\n")
