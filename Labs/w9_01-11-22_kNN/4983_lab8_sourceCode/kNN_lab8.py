import numpy as np
import pandas as pd
import random
import collections
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from math import sqrt


def kNN(x, Xtrain, Ytrain, k):
    """
    Performs multiclass classification for the provided datasets using Euclidean Distance

    :param x: a test sample
    :param Xtrain: array of input vectors of the training set
    :param Ytrain: array of output values of the training set
    :param k: an int, number of nearest neighbors
    :return: an int, predicted value for the test sample
    """
    x_coord_test = x[0]
    y_coord_test = x[1]

    distances_arr = []
    output_value_arr = []

    for vector in Xtrain:
        if x != vector:  # prevents calculating distance of test sample from itself
            x_vector_coord = vector[0]
            y_vector_coord = vector[1]

            euclid_distance = round(sqrt((x_coord_test - x_vector_coord)**2 + (y_coord_test - y_vector_coord)**2), 5)
            distances_arr.append((euclid_distance, vector))

    distances_arr.sort()
    for neighbor in range(0, k):
        index = Xtrain.index(distances_arr.pop(0)[1])  # gets index of closest neighbor from Xtrain
        output_value_arr.append(Ytrain[index])  # extracts the output value of the closest neighbor

    # initializes Counter object from collections
    counter = collections.Counter(output_value_arr)

    # gets the most frequent value from the output values of the training set
    predicted_val = counter.most_common()[0][0]

    return predicted_val


def kNN_kfoldCV(x, y, p, K):
    """
    Finds the training sample_error and cross-validation sample_error for the specified number of K Nearest Neighbors.

    :param x: training input
    :param y: training output
    :param p: an integer, the number of nearest neighbors
    :param K: an integer, the number of folds
    :return: a tuple, containing the MAE of the Training dataset, and the MAE of the Cross Validation dataset
    """
    ZERO_INDEX_OFFSET = 1

    x = x.to_numpy()
    y = y.to_numpy()

    ratio = int(len(x) / K)
    for fold in range(1, K + ZERO_INDEX_OFFSET):
        lower_range = fold - 1

        # slices the dataset for the validation dataset and indices
        training_set_lower_half = [x[:lower_range * ratio].tolist(), y[:lower_range * ratio].tolist()]
        training_set_upper_half = [x[fold * ratio:].tolist(), y[fold * ratio:].tolist()]

        # combines the sliced training dataset and indices
        training_set = [training_set_lower_half[0] + training_set_upper_half[0], training_set_lower_half[1] + training_set_upper_half[1]]

        # combines the validation dataset and indices
        validation_set = [x[lower_range * ratio:fold * ratio], y[lower_range * ratio:fold * ratio]]

        # trains the model and gets the prediction for the training data
        model = KNeighborsClassifier(n_neighbors=p)
        train_labels_arr = np.array(training_set[1]).reshape(len(training_set[1]))
        model.fit(training_set[0], train_labels_arr)

        pred_train = model.predict(training_set[0])
        pred_validation = model.predict(validation_set[0])

        # gets the number of errors from the training and cross-validation datasets
        train_err = 0
        for index in range(0, len(training_set[0])):
            if pred_train[index] != training_set[1][index]:
                train_err += 1

        validation_err = 0
        for index in range(0, len(validation_set[0])):
            if pred_validation[index] != validation_set[1][index]:
                validation_err += 1

    train_error = train_err / len(training_set[0])
    cv_error = validation_err / len(validation_set[0])
    return train_error, cv_error


if __name__ == '__main__':
    print("==================== Part 2: k-NN Implementation ====================")

    input_vector_arr = [[1, 1], [2, 3], [3, 2], [3, 4], [2, 5]]
    output_value_arr = [0, 0, 0, 1, 1]

    # gets the predicted values for 3 nearest neighbors from the test dataset
    knn_predicted_val_arr = []
    for vector in input_vector_arr:
        knn_predicted_val_arr.append(kNN(vector, input_vector_arr, output_value_arr, 3))

    # trains the sklearn model and retrieves the predicted value(s) from it
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(input_vector_arr, output_value_arr)
    pred = model.predict(input_vector_arr)

    knn_error = 0
    sklearn_error = 0
    for index in range(0, len(knn_predicted_val_arr)):
        if knn_predicted_val_arr[index] != output_value_arr[index]:
            knn_error += 1

        if pred[index] != output_value_arr[index]:
            sklearn_error += 1
    print(f"Step 4: kNN Error: {knn_error / len(output_value_arr)} - KNeighborsClassifier Error: {sklearn_error / len(output_value_arr)}\n")

    print("==================== Part 3: Classification of Handwritten Digits ====================")

    data = pd.read_csv("data_lab8.csv")

    num_rows = data.shape[0]

    shuffled_indices = list(range(num_rows))
    random.seed(42)
    random.shuffle(shuffled_indices)

    train_ratio = 0.75
    train_set_size = int(num_rows * train_ratio)

    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:num_rows]

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    train_features = train_data.drop(columns="digit", inplace=False)
    train_labels = train_data.loc[:, ["digit"]]
    test_features = test_data.drop(columns="digit", inplace=False)
    test_labels = test_data.loc[:, ["digit"]]

    k = [x for x in range(1, 22, 2)]
    train_errors = []
    cv_errors = []
    for neighbor in k:
        errors = kNN_kfoldCV(train_features, train_labels, neighbor, 10)

        train_errors.append(errors[0])
        cv_errors.append(errors[1])

    plt.plot(k, train_errors, label="Training Error")
    plt.plot(k, cv_errors, label="CV Error")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    plt.title(f"Train Errors & CV Errors vs Number of Nearest Neighbors")
    plt.xlabel("Number of Nearest Neighbors")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend(loc=9)
    plt.tight_layout()
    plt.show()
    plt.close()

    print("Step 5: The best value for K seems to be around 11 as that value seems to provide the optimal\n"
          "balance between bias and variance. This is because there seems to be a drop in the\n"
          "cross-validation error when k = 11\n")

    # models and predicts using 11 nearest neighbors
    model = KNeighborsClassifier(n_neighbors=11)
    train_labels_arr = np.array(train_labels).reshape(len(train_labels))
    model.fit(train_features, train_labels_arr)
    output_vals = model.predict(test_features)

    # calculates the error rate at 11 nearest neighbors
    err = 0
    for index in range(0, len(output_vals)):
        if output_vals[index] != test_labels.iloc[index][0]:
            err += 1

    error_rate = err / len(output_vals)
    print(f"Step 6: The error rate for a K-NN value of 11 is {error_rate}\n")
