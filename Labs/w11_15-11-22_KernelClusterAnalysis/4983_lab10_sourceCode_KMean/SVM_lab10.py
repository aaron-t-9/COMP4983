import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def svc_kfoldCV(x, y, C, K, kernel="linear"):
    """
    Finds the training sample_error and cross-validation sample_error for the specified penalty parameter

    :param x: training input
    :param y: training output
    :param C: an integer, the penalty parameter
    :param K: an integer, the number of folds
    :return: a tuple, containing the MAE of the Training dataset, and the MAE of the Cross Validation dataset
    """
    ZERO_INDEX_OFFSET = 1

    train_error_arr = []
    cv_error_arr = []

    x = x.values.tolist()
    y = y.values.tolist()

    ratio = int(len(x) / K)
    for fold in range(1, K + ZERO_INDEX_OFFSET):
        lower_range = fold - 1

        # slices the dataset for the validation dataset
        training_set_lower_half = [x[:lower_range * ratio], y[:lower_range * ratio]]
        training_set_upper_half = [x[fold * ratio:], y[fold * ratio:]]

        # combines the sliced training dataset
        training_set = [training_set_lower_half[0] + training_set_upper_half[0], training_set_lower_half[1] + training_set_upper_half[1]]
        training_set[1] = np.array(training_set[1]).flatten().tolist()
        # combines the validation dataset
        validation_set = [x[lower_range * ratio:fold * ratio], y[lower_range * ratio:fold * ratio]]

        # trains the model and gets the prediction for the training data
        model = SVC(kernel=kernel, C=C)
        model.fit(training_set[0], training_set[1])

        train_pred = model.predict(training_set[0])
        train_error_arr.append(np.mean(abs(train_pred - training_set[1])))

        cv_pred = model.predict(validation_set[0])
        cv_error_arr.append(np.mean(abs(cv_pred - validation_set[1])))

    train_error = np.mean(train_error_arr)
    cv_error = np.mean(cv_error_arr)
    return train_error, cv_error


if __name__ == '__main__':
    """
    Part 3
    """
    data = pd.read_csv("./data_lab10.csv")

    """ 
    Part 4
    """
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

    train_features = train_data.drop(columns="y", inplace=False)
    train_labels = train_data.loc[:, ["y"]]
    test_features = test_data.drop(columns="y", inplace=False)
    test_labels = test_data.loc[:, ["y"]]

    """
    Part 5
    """
    C = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]

    train_errors = []
    cv_errors = []
    for penalty_param in C:
        errors = svc_kfoldCV(train_features, train_labels, penalty_param, 10)

        train_errors.append(errors[0])
        cv_errors.append(errors[1])

    plt.plot(C, train_errors, label="Training Error")
    plt.plot(C, cv_errors, label="Cross-Validation Error")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    plt.title("Training Error and CV Error vs Penalty Parameter for Linear Kernel")
    plt.xlabel("Penalty Parameter")
    plt.xscale("log")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend(loc=9)
    plt.tight_layout()
    plt.show()
    plt.close()

    """
    Part 6
    """
    # gets the minimum cross-validation error and finds the penalty parameter used
    min_cv_err = np.array(cv_errors).min()
    index = cv_errors.index(min_cv_err)
    print("\n===== Part 6 =====")
    print(f"The best value of C is {C[index]} as it produced the lowest cross-validation error of {min_cv_err:.3f}\n\n")

    """
    Part 7
    """
    model = SVC(kernel="linear", C=C[index])
    train_labels_temp = train_labels.values.flatten()
    model.fit(train_features, train_labels_temp)
    test_pred = model.predict(test_features)
    test_labels = test_labels.values.flatten()

    err_count = 0
    for index in range(0, len(test_pred)):
        if test_pred[index] != test_labels[index]:
            err_count += 1

    print("\n===== Part 7 =====")
    print(f"Using the best value of C, the error rate on the test set is {err_count/len(test_pred)}\n\n")

    """
    Part 8
    """
    plt.scatter(test_features.iloc[:, 0], test_features.iloc[:, 1], label="Scatter plot of the test samples")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    plt.title("Scatter Plot of Test Samples with the Decision Boundary & SVC Margin", fontsize=11)
    plt.xlabel("x-value")
    plt.ylabel("y-value")
    plt.grid(True)
    plt.legend(loc=9)
    plt.tight_layout()
    plot_svc_decision_function(model)
    plt.show()
    plt.close()

    """
    Part 9
    """
    print("\n===== Part 9 =====")
    train_errors = []
    cv_errors = []
    for penalty_param in C:
        errors = svc_kfoldCV(train_features, train_labels, penalty_param, 10, "rbf")

        train_errors.append(errors[0])
        cv_errors.append(errors[1])

    plt.plot(C, train_errors, label="Training Error")
    plt.plot(C, cv_errors, label="Cross-Validation Error")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    plt.title("Training Error and CV Error vs Penalty Parameter for RBF Kernel")
    plt.xlabel("Penalty Parameter")
    plt.xscale("log")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend(loc=9)
    plt.tight_layout()
    plt.show()
    plt.close()

    # gets the minimum cross-validation error and finds the penalty parameter used
    min_cv_err = np.array(cv_errors).min()
    index = cv_errors.index(min_cv_err)
    print(f"The best value of C is {C[index]} as it produced the lowest cross-validation error of {min_cv_err:.3f}")

    model = SVC(kernel="rbf", C=C[index])
    model.fit(train_features, train_labels_temp)
    test_pred = model.predict(test_features)

    err_count = 0
    for index in range(0, len(test_pred)):
        if test_pred[index] != test_labels[index]:
            err_count += 1

    print(f"Using the best value of C, the error rate on the test set is {err_count/len(test_pred)}")

    plt.scatter(test_features.iloc[:, 0], test_features.iloc[:, 1], label="Scatter plot of the test samples")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    plt.title("Scatter Plot of Test Samples with the Decision Boundary & SVC Margin", fontsize=11)
    plt.xlabel("x-value")
    plt.ylabel("y-value")
    plt.grid(True)
    plt.legend(loc=9)
    plt.tight_layout()
    plot_svc_decision_function(model)
    plt.show()
    plt.close()
