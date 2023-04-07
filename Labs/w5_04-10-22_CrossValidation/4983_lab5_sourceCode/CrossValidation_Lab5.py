import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# suppress the FutureWarnings produced by pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('data_lab5.csv')

x = data["x"]
y = data["y"]

"""
Part 2: K-fold Cross-Validation
"""


def poly_kfoldCV(x, y, p, K):
    """
    Finds the training sample_error and cross-validation sample_error for the specified degree of polynomials and
    number of folds.

    :param x: training input
    :param y: training output
    :param p: an integer, the degree of the fitting polynomial
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
        training_model = np.polyfit(training_set[0], training_set[1], p)
        training_predict = np.polyval(training_model, training_set[0])
        train_error_arr.append(np.mean(abs(training_set[1] - training_predict)))

        # combines the validation dataset
        validation_set = [x[lower_range * ratio:fold * ratio], y[lower_range * ratio:fold * ratio]]

        # gets the prediction for the validation dataset
        validation_predict = np.polyval(training_model, validation_set[0])
        cv_error_arr.append(np.mean(abs(validation_set[1] - validation_predict)))

    train_error = np.mean(train_error_arr)
    cv_error = np.mean(cv_error_arr)
    return train_error, cv_error


"""
Part 3: Model Assessment and Selection
"""

sample_error = poly_kfoldCV(x, y, 1, 5)
print(f"\nTraining sample_error = 1.0355?  {round(sample_error[0], 4) == 1.0355}\nCross-Validation sample_error = 1.0848? {round(sample_error[1], 4) == 1.0848}\n")


degrees = [x for x in range(1, 16)]

train_errors = []
cv_errors = []
for degree in degrees:
    errors = poly_kfoldCV(x, y, degree, 5)

    train_errors.append(errors[0])
    cv_errors.append(errors[1])

plt.plot(degrees, train_errors, label="Training Error")
plt.plot(degrees, cv_errors, label="Cross-Validation Error")
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontsize=10
)
plt.title("Line plot of Training Error and Validation Error vs Polynomial Degree")
plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.grid(True)
plt.legend(loc=9)
plt.tight_layout()
plt.show()
plt.close()
print("\n" + "="*30 + "   Part 3: Model Assessment and Selection   " + "="*30 + "\n")
print("Based on the plot generated from Part 3, it seems like the 5th degree polynomial provides a tradeoff between\n"
      "bias and variance. The polynomials before 5 produce low variance but high bias (due to underfitting), while \n"
      "polynomial degrees after 6 start to increase in variance while the bias lowers (due to overfitting).\n\n")

"""
Part 4: Learning Curve
"""

N = [x for x in range(20, 101, 5)]
degrees_part4 = [1, 2, 7, 10, 16]

for degree in degrees_part4:
    train_errors = []
    cv_errors = []

    for size in N:
        x_dataset = x[:size]
        y_dataset = y[:size]

        errors = poly_kfoldCV(x_dataset, y_dataset, degree, 5)

        train_errors.append(errors[0])
        cv_errors.append(errors[1])

    plt.plot(N, train_errors, label="Training Error")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    plt.title(f"Train Error vs Sample Size for a {degree} degree polynomial")
    plt.xlabel("Sample size")
    plt.ylabel("Training Error")
    plt.grid(True)
    plt.legend(loc=9)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.plot(N, cv_errors, label="Cross-Validation Error", color="orange")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontsize=10
    )
    plt.title(f"Cross-Validation Error vs Sample Size for a {degree} deg polynomial")
    plt.xlabel("Sample size")
    plt.ylabel("CV Error")
    plt.grid(True)
    plt.legend(loc=9)
    plt.tight_layout()
    plt.show()
    plt.close()

print("\n" + "="*30 + "   Part 4: Learning Curves   " + "="*30 + "\n")

print("\nNOTE: Sometimes the 10 and 16 degree polynomial graphs aren't and I'm not sure why. Maybe try re-running the "
      "code \nuntil they are generated. Also 'RankWarnings' are thrown due to the extreme values when "
      "using \nnumpy.polyfit() for higher degree polynomials.\n")

print("\nPart 3.a.")
print("The 1 degree polynomial has the highest bias. The Training Error is highest within the line plot for Training "
      "Error \nvs Sample Size for a 1 degree polynomial")

print("\nPart 3.b.")
print("Although the scale on the graph is difficult to read due to extreme values, it seems the 16 degree polynomial "
      "has the \nhighest variance. This would make sense as when increasing the degree of polynomial the model tends"
      "to be overfitted \nthus resulting in a higher variance, but lower bias.")

print("\nPart 4.a.")
print("If only the first 50 samples were provided, I would use a 2nd degree polynomial. This is because the bias and "
      "variance is lowest \nwhen compared to the other degree polynomials. Although, the model may be underfitted to "
      "the data due to the low complexity.")

print("\nPart 4.b.")
print("I would choose the 7 degree polynomial if 80 samples were provided. The 10 and 16 degree polynomials may be "
      "better \nbut the scale of the plots make it impossible to view the variance when the sample size increases, "
      "but on the \nother hand the 10 and 16 degree polynomials would likely be subject to overfitting.")
