import GenData_Lab4 as lab4
import numpy
import matplotlib.pyplot as plt

p = [1, 3, 5, 9, 15]  # contains the degrees of the polynomials
actual_y_at_five = lab4.f(5)  # gets the actual value of f(x) where x = 5

# creates the 1000 data sets
num_of_datasets = 1000
datasets = []
for _ in range(0, num_of_datasets):
    datasets.append(lab4.genNoisyData())

"""
Part 4:
"""
# initialize arrays as empty
fitted_datasets = []
y_predictions = []

# iterate through the degrees
for degree in p:

    # trains the data for the current degree and places values into a list
    for dataset in datasets:
        fitted_datasets.append(numpy.polyfit(dataset[0], dataset[1], degree))

    # gets the predicted values where x = 5 for the fitted datasets and places values into a list
    for fitted_dataset in fitted_datasets:
        y_predictions.append(numpy.polyval(fitted_dataset, 5))

    plt.hist(y_predictions, bins=30)
    # mean_of_pred_line = plt.axvline(numpy.mean(y_predictions), color='red', linestyle='-', label=('mean of $y^{pred}$(x=5)'))
    # actual_y_line = plt.axvline(actual_y_at_five, color='black', linestyle='-', label='f(x = 5)')
    # plt.xticks(
    #     rotation=90,
    #     horizontalalignment='right',
    #     fontsize=10
    # )
    # plt.title(f"Histogram for p = {degree}")
    # plt.xlabel("$y^{pred}$ (x = 5)")
    # plt.ylabel("Counts")
    # plt.legend(handles=[mean_of_pred_line, actual_y_line])
    # plt.xlim([3.5, 6.5])
    # plt.tight_layout()
    plt.show()
    plt.close()

    print("\n" + "="*15 + f" {degree} Degree Polynomial " + "="*15)
    print(f"Mean of predicted y-values: {numpy.mean(y_predictions)}\nActual y-value: {actual_y_at_five}")
    print(f"Bias = {abs(numpy.mean(y_predictions) - actual_y_at_five)}")
    print(f"Variance = {abs(numpy.var(y_predictions))}")
    print("_"*50 + "\n")

    fitted_datasets = []
    y_predictions = []

"""
Part 5:
"""
print("\n\033[1m" + "As observed from the output above, the 1st degree polynomial gives the lowest variance, whereas the 15th degree "
      "polynomial gives the lowest bias." + "\n\033[0m")
