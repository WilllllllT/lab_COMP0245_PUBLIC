import os
import numpy as np
from matplotlib import pyplot as plt


# Load the data from the file
optimal_result_array = np.load("lab_COMP0245_PUBLIC/optimal_result_arrayEI.npy")
all_tracking_errors_array = np.load("lab_COMP0245_PUBLIC/all_tracking_errors_arrayEI.npy")

def plot_tracking_error(tracking_errors_array):

    size = tracking_errors_array.shape[0]
    # plot the tracking errors over the top of each other
    for i in range(size):
        plt.plot(tracking_errors_array[i], label='Iteration '+str(i))
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.show()

def find_optimal_results(optimal_result_array):
    # seperate each of the results into their own arrays
    optimal_kp = optimal_result_array[:, :7]
    optimal_kd = optimal_result_array[:, 7:14]
    optimal_tracking_error = optimal_result_array[:, -1]

    # find the best tracking error result
    best_result = optimal_result_array[np.argmin(optimal_tracking_error)]

    # Find average tracking error
    average_tracking_error = np.mean(optimal_tracking_error)

    return best_result, optimal_kp, optimal_kd, optimal_tracking_error, average_tracking_error

def plot_optimal_results(optimal_kp, optimal_kd, optimal_tracking_error, average_tracking_error):
    # plot each iteration of all optimal kp values, using different coloured lines to plot each of the 7 values in the array
    for i in range(7):
        plt.plot(optimal_kp[:, i], label='Kp'+str(i+1))
    plt.xlabel('Iteration')
    plt.ylabel('Kp')
    plt.legend()
    plt.show()

    # plot each iteration of all optimal kd values, using different coloured lines to plot each of the 7 values in the array
    for i in range(7):
        plt.plot(optimal_kd[:, i], label='Kd'+str(i+1))
    plt.xlabel('Iteration')
    plt.ylabel('Kd')
    plt.legend()
    plt.show()


    # Plot the average tracking error
    plt.plot(optimal_tracking_error, label='Tracking Error')
    plt.axhline(average_tracking_error, color='r', linestyle='--', label='Average Tracking Error')
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.show()

def convergence_index(tracking_errors):
    # print the the point where the tracking value reached a certain threshold (convergence_limit) for more than 3 consecutive tracking errors in each of the itereations
    convergence_limit = 250
    for i in range(tracking_errors.shape[0]):
        count = 0
        for j in range(tracking_errors.shape[1]):
            if tracking_errors[i, j] < convergence_limit:
                count += 1
            else:
                count = 0
            if count == 5:
                print('Convergence index for iteration '+str(i)+': '+str(j-2))
                break


def main():
    plot_tracking_error(all_tracking_errors_array)
    best_result, optimal_kp, optimal_kd, optimal_tracking_error, average_tracking_error = find_optimal_results(optimal_result_array)
    plot_optimal_results(optimal_kp, optimal_kd, optimal_tracking_error, average_tracking_error)
    convergence_index(all_tracking_errors_array)
    print(best_result)


if __name__ == "__main__":
    main()