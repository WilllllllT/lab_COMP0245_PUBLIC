import os
import numpy as np
from matplotlib import pyplot as plt

print(os.path.abspath("."))
# Load the data from the files
optimal_result_arrayPI = np.load('optimal_result_arrayPI.npy')
optimal_result_arrayEI = np.load('optimal_result_arrayEI.npy')
optimal_result_arrayLCB = np.load('optimal_result_arrayLCB.npy')

all_tracking_errors_arrayPI = np.load('all_tracking_errors_arrayPI.npy')
all_tracking_errors_arrayEI = np.load('all_tracking_errors_arrayEI.npy')
all_tracking_errors_arrayLCB = np.load('all_tracking_errors_arrayLCB.npy')



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

def plot_best_tracking_error(EI, PI, LCB):
    # plot the best tracking error 1 out of 10 for each of the acquisition functions
    plt.plot(PI, label='PI')
    plt.plot(EI, label='EI')
    plt.plot(LCB, label='LCB')
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.show()


def main():
    # plot the tracking errors for each of the acquisition functions
    best_resultPI, optimal_kpPI, optimal_kdPI, optimal_tracking_errorPI, average_tracking_errorPI = find_optimal_results(optimal_result_arrayPI)
    best_resultEI, optimal_kpEI, optimal_kdEI, optimal_tracking_errorEI, average_tracking_errorEI = find_optimal_results(optimal_result_arrayEI)
    best_resultLCB, optimal_kpLCB, optimal_kdLCB, optimal_tracking_errorLCB, average_tracking_errorLCB = find_optimal_results(optimal_result_arrayLCB)

    plot_best_tracking_error(all_tracking_errors_arrayEI[np.argmin(optimal_tracking_errorEI)], all_tracking_errors_arrayPI[np.argmin(optimal_tracking_errorPI)], all_tracking_errors_arrayLCB[np.argmin(optimal_tracking_errorLCB)])


    plot_tracking_error(all_tracking_errors_arrayEI)
    convergence_index(all_tracking_errors_arrayPI)


if __name__ == "__main__":
    main()