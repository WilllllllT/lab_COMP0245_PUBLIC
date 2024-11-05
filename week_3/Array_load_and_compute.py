import os
import numpy as np
from matplotlib import pyplot as plt

print(os.path.abspath("."))
# Load the data from the files
optimal_result_arrayPI = np.load('optimal_result_arrayPI.npy')
optimal_result_arrayEI = np.load('optimal_result_arrayEI.npy')
optimal_result_arrayLCB = np.load('optimal_result_arrayLCB.npy')

all_tracking_errors_arrayPI = np.load('l_tracking_errors_arrayPI.npy')
all_tracking_errors_arrayEI = np.load('l_tracking_errors_arrayEI.npy')
all_tracking_errors_arrayLCB = np.load('l_tracking_errors_arrayLCB.npy')



def plot_tracking_error(tracking_errors_array, name):

    size = tracking_errors_array.shape[0]
    # plot the tracking errors over the top of each other
    for i in range(size):
        plt.plot(np.log(tracking_errors_array[i]), label='Iteration '+str(i))
    plt.title(f"A plot of the log of tracking errors for {name}")
    plt.xlabel('Iteration')
    plt.ylabel('log(Tracking Error)')
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

def plot_optimal_results(optimal_kp, optimal_kd, optimal_tracking_error, average_tracking_error, name):
    # scatter plot of optimal kp values with joint on the x axis and kp value on the y axis
    for i in range(7):
        plt.scatter([i+1]*optimal_kp.shape[0], optimal_kp[:, i])
    plt.title(f"Optimal Kp for each iteration using the {name} acquisition function")
    plt.xlabel('Joint')
    plt.ylabel('Kp')
    plt.show()

    # scatter plot of optimal kp values with joint on the x axis and kp value on the y axis
    for i in range(7):
        plt.scatter([i+1]*optimal_kd.shape[0], optimal_kd[:, i])
    plt.title(f"Optimal Kd for each iteration using the {name} acquisition function")
    plt.xlabel('Joint')
    plt.ylabel('Kp')
    plt.show()


    # Plot the average tracking error
    plt.plot(optimal_tracking_error, label='Tracking Error')
    plt.axhline(average_tracking_error, color='r', linestyle='--', label='Average Tracking Error = '+str(round(average_tracking_error, 2)))
    # add the best tracking error to the plot
    plt.scatter(np.argmin(optimal_tracking_error), np.min(optimal_tracking_error), color='g', label='Best Tracking Error = '+str(round(np.min(optimal_tracking_error), 2)))
    plt.title(f"Tracking Error for each iteration using the {name} acquisition function")
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.show()

def convergence_index(tracking_errors, name):
    # print the the point where the tracking value reached a certain threshold (convergence_limit) for more than 3 consecutive tracking errors in each of the itereations
    convergence_index = []
    convergence_limit = 200
    for i in range(tracking_errors.shape[0]):
        count = 0
        for j in range(tracking_errors.shape[1]):
            if tracking_errors[i, j] < convergence_limit:
                count += 1
            else:
                count = 0
            if count == 5:
                convergence_index.append(i)
                print(f'Convergence index for {name} iteration '+str(i)+': '+str(j-2))
                break
    return convergence_index

def plot_best_tracking_error(EI, PI, LCB):
    # plot the best tracking error 1 out of 10 for each of the acquisition functions
    plt.plot(PI, label='PI')
    plt.plot(EI, label='EI')
    plt.plot(LCB, label='LCB')
    plt.title("The Best Tracking Error for each acquisition function")
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.show()

def plot_best_result(best_resultEI, best_resultPI, best_resultLCB):
    # plot the best result for kp and each of the acquisition functions
    plt.plot(best_resultPI[:7], label='PI')
    plt.plot(best_resultEI[:7], label='EI')
    plt.plot(best_resultLCB[:7], label='LCB')
    plt.title("Optimal Kp values for each acquisition function")
    plt.xlabel('Joint')
    plt.ylabel('Kp')
    plt.legend()
    plt.show()

    # plot the best result for kd and each of the acquisition functions
    plt.plot(best_resultPI[7:14], label='PI')
    plt.plot(best_resultEI[7:14], label='EI')
    plt.plot(best_resultLCB[7:14], label='LCB')
    plt.title("Optimal Kd values for each acquisition function")
    plt.xlabel('Joint')
    plt.ylabel('Kd')
    plt.legend()
    plt.show()

def plot_converged_errors(tracking_errors_array, name):
    # plot the tracking errors for each of the acquisition functions that reached convergence based on the convergence index
    index = convergence_index(tracking_errors_array, name)
    for i in index:
        plt.plot(tracking_errors_array[i], label='Iteration '+str(i))
    plt.title(f"A plot of the tracking errors for {name} that reached convergence")
    plt.xlabel('Iteration')
    plt.ylabel('Tracking Error')
    plt.legend()
    plt.show()

def print_best_tracking_error(EI, PI, LCB):
    # print the best tracking error for each of the acquisition functions
    print('Best Tracking Error for PI: ', np.min(PI))
    print('Best Tracking Error for EI: ', np.min(EI))
    print('Best Tracking Error for LCB: ', np.min(LCB))

    


def main():
    # plot the tracking errors for each of the acquisition functions
    best_resultPI, optimal_kpPI, optimal_kdPI, optimal_tracking_errorPI, average_tracking_errorPI = find_optimal_results(optimal_result_arrayPI)
    best_resultEI, optimal_kpEI, optimal_kdEI, optimal_tracking_errorEI, average_tracking_errorEI = find_optimal_results(optimal_result_arrayEI)
    best_resultLCB, optimal_kpLCB, optimal_kdLCB, optimal_tracking_errorLCB, average_tracking_errorLCB = find_optimal_results(optimal_result_arrayLCB)

    print_best_tracking_error(optimal_tracking_errorEI, optimal_tracking_errorPI, optimal_tracking_errorLCB)

    plot_tracking_error(all_tracking_errors_arrayPI, 'PI')
    plot_tracking_error(all_tracking_errors_arrayEI, 'EI')
    plot_tracking_error(all_tracking_errors_arrayLCB, 'LCB')
    
    #plot best tracking error
    plot_best_tracking_error(optimal_tracking_errorEI, optimal_tracking_errorPI, optimal_tracking_errorLCB)

    #plot convereged errors
    plot_converged_errors(all_tracking_errors_arrayPI, 'PI')
    plot_converged_errors(all_tracking_errors_arrayEI, 'EI')
    plot_converged_errors(all_tracking_errors_arrayLCB, 'LCB')

    #plot optimal results
    plot_optimal_results(optimal_kpPI, optimal_kdPI, optimal_tracking_errorPI, average_tracking_errorPI, 'PI')
    plot_optimal_results(optimal_kpEI, optimal_kdEI, optimal_tracking_errorEI, average_tracking_errorEI, 'EI')
    plot_optimal_results(optimal_kpLCB, optimal_kdLCB, optimal_tracking_errorLCB, average_tracking_errorLCB, 'LCB')

    #plot best result
    plot_best_result(best_resultPI, best_resultEI, best_resultLCB)
    print(best_resultPI)

    print

    


if __name__ == "__main__":
    main()