from skopt import gp_minimize
from skopt.space import Real
import numpy as np
import os
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from gp_functions import fit_gp_model_1d, plot_gp_results_1d
import time


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir, use_gui=False)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

acq_func = 'LCB' # Acquisition function for Bayesian optimization

print(f"Initial joint angles: {init_joint_angles}")

# Sinusoidal reference
# Specify different amplitude values for each joint
amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
# Specify different frequency values for each joint
frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

# Convert lists to NumPy arrays for easier manipulation in computations
amplitude = np.array(amplitudes)
frequency = np.array(frequencies)
ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference


# Global lists to store data
kp0_values = []
kd0_values = []
kp_all = []
kd_all = []
tracking_errors = []
optimal_result = []
tracking_errors_array = []

def simulate_with_given_pid_values(sim_, kp, kd, episode_duration=10):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    
    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
    
        # Compute sinusoidal reference trajectory
        q_des, qd_des = ref.get_values(current_time)  # Desired position and velocity
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        
        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
       
    # Calculate tracking error
    q_mes_all = np.array(q_mes_all)
    q_des_all = np.array(q_d_all)

    tracking_error = np.sum((q_mes_all - q_des_all)**2)  # Sum of squared error as the objective
    # print tracking error
    print("Tracking error: ", tracking_error)
    # print PD gains
    print("kp: ", kp)
    print("kd: ", kd)
    
    return tracking_error


# Objective function for optimization
def objective(params):
    kp = np.array(params[:7])  # First 7 elements correspond to kp
    kd = np.array(params[7:])  # Last 7 elements correspond to kd
    episode_duration = 10
    
    # TODO Call the simulation with given kp and kd values
    tracking_error = simulate_with_given_pid_values(sim, kp, kd, episode_duration)
    tracking_errors.append(tracking_error)



    # TODO Collect data for the first kp and kd  
    kp0_values.append(kp[0])
    kd0_values.append(kd[0])

    kp_all.append(kp)
    kd_all.append(kd)
    
    
    return tracking_error

def plot_tracking_error(tracking_errors):
    plt.figure(figsize=(10, 8))
    plt.plot(tracking_errors)
    plt.xlabel("Iteration")
    plt.ylabel("Tracking error")
    plt.title("Tracking error vs. Iteration")
    plt.show()



def main():
    # Define the search space for Kp and Kd
   # Define the search space as before
    space = [
        Real(0.1, 1000, name=f'kp{i}') for i in range(7)
    ] + [
        Real(0.0, 200, name=f'kd{i}') for i in range(7)
    ]

    rbf_kernel = RBF(
    length_scale=1.0,            # Initial length scale
    length_scale_bounds=(1e-2, 1e2)  # Bounds for length scale
    )


    gp = GaussianProcessRegressor(
    kernel=rbf_kernel,
    normalize_y=True,
    n_restarts_optimizer=10  # Optional for better hyperparameter optimization
    )

    #create np arrays to store the data
    optimal_result_array = []
    all_tracking_errors_array = []

    acq_func = 'EI'

    #start the times
    start = time.time()

    shape = (10, 150)



    for i in range(shape[0]):    
        # Perform Bayesian optimization
        result = gp_minimize(
        objective,
        space,
        n_calls=shape[1],  # Number of iterations
        base_estimator=gp,  # Use the custom Gaussian Process Regressor
        acq_func=acq_func,      # TODO change this LCB': Lower Confidence Bound 'EI': Expected Improvement 'PI': Probability of Improvement
        random_state=10+i)

        
        # Extract the optimal values and reshape the array to add them to the next iteraion
        optimal_result = np.concatenate((result.x, [result.fun]))

        # concatinate all data to a large array and save the file
        #data = np.concatenate((kp_all_array, kd_all_array, tracking_errors_array.reshape(-1, 1)), axis=1)

        # add the optimal result to the optimal result array
        optimal_result_array.append(optimal_result)

    #convert the arrays to np arrays
    optimal_result_array = np.array(optimal_result_array)
    all_tracking_errors_array = np.array(tracking_errors).reshape(shape)

    np.save(f"optimal_result_array{acq_func}.npy", optimal_result_array)
    np.save(f"all_tracking_errors_array{acq_func}.npy", all_tracking_errors_array)

    # print time
    print("Time: ", time.time()-start)

    print(optimal_result_array.shape)
    print(all_tracking_errors_array.shape)
    print(optimal_result_array)
    print(all_tracking_errors_array)


    #VALERIOS CODE#

    # Prepare data
    kp0_values_array = np.array(kp0_values).reshape(-1, 1)
    kd0_values_array = np.array(kd0_values).reshape(-1, 1)
    
    

    print(optimal_result.shape)
    print(tracking_errors_array.shape)


    # Fit GP models
    gp_kp0 = fit_gp_model_1d(kp0_values_array, tracking_errors_array)
    gp_kd0 = fit_gp_model_1d(kd0_values_array, tracking_errors_array)

    # Plot the results
    plot_tracking_error(tracking_errors)
    plot_gp_results_1d(kp0_values_array, kd0_values_array, tracking_errors_array, gp_kp0, gp_kd0)


    print(f"Optimal Kp: {optimal_result[0]} \tOptimal Kd: {optimal_result[7]} \tOptimal tracking error: {optimal_result[-1]}")

if __name__ == "__main__":
    main()