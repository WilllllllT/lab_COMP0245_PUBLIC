import re
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)
        # print (tau_mes.shape)      

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # TODO Compute regressor and store it
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        # print(cur_regressor.shape)

        # Store regressor and measured torque
        tau_mes_all.append(tau_mes)
        regressor_all.append(cur_regressor) #7x70
        
        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")


    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    # Stack all regressor matrices and measured torques, using np.vstack and np.hstack to match the dimensions
    shift = 100 # Set a initial time shift for the start of the data collection (ms)
    regressor_all = np.vstack(regressor_all[shift + 1:]) # 70000x70
    tau_mes_all = np.hstack(tau_mes_all[shift + 1:]) # 70000x1

    print (regressor_all.shape) # Check the shape of the stacked regressor matrix
    print (tau_mes_all.shape) # Check the shape of the stacked torque matrix

    # Compute the dynamic parameters using pseudoinverse
    a = np.linalg.pinv(regressor_all) @ tau_mes_all # 70x1
    print(f"Estimated parameters: {a}") 
    
    # TODO compute the metrics for the linear model
    # Predicted torques using the estimated parameters
    tau_pred = regressor_all @ a # 70000x1

    # Compute metrics
    rss = np.sum((tau_mes_all - tau_pred) ** 2)
    tss = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2)
    r2 = 1 - (rss / tss)
    adjusted_r2 = 1 - (1 - r2) * (len(tau_mes_all) - 1) / (len(tau_mes_all) - len(a) - 1)
    f_statistic = (tss - rss) / len(a) / (rss / (len(tau_mes_all) - len(a) - 1)) # 
    
    #confidence interval
    mse = rss / len(tau_mes_all -1) # -1 to prevent division by zero
    se = np.sqrt(abs(np.diag(mse * np.linalg.pinv(regressor_all.T @ regressor_all)))) # 70x1, use abs to prevent negative values, and mse to replace standard deviation
    conf_interval = 1.96 * se

    # Plot the torque prediction error for each joint in separate subplots
    # Determine the number of rows for the two-column layout
    noise_level = dyn_model.GetConfigurationVariable("robot_noise")[0]["joint_cov"]

    plt.errorbar(np.arange(len(a)), a, yerr=1.96 * se, fmt='o', capsize=5, label='Estimated Parameters')
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title(f'Estimated System Parameters with 95% of Confidence Interval (noise_level = {noise_level})')
    plt.savefig(f'Estimated System Parameters with 95% of Confidence Interval (noise_level = {noise_level}).png', dpi=300)
    plt.show()

    # prediction interval
    pred_se = []
    for i in range(len(regressor_all) // num_joints):
        x0 = tau_mes_all[i * num_joints : (i + 1) * num_joints].reshape(-1, 1) # 7x1
        X = regressor_all[i * num_joints : (i + 1) * num_joints] # 7x70
        inv = np.linalg.inv(X @ X.T) #7X7
        pred_se.append(np.sqrt(mse * (1 + x0.T @ inv @ x0))) # 1x1
    pred_se = np.vstack(pred_se) # 10000x1
    pred_interval = (tau_pred.reshape(-1, num_joints) - 1.96 * pred_se, tau_pred.reshape(-1, num_joints) + 1.96 * pred_se)

    # pred_interval = 1.96 * np.sqrt(mse * (1 + np.diag(tau_stack.T @ np.linalg.pinv(regressor_stack.T @ regressor_stack) @ tau_stack)))

    print(f"Adjusted R-squared: {adjusted_r2}")
    print(f"F-statistic: {f_statistic}")
    print(f"Mean squared error: {mse}")
    print(f"Standard error: {se}")
    print(f"Confidence interval: {conf_interval}")
    print(f"Prediction interval: {pred_interval}")


    # TODO plot the torque prediction error for each joint (optional)
   
    # Plot torque prediction error for each joint
    torque_error = tau_mes_all - tau_pred

    # Plot the torque prediction error for each joint
    time_steps = len(torque_error) // num_joints  # Total number of time steps

    # #Start plotting after 100 time steps
    x_values = np.arange(shift, time_steps + shift)  # Time steps starting from 100
    # # x_values = np.arange(100, 100 + (time_steps - time_offset) * time_step * 1000, time_step * 1000)
    # # x_values = np.linspace(100, 100 + (time_steps - time_offset - 1) * time_step * 1000, num=time_steps - time_offset)
    # initial_tick = 100
    # interval = 2000 # Define your desired interval for the x-axis ticks
    # tick_positions = np.arange(0, x_values[-1] + interval - initial_tick, interval)
    # tick_positions[0] = initial_tick
    # print (tick_positions)

    # Reshape the torque error array to have a shape (time_steps, num_joints) for easier plotting
    torque_error_reshaped = np.reshape(torque_error, (time_steps, num_joints))

    rows = (num_joints + 1) // 2  # Adding 1 ensures that odd numbers get an extra row

    plt.figure(figsize=(13, 9))
    for joint_idx in range(num_joints):
        plt.subplot(rows, 2, joint_idx + 1)  # Use a two-column layout
        # plt.plot(torque_error_reshaped[:, joint_idx])
        plt.plot(x_values, torque_error_reshaped[:, joint_idx])
        plt.xlabel(f'Time Steps (starting from $t_{{{100}}}$)')
        plt.ylabel(f'Joint {joint_idx + 1} Error')
        plt.title(f'Torque Prediction Error for Joint {joint_idx + 1}')
        plt.tight_layout()
    plt.legend()
    plt.suptitle(f'Torque Prediction Error for Each Joint since $t_{{{100}}}$ (noise_level = {noise_level})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
    plt.savefig(f'Torque Prediction Error for Each Joint since $t_{{{100}}}$ (noise_level = {noise_level}).png', dpi=300)
    plt.show()

    # plot measured torque versus predicted torque versus predicion interval
    plt.figure(figsize=(13, 9))
    for joint_idx in range(num_joints):
        plt.subplot(rows, 2, joint_idx + 1)
        # plt.plot(tau_mes_all[joint_idx::num_joints], label='Measured Torque')  # Plot only the torque for the current joint
        # plt.plot(tau_pred[joint_idx::num_joints], label='Predicted Torque')  # Plot only the predicted torque for the current joint
        plt.plot(x_values, tau_mes_all[joint_idx::num_joints], label='Measured Torque')  # Plot only the torque for the current joint
        plt.plot(x_values, tau_pred[joint_idx::num_joints], label='Predicted Torque')  # Plot only the predicted torque for the current joint
        plt.fill_between(x_values, pred_interval[0][:, joint_idx], pred_interval[1][:, joint_idx], alpha=0.5, label='Prediction Interval', color = 'gray') 
        plt.xlabel(f'Time Steps (starting from $t_{{{shift}}}$)')
        plt.ylabel('Torque')
        plt.title(f'Measured Torque vs Predicted Torque for Joint {joint_idx + 1}')
        # plt.xticks(tick_positions)
        plt.tight_layout()
    plt.legend()
    plt.suptitle(f'Measured Torque vs Predicted Torque for Each Joint since $t_{{{100}}}$ (noise = {noise_level})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
    plt.savefig(f'Measured Torque vs Predicted Torque for Each Joint since $t_{{{100}}}$ (noise = {noise_level}).png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
