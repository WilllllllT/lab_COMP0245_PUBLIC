import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def get_syntetic_data(m = 1.0, b = 10, k_p = 50, k_d = 10, dt = 0.01, num_samples = 1000, batch_size = 32):
    # Generate synthetic data for trajectory tracking
    t = np.linspace(0, 10, num_samples)
    q_target = np.sin(t)
    dot_q_target = np.cos(t)

    # Initial conditions for training data generation
    q = 0
    dot_q = 0
    X = []
    Y = []

    for i in range(num_samples):
        # PD control output
        tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
        # Ideal motor dynamics (variable mass for realism)
        #m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
        ddot_q_real = (tau - b * dot_q) / m

        # Calculate error
        ddot_q_ideal = (tau) / m
        ddot_q_error = ddot_q_ideal - ddot_q_real

        # Store data
        X.append([q, dot_q, q_target[i], dot_q_target[i]])
        Y.append(ddot_q_error)

        # Update state
        dot_q += ddot_q_real * dt
        q += dot_q * dt

    # Convert data for PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    # Dataset and DataLoader
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, t, q_target, dot_q_target

def train_model(model, train_loader, optimizer, criterion, epochs = 1000):
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}')

    return train_losses

def test_model(model, t, dt, q_target, dot_q_target, k_p, k_d, m, b):
    q_real = []
    q_real_corrected = []

    q_test = 0
    dot_q_test = 0
    # integration with only PD Control
    for i in range(len(t)):
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        ddot_q_real = (tau - b * dot_q_test) / m
        dot_q_test += ddot_q_real * dt
        q_test += dot_q_test * dt
        q_real.append(q_test)

    q_test = 0
    dot_q_test = 0
    for i in range(len(t)):
        # Apply MLP correction
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32)
        correction = model(inputs.unsqueeze(0)).item()
        ddot_q_corrected =(tau - b * dot_q_test + correction) / m
        dot_q_test += ddot_q_corrected * dt
        q_test += dot_q_test * dt
        q_real_corrected.append(q_test)

    return q_real, q_real_corrected

# MLP Model Definition
class ShallowCorrectorMLP(nn.Module):
    def __init__(self, hidden_size=64):  # Add hidden_size parameter with default value
        super(ShallowCorrectorMLP, self).__init__()
        self.hidden_size = hidden_size
        self._build_model()  # Call the model-building function

    def _build_model(self):
        # Use self.hidden_size to set the hidden layer size
        self.layers = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        print(f'Model built with hidden size: {self.hidden_size}')

    def set_hidden_size(self, hidden_size):
        # Update the hidden layer size and rebuild the model
        self.hidden_size = hidden_size
        self._build_model()

    def forward(self, x):
        return self.layers(x)
    
class DeepCorrectorMLP(nn.Module):
    def __init__(self, hidden_size=[64, 64]):  # Add hidden_size parameter with default value
        super(DeepCorrectorMLP, self).__init__()
        self.hidden_size = hidden_size[0], hidden_size[1]
        self._build_model()  # Call the model-building function

    def _build_model(self):
        # Use self.hidden_size to set the hidden layer size
        self.layers = nn.Sequential(
            nn.Linear(4, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1)
        )
        print(f'Model built with hidden size: {self.hidden_size}')

    def set_hidden_size(self, hidden_size):
        # Update the hidden layer size and rebuild the model
        self.hidden_size = hidden_size[0], hidden_size[1]
        self._build_model()

    def forward(self, x):
        return self.layers(x)
    

def plot_results(t, q_target, q_real, q_real_corrected):
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, q_target, 'r-', label='Target')
    plt.plot(t, q_real, 'b--', label='PD Only')
    plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction')
    plt.title('Trajectory Tracking with and without MLP Correction')
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.legend()
    plt.show()

def plot_results_all(t_all, q_target_all, q_real_all, q_real_corrected_all):
    # Plot results
    plt.figure(figsize=(12, 6))
    for i in range(len(q_real_all)):
        plt.plot(t_all[i], q_target_all[i], 'r-', label='Target, hidden size = ' + str((i+1)*32))
        plt.plot(t_all[i], q_real_all[i], 'b--', label='PD Only, hidden size = ' + str((i+1)*32))
        plt.plot(t_all[i], q_real_corrected_all[i], 'g:', label='PD + MLP Correction, hidden size = ' + str((i+1)*32))
    plt.title('Trajectory Tracking with and without MLP Correction')
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.legend()
    plt.show()

def plot_loss(train_losses):
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses)
    plt.grid()
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def plot_loss_all(train_losses_all):
    # Plot loss
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(train_losses_all[i],'-', label='hidden size = ' + str(1))
    plt.grid()
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def plot_prediction_accuracy(t, q_target, q_real, q_real_corrected):
    # Calculate prediction accuracy
    error_pd = np.abs(q_target - q_real)
    error_corrected = np.abs(q_target - q_real_corrected)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, error_pd, 'b--', label='PD Only')
    plt.plot(t, error_corrected, 'g:', label='PD + MLP Correction')
    plt.title('Prediction Accuracy with and without MLP Correction')
    plt.xlabel('Time [s]')
    plt.ylabel('Position Error')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    # Constants
    m = 1.0  # Mass (kg)
    b = 10  # Friction coefficient
    k_p = 50  # Proportional gain
    k_d = 10   # Derivative gain
    dt = 0.01  # Time step
    num_samples = 1000  # Number of samples in dataset

    # Generate synthetic data for trajectory tracking
    train_loader, t, q_target, dot_q_target = get_syntetic_data(m, b, k_p, k_d, dt, num_samples)

    t_all = []
    q_target_all = []
    q_real_all = []
    q_real_corrected_all = []
    train_losses_all = []

    for i in [1.0, 0.01, 0.001, 0.0001, 0.00001]:
        # Model, Loss, Optimizer
        model = DeepCorrectorMLP()       
        # model.set_hidden_size([32,128])
        model.set_hidden_size([128,128])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=i)

        # Training Loop
        epochs = 1000

        #train losses
        train_losses = train_model(model, train_loader, optimizer, criterion, epochs)
        
        # integration with only PD Control
        q_real, q_real_corrected = test_model(model, t, dt, q_target, dot_q_target, k_p, k_d, m, b)

        t_all.append(t)
        q_target_all.append(q_target)
        q_real_all.append(q_real)
        q_real_corrected_all.append(q_real_corrected)
        train_losses_all.append(train_losses)

    plot_results_all(t_all, q_target_all, q_real_all, q_real_corrected_all)
    plot_loss_all(train_losses_all)

    plot_results(t, q_target, q_real, q_real_corrected)
    plot_loss(train_losses)
    plot_prediction_accuracy(t, q_target, q_real, q_real_corrected)
    
    return 0

if __name__ == '__main__':
    main()